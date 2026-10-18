"""Build on an ephemeral Windows Actions runner; upload to a new draft only.

No models are downloaded. Cleanup is limited to this run's marked temporary
directory and never includes the source checkout or the running base Python.
The stable release, existing attachments, and user installations are untouched.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import importlib.util
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import subprocess
import sys
import tempfile
import time
import tomllib
from urllib.request import Request, urlopen
import zipfile

ROOT = Path(__file__).resolve().parents[1]
REPOSITORY = "humerus-sketch/fanyiji-voice-next"
APPLICATION = "fanyiji-voice-next"
GIB = 1024**3
PART_SIZE = 1536 * 1024**2
LLAMA_COMMIT = "ceefd16daa2093730bf3340315fa2b84b286ddbc"
LLAMA_ASSETS = (
    ("llama-b11696-bin-win-cuda-12.4-x64.zip", 266794276,
     "a4cc6b5bc50ac057625f24ab9aa63ed497836ba5dcd10d9552954b73755b4011"),
    ("cudart-llama-bin-win-cuda-12.4-x64.zip", 391443627,
     "8c79a9b226de4b3cacfd1f83d24f962d0773be79f1e7b75c6af4ded7e32ae1d6"),
)
OLD_CLIENT_COMMIT = "7b466fb1c396704aab5aa82ff483a78936636d9d"
CI_PYTHON_VERSION = (3, 12, 10)
PYTHON_MANIFEST = "https://raw.githubusercontent.com/actions/python-versions/main/versions-manifest.json"
BOOTSTRAP_FILES = {"FanyiJiVoiceNext.exe", "_updater/FanyiJiUpdate.exe", "README.md",
                   "build-info.json", "release-defaults.json", "_updater/runtime-bootstrap.json"}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def save(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def run(command, *, cwd=None, capture=False, timeout=None):
    result = subprocess.run([str(item) for item in command], cwd=cwd, check=True,
                            capture_output=capture, text=True, encoding="utf-8", timeout=timeout)
    return result.stdout.strip() if capture else None


def state(root):
    root = Path(root).absolute()
    if root.is_symlink() or root.is_junction() or not (root / "state.json").is_file():
        raise ValueError("Missing owned CI workspace marker")
    value = json.loads((root / "state.json").read_text(encoding="utf-8"))
    if value.get("owned_root") != str(root.resolve()) or value.get("run_id") != os.environ.get("GITHUB_RUN_ID"):
        raise ValueError("Workspace identity does not match this Actions run")
    return root, value


def disk(root, phase, required):
    usage = shutil.disk_usage(root)
    report = {"phase": phase, "free": usage.free, "required": required, "total": usage.total,
              "documentation_only_guarantees_14_gb_for_standard_windows": True}
    save(root / f"disk-{phase}.json", report)
    print(json.dumps(report), flush=True)
    if usage.free < required:
        raise RuntimeError(f"Insufficient runner disk at {phase}: {usage.free / GIB:.2f} GiB free; "
                           f"need {required / GIB:.2f} GiB. No paid runner or host cleanup was requested.")


def remove_owned(root, target):
    root = root.resolve()
    target = Path(target).absolute()
    if target.is_symlink() or target.is_junction():
        raise ValueError("Cleanup refuses links or junctions")
    target = target.resolve()
    if (not target.is_relative_to(root) or target == root
            or Path(sys.executable).resolve().is_relative_to(target)):
        raise ValueError("Cleanup may only remove this run's owned, inactive temporary trees")
    if not target.exists():
        return
    for directory, directories, files in os.walk(target, followlinks=False):
        for name in (*directories, *files):
            path = Path(directory) / name
            if path.is_symlink() or path.is_junction() or not path.resolve().is_relative_to(root):
                raise ValueError("Refusing CI cleanup containing links or junctions")
    shutil.rmtree(target)


def prepare(root, commit, staging_tag, final_tag):
    if os.environ.get("GITHUB_ACTIONS") != "true" or os.name != "nt" or sys.version_info[:3] != CI_PYTHON_VERSION:
        raise ValueError("Build commands require a Windows Actions runner with Python 3.12.10")
    if not re.fullmatch(r"[0-9a-f]{40}", commit) or run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture=True) != commit:
        raise ValueError("Checkout must equal the exact requested immutable commit")
    version = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]["version"]
    if (version != "0.5.3" or final_tag != f"v{version}"
            or not re.fullmatch(rf"v{re.escape(version)}-r3-build(?:-[A-Za-z0-9.-]+)?", staging_tag)):
        raise ValueError("Use the 0.5.3 final URL and a NEW, independent r3 staging draft tag")
    root = Path(root).absolute()
    parents = [Path(value).resolve() for key in ("RUNNER_TEMP", "TEMP") if (value := os.environ.get(key))]
    if (root.exists() or root.is_symlink() or not any(root.parent.resolve() == parent for parent in parents)
            or not re.fullmatch(r"fanyiji-release-\d+-\d+", root.name)):
        raise ValueError("A fresh immediate child of this runner's temporary directory is required")
    root.mkdir()
    value = {"schema_version": 1, "owned_root": str(root.resolve()), "run_id": os.environ["GITHUB_RUN_ID"],
             "source_commit": commit, "version": version, "staging_tag": staging_tag, "final_tag": final_tag,
             "build_id": f"voice-ab-{version}-20261019-r3", "model_weights_bundled": False,
             "python_version": sys.version.split()[0], "python_version_manifest": PYTHON_MANIFEST,
             "python_patch_selection": "3.12.13 has no Windows x64 Actions asset; use official 3.12.10 Windows x64",
             "public_download_or_install_verified": False}
    save(root / "state.json", value)
    # The actual runner may have more space than the documented 14 GB. Measure
    # before downloading wheels; never free space by deleting host software.
    disk(root, "before-dependencies", 45 * GIB)
    source = root / "source"
    source.mkdir()
    tracked = subprocess.check_output(["git", "ls-files", "-z"], cwd=ROOT).decode("utf-8").split("\0")
    for relative in filter(None, tracked):
        path = PurePosixPath(relative)
        original = ROOT / path
        if (path.is_absolute() or ".." in path.parts or original.is_symlink() or original.is_junction()
                or not original.is_file() or not original.resolve().is_relative_to(ROOT.resolve())):
            raise ValueError(f"Non-regular tracked source: {relative}")
        destination = source / path
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(original, destination)
    (root / "release").mkdir()
    print(f"Prepared exact source {commit}; no weights or user files copied", flush=True)


def download_pinned(url, destination, expected_size, expected_sha):
    if destination.exists():
        raise FileExistsError(destination)
    digest, length = hashlib.sha256(), 0
    with urlopen(Request(url, headers={"User-Agent": "FanyiJi-release-CI"}), timeout=120) as response, destination.open("xb") as output:
        for block in iter(lambda: response.read(1024**2), b""):
            length += len(block)
            if length > expected_size:
                raise ValueError("Official runtime exceeds pinned size")
            digest.update(block)
            output.write(block)
    if length != expected_size or digest.hexdigest() != expected_sha:
        raise ValueError(f"Pinned official runtime size/SHA256 mismatch: {destination.name}")


def llama(root):
    root, value = state(root)
    runtime = root / "source/runtime/llama-b11696-cuda12.4"
    runtime.mkdir(parents=True)
    downloads = root / "llama-downloads"
    downloads.mkdir()
    collected = {}
    for name, size, checksum in LLAMA_ASSETS:
        archive = downloads / name
        download_pinned(f"https://github.com/ggml-org/llama.cpp/releases/download/b11696/{name}", archive, size, checksum)
        with zipfile.ZipFile(archive) as bundle:
            for entry in bundle.infolist():
                path = PurePosixPath(entry.filename)
                if (entry.is_dir() or path.is_absolute() or ".." in path.parts or "\\" in entry.filename
                        or stat.S_ISLNK(entry.external_attr >> 16)):
                    continue
                if path.suffix.casefold() != ".dll" and path.name != "llama-server.exe" and not path.name.upper().startswith("LICENSE"):
                    continue
                destination = runtime / path.name
                digest = hashlib.sha256()
                if destination.exists():
                    with bundle.open(entry) as stream:
                        for block in iter(lambda: stream.read(1024**2), b""):
                            digest.update(block)
                    if sha256(destination) != digest.hexdigest():
                        raise ValueError(f"Official archives contain conflicting runtime file: {path.name}")
                    continue
                with bundle.open(entry) as stream, destination.open("xb") as output:
                    shutil.copyfileobj(stream, output, 1024**2)
                collected[path.name] = sha256(destination)
    if not (runtime / "llama-server.exe").is_file():
        raise ValueError("Pinned official archives lack llama-server.exe")
    # Source license is bound to the exact official b11696 commit, not main.
    license_path = runtime / "LICENSE-llama.cpp.txt"
    if license_path.exists():
        raise ValueError("Unexpected duplicate llama source license")
    with urlopen(f"https://raw.githubusercontent.com/ggml-org/llama.cpp/{LLAMA_COMMIT}/LICENSE", timeout=60) as response:
        content = response.read(128 * 1024)
    if b"MIT License" not in content or b"Permission is hereby granted" not in content:
        raise ValueError("Pinned official llama source license is missing or invalid")
    license_path.write_bytes(content)
    collected[license_path.name] = sha256(license_path)
    save(root / "release/llama-runtime-provenance-r3.json", {"commit": LLAMA_COMMIT, "files": collected,
         "archives": [{"name": name, "size": size, "sha256": checksum} for name, size, checksum in LLAMA_ASSETS]})
    remove_owned(root, downloads)
    print(f"Verified official llama runtime: {len(collected)} files", flush=True)


def build(root):
    root, value = state(root)
    source = root / "source"
    python = root / "venv/Scripts/python.exe"
    disk(root, "before-freeze", 14 * GIB)
    run([python, "-m", "pytest", "-q"], cwd=source, timeout=1800)
    run([python, source / "tools/build.py", "--edition", "voice", "--with-updater", "--with-ab-engines",
         "--output", root / "frozen"], cwd=source, timeout=3600)
    checks = root / "frozen/checks"
    for filename in ("runtime.json", "ui.json"):
        report = json.loads((checks / filename).read_text(encoding="utf-8"))
        if report.get("audio_capture") is not False or report.get("settings_written") is not False:
            raise ValueError("Frozen checks did not prove audio/settings isolation")
    value["source_tests_passed"] = True
    value["frozen_checks_passed"] = True
    save(root / "state.json", value)


def modules(source):
    sys.path.insert(0, str(source / "src"))
    sys.path.insert(0, str(source / "tools"))
    import package_release
    from fanyiji_next.app_update import _manifest, parse_runtime_bundle_descriptor
    from fanyiji_next.release_profile import write_release_defaults
    return package_release, _manifest, parse_runtime_bundle_descriptor, write_release_defaults


def split_archive(archive, *, repository, version, part_size=PART_SIZE):
    parts, whole = [], hashlib.sha256()
    with archive.open("rb") as stream:
        remaining = archive.stat().st_size
        while remaining:
            path = archive.with_name(f"{archive.name}.{len(parts) + 1:03d}")
            digest, size = hashlib.sha256(), min(remaining, part_size)
            with path.open("xb") as output:
                pending = size
                while pending:
                    block = stream.read(min(pending, 4 * 1024**2))
                    if not block:
                        raise ValueError("Runtime archive changed during split")
                    output.write(block)
                    digest.update(block)
                    whole.update(block)
                    pending -= len(block)
            parts.append({"name": path.name, "size": size, "sha256": digest.hexdigest(),
                          "url": f"https://github.com/{repository}/releases/download/v{version}/{path.name}"})
            remaining -= size
    return parts, whole.hexdigest()


def package(root):
    root, value = state(root)
    if value.get("frozen_checks_passed") is not True or value.get("source_tests_passed") is not True:
        raise ValueError("Source and frozen checks must pass before packaging")
    source, output = root / "source", root / "release"
    release, manifest_parser, descriptor_parser, defaults = modules(source)
    folder = root / "frozen/FanyiJiVoiceNext"
    bootstrap = root / "bootstrap"
    bootstrap.mkdir()
    for relative in ("FanyiJiVoiceNext.exe", "_updater/FanyiJiUpdate.exe", "build-info.json"):
        destination = bootstrap / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(folder / relative, destination)
    shutil.copy2(source / "README.md", bootstrap / "README.md")
    # Base Python runs this phase. PyInstaller has finished; removing only the
    # owned venv/intermediates prevents a second 7 GB program-tree copy.
    remove_owned(root, source / "build")
    remove_owned(root, root / "venv")
    defaults(folder / "release-defaults.json")
    info = json.loads((folder / "build-info.json").read_text(encoding="utf-8"))
    info.update(portable_release=True, model_download_required=True, model_weights_bundled=False,
                user_settings_bundled=False, llama_vc_runtime_bundled=True,
                ci_source_commit=value["source_commit"], runtime_build_id=value["build_id"])
    save(folder / "build-info.json", info)
    marker = {"schema_version": 1, "application": APPLICATION, "version": value["version"], "build_id": value["build_id"]}
    save(folder / "_updater/runtime-ready.json", marker)
    public_files = list(release.regular_files(folder))
    for path in public_files:
        relative = path.relative_to(folder).as_posix()
        release.safe_update_path(relative)
        if not ((len(PurePosixPath(relative).parts) == 1 and relative in release.PROGRAM_ROOT_FILES | {"release-defaults.json"})
                or PurePosixPath(relative).parts[0] in release.PROGRAM_ROOT_DIRS):
            raise ValueError(f"Unexpected public program path: {relative}")
    payload_size = sum(path.stat().st_size for path in public_files)
    disk(root, "before-compression", payload_size + GIB)
    archive = output / f"FanyiJiVoiceNext-{value['version']}-runtime-r3.zip"
    update = release.create_update_zip(folder, archive, value["version"])
    manifest_bytes = (folder / release.UPDATE_MANIFEST).read_bytes()
    rows = manifest_parser(manifest_bytes, value["version"])
    if len(manifest_bytes) > 2 * 1024**2 or archive.stat().st_size > 8 * GIB:
        raise ValueError("Runtime archive or manifest exceeds updater limits")
    full = json.loads(manifest_bytes)
    full["files"].append({"path": release.UPDATE_MANIFEST, "size": len(manifest_bytes),
                          "sha256": hashlib.sha256(manifest_bytes).hexdigest()})
    save(output / "full-manifest-r3.json", full)
    remove_owned(root, folder)
    disk(root, "before-split", archive.stat().st_size + GIB)
    parts, checksum = split_archive(archive, repository=REPOSITORY, version=value["version"])
    if not 1 <= len(parts) <= 16 or checksum != update["sha256"]:
        raise ValueError("Raw runtime parts do not match the original archive")
    descriptor = {"schema_version": 1, "application": APPLICATION, "platform": "windows-x64", "kind": "runtime-bundle",
                  "version": value["version"], "build_id": value["build_id"], "name": archive.name,
                  "size": archive.stat().st_size, "sha256": checksum, "parts": parts}
    descriptor_parser(descriptor, repository=REPOSITORY)
    save(output / "runtime-bundle-r3.json", descriptor)
    save(bootstrap / "_updater/runtime-bootstrap.json", descriptor)
    info.update(ab_engines_bundled=False, runtime_bootstrap_required=True)
    save(bootstrap / "build-info.json", info)
    defaults(bootstrap / "release-defaults.json")
    bootstrap_update = release.create_update_zip(bootstrap, output / f"FanyiJiVoiceNext-{value['version']}-update-r3.zip", value["version"])
    bootstrap_update["url"] = f"https://github.com/{REPOSITORY}/releases/download/{value['final_tag']}/{bootstrap_update['name']}"
    # Validate with the actual published 0.5.0 client from immutable Git history,
    # rather than assuming the current updater proves old-client compatibility.
    old_bytes = subprocess.check_output(["git", "show", f"{OLD_CLIENT_COMMIT}:src/fanyiji_next/app_update.py"], cwd=ROOT)
    old_path = root / "published-client-050.py"
    old_path.write_bytes(old_bytes)
    spec = importlib.util.spec_from_file_location("ci_published_client_050", old_path)
    old = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = old
    spec.loader.exec_module(old)
    accepted = old._manifest((bootstrap / release.UPDATE_MANIFEST).read_bytes(), value["version"])
    accepted_paths = {row["path"] for row in accepted}
    if accepted_paths != BOOTSTRAP_FILES or bootstrap_update["size"] >= old.MAX_DOWNLOAD_BYTES:
        raise ValueError("Small bootstrap is not compatible with the published 0.5.0 client")
    feed = {"schema_version": 1, "application": APPLICATION, "platform": "windows-x64",
            "version": value["version"], "program": bootstrap_update}
    save(output / "update-feed.json", feed)
    save(output / "bootstrap-package-report-r3.json", {"version": value["version"], "source_commit": value["source_commit"],
         "old_050_manifest_accepted": True, "old_client_commit": OLD_CLIENT_COMMIT, "files": sorted(accepted_paths),
         "update": bootstrap_update, "runtime_archive": checksum, "model_weights_bundled": False,
         "public_download_or_install_verified": False})
    save(output / "runtime-assembly-report-r3.json", {"version": value["version"], "build_id": value["build_id"],
         "archive": descriptor, "source_commit": value["source_commit"], "program_files": len(rows),
         "python_version": value["python_version"], "python_version_manifest": value["python_version_manifest"],
         "uncompressed_size": payload_size, "model_weights_bundled": False, "user_settings_bundled": False,
         "source_tests_passed": True, "frozen_checks_passed": True, "public_download_or_install_verified": False})
    notes = (f"# 翻译姬 NEXT {value['version']} — GitHub 原生构建候选\n\n"
             "本草稿尚未正式发布。识别、翻译和 Silero VAD 模型权重均为 0，全部在软件内按需下载。\n\n"
             "大体积附件是 Torch、CUDA、cuDNN 等程序运行库。现有客户端先安装小 bootstrap 更新，"
             "再自动下载并校验运行库分卷；保留模型和用户设置。新安装需合并全部 rawparts 后解压。\n\n"
             "提供国内源、失败切换重试、断点续传和 SHA256 校验。B 主权重没有同 SHA 的 ModelScope 国内副本，"
             "仍使用 HF-Mirror/Hugging Face，不能保证所有国内网络均可访问。\n\n"
             f"源提交：`{value['source_commit']}`。构建和离线冻结验证通过不代表最终公网更新下载、安装已通过；"
             "正式发布前还需完成最终资产地址的真实更新验证。\n\n"
             f"构建 Python：`{value['python_version']}`。Actions 官方 manifest 未提供 Python 3.12.13 Windows x64，"
             f"因此使用同一 3.12 系列中提供 Windows 安装包的 3.12.10；[官方 manifest]({PYTHON_MANIFEST})。\n")
    (output / "release-notes-r3.md").write_text(notes, encoding="utf-8")
    public = [output / part["name"] for part in parts] + [output / name for name in
              ("runtime-bundle-r3.json", "full-manifest-r3.json", "runtime-assembly-report-r3.json",
               bootstrap_update["name"], "update-feed.json", "bootstrap-package-report-r3.json", "llama-runtime-provenance-r3.json")]
    entries = [{"name": path.name, "size": path.stat().st_size, "sha256": sha256(path)} for path in public]
    sums = output / "SHA256SUMS-r3.txt"
    sums.write_text(f"{checksum}  {archive.name}\n" + "".join(f"{row['sha256']}  {row['name']}\n" for row in entries), encoding="utf-8")
    entries.append({"name": sums.name, "size": sums.stat().st_size, "sha256": sha256(sums)})
    if any(row["size"] >= 2 * GIB for row in entries):
        raise ValueError("A public GitHub asset exceeds the strict 2 GiB limit")
    save(output / "public-assets-r3.json", {"source_commit": value["source_commit"], "assets": entries})
    value.update(packaged=True, archive_sha256=checksum, part_count=len(parts))
    save(root / "state.json", value)
    print(json.dumps({"runtime_size": descriptor["size"], "parts": len(parts), "bootstrap_size": bootstrap_update["size"],
                      "model_weights": 0, "public_network_validation": "not run"}), flush=True)


def release_id_for_tag(tag):
    # REST /releases/tags/{tag} omits drafts even when the caller can read the
    # release by ID. gh release view also enumerates accessible drafts.
    result = subprocess.run(["gh", "release", "view", tag, "--repo", REPOSITORY, "--json", "databaseId"],
                            capture_output=True, text=True, encoding="utf-8")
    if result.returncode == 0:
        value = json.loads(result.stdout)
        identifier = value.get("databaseId")
        if type(identifier) is not int or identifier <= 0:
            raise ValueError("GitHub release lookup returned an invalid database ID")
        return identifier
    if "release not found" in result.stderr.casefold():
        return None
    raise RuntimeError("Cannot confirm whether the staging release exists; release creation stopped")


def upload_draft(root):
    root, value = state(root)
    if not value.get("packaged") or not os.environ.get("GH_TOKEN"):
        raise ValueError("Verified package and scoped GitHub token are required")
    output = root / "release"
    rows = json.loads((output / "public-assets-r3.json").read_text(encoding="utf-8"))["assets"]
    for row in rows:
        path = output / row["name"]
        if path.name != row["name"] or path.stat().st_size != row["size"] or sha256(path) != row["sha256"]:
            raise ValueError("Public asset changed after packaging")
    tag = value["staging_tag"]
    if release_id_for_tag(tag) is not None:
        raise ValueError("Staging release already exists; no deletion, replacement, or duplicate upload attempted")
    tag_probe = subprocess.run(["gh", "api", f"repos/{REPOSITORY}/git/ref/tags/{tag}"], capture_output=True, text=True)
    if tag_probe.returncode == 0 or "404" not in tag_probe.stderr:
        raise ValueError("Staging Git tag exists or its absence is unconfirmed; use a fresh staging name")
    run(["gh", "release", "create", tag, "--repo", REPOSITORY, "--draft", "--prerelease", "--target", value["source_commit"],
         "--title", f"NEXT {value['version']} r3 build candidate", "--notes-file", output / "release-notes-r3.md"])
    identifier = release_id_for_tag(tag)
    if identifier is None:
        raise ValueError("Created staging draft could not be located; no upload attempted")
    remote = json.loads(run(["gh", "api", f"repos/{REPOSITORY}/releases/{identifier}"], capture=True))
    if (remote["draft"] is not True or remote["tag_name"] != tag
            or remote["target_commitish"] != value["source_commit"]):
        raise ValueError("Independent staging release identity mismatch")
    value.update(draft_release_id=remote["id"], draft_release_url=remote["html_url"])
    save(root / "state.json", value)

    def upload(row):
        started = time.monotonic()
        run(["gh", "release", "upload", tag, output / row["name"], "--repo", REPOSITORY])
        print(f"Upload returned success: {row['name']} ({time.monotonic() - started:.1f}s); verifying server digest", flush=True)

    with ThreadPoolExecutor(max_workers=3) as pool:
        list(pool.map(upload, rows))
    remote = json.loads(run(["gh", "api", f"repos/{REPOSITORY}/releases/{remote['id']}"], capture=True))
    assets = {entry["name"]: entry for entry in remote["assets"]}
    for row in rows:
        asset = assets.get(row["name"], {})
        if asset.get("size") != row["size"] or asset.get("digest") != "sha256:" + row["sha256"] or asset.get("state") != "uploaded":
            raise ValueError(f"GitHub has not confirmed exact bytes/SHA256: {row['name']}")
    if remote["draft"] is not True:
        raise ValueError("Staging release unexpectedly became public")
    save(output / "draft-upload-verification-r3.json", {"release_id": remote["id"], "draft": True,
         "target_commitish": remote["target_commitish"], "confirmed_assets": rows,
         "final_asset_urls": f"https://github.com/{REPOSITORY}/releases/download/{value['final_tag']}/",
         "public_download_or_install_verified": False})
    print(f"GitHub confirmed {len(rows)} staging assets by size and SHA256; draft retained", flush=True)


def self_test():
    from unittest.mock import patch

    command = ["gh", "release", "view", "v0.5.3-r3-build", "--repo", REPOSITORY, "--json", "databaseId"]
    with patch.object(subprocess, "run", return_value=subprocess.CompletedProcess(command, 0, '{"databaseId":416604888}', "")):
        assert release_id_for_tag("v0.5.3-r3-build") == 416604888
    with patch.object(subprocess, "run", return_value=subprocess.CompletedProcess(command, 1, "", "release not found")):
        assert release_id_for_tag("v0.5.3-r3-build") is None
    for result in (subprocess.CompletedProcess(command, 1, "", "authentication failed"),
                   subprocess.CompletedProcess(command, 0, '{"databaseId":true}', "")):
        with patch.object(subprocess, "run", return_value=result):
            try:
                release_id_for_tag("v0.5.3-r3-build")
            except (RuntimeError, ValueError):
                pass
            else:
                raise AssertionError("Unknown GitHub lookup result was accepted")
    with tempfile.TemporaryDirectory(prefix="fanyiji-ci-tool-test-") as temporary:
        root = Path(temporary)
        archive = root / "synthetic.zip"
        archive.write_bytes(bytes(range(256)) * 19)
        parts, checksum = split_archive(archive, repository=REPOSITORY, version="0.5.3", part_size=1024)
        assert b"".join((root / row["name"]).read_bytes() for row in parts) == archive.read_bytes()
        assert checksum == sha256(archive) and sum(row["size"] for row in parts) == archive.stat().st_size
        assert all(sha256(root / row["name"]) == row["sha256"] for row in parts)
        assert all("/v0.5.3/" in row["url"] for row in parts)
        try:
            remove_owned(root, ROOT)
        except ValueError:
            pass
        else:
            raise AssertionError("Cleanup escaped owned CI root")
    print("CI packaging synthetic split/SHA/cleanup-boundary checks passed; no build or network operation run")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "llama", "build", "package", "upload-draft", "self-test"))
    parser.add_argument("--work-root", type=Path)
    parser.add_argument("--commit")
    parser.add_argument("--staging-tag")
    parser.add_argument("--final-tag")
    args = parser.parse_args()
    if args.command == "self-test":
        return self_test()
    if args.work_root is None:
        parser.error("--work-root is required")
    if args.command == "prepare":
        return prepare(args.work_root, args.commit, args.staging_tag, args.final_tag)
    return {"llama": llama, "build": build, "package": package, "upload-draft": upload_draft}[args.command](args.work_root)


if __name__ == "__main__":
    main()
