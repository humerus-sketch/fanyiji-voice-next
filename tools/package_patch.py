"""Build a schema-1 program overlay against the published 0.5.0 baseline.

The 0.5.0 client accepts replacement files, not binary deltas or baseline
constraints. Keep future patches cumulative against this same baseline until
all supported clients can enforce a different baseline themselves.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path, PurePosixPath
import re
import sys

TOOLS = Path(__file__).resolve().parent
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))
import package_release as release
from fanyiji_next import app_update as updater

BASELINE_VERSION = "0.5.0"
BASELINE_MANIFEST_SHA256 = "55144ddc2cf81bb044432b4aa38c7ee44b20c7324b24fa4455c0440dd7985445"


def _no_links(path: Path) -> None:
    if any(part.is_symlink() or part.is_junction() for part in (path, *path.parents)):
        raise ValueError(f"不允许符号链接或目录联接：{path}")


def _relative(value: str) -> PurePosixPath:
    if not isinstance(value, str) or not value or any(c in value for c in '\\:\x00<>"|?*'):
        raise ValueError("基线清单包含非法路径")
    path = PurePosixPath(value)
    if (path.is_absolute() or path.as_posix() != value or
            any(part in {".", "..", ""} or part.endswith((".", " ")) or
                updater._DEVICE_NAME.fullmatch(part) or any(ord(c) < 32 for c in part)
                for part in path.parts)):
        raise ValueError(f"基线清单包含不安全路径：{value}")
    return path


def _unique_keys(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError("JSON 包含重复字段")
        value[key] = item
    return value


def _read_json(path: Path):
    _no_links(path)
    if not path.is_file() or path.stat().st_size > 16 * 1024**2:
        raise ValueError(f"JSON 文件不存在或过大：{path}")
    return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_unique_keys)


def _is_release_extra(relative: str) -> bool:
    return (relative in {"release-defaults.json", "公开包说明.md", release.UPDATE_MANIFEST}
            or relative.startswith(("runtime/", "licenses/model-notices/")))


def _validate_program_path(relative: str) -> None:
    release.safe_update_path(relative)
    updater._program_path(updater._safe_path(relative))


def _baseline(folder: Path, manifest: Path, expected_sha256: str, expected_version: str):
    if not re.fullmatch(r"[a-f0-9]{64}", expected_sha256):
        raise ValueError("基线清单 SHA256 无效")
    _no_links(manifest)
    if release.sha256(manifest) != expected_sha256:
        raise ValueError("基线 full-manifest.json 与指定发布 SHA256 不匹配")
    value = _read_json(manifest)
    if (not isinstance(value, dict) or set(value) != {"schema_version", "application", "version", "files"}
            or type(value["schema_version"]) is not int or value["schema_version"] != 1
            or value["application"] != release.APPLICATION or value["version"] != expected_version
            or not isinstance(value["files"], list) or not value["files"]):
        raise ValueError("基线发布清单的产品、版本或格式不匹配")
    listed, case_paths = {}, {}
    for row in value["files"]:
        if (not isinstance(row, dict) or set(row) != {"path", "sha256", "size"}
                or type(row["size"]) is not int or row["size"] < 0
                or not isinstance(row["sha256"], str) or not re.fullmatch(r"[a-f0-9]{64}", row["sha256"])):
            raise ValueError("基线发布文件项无效")
        path = _relative(row["path"])
        for length in range(1, len(path.parts) + 1):
            spelling = "/".join(path.parts[:length])
            key = spelling.casefold()
            if key in case_paths and case_paths[key] != spelling:
                raise ValueError("基线清单含大小写冲突")
            case_paths[key] = spelling
        if row["path"] in listed:
            raise ValueError("基线清单包含重复文件")
        listed[row["path"]] = row
    actual = {file.relative_to(folder).as_posix(): file for file in release.regular_files(folder)}
    if set(actual) != set(listed):
        raise ValueError("基线目录文件与发布 full-manifest.json 不一致")
    verified, model_count = {}, 0
    for relative, file in actual.items():
        model_root = PurePosixPath(relative).parts[0].casefold()
        if model_root == "models" or model_root.endswith("_models"):
            model_count += 1  # Model contents are deliberately not reread or copied.
            continue
        _validate_program_path(relative)
        row = listed[relative]
        if file.stat().st_size != row["size"] or release.sha256(file) != row["sha256"]:
            raise ValueError(f"基线程序文件与发布清单不匹配：{relative}")
        verified[relative] = row
    info = _read_json(folder / "build-info.json")
    if (not isinstance(info, dict) or info.get("application") != release.APPLICATION
            or info.get("version") != expected_version):
        raise ValueError("基线 build-info.json 版本或产品不匹配")
    if "FanyiJiVoiceNext.exe" not in verified or "_updater/FanyiJiUpdate.exe" not in verified:
        raise ValueError("基线缺少程序或已安装的独立更新助手")
    return verified, model_count


def package_patch(*, program_folder: Path, baseline_folder: Path, baseline_manifest: Path,
                  output: Path, version: str, github_repo: str = updater.DEFAULT_REPOSITORY,
                  baseline_version: str = BASELINE_VERSION,
                  baseline_manifest_sha256: str = BASELINE_MANIFEST_SHA256) -> dict:
    """Validate immutable inputs before creating a fresh, model-free overlay."""
    try:
        if updater._version_key(version) <= updater._version_key(baseline_version):
            raise ValueError("补丁版本必须高于基线版本")
    except updater.UpdateError as exc:
        raise ValueError("补丁或基线版本无效") from exc
    if not isinstance(github_repo, str) or not updater._REPOSITORY.fullmatch(github_repo):
        raise ValueError("GitHub 仓库必须为 owner/repo")
    program_folder, baseline_folder, baseline_manifest, output = (
        Path(path).absolute() for path in (program_folder, baseline_folder, baseline_manifest, output))
    release.assert_build_input(program_folder, purpose="patch program folder")
    release.assert_build_input(baseline_folder, purpose="patch baseline program")
    for path in (program_folder, baseline_folder, baseline_manifest, output):
        _no_links(path)
    if output.exists():
        raise FileExistsError(f"输出目录已存在，请指定新目录：{output}")
    if any(output.resolve().is_relative_to(folder.resolve()) for folder in (program_folder, baseline_folder)):
        raise ValueError("输出目录不能位于输入目录内")
    if program_folder.resolve() == baseline_folder.resolve():
        raise ValueError("新构建与基线必须为不同目录")
    files = list(release.regular_files(program_folder))
    spellings = {}
    for file in files:
        relative = file.relative_to(program_folder).as_posix()
        if relative == release.UPDATE_MANIFEST:
            raise ValueError("新构建不得带入旧的 update-manifest.json")
        _validate_program_path(relative)
        updater._case_path(PurePosixPath(relative), spellings)
        # The build has no model directory: do not silently skip weights here.
        model_root = PurePosixPath(relative).parts[0].casefold()
        if model_root == "models" or model_root.endswith("_models"):
            raise ValueError("补丁构建不得包含模型")
        if relative == "_internal/certifi/cacert.pem":
            pem = file.read_bytes()
            if b"PRIVATE KEY" in pem.upper() or b"-----BEGIN CERTIFICATE-----" not in pem:
                raise ValueError("certifi 公共证书文件含非证书内容或私钥")
    new_info = _read_json(program_folder / "build-info.json")
    if (not isinstance(new_info, dict) or new_info.get("application") != release.APPLICATION
            or new_info.get("version") != version):
        raise ValueError("新构建版本或产品与补丁不一致")
    baseline, model_count = _baseline(baseline_folder, baseline_manifest,
                                      baseline_manifest_sha256, baseline_version)
    new = {}
    for file, relative in release._program_files(program_folder):
        new[relative] = {"path": relative, "size": file.stat().st_size, "sha256": release.sha256(file)}
    if "FanyiJiVoiceNext.exe" not in new:
        raise ValueError("新构建缺少 FanyiJiVoiceNext.exe")
    required_base = {name for name in baseline if not _is_release_extra(name)}
    removed = sorted(required_base - set(new))
    if removed:
        raise ValueError("新构建删除了基线程序文件，须人工评估，不能制作覆盖补丁：" + ", ".join(removed))
    changed = [name for name, row in new.items() if name not in baseline or row != baseline[name]]
    included = sorted(set(changed) | {"FanyiJiVoiceNext.exe"})
    unchanged = sorted(set(new) - set(changed))
    preserved = sorted(set(baseline) - set(new))
    output.mkdir(parents=True, exist_ok=False)
    patch_folder = output / "program-patch"
    patch_folder.mkdir()
    for relative in included:
        release._copy_file(program_folder / relative, program_folder, patch_folder, Path(relative))
        copied = patch_folder / relative
        if copied.stat().st_size != new[relative]["size"] or release.sha256(copied) != new[relative]["sha256"]:
            raise ValueError(f"打包期间新构建发生变化：{relative}")
    update = release.create_update_zip(patch_folder, output / f"FanyiJiVoiceNext-{version}-update.zip", version)
    # Validate against the unchanged client schema, not an invented delta schema.
    updater._manifest((patch_folder / release.UPDATE_MANIFEST).read_bytes(), version)
    update["url"] = f"https://github.com/{github_repo}/releases/download/v{version}/{update['name']}"
    release.write_json(output / "update-feed.json", {"schema_version": 1, "application": release.APPLICATION,
                       "platform": updater.PLATFORM, "version": version, "program": update})
    report = {"schema_version": 1, "application": release.APPLICATION, "version": version,
              "baseline_version": baseline_version, "baseline_manifest_sha256": baseline_manifest_sha256,
              "program_update": update, "package_kind": "cumulative_program_overlay",
              "baseline_constraints_enforced_by_old_client": False,
              "baseline_verified_program_file_count": len(baseline),
              "baseline_program_files": [baseline[name] for name in sorted(baseline)],
              "included_program_files": [new[name] for name in included],
              "changed_program_file_count": len(changed), "added_program_files": sorted(set(new) - set(baseline)),
              "unchanged_program_file_count": len(unchanged),
              "unchanged_program_bytes": sum(new[name]["size"] for name in unchanged),
              "preserved_release_files": preserved, "preserved_model_file_count": model_count,
              "model_weights_rehashed": False, "models_copied": False, "runtime_copied": any(
                  name.startswith("runtime/") for name in included), "removed_program_files": [],
              "baseline_note": "Requires the complete published baseline; the 0.5.0 updater does not check baseline hashes.",
              "rollback_note": "Copy/start errors roll back touched files; a child that starts then crashes is not automatically rolled back."}
    release.write_json(output / "patch-report.json", report)
    checksum_files = [output / update["name"], output / "update-feed.json", output / "patch-report.json"]
    (output / "SHA256SUMS.txt").write_text("".join(f"{release.sha256(path)}  {path.name}\n" for path in checksum_files),
                                         encoding="utf-8")
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="制作兼容 0.5.0 更新器的累积程序小补丁，不复制模型")
    for name in ("program-folder", "baseline-folder", "baseline-manifest", "output"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--version", required=True)
    parser.add_argument("--github-repo", default=updater.DEFAULT_REPOSITORY)
    parser.add_argument("--baseline-version", default=BASELINE_VERSION)
    parser.add_argument("--baseline-manifest-sha256", default=BASELINE_MANIFEST_SHA256)
    args = parser.parse_args(argv)
    report = package_patch(**vars(args))
    print(json.dumps({key: report[key] for key in ("version", "baseline_version", "program_update",
                     "changed_program_file_count", "unchanged_program_file_count")}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
