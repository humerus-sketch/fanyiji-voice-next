"""Program overlays keep the published runtime baseline and old-client schema."""
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import zipfile

import pytest

from fanyiji_next import app_update as updater


@pytest.fixture
def patcher():
    path = Path(__file__).resolve().parents[1] / "tools/package_patch.py"
    spec = importlib.util.spec_from_file_location("next_package_patch", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_files(root, files):
    for relative, data in files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)


def save_baseline_manifest(patcher, args, *, version="0.5.0"):
    folder = args["baseline_folder"]
    patcher.release.write_json(args["baseline_manifest"], {
        "schema_version": 1, "application": updater.APPLICATION, "version": version,
        "files": [{"path": file.relative_to(folder).as_posix(), "size": file.stat().st_size,
                   "sha256": patcher.release.sha256(file)} for file in patcher.release.regular_files(folder)]})
    args["baseline_manifest_sha256"] = patcher.release.sha256(args["baseline_manifest"])


@pytest.fixture
def inputs(tmp_path, patcher):
    baseline, program = tmp_path / "baseline/FanyiJiVoiceNext", tmp_path / "new/FanyiJiVoiceNext"
    baseline.mkdir(parents=True)
    program.mkdir(parents=True)
    files = {
        "FanyiJiVoiceNext.exe": b"MZ old program",
        "build-info.json": b'{"application":"fanyiji-voice-next","version":"0.5.0"}',
        "_internal/library.dll": b"unchanged runtime",
        "_internal/changed.dll": b"old dependency",
        "_internal/retained.pyd": b"unchanged Python extension",
        "licenses/dependency.txt": b"unchanged dependency license",
        "_updater/FanyiJiUpdate.exe": b"existing independent helper",
    }
    write_files(baseline, files)
    write_files(program, files)
    write_files(baseline, {
        "runtime/llama/server.exe": b"baseline external runtime",
        "release-defaults.json": b'{"default":"preserved"}',
        "公开包说明.md": b"public package notice",
        "licenses/model-notices/LICENSE": b"preserved model notices",
        "models/model.gguf": b"baseline model; never read by the patch builder",
        "update-manifest.json": b"old transport manifest; not the baseline authority",
    })
    write_files(program, {
        "FanyiJiVoiceNext.exe": b"MZ new program",
        "build-info.json": b'{"application":"fanyiji-voice-next","version":"0.5.1"}',
        "_internal/changed.dll": b"new dependency",
        "_internal/added.dll": b"new file",
    })
    args = dict(program_folder=program, baseline_folder=baseline,
                baseline_manifest=tmp_path / "full-manifest.json", output=tmp_path / "patch",
                version="0.5.1", github_repo="example/voice-next")
    save_baseline_manifest(patcher, args)
    return args


def test_patch_is_complete_changed_file_overlay_and_schema1(patcher, inputs, monkeypatch):
    original_sha = patcher.release.sha256

    def no_model_reads(path):
        assert "models" not in Path(path).parts, "The patch builder must not read model weights"
        return original_sha(path)

    monkeypatch.setattr(patcher.release, "sha256", no_model_reads)
    result = patcher.package_patch(**inputs)
    output = inputs["output"]
    expected = {"FanyiJiVoiceNext.exe", "build-info.json", "_internal/changed.dll", "_internal/added.dll"}
    with zipfile.ZipFile(output / result["program_update"]["name"]) as archive:
        manifest = json.loads(archive.read("update-manifest.json"))
        assert set(manifest) == {"schema_version", "application", "version", "files"}
        assert {r["path"] for r in updater._manifest(archive.read("update-manifest.json"), "0.5.1")} == expected
        assert set(archive.namelist()) == expected | {"update-manifest.json"}
        for row in manifest["files"]:
            assert hashlib.sha256(archive.read(row["path"])).hexdigest() == row["sha256"]
    feed = json.loads((output / "update-feed.json").read_text(encoding="utf-8"))
    assert set(feed) == {"schema_version", "application", "platform", "version", "program"}
    assert set(feed["program"]) == {"name", "url", "sha256", "size"}
    assert feed["program"]["url"].endswith("/v0.5.1/FanyiJiVoiceNext-0.5.1-update.zip")
    assert result["unchanged_program_file_count"] == 4
    assert result["model_weights_rehashed"] is result["models_copied"] is result["runtime_copied"] is False
    assert result["preserved_model_file_count"] == 1
    assert "runtime/llama/server.exe" in result["preserved_release_files"]
    assert "release-defaults.json" in result["preserved_release_files"]
    assert "licenses/model-notices/LICENSE" in result["preserved_release_files"]
    for line in (output / "SHA256SUMS.txt").read_text(encoding="utf-8").splitlines():
        digest, name = line.split("  ", 1)
        assert original_sha(output / name) == digest


def test_patch_applies_without_replacing_unchanged_runtime_or_models(patcher, inputs, tmp_path):
    result = patcher.package_patch(**inputs)
    installed = tmp_path / "installed"
    shutil.copytree(inputs["baseline_folder"], installed)
    item = result["program_update"]
    info = updater.UpdateInfo("0.5.1", item["name"], item["url"], item["sha256"], item["size"],
                              "https://github.com/example/voice-next/releases/tag/v0.5.1", "patch")
    prepared = updater.UpdateClient(repository="example/voice-next", cache_dir=tmp_path / "cache").prepare(
        updater.DownloadedUpdate(info, inputs["output"] / item["name"]), installed)
    backup = updater.apply_update(prepared.job_path, restart=False)
    assert (installed / "FanyiJiVoiceNext.exe").read_bytes() == b"MZ new program"
    assert (backup / "FanyiJiVoiceNext.exe").read_bytes() == b"MZ old program"
    assert (installed / "_internal/library.dll").read_bytes() == b"unchanged runtime"
    assert (installed / "models/model.gguf").read_bytes() == b"baseline model; never read by the patch builder"
    assert not (backup / "runtime/llama/server.exe").exists()
    assert json.loads((installed / "build-info.json").read_text())["version"] == "0.5.1"


def test_baseline_manifest_requires_published_digest(patcher, inputs):
    inputs["baseline_manifest_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="发布 SHA256"):
        patcher.package_patch(**inputs)
    assert not inputs["output"].exists()


@pytest.mark.parametrize("relative", ["FanyiJiVoiceNext.exe", "_internal/library.dll", "runtime/llama/server.exe"])
def test_every_non_model_baseline_file_is_verified(patcher, inputs, relative):
    (inputs["baseline_folder"] / relative).write_bytes(b"tampered baseline")
    with pytest.raises(ValueError, match="基线程序文件"):
        patcher.package_patch(**inputs)
    assert not inputs["output"].exists()


@pytest.mark.parametrize("target", ["manifest", "build-info"])
def test_baseline_version_mismatch_rejected(patcher, inputs, target):
    if target == "build-info":
        (inputs["baseline_folder"] / "build-info.json").write_text(
            '{"application":"fanyiji-voice-next","version":"0.4.8"}', encoding="utf-8")
    save_baseline_manifest(patcher, inputs, version="0.4.8" if target == "manifest" else "0.5.0")
    with pytest.raises(ValueError, match="基线.*版本"):
        patcher.package_patch(**inputs)
    assert not inputs["output"].exists()


def test_removed_program_file_requires_manual_review(patcher, inputs):
    (inputs["program_folder"] / "_internal/retained.pyd").rename(inputs["program_folder"] / "_internal/replaced.pyd")
    with pytest.raises(ValueError, match="删除了基线程序文件"):
        patcher.package_patch(**inputs)
    assert not inputs["output"].exists()


@pytest.mark.parametrize("relative", ["settings.json", "voice.settings.json", "private.key", "source/secret.txt",
                                      "models/model.gguf", "_internal/model.bin", "_internal/private.pem", "logs/audio.wav"])
def test_private_or_non_program_files_rejected_before_output(patcher, inputs, relative):
    write_files(inputs["program_folder"], {relative: b"private fixture"})
    with pytest.raises((ValueError, updater.UpdateError)):
        patcher.package_patch(**inputs)
    assert not inputs["output"].exists()


@pytest.mark.parametrize("relative", ["../escape.dll", "C:/escape.dll", "_internal/../escape.dll", "_internal//a.dll", "_internal/CON.dll"])
def test_unsafe_baseline_manifest_paths_rejected(patcher, inputs, relative):
    manifest = json.loads(inputs["baseline_manifest"].read_text(encoding="utf-8"))
    manifest["files"][0]["path"] = relative
    patcher.release.write_json(inputs["baseline_manifest"], manifest)
    inputs["baseline_manifest_sha256"] = patcher.release.sha256(inputs["baseline_manifest"])
    with pytest.raises(ValueError, match="路径"):
        patcher.package_patch(**inputs)
    assert not inputs["output"].exists()


@pytest.mark.parametrize("where", ["new", "baseline", "manifest"])
def test_links_rejected(patcher, inputs, tmp_path, where):
    if where == "manifest":
        target = tmp_path / "manifest-link.json"
        source = inputs["baseline_manifest"]
    else:
        root = inputs["program_folder" if where == "new" else "baseline_folder"]
        target, source = root / "_internal/linked.dll", root / "_internal/library.dll"
    try:
        target.symlink_to(source)
    except OSError:
        pytest.skip("Symlink creation is unavailable")
    if where == "manifest":
        inputs["baseline_manifest"] = target
    with pytest.raises(ValueError, match="链接|联接"):
        patcher.package_patch(**inputs)
    assert not inputs["output"].exists()


def test_existing_output_is_preserved(patcher, inputs):
    inputs["output"].mkdir()
    marker = inputs["output"] / "keep.txt"
    marker.write_bytes(b"existing output")
    with pytest.raises(FileExistsError):
        patcher.package_patch(**inputs)
    assert marker.read_bytes() == b"existing output"


def test_nested_output_rejected(patcher, inputs):
    inputs["output"] = inputs["program_folder"] / "patch-output"
    with pytest.raises(ValueError, match="输入目录"):
        patcher.package_patch(**inputs)
    assert not inputs["output"].exists()


def test_executable_is_included_even_when_unchanged(patcher, inputs):
    (inputs["program_folder"] / "FanyiJiVoiceNext.exe").write_bytes(b"MZ old program")
    result = patcher.package_patch(**inputs)
    assert "FanyiJiVoiceNext.exe" in {row["path"] for row in result["included_program_files"]}


@pytest.mark.parametrize("directory", ["FanyiJiVoiceNext_models", "直播字幕_models", "custom_MODELS"])
def test_patch_preserves_downloaded_model_root_without_reading_weights(patcher, inputs, tmp_path, monkeypatch, directory):
    write_files(inputs["baseline_folder"], {directory + "/speech/model.safetensors": b"downloaded weight"})
    save_baseline_manifest(patcher, inputs)
    original = patcher.release.sha256

    def no_model_reads(path):
        assert not any(part.casefold().endswith("_models") for part in Path(path).parts)
        return original(path)

    monkeypatch.setattr(patcher.release, "sha256", no_model_reads)
    result = patcher.package_patch(**inputs)
    assert result["preserved_model_file_count"] == 2
    installed = tmp_path / "installed-external"
    shutil.copytree(inputs["baseline_folder"], installed)
    item = result["program_update"]
    info = updater.UpdateInfo("0.5.1", item["name"], item["url"], item["sha256"], item["size"],
                              "https://github.com/example/voice-next/releases/tag/v0.5.1", "patch")
    prepared = updater.UpdateClient(repository="example/voice-next", cache_dir=tmp_path / "cache").prepare(
        updater.DownloadedUpdate(info, inputs["output"] / item["name"]), installed)
    updater.apply_update(prepared.job_path, restart=False)
    assert (installed / directory / "speech/model.safetensors").read_bytes() == b"downloaded weight"


@pytest.mark.parametrize("relative", ["FanyiJiVoiceNext_models/catalog.json", "_internal/data/model.bin",
                                      "_internal/data/vad.onnx", "_internal/data/weight.safetensors"])
def test_new_overlay_cannot_inject_external_models(patcher, inputs, relative):
    write_files(inputs["program_folder"], {relative: b"must be rejected"})
    with pytest.raises((ValueError, updater.UpdateError)):
        patcher.package_patch(**inputs)
    assert not inputs["output"].exists()
