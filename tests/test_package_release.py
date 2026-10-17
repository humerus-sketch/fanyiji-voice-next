"""Portable publishing carries runtime models, never a developer profile or recording."""
import importlib.util
import json
from pathlib import Path
import zipfile

import pytest

from fanyiji_next.release_profile import load_release_defaults, release_defaults_payload, write_release_defaults
from fanyiji_next.settings import SettingsError, with_local_models


@pytest.fixture
def packager():
    path = Path(__file__).resolve().parents[1] / "tools/package_release.py"
    spec = importlib.util.spec_from_file_location("next_package", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def inputs(tmp_path, packager):
    program = tmp_path / "fresh-build/FanyiJiVoiceNext"
    models = tmp_path / "models"
    runtime = tmp_path / "runtime"
    program.mkdir(parents=True)
    (program / "FanyiJiVoiceNext.exe").write_bytes(b"MZ frozen program fixture")
    (program / "build-info.json").write_text('{"version":"0.5.0","model_weights_bundled":false}', encoding="utf-8")
    for relative in ("_internal/dependency.dll", "_internal/sherpa_onnx/__init__.py", "_internal/ctranslate2/models/__init__.py", "licenses/av/AUTHORS.py", "_updater/FanyiJiUpdate.exe"):
        target = program / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"public runtime fixture")
    for name in packager.VC_RUNTIME_NAMES:
        (program / "_internal" / name).write_bytes(("canonical VC fixture " + name).encode())
    for directory, names in packager.MODEL_FILES.items():
        for name in names:
            target = models / directory / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes((directory + "/" + name).encode())
    (models / "asr/LICENSE").write_text("model license fixture", encoding="utf-8")
    for relative in ("source/nllb/pytorch_model.bin", "self_test/ja.wav", "private.settings.json"):
        target = models / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"must never be copied")
    runtime.mkdir()
    for name in ("llama-server.exe", "ggml.dll", "ggml-cuda.dll", "cublas64_12.dll", "LICENSE-LLVM-OpenMP", "llama-quantize.exe"):
        (runtime / name).write_bytes(name.encode())
    return dict(program_folder=program, models=models, runtime=runtime, output=tmp_path / "release", version="0.5.0")


def test_release_defaults_keep_current_engines_without_user_paths(tmp_path):
    path = tmp_path / "release-defaults.json"
    write_release_defaults(path)
    settings = load_release_defaults(path)
    assert settings.recognition_backend == "whisper"
    assert settings.recognition_device == "cuda" and settings.recognition_compute_type == "int8_float16"
    assert settings.translation_backend == "hymt" and settings.translation_device == "cuda"
    assert settings.system_audio_processing == "mdx_gtcrn"
    assert not settings.microphone.enabled and settings.system.enabled
    assert settings.microphone.device_index is settings.system.device_index is None
    assert all(not value for key, value in release_defaults_payload().items()
               if key.endswith(("_model", "_tokens", "_model_dir", "_tokenizer_dir", "_model_file", "_server_exe", "_spm")))
    assert list(tmp_path.iterdir()) == [path]
    before = path.read_bytes()
    with pytest.raises(FileExistsError):
        write_release_defaults(path)
    assert path.read_bytes() == before


@pytest.mark.parametrize("change", [{"hymt_model_file": "C:/private/model.gguf"}, {"unknown": True}, {"recognition_backend": "sensevoice"}, {"system": {"enabled": True, "direction": "ja_to_zh", "device_index": 7}}])
def test_release_defaults_reject_extra_settings_paths_devices_and_changed_defaults(tmp_path, change):
    path = tmp_path / "release-defaults.json"
    path.write_text(json.dumps(release_defaults_payload() | change), encoding="utf-8")
    with pytest.raises(SettingsError):
        load_release_defaults(path)


def test_full_package_and_update_manifest_are_exact_and_portable(packager, inputs):
    result = packager.package_release(**inputs, github_repo="example/voice-next")
    output = inputs["output"]
    folder = Path(result["portable_folder"])
    assert not (folder / "models/source").exists()
    assert not (folder / "models/self_test").exists()
    assert not (folder / "models/private.settings.json").exists()
    assert (folder / "models/asr/LICENSE").read_text(encoding="utf-8") == "model license fixture"
    assert (folder / "runtime/llama-b11696-cuda12.4/llama-server.exe").is_file()
    assert not (folder / "runtime/llama-b11696-cuda12.4/llama-quantize.exe").exists()
    settings = with_local_models(load_release_defaults(folder / "release-defaults.json"), folder / "models")
    assert settings.whisper_model_dir == str((folder / "models/whisper-large-v3-ct2").resolve())
    assert settings.hymt_server_exe == str((folder / "runtime/llama-b11696-cuda12.4/llama-server.exe").resolve())
    inventory = json.loads((folder / "models/model-inventory.json").read_text(encoding="utf-8"))
    assert len(inventory["models"]) == 8
    for row in inventory["models"]:
        assert packager.sha256(folder / row["path"]) == row["sha256"]
    info = json.loads((folder / "build-info.json").read_text(encoding="utf-8"))
    assert info["model_weights_bundled"] and not info["user_settings_bundled"]
    archive_path = output / result["program_update"]["name"]
    with zipfile.ZipFile(archive_path) as archive:
        manifest = json.loads(archive.read("update-manifest.json"))
        assert manifest["application"] == "fanyiji-voice-next" and manifest["schema_version"] == 1
        assert {entry["path"] for entry in manifest["files"]} == set(archive.namelist()) - {"update-manifest.json"}
        assert all(not name.startswith("models/") for name in archive.namelist())
        assert "_updater/FanyiJiUpdate.exe" in archive.namelist()
        for row in manifest["files"]:
            assert len(archive.read(row["path"])) == row["size"]
            assert packager.hashlib.sha256(archive.read(row["path"])).hexdigest() == row["sha256"]
    feed = json.loads((output / "update-feed.json").read_text(encoding="utf-8"))
    assert feed["program"]["url"] == "https://github.com/example/voice-next/releases/download/v0.5.0/FanyiJiVoiceNext-0.5.0-update.zip"
    assert feed["program"]["sha256"] == packager.sha256(archive_path)
    assert feed["program"]["size"] == archive_path.stat().st_size
    full_manifest = json.loads((output / "full-manifest.json").read_text(encoding="utf-8"))
    assert {row["path"] for row in full_manifest["files"]} == {file.relative_to(folder).as_posix() for file in folder.rglob("*") if file.is_file()}
    for row in full_manifest["files"]:
        assert packager.sha256(folder / row["path"]) == row["sha256"]


def test_existing_output_never_overwrites_any_file(packager, inputs):
    inputs["output"].mkdir()
    marker = inputs["output"] / "keep.txt"
    marker.write_text("old release", encoding="utf-8")
    with pytest.raises(FileExistsError):
        packager.package_release(**inputs)
    assert marker.read_text(encoding="utf-8") == "old release"


@pytest.mark.parametrize("relative", ["settings.json", "voice.settings.json", "private.key", "artifacts/a.wav", "models/model.onnx", "source/secret.txt", "_internal/model.bin", "_internal/private.pem", "_internal/ctranslate2/models/model.bin"])
def test_unexpected_program_data_is_rejected_before_output(packager, inputs, relative):
    target = inputs["program_folder"] / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(b"private fixture")
    with pytest.raises(ValueError):
        packager.package_release(**inputs)
    assert not inputs["output"].exists()


@pytest.mark.parametrize("relative", ["../escape.exe", "C:/escape.exe", "\\server/share/file", "a\\b.dll", "/absolute/file", "a//b.dll", "a/./b.dll", "settings/a.json", "Models/a.bin", "logs/a.txt"])
def test_update_paths_reject_escaping_or_preserved_data(packager, relative):
    with pytest.raises(ValueError):
        packager.safe_update_path(relative)


def test_missing_required_model_fails_without_partial_output(packager, inputs):
    (inputs["models"] / "vad/silero_vad.onnx").rename(inputs["models"] / "vad/missing.onnx")
    with pytest.raises(ValueError, match="缺少运行模型"):
        packager.package_release(**inputs)
    assert not inputs["output"].exists()


def test_build_version_mismatch_fails_before_copy(packager, inputs):
    inputs["version"] = "0.6.0"
    with pytest.raises(ValueError, match="版本"):
        packager.package_release(**inputs)
    assert not inputs["output"].exists()


def test_without_repo_does_not_publish_placeholder_urls(packager, inputs):
    packager.package_release(**inputs)
    assert not (inputs["output"] / "update-feed.json").exists()


def test_model_notices_are_in_program_update(packager, inputs, tmp_path):
    notices = tmp_path / "notices"
    notices.mkdir()
    (notices / "Hy-MT2-LICENSE.txt").write_text("public model license fixture", encoding="utf-8")
    result = packager.package_release(**inputs, model_notices=notices)
    with zipfile.ZipFile(inputs["output"] / result["program_update"]["name"]) as archive:
        assert archive.read("licenses/model-notices/Hy-MT2-LICENSE.txt") == b"public model license fixture"


def test_output_cannot_be_nested_in_inputs(packager, inputs):
    inputs["output"] = inputs["program_folder"] / "nested"
    with pytest.raises(ValueError, match="输入目录"):
        packager.package_release(**inputs)


def test_source_symlink_is_rejected(packager, inputs):
    link = inputs["models"] / "linked"
    try:
        link.symlink_to(inputs["runtime"], target_is_directory=True)
    except OSError:
        pytest.skip("Windows symlink privilege is unavailable")
    with pytest.raises(ValueError, match="链接"):
        packager.package_release(**inputs)


@pytest.mark.parametrize("content", [b"private fixture", b"-----BEGIN CERTIFICATE-----\n-----BEGIN PRIVATE KEY-----"])
def test_canonical_certifi_bundle_cannot_hide_private_key(packager, inputs, content):
    target = inputs["program_folder"] / "_internal/certifi/cacert.pem"
    target.parent.mkdir(parents=True)
    target.write_bytes(content)
    with pytest.raises(ValueError, match="证书"):
        packager.package_release(**inputs)
    assert not inputs["output"].exists()


def test_public_package_excludes_mdx_even_when_locally_present_without_changing_defaults(packager, inputs):
    source_metadata = inputs["models"] / "audio-processing/model-sources.json"
    source_metadata.write_text(json.dumps({"models": [{"file": "Kim_Vocal_2.onnx", "source": "https://example.test/mdx"},
                                                      {"file": "gtcrn_simple.onnx", "source": "https://example.test/gtcrn"}]}), encoding="utf-8")
    source_before = source_metadata.read_bytes()
    result = packager.package_release(**inputs, include_mdx=False)
    folder = Path(result["portable_folder"])
    assert result["model_count"] == 7 and not result["mdx_bundled"]
    assert (inputs["models"] / "audio-processing/Kim_Vocal_2.onnx").is_file()
    assert not (folder / "models/audio-processing/Kim_Vocal_2.onnx").exists()
    assert (folder / "models/audio-processing/gtcrn_simple.onnx").is_file()
    inventory = json.loads((folder / "models/model-inventory.json").read_text(encoding="utf-8"))
    assert not inventory["mdx_bundled"] and len(inventory["models"]) == 7
    assert inventory["excluded_models"][0]["bundled"] is False
    provenance = json.loads((folder / "models/audio-processing/model-sources.json").read_text(encoding="utf-8"))
    assert provenance["models"][0]["bundled_in_this_package"] is False
    assert provenance["models"][1]["bundled_in_this_package"] is True
    assert source_metadata.read_bytes() == source_before
    assert load_release_defaults(folder / "release-defaults.json").system_audio_processing == "mdx_gtcrn"
    note = (folder / "公开包说明.md").read_text(encoding="utf-8")
    assert "不附带" in note and "声音与模型" in note and "非商业" in note
    with zipfile.ZipFile(inputs["output"] / result["program_update"]["name"]) as archive:
        assert "公开包说明.md" in archive.namelist()


def test_public_package_does_not_require_mdx_local_file(packager, inputs):
    (inputs["models"] / "audio-processing/Kim_Vocal_2.onnx").rename(inputs["models"] / "audio-processing/local-only.onnx")
    result = packager.package_release(**inputs, include_mdx=False)
    assert not (Path(result["portable_folder"]) / "models/audio-processing/local-only.onnx").exists()


@pytest.mark.parametrize("flags,expected", [([], True), (["--exclude-mdx"], False)])
def test_cli_passes_explicit_public_model_scope_without_changing_legacy_default(packager, inputs, monkeypatch, flags, expected):
    calls = []
    monkeypatch.setattr(packager, "package_release", lambda **values: calls.append(values) or {"ok": True})
    options = []
    for key, value in inputs.items():
        options.extend(("--" + key.replace("_", "-"), str(value)))
    assert packager.main([*options, *flags]) == 0
    assert calls[0]["include_mdx"] is expected


def test_llama_runtime_copies_canonical_vc_dlls_and_manifests_them(packager, inputs):
    alternate = inputs["program_folder"] / "_internal/PySide6"
    alternate.mkdir()
    (alternate / "MSVCP140.dll").write_bytes(b"older Qt DLL must not win")
    result = packager.package_release(**inputs)
    folder = Path(result["portable_folder"])
    for name in packager.VC_RUNTIME_NAMES:
        target = folder / "runtime/llama-b11696-cuda12.4" / name
        assert target.read_bytes() == ("canonical VC fixture " + name).encode()
    with zipfile.ZipFile(inputs["output"] / result["program_update"]["name"]) as archive:
        manifest = json.loads(archive.read("update-manifest.json"))
        entries = {row["path"]: row for row in manifest["files"]}
        for name in packager.VC_RUNTIME_NAMES:
            relative = "runtime/llama-b11696-cuda12.4/" + name
            assert entries[relative]["sha256"] == packager.hashlib.sha256(archive.read(relative)).hexdigest()


def test_existing_runtime_vc_dll_takes_priority(packager, inputs):
    for name in packager.VC_RUNTIME_NAMES:
        (inputs["runtime"] / name).write_bytes(b"existing portable runtime VC")
    result = packager.package_release(**inputs)
    folder = Path(result["portable_folder"])
    for name in packager.VC_RUNTIME_NAMES:
        assert (folder / "runtime/llama-b11696-cuda12.4" / name).read_bytes() == b"existing portable runtime VC"


def test_unique_nested_vc_dll_is_used_only_if_canonical_missing(packager, inputs):
    canonical = inputs["program_folder"] / "_internal/msvcp140.dll"
    canonical.rename(canonical.with_suffix(".original"))
    alternate = inputs["program_folder"] / "_internal/PySide6/MSVCP140.dll"
    alternate.parent.mkdir()
    alternate.write_bytes(b"only nested source")
    result = packager.package_release(**inputs)
    assert (Path(result["portable_folder"]) / "runtime/llama-b11696-cuda12.4/msvcp140.dll").read_bytes() == b"only nested source"


@pytest.mark.parametrize("nested_copies", [0, 2])
def test_missing_or_ambiguous_vc_dll_fails_without_relying_on_system32(packager, inputs, nested_copies):
    canonical = inputs["program_folder"] / "_internal/msvcp140.dll"
    canonical.rename(canonical.with_suffix(".original"))
    for index in range(nested_copies):
        alternate = inputs["program_folder"] / f"_internal/vendor{index}/msvcp140.dll"
        alternate.parent.mkdir()
        alternate.write_bytes(bytes([index]))
    with pytest.raises(ValueError, match="唯一"):
        packager.package_release(**inputs)
    assert not inputs["output"].exists()
