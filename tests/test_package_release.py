"""Program packages contain no model weights or developer data."""
import importlib.util
import json
from pathlib import Path
import zipfile

import pytest

from fanyiji_next.release_profile import load_release_defaults, release_defaults_payload, write_release_defaults
from fanyiji_next.settings import SettingsError


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
    models.mkdir()
    for relative in ("asr/model.int8.onnx", "vad/silero_vad.onnx", "whisper-large-v3-ct2/model.bin",
                     "hymt2-1.8b/Hy-MT2-1.8B-Q8_0.gguf", "source/nllb/pytorch_model.bin",
                     "self_test/ja.wav", "private.settings.json"):
        target = models / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"must never be read or copied")
    runtime.mkdir()
    for name in ("llama-server.exe", "ggml.dll", "ggml-cuda.dll", "cublas64_12.dll", "LICENSE-LLVM-OpenMP", "llama-quantize.exe"):
        (runtime / name).write_bytes(name.encode())
    return dict(program_folder=program, models=models, runtime=runtime, output=tmp_path / "release", version="0.5.0")


@pytest.fixture
def release_defaults_050():
    # Published 0.5.0 payload, independent of the current defaults generator.
    return {
        "schema_version": 1,
        "microphone": {"enabled": False, "direction": "zh_to_ja", "device_index": None},
        "system": {"enabled": True, "direction": "ja_to_zh", "device_index": None},
        "asr_model": "", "asr_tokens": "", "translation_model_dir": "", "translation_spm": "",
        "font_size": 30, "show_original": True, "always_on_top": False, "window_opacity": 100,
        "recognition_backend": "whisper", "whisper_model_dir": "", "recognition_device": "cuda",
        "recognition_compute_type": "int8_float16", "translation_backend": "hymt",
        "nllb_model_dir": "", "nllb_tokenizer_dir": "", "hymt_model_file": "", "hymt_server_exe": "",
        "translation_device": "cuda", "translation_compute_type": "int8_float16", "vad_model": "",
        "system_audio_processing": "mdx_gtcrn", "mdx_model_file": "", "gtcrn_model_file": "",
    }


def test_release_defaults_select_downloadable_a_without_user_paths(tmp_path):
    path = tmp_path / "release-defaults.json"
    write_release_defaults(path)
    settings = load_release_defaults(path)
    assert settings.model_combination == "A" and settings.recognition_backend == "qwen3_asr"
    assert settings.recognition_device == "cuda" and settings.recognition_compute_type == "bfloat16"
    assert settings.translation_backend == "translategemma" and settings.translation_device == "cuda"
    assert settings.system_audio_processing == "raw"
    assert not settings.microphone.enabled and settings.system.enabled
    assert settings.microphone.device_index is settings.system.device_index is None
    assert settings.window_transparency_enabled is True and settings.window_opacity == 100
    assert release_defaults_payload()["window_transparency_enabled"] is True
    assert all(not value for key, value in release_defaults_payload().items()
               if key.endswith(("_model", "_tokens", "_model_dir", "_tokenizer_dir", "_model_file", "_server_exe", "_spm", "_root")))
    assert list(tmp_path.iterdir()) == [path]
    before = path.read_bytes()
    with pytest.raises(FileExistsError):
        write_release_defaults(path)
    assert path.read_bytes() == before


def test_published_050_defaults_gain_only_the_missing_transparency_flag_without_rewriting(tmp_path, release_defaults_050):
    path = tmp_path / "release-defaults.json"
    path.write_text(json.dumps(release_defaults_050), encoding="utf-8")
    before = path.read_bytes()
    settings = load_release_defaults(path)
    assert settings.window_transparency_enabled is True and settings.window_opacity == 100
    assert settings.recognition_backend == "whisper" and settings.translation_backend == "hymt"
    assert settings.system_audio_processing == "mdx_gtcrn"
    assert path.read_bytes() == before


@pytest.mark.parametrize("change", [
    {"unknown": True}, {"window_transparency_enabled": False}, {"window_transparency_enabled": 1},
    {"recognition_backend": "sensevoice"}, {"translation_backend": "m2m100"},
    {"window_opacity": 85}, {"hymt_model_file": "C:/private/model.gguf"},
])
def test_legacy_defaults_migration_keeps_all_other_strict_checks(tmp_path, release_defaults_050, change):
    path = tmp_path / "release-defaults.json"
    path.write_text(json.dumps(release_defaults_050 | change), encoding="utf-8")
    before = path.read_bytes()
    with pytest.raises(SettingsError):
        load_release_defaults(path)
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
    assert not (folder / "models").exists()
    assert not (folder / "FanyiJiVoiceNext_models").exists()
    assert (folder / "runtime/llama-b11696-cuda12.4/llama-server.exe").is_file()
    assert not (folder / "runtime/llama-b11696-cuda12.4/llama-quantize.exe").exists()
    assert load_release_defaults(folder / "release-defaults.json").recognition_backend == "qwen3_asr"
    info = json.loads((folder / "build-info.json").read_text(encoding="utf-8"))
    assert not info["model_weights_bundled"] and not info["user_settings_bundled"]
    assert info["external_model_directory"] == "<executable-stem>_models"
    assert info["model_download_required"] and result["model_count"] == 0
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


def test_models_argument_is_optional_and_no_missing_weight_blocks_program_packaging(packager, inputs):
    inputs.pop("models")
    result = packager.package_release(**inputs)
    assert result["model_count"] == 0
    assert not (Path(result["portable_folder"]) / "models").exists()


def test_legacy_models_argument_is_never_read_or_required(packager, inputs, monkeypatch):
    model_root = inputs["models"]
    original = packager.regular_files

    def only_program_and_runtime(path):
        assert Path(path) != model_root, "Legacy models must not be enumerated"
        return original(path)

    monkeypatch.setattr(packager, "regular_files", only_program_and_runtime)
    inputs["models"] = model_root / "does-not-exist"
    result = packager.package_release(**inputs)
    assert result["model_count"] == 0


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
    link = inputs["program_folder"] / "_internal/linked"
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


@pytest.mark.parametrize("include_mdx", [True, False])
def test_legacy_mdx_flag_cannot_include_any_weight(packager, inputs, include_mdx):
    result = packager.package_release(**inputs, include_mdx=include_mdx)
    folder = Path(result["portable_folder"])
    assert result["model_count"] == 0 and not result["mdx_bundled"]
    assert not (folder / "models").exists()
    note = (folder / "公开包说明.md").read_text(encoding="utf-8")
    assert "不内置" in note and "下载" in note and "_models" in note
    with zipfile.ZipFile(inputs["output"] / result["program_update"]["name"]) as archive:
        assert "公开包说明.md" in archive.namelist()
        assert all(Path(name).suffix.casefold() not in packager.WEIGHT_SUFFIXES
                   for name in archive.namelist())


def test_cli_does_not_require_model_directory(packager, inputs, monkeypatch):
    calls = []
    monkeypatch.setattr(packager, "package_release", lambda **values: calls.append(values) or {"ok": True})
    options = []
    for key, value in inputs.items():
        if key != "models":
            options.extend(("--" + key.replace("_", "-"), str(value)))
    assert packager.main(options) == 0
    assert calls[0]["models"] is None


@pytest.mark.parametrize("relative", ["FanyiJiVoiceNext_models/manifest.json", "直播字幕_models/config.json",
                                      "_internal/data/weights.bin", "_internal/data/pytorch_model.bin",
                                      "_internal/data/tokenizer.model", "_internal/data/vad.onnx"])
def test_model_tree_or_weight_in_new_build_is_rejected(packager, inputs, relative):
    target = inputs["program_folder"] / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(b"model fixture")
    with pytest.raises(ValueError):
        packager.package_release(**inputs)
    assert not inputs["output"].exists()


def test_update_zip_rejects_model_directory_instead_of_silently_dropping_it(packager, inputs, tmp_path):
    target = inputs["program_folder"] / "models/vad/model.onnx"
    target.parent.mkdir(parents=True)
    target.write_bytes(b"model fixture")
    with pytest.raises(ValueError, match="模型"):
        packager.create_update_zip(inputs["program_folder"], tmp_path / "bad.zip", "0.5.0")
    assert not (tmp_path / "bad.zip").exists()


def test_exact_sentencepiece_normalization_tables_are_runtime_data(packager, inputs):
    for relative in packager.RUNTIME_DATA_FILES:
        target = inputs["program_folder"] / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"unicode normalization table fixture")
    result = packager.package_release(**inputs)
    with zipfile.ZipFile(inputs["output"] / result["program_update"]["name"]) as archive:
        assert packager.RUNTIME_DATA_FILES <= set(archive.namelist())


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


@pytest.mark.parametrize("package", ["ctranslate2", "transformers", "funasr", "torchaudio", "torchaudio/prototype"])
def test_runtime_models_code_is_not_a_user_weight_directory(packager, package):
    packager.safe_update_path(f"_internal/{package}/models/qwen3_asr/modeling.py")
    with pytest.raises(ValueError):
        packager.safe_update_path(f"_internal/{package}/models/qwen3_asr/model.safetensors")
    with pytest.raises(ValueError):
        packager.safe_update_path(f"_internal/{package}/models/user_models/config.json")


def test_fresh_build_with_runtime_is_packaged_without_duplicate_copy(packager, inputs):
    target = inputs["program_folder"] / "runtime/llama-b11696-cuda12.4/llama-server.exe"
    target.parent.mkdir(parents=True)
    target.write_bytes(b"fresh build runtime copy")
    report = packager.package_release(**inputs)
    output = Path(report["portable_folder"]) / "runtime/llama-b11696-cuda12.4/llama-server.exe"
    assert output.read_bytes() == (inputs["runtime"] / "llama-server.exe").read_bytes()
    assert report["model_count"] == 0


@pytest.mark.parametrize("matching", [True, False])
def test_vendored_public_certificate_must_match_canonical_bundle(packager, inputs, matching):
    import certifi
    canonical = inputs["program_folder"] / "_internal/certifi/cacert.pem"
    canonical.parent.mkdir(parents=True)
    canonical.write_bytes(Path(certifi.where()).read_bytes())
    target = inputs["program_folder"] / "_internal/aliyunsdkcore/vendored/requests/packages/certifi/cacert.pem"
    target.parent.mkdir(parents=True)
    target.write_bytes(canonical.read_bytes() + (b"" if matching else b"# changed"))
    if matching:
        result = packager.package_release(**inputs)
        assert (Path(result["portable_folder"]) / target.relative_to(inputs["program_folder"])).read_bytes() == canonical.read_bytes()
    else:
        with pytest.raises(ValueError, match="证书"):
            packager.package_release(**inputs)
        assert not inputs["output"].exists()


def test_shipped_torch_heuristic_code_is_present_in_program_and_update(packager, inputs):
    names = ("__init__", "_MixedMMA100", "_MixedMMH100", "_MMRankingA100", "_MMRankingH100", "_PadMMA100")
    files = {}
    for name in names:
        relative = f"_internal/torch/_inductor/autoheuristic/artifacts/{name}.py"
        target = inputs["program_folder"] / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        files[relative] = ("# public torch heuristic " + name).encode()
        target.write_bytes(files[relative])
    result = packager.package_release(**inputs)
    with zipfile.ZipFile(inputs["output"] / result["program_update"]["name"]) as archive:
        for relative, content in files.items():
            assert archive.read(relative) == content
            assert (Path(result["portable_folder"]) / relative).read_bytes() == content


@pytest.mark.parametrize("relative", [
    "artifacts/_MMRankingA100.py", "_internal/other/artifacts/_MMRankingA100.py",
    "_internal/torch/_inductor/autoheuristic/artifacts/private.py",
    "_internal/torch/_inductor/autoheuristic/artifacts/_MMRankingA100.pyc",
    "_internal/torch/_inductor/autoheuristic/artifacts/_MMRankingA100.pyd",
    "_internal/torch/_inductor/autoheuristic/artifacts/model.gguf",
    "_internal/torch/_inductor/autoheuristic/artifacts/model.safetensors",
    "_internal/torch/_inductor/autoheuristic/artifacts/nested/_MMRankingA100.py",
    "_internal/torch/_inductor/autoheuristic/artifacts/settings.json",
])
def test_torch_artifacts_exception_cannot_admit_other_payload(packager, inputs, relative):
    target = inputs["program_folder"] / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(b"must remain excluded")
    with pytest.raises(ValueError):
        packager.package_release(**inputs)
    assert not inputs["output"].exists()


@pytest.mark.parametrize("relative", [
    "_internal/jieba/lac_small/model_baseline/word_emb",
    "_internal/jieba/lac_small/model_baseline/crfw",
    "_internal/jieba/lac_small/MODEL_BASELINE/weights_without_extension",
    "_internal/jieba/lac_small/model_baseline/__init__.py",
    "_internal/funasr/__pycache__/model.cpython-312.pyc",
])
def test_hidden_dependency_checkpoint_and_import_cache_are_never_packaged(packager, inputs, relative):
    target = inputs["program_folder"] / relative
    target.parent.mkdir(parents=True)
    target.write_bytes(b"not public program payload")
    with pytest.raises(ValueError):
        packager.package_release(**inputs)
    assert not inputs["output"].exists()
