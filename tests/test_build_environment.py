"""Prevent unrelated tool DLLs from contaminating a portable application."""
import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def builder():
    path = Path(__file__).resolve().parents[1] / "tools/build.py"
    spec = importlib.util.spec_from_file_location("next_build", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def fake_gpu_wheels(builder, tmp_path, monkeypatch):
    root = tmp_path / "site-packages"
    records = {}
    files = {
        "nvidia-cublas-cu12": ["nvidia/cublas/bin/cublas64_12.dll", "nvidia/cublas/bin/cublasLt64_12.dll",
                              "nvidia/cublas/LICENSE.txt", "nvidia/cublas/lib/cublas.lib"],
        "nvidia-cudnn-cu12": ["nvidia/cudnn/bin/cudnn64_9.dll", "nvidia/cudnn/bin/cudnn_ops64_9.dll",
                             "nvidia/cudnn/bin/cudnn_engines_runtime_compiled64_9.dll",
                             "nvidia/cudnn/LICENSE.txt"],
    }
    for name, names in files.items():
        paths = [Path(value) for value in names]
        for relative in paths:
            source = root / relative
            source.parent.mkdir(parents=True, exist_ok=True)
            source.write_bytes(b"fake wheel file; never loaded")
        records[name] = SimpleNamespace(files=paths, requires=[], locate_file=lambda item: root / item)
    original = builder.importlib.metadata.distribution
    monkeypatch.setattr(builder.importlib.metadata, "distribution", lambda name: records[name] if name in records else original(name))
    return root, records


def test_build_environment_excludes_unrelated_dll_and_python_paths(monkeypatch, builder):
    monkeypatch.setenv("PATH", "C:/unrelated/poppler/bin;C:/unrelated/libheif/bin")
    monkeypatch.setenv("PYTHONPATH", "C:/unrelated/application")
    monkeypatch.setenv("PYTHONHOME", "C:/unrelated/python")
    environment = builder.build_environment()
    assert "unrelated" not in environment["PATH"]
    assert "PYTHONPATH" not in environment and "PYTHONHOME" not in environment
    assert "System32" in environment["PATH"]
    assert "unrelated" in os.environ["PATH"]


def test_prototype_uses_its_own_entrypoint_and_excludes_voice_modules(builder, tmp_path, monkeypatch):
    monkeypatch.setattr(builder.importlib.metadata, "distribution", lambda name: pytest.fail("prototype must not inspect GPU wheels"))
    command = builder.build_command(tmp_path / "dist", tmp_path / "work", "prototype")
    excluded = {command[index + 1] for index, item in enumerate(command) if item == "--exclude-module"}
    assert {"fanyiji_next.app", "fanyiji_next.controller", "fanyiji_next.audio_runtime"} <= excluded
    assert set(builder.VOICE_MODULES) <= excluded
    assert {"fanyiji_next.native_runtime", "nvidia", "nvidia.cublas", "nvidia.cudnn"} <= excluded
    assert "--collect-all" not in command
    assert "--add-binary" not in command and "--copy-metadata" not in command
    assert Path(command[-1]).name == "run_prototype.py"


def test_voice_explicitly_collects_all_native_backends(builder, tmp_path, fake_gpu_wheels):
    command = builder.build_command(tmp_path / "dist", tmp_path / "work", "voice")
    collected = {command[index + 1] for index, item in enumerate(command) if item == "--collect-all"}
    assert collected == {"numpy", "sherpa_onnx", "ctranslate2", "sentencepiece", "pyaudiowpatch", "soxr"}
    modules = {command[index + 1] for index, item in enumerate(command) if item == "--collect-submodules"}
    binaries = {command[index + 1] for index, item in enumerate(command) if item == "--collect-binaries"}
    metadata = {command[index + 1] for index, item in enumerate(command) if item == "--copy-metadata"}
    assert modules == {"faster_whisper", "tokenizers", "av", "onnxruntime", "huggingface_hub", "tqdm"}
    assert {"tokenizers", "av", "onnxruntime"} <= binaries
    assert {"faster-whisper", "tokenizers", "av", "onnxruntime", "huggingface-hub", "tqdm"} <= metadata
    assert {"nvidia-cublas-cu12", "nvidia-cudnn-cu12"} <= metadata
    assert not {"faster_whisper", "onnxruntime"} & collected
    assert "--collect-data" not in command
    assert "--exclude-module" not in command
    assert Path(command[-1]).name == "run.py"
    assert "sherpa-onnx-core" in builder.license_packages("voice")
    assert "sherpa-onnx-core" not in builder.license_packages("prototype")
    assert set(builder.VOICE_METADATA) <= set(builder.license_packages("voice"))
    assert not set(builder.VOICE_METADATA) & set(builder.license_packages("prototype"))


def test_existing_output_is_preserved_before_any_build_process(builder, tmp_path, monkeypatch):
    output = tmp_path / "existing"
    output.mkdir()
    marker = output / "keep.txt"
    marker.write_text("existing release", encoding="utf-8")
    monkeypatch.setattr(builder.subprocess, "run", lambda *args, **kwargs: pytest.fail("must not launch a build"))
    with pytest.raises(SystemExit) as error:
        builder.main(["--edition", "voice", "--output", str(output)])
    assert error.value.code == 2
    assert marker.read_text(encoding="utf-8") == "existing release"


def test_default_editions_get_separate_versioned_outputs(builder, tmp_path, monkeypatch):
    monkeypatch.setattr(builder, "ROOT", tmp_path)
    (tmp_path / "pyproject.toml").write_text('[project]\nversion="0.2.0"\n', encoding="utf-8")
    for filename in ("run.py", "run_prototype.py", "README.md"):
        (tmp_path / filename).write_text("fixture", encoding="utf-8")
    monkeypatch.setattr(builder.importlib.metadata, "distribution", lambda name: object())
    monkeypatch.setattr(builder, "nvidia_binaries", lambda: [])
    monkeypatch.setattr(builder, "collect_licenses", lambda folder, edition: None)
    checks = []
    monkeypatch.setattr(builder, "run_frozen_checks", lambda folder, path, edition, environment: checks.append(edition))

    def fake_package(command, **kwargs):
        output = Path(command[command.index("--distpath") + 1])
        (output / "FanyiJiVoiceNext").mkdir()

    monkeypatch.setattr(builder.subprocess, "run", fake_package)
    builder.main(["--edition", "prototype"])
    builder.main(["--edition", "voice"])
    assert checks == ["prototype", "voice"]
    for edition in checks:
        folder = tmp_path / "dist" / f"0.2.0-{edition}" / "FanyiJiVoiceNext"
        report = json.loads((folder / "build-info.json").read_text(encoding="utf-8"))
        assert report["version"] == "0.2.0" and report["edition"] == edition
        assert report["voice_backend_bundled"] == (edition == "voice")
        assert report["gpu_runtime_bundled"] == (edition == "voice")
        assert not report["model_weights_bundled"] and not report["user_settings_bundled"]


@pytest.mark.parametrize("filename", ["weights.onnx", "model.bin", "sentencepiece.bpe.model", "settings.json", "Cookies", "private.pem",
                                     "faster_whisper/assets/silero_vad_v6.onnx", "onnxruntime/datasets/logreg_iris.onnx"])
def test_payload_rejects_model_weights_and_private_files(builder, tmp_path, filename):
    target = tmp_path / "_internal" / filename
    target.parent.mkdir(parents=True)
    target.write_bytes(b"fixture")
    with pytest.raises(RuntimeError, match="模型或用户资料"):
        builder.assert_public_payload(tmp_path)


def test_payload_allows_runtime_libraries_and_licenses(builder, tmp_path):
    (tmp_path / "onnxruntime.dll").write_bytes(b"fixture")
    (tmp_path / "LICENSE.txt").write_text("fixture", encoding="utf-8")
    builder.assert_public_payload(tmp_path)


def test_payload_allows_only_the_installed_certifi_bundle_at_its_canonical_path(builder, tmp_path):
    import certifi

    target = tmp_path / "_internal/certifi/cacert.pem"
    target.parent.mkdir(parents=True)
    target.write_bytes(Path(certifi.where()).read_bytes())
    builder.assert_public_payload(tmp_path)
    (target.parent / "private.pem").write_bytes(b"another PEM is still forbidden")
    with pytest.raises(RuntimeError, match="模型或用户资料"):
        builder.assert_public_payload(tmp_path)


@pytest.mark.parametrize("relative", ["certifi/cacert.pem", "_internal/other/cacert.pem",
                                     "_internal/certifi/other.pem", "_internal/certifi/nested/cacert.pem",
                                     "_internal/certifi/cacert.key", "_internal/certifi/CACERT.PEM"])
def test_identical_certifi_content_cannot_allow_pem_or_key_at_other_paths(builder, tmp_path, relative):
    import certifi

    target = tmp_path / relative
    target.parent.mkdir(parents=True)
    target.write_bytes(Path(certifi.where()).read_bytes())
    with pytest.raises(RuntimeError, match="模型或用户资料"):
        builder.assert_public_payload(tmp_path)


def test_canonical_certifi_bundle_with_a_different_sha_is_rejected(builder, tmp_path):
    import certifi

    target = tmp_path / "_internal/certifi/cacert.pem"
    target.parent.mkdir(parents=True)
    content = bytearray(Path(certifi.where()).read_bytes())
    content[-1] ^= 1
    target.write_bytes(content)
    with pytest.raises(RuntimeError, match="模型或用户资料"):
        builder.assert_public_payload(tmp_path)


@pytest.mark.parametrize("marker", [b"-----BEGIN PRIVATE KEY-----", b"-----BEGIN RSA PRIVATE KEY-----",
                                   b"-----BEGIN ENCRYPTED PRIVATE KEY-----"])
def test_canonical_bundle_private_key_is_rejected_even_if_installed_hash_matches(builder, tmp_path, monkeypatch, marker):
    import certifi

    installed = tmp_path / "installed-fixture.pem"
    installed.write_bytes(marker + b"\nfake private material, no real key\n")
    target = tmp_path / "program/_internal/certifi/cacert.pem"
    target.parent.mkdir(parents=True)
    target.write_bytes(installed.read_bytes())
    monkeypatch.setattr(certifi, "where", lambda: str(installed))
    with pytest.raises(RuntimeError, match="模型或用户资料"):
        builder.assert_public_payload(tmp_path / "program")


def prototype_report():
    return {"ok": True, "prototype_only": True, "voice_backend_loaded": False, "forbidden_modules": [],
            "audio_capture": False, "model_loading": False, "settings_written": False}


@pytest.mark.parametrize("field,value", [("audio_capture", True), ("model_loading", True), ("settings_written", True),
                                         ("voice_backend_loaded", True), ("forbidden_modules", ["numpy"])])
def test_frozen_prototype_report_rejects_unsafe_or_incomplete_checks(builder, field, value):
    report = prototype_report()
    report[field] = value
    with pytest.raises(RuntimeError):
        builder.validate_frozen_report(report, "prototype", "--smoke-test")


def test_frozen_checks_only_use_safe_modes_and_an_isolated_profile(builder, tmp_path, monkeypatch):
    folder = tmp_path / "program"
    folder.mkdir()
    calls = []

    def fake_check(command, **kwargs):
        calls.append((command, kwargs))
        path = Path(command[command.index("--report") + 1])
        path.write_text(json.dumps(prototype_report()), encoding="utf-8")

    monkeypatch.setattr(builder.subprocess, "run", fake_check)
    original = {"PATH": "fixture"}
    builder.run_frozen_checks(folder, tmp_path / "checks", "prototype", original)
    assert [command[1] for command, _ in calls] == ["--check-runtime", "--smoke-test"]
    for command, options in calls:
        assert options["env"]["QT_QPA_PLATFORM"] == "offscreen"
        assert options["env"]["LOCALAPPDATA"] == str(tmp_path / "checks" / "isolated-profile")
        assert options["env"]["APPDATA"] == str(tmp_path / "checks" / "isolated-profile" / "roaming")
        assert options["env"]["HF_HOME"] == str(tmp_path / "checks" / "isolated-profile" / "huggingface")
        assert options["env"]["HF_HUB_OFFLINE"] == "1"
        assert options["env"]["PATH"] == os.pathsep.join((str(Path("C:/Windows/System32")), str(Path("C:/Windows"))))
        assert "--demo" not in command and "--start" not in command
    assert original == {"PATH": "fixture"}


def test_voice_runtime_and_ui_reports_require_explicit_success(builder):
    safe = {"audio_capture": False, "models_loaded": False, "settings_written": False}
    builder.validate_frozen_report({**safe, "runtime_imports_passed": True}, "voice", "--check-runtime")
    builder.validate_frozen_report({**safe, "ui_smoke": "passed"}, "voice", "--smoke-test")
    with pytest.raises(RuntimeError):
        builder.validate_frozen_report({**safe, "runtime_imports_passed": False}, "voice", "--check-runtime")
    with pytest.raises(RuntimeError):
        builder.validate_frozen_report(safe, "voice", "--smoke-test")


def test_licenses_include_base_dependency_not_optional_conversion_extra(builder, tmp_path, monkeypatch):
    root = tmp_path / "site-packages"
    records = {}
    for name, requires in (("faster-whisper", ["tokenizers>=0.13", "torch; extra == 'conversion'"]),
                           ("tokenizers", [])):
        relative = Path(f"{name}-1.dist-info/licenses/LICENSE.txt")
        source = root / relative
        source.parent.mkdir(parents=True)
        source.write_text(name + " license", encoding="utf-8")
        records[name] = SimpleNamespace(requires=requires, files=[relative], locate_file=lambda item: root / item)
    requested = []
    def distribution(name):
        requested.append(name)
        return records[name]
    monkeypatch.setattr(builder, "license_packages", lambda edition: ("faster-whisper",))
    monkeypatch.setattr(builder.importlib.metadata, "distribution", distribution)
    output = tmp_path / "program"
    builder.collect_licenses(output, "voice")
    assert requested == ["faster-whisper", "tokenizers"]
    assert (output / "licenses/faster-whisper/licenses/LICENSE.txt").read_text(encoding="utf-8") == "faster-whisper license"
    assert (output / "licenses/tokenizers/licenses/LICENSE.txt").read_text(encoding="utf-8") == "tokenizers license"
    assert not (output / "licenses/torch").exists()


def test_gpu_manifest_collects_all_split_dlls_into_their_namespace_bins(builder, tmp_path, fake_gpu_wheels):
    root, _records = fake_gpu_wheels
    command = builder.build_command(tmp_path / "dist", tmp_path / "work", "voice")
    values = [command[index + 1].rsplit(":", 1) for index, item in enumerate(command) if item == "--add-binary"]
    actual = {(Path(source), destination) for source, destination in values}
    expected = {
        (root / "nvidia/cublas/bin/cublas64_12.dll", "nvidia/cublas/bin"),
        (root / "nvidia/cublas/bin/cublasLt64_12.dll", "nvidia/cublas/bin"),
        (root / "nvidia/cudnn/bin/cudnn64_9.dll", "nvidia/cudnn/bin"),
        (root / "nvidia/cudnn/bin/cudnn_ops64_9.dll", "nvidia/cudnn/bin"),
        (root / "nvidia/cudnn/bin/cudnn_engines_runtime_compiled64_9.dll", "nvidia/cudnn/bin"),
    }
    assert actual == expected
    assert all(path.suffix == ".dll" for path, _destination in actual)


def test_gpu_wheel_licenses_are_retained(builder, tmp_path, fake_gpu_wheels, monkeypatch):
    monkeypatch.setattr(builder, "license_packages", lambda edition: ("nvidia-cublas-cu12", "nvidia-cudnn-cu12"))
    output = tmp_path / "program"
    builder.collect_licenses(output, "voice")
    for name, component in (("nvidia-cublas-cu12", "cublas"), ("nvidia-cudnn-cu12", "cudnn")):
        assert (output / "licenses" / name / "nvidia" / component / "LICENSE.txt").is_file()


def test_incomplete_gpu_wheel_is_rejected_before_output_or_build_process(builder, tmp_path, fake_gpu_wheels, monkeypatch):
    root, _records = fake_gpu_wheels
    (root / "nvidia/cublas/bin/cublas64_12.dll").unlink()
    output = tmp_path / "new-release"
    monkeypatch.setattr(builder.subprocess, "run", lambda *args, **kwargs: pytest.fail("must not build incomplete GPU runtime"))
    with pytest.raises(SystemExit) as error:
        builder.main(["--edition", "voice", "--output", str(output)])
    assert error.value.code == 2 and not output.exists()


def test_gpu_records_outside_declared_namespace_are_not_collected(builder, tmp_path, fake_gpu_wheels):
    root, records = fake_gpu_wheels
    extra = Path("unrelated/bin/private.dll")
    (root / extra).parent.mkdir(parents=True)
    (root / extra).write_bytes(b"unrelated fixture")
    records["nvidia-cublas-cu12"].files.extend((extra, Path("../private.dll")))
    binaries = builder.nvidia_binaries()
    assert len(binaries) == 5
    assert all(source.name != "private.dll" for source, _destination in binaries)
