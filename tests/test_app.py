"""The real entry point stays idle; validation modes use no personal audio."""
import json
import os
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication, QMessageBox

from fanyiji_next import app as application
from fanyiji_next import audio_runtime, settings, studio
from fanyiji_next import verification


def forbidden(*_args, **_kwargs):
    raise AssertionError("Audio capture, model loading or settings access was unexpected")


@pytest.fixture
def stable_version(monkeypatch):
    monkeypatch.setattr(application, "__version__", "0.5.1")


def test_voice_smoke_uses_fake_runtime_and_never_reads_or_writes_settings(monkeypatch, tmp_path):
    monkeypatch.setattr(audio_runtime.VoiceRuntime, "start", forbidden)
    monkeypatch.setattr(audio_runtime, "_LocalEngine", forbidden)
    monkeypatch.setattr(settings, "load_settings", forbidden)
    monkeypatch.setattr(settings, "save_settings", forbidden)
    report = tmp_path / "ui.json"
    assert application.main(["--smoke-test", "--report", str(report)]) == 0
    result = json.loads(report.read_text(encoding="utf-8"))
    assert result["ok"]
    assert result["audio_capture"] is False and result["models_loaded"] is False
    assert result["settings_written"] is False
    assert not result["legacy_modules_loaded"]
    assert result["checks"]["restart_preserves_history"]
    assert result["checks"]["runtime_settings_locked"]
    assert result["checks"]["settings_unlock_after_stop"]
    assert result["checks"]["stop_preserves_history"]


def test_normal_startup_only_suggests_local_models_and_waits_for_manual_start(monkeypatch, tmp_path):
    qt_app = QApplication.instance() or QApplication([])
    windows = []
    real_window = studio.StudioWindow

    def create_window(*args, **kwargs):
        window = real_window(*args, **kwargs)
        windows.append(window)
        return window

    def inspect_idle(_self):
        window = windows[-1]
        assert not window.prototype
        assert window.settings.font_size == 30
        assert window._controller.state == "stopped"
        assert window._controller._runtime is None
        assert window.settings.asr_model == str(tmp_path / "models/asr/model.int8.onnx")
        assert window._settings_source == str(target.resolve())
        assert str(target.resolve()) in window.model_button.toolTip()
        window.close()
        return 0

    monkeypatch.setattr(studio, "StudioWindow", create_window)
    monkeypatch.setattr(QApplication, "exec", inspect_idle)
    monkeypatch.setattr(audio_runtime.VoiceRuntime, "start", forbidden)
    monkeypatch.setattr(audio_runtime, "_LocalEngine", forbidden)
    monkeypatch.setattr(settings, "save_settings", forbidden)
    model_root = tmp_path / "models"
    (model_root / "asr").mkdir(parents=True)
    (model_root / "asr/model.int8.onnx").write_bytes(b"fixture")
    target = tmp_path / "config/settings.json"
    assert application.main(["--model-root", str(model_root), "--settings-file", str(target)]) == 0
    assert not target.exists()
    for window in windows:
        window.deleteLater()
    qt_app.processEvents()


def test_runtime_check_imports_both_new_backends_without_loading_models(monkeypatch, tmp_path):
    imported = []
    def fake_import(name):
        imported.append(name)
        return SimpleNamespace(__version__="fixture")
    monkeypatch.setattr(application.importlib, "import_module", fake_import)
    monkeypatch.setattr(application, "_legacy_modules", lambda: [])
    monkeypatch.setattr(audio_runtime, "_LocalEngine", forbidden)
    monkeypatch.setattr(settings, "load_settings", forbidden)
    path = tmp_path / "imports.json"
    assert application.main(["--check-runtime", "--report", str(path)]) == 0
    report = json.loads(path.read_text(encoding="utf-8"))
    assert {"faster_whisper", "tokenizers", "av", "onnxruntime", "huggingface_hub", "tqdm"} <= set(imported)
    assert report["runtime_imports_passed"] and not report["models_loaded"]
    assert not report["audio_capture"] and not report["settings_written"]


def test_missing_new_runtime_dependency_fails_check_with_clear_module(monkeypatch, tmp_path):
    def fake_import(name):
        if name == "faster_whisper":
            raise ImportError("fixture missing faster-whisper")
        return SimpleNamespace(__version__="fixture")
    monkeypatch.setattr(application.importlib, "import_module", fake_import)
    monkeypatch.setattr(application, "_legacy_modules", lambda: [])
    path = tmp_path / "imports.json"
    assert application.main(["--check-runtime", "--report", str(path)]) == 2
    report = json.loads(path.read_text(encoding="utf-8"))
    assert not report["modules"]["faster_whisper"]["available"]
    assert "fixture missing" in report["modules"]["faster_whisper"]["error"]


def test_verify_cli_forwards_backend_device_and_precision_without_user_settings(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(verification, "main", lambda argv: calls.append(argv) or 7)
    monkeypatch.setattr(settings, "load_settings", forbidden)
    monkeypatch.setattr(settings, "save_settings", forbidden)
    monkeypatch.setattr(audio_runtime, "_LocalEngine", forbidden)
    options = ["--recognition-backend", "whisper", "--translation-backend", "nllb",
               "--recognition-device", "cuda", "--translation-device", "cuda",
               "--recognition-compute-type", "int8_float16", "--translation-compute-type", "float16"]
    report = tmp_path / "report.json"
    assert application.main(["--verify-models", "--model-root", str(tmp_path / "models"),
                             "--report", str(report), *options]) == 7
    assert calls == [["--model-root", str(tmp_path / "models"), "--report", str(report), *options]]


def test_verify_cli_default_does_not_forward_gui_settings_or_overrides(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(verification, "main", lambda argv: calls.append(argv) or 0)
    monkeypatch.setattr(settings, "load_settings", forbidden)
    assert application.main(["--verify-models", "--model-root", str(tmp_path)]) == 0
    assert calls == [["--model-root", str(tmp_path)]]


def test_invalid_backend_is_rejected_by_argparse_before_runtime_access(monkeypatch):
    monkeypatch.setattr(audio_runtime, "_LocalEngine", forbidden)
    with pytest.raises(SystemExit) as error:
        application.main(["--recognition-backend", "unrecognized"])
    assert error.value.code == 2


@pytest.mark.parametrize("explicit", [False, True])
def test_gui_preserves_saved_backend_unless_cli_explicitly_overrides(monkeypatch, tmp_path, explicit):
    from fanyiji_next.models import AppSettings

    qt_app = QApplication.instance() or QApplication([])
    target = tmp_path / "settings.json"
    target.write_text("fixture; reading is mocked", encoding="utf-8")
    saved = AppSettings(recognition_backend="whisper", translation_backend="nllb", system_audio_processing="mdx_gtcrn",
                        mdx_model_file="D:/chosen/mdx.onnx", gtcrn_model_file="D:/chosen/gtcrn.onnx")
    monkeypatch.setattr(settings, "load_settings", lambda path: saved)
    monkeypatch.setattr(settings, "save_settings", forbidden)
    monkeypatch.setattr(audio_runtime, "_LocalEngine", forbidden)
    windows = []
    real_window = studio.StudioWindow
    def create_window(*args, **kwargs):
        window = real_window(*args, **kwargs)
        windows.append(window)
        return window
    def inspect_idle(_self):
        window = windows[-1]
        assert window.settings.recognition_backend == ("sensevoice" if explicit else "whisper")
        assert window.settings.translation_backend == ("m2m100" if explicit else "nllb")
        assert window.settings.system_audio_processing == ("raw" if explicit else "mdx_gtcrn")
        assert window.settings.mdx_model_file == saved.mdx_model_file
        assert window.settings.gtcrn_model_file == saved.gtcrn_model_file
        assert window._settings_source == str(target.resolve())
        assert str(target.resolve()) in window.mode_label.toolTip()
        assert window._controller._runtime is None
        window.close()
        return 0
    monkeypatch.setattr(studio, "StudioWindow", create_window)
    monkeypatch.setattr(QApplication, "exec", inspect_idle)
    options = ["--recognition-backend", "sensevoice", "--translation-backend", "m2m100", "--system-audio-processing", "raw"] if explicit else []
    assert application.main(["--settings-file", str(target), "--model-root", str(tmp_path / "models"), *options]) == 0
    for window in windows:
        window.deleteLater()
    qt_app.processEvents()


def test_executable_cli_forwards_audio_processing_mode_and_paths_without_user_settings(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(verification, "main", lambda argv: calls.append(argv) or 0)
    monkeypatch.setattr(settings, "load_settings", forbidden)
    monkeypatch.setattr(settings, "save_settings", forbidden)
    options = ["--system-audio-processing", "mdx_gtcrn", "--mdx-model-file", "D:/chosen-mdx.onnx",
               "--gtcrn-model-file", "D:/chosen-gtcrn.onnx"]
    assert application.main(["--verify-models", "--model-root", str(tmp_path), *options]) == 0
    assert calls == [["--model-root", str(tmp_path), *options]]


def make_project(tmp_path):
    project = tmp_path / "project"
    source = project / "src/fanyiji_next/app.py"
    source.parent.mkdir(parents=True)
    source.write_text("# fixture source marker\n", encoding="utf-8")
    (project / "pyproject.toml").write_text('[project]\nname = "fanyiji-voice-next"\n', encoding="utf-8")
    return project, source


def write_profile(path, **overrides):
    from fanyiji_next.models import AppSettings

    settings_value = AppSettings(**overrides)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"schema_version": 1, **asdict(settings_value)}), encoding="utf-8")
    return settings_value


def probe_normal_startup(monkeypatch, arguments, check):
    qt_app = QApplication.instance() or QApplication([])
    windows = []
    real_window = studio.StudioWindow

    def create_window(*args, **kwargs):
        window = real_window(*args, **kwargs)
        windows.append(window)
        return window

    def inspect_idle(_self):
        window = windows[-1]
        assert window._controller._runtime is None
        assert window._controller.state == "stopped"
        check(window)
        window.close()
        return 0

    monkeypatch.setattr(studio, "StudioWindow", create_window)
    monkeypatch.setattr(QApplication, "exec", inspect_idle)
    monkeypatch.setattr(audio_runtime, "VoiceRuntime", forbidden)
    monkeypatch.setattr(audio_runtime, "_LocalEngine", forbidden)
    monkeypatch.setattr(settings, "save_settings", forbidden)
    try:
        assert application.main(arguments) == 0
    finally:
        for window in windows:
            window.deleteLater()
        qt_app.processEvents()


@pytest.mark.parametrize("existing", [False, True])
def test_portable_release_defaults_preserve_saved_profile_and_relocate_models(monkeypatch, tmp_path, existing):
    from fanyiji_next.release_profile import write_release_defaults

    portable = tmp_path / "moved-portable"
    portable.mkdir()
    write_release_defaults(portable / "release-defaults.json")
    model = portable / "models/whisper-large-v3-ct2/model.bin"
    model.parent.mkdir(parents=True)
    model.write_bytes(b"fixture")
    appdata = tmp_path / "isolated-appdata"
    target = appdata / "FanyiJiVoiceNext/settings.json"
    if existing:
        write_profile(target, recognition_backend="sensevoice", translation_backend="m2m100")
    before = target.read_bytes() if existing else None
    monkeypatch.setenv("LOCALAPPDATA", str(appdata))
    monkeypatch.setattr(application.sys, "frozen", True, raising=False)
    monkeypatch.setattr(application.sys, "executable", str(portable / "FanyiJiVoiceNext.exe"))
    monkeypatch.setattr(application, "_discover_project_root", lambda: None)

    def check(window):
        assert window.settings.recognition_backend == ("sensevoice" if existing else "whisper")
        assert window.settings.translation_backend == ("m2m100" if existing else "hymt")
        if not existing:
            assert window.settings.system_audio_processing == "mdx_gtcrn"
            assert window.settings.whisper_model_dir == str(model.parent)
        assert window.update_button.isVisible()

    probe_normal_startup(monkeypatch, [], check)
    assert (target.read_bytes() if target.exists() else None) == before


def test_update_restart_preserves_explicit_profile_and_model_cli(monkeypatch, tmp_path):
    target = tmp_path / "custom-settings.json"
    write_profile(target)
    command = ["--settings-file", str(target), "--require-settings", "--model-root", str(tmp_path / "chosen-models")]
    monkeypatch.setattr(application, "_discover_project_root", lambda: None)
    probe_normal_startup(monkeypatch, command, lambda window: (
        window._update_restart_args == tuple(command) or pytest.fail("restart changed profile/model selection")
    ))


@pytest.mark.parametrize("frozen", [False, True])
@pytest.mark.parametrize("appdata_name", ["normal-local", "packages-localcache"])
def test_project_profile_wins_for_source_and_standard_dist_under_different_appdata(monkeypatch, tmp_path, frozen, appdata_name, stable_version):
    project, source = make_project(tmp_path)
    profile = project / "settings/voice.settings.json"
    saved = write_profile(profile, recognition_backend="whisper", translation_backend="hymt", system_audio_processing="mdx_gtcrn")
    appdata = tmp_path / appdata_name
    legacy = appdata / "FanyiJiVoiceNext/settings.json"
    write_profile(legacy, recognition_backend="sensevoice", translation_backend="m2m100")
    monkeypatch.setenv("LOCALAPPDATA", str(appdata))
    monkeypatch.setattr(application, "__file__", str(source))
    monkeypatch.setattr(application.sys, "frozen", frozen, raising=False)
    executable = project / "dist/0.4.6-voice-verified/FanyiJiVoiceNext/FanyiJiVoiceNext.exe"
    monkeypatch.setattr(application.sys, "executable", str(executable))
    unrelated_cwd = tmp_path / "elsewhere"
    unrelated_cwd.mkdir()
    monkeypatch.chdir(unrelated_cwd)
    assert application._discover_project_root() == project

    def check(window):
        assert window.settings.recognition_backend == saved.recognition_backend
        assert window.settings.translation_backend == saved.translation_backend
        assert window.settings.system_audio_processing == saved.system_audio_processing
        assert window._settings_source == str(profile.resolve())
        assert window._controller._settings_file == profile.resolve()
    probe_normal_startup(monkeypatch, [], check)


def test_project_direct_exe_discovers_models_from_same_project_root(monkeypatch, tmp_path, stable_version):
    project, _source = make_project(tmp_path)
    profile = project / "settings/voice.settings.json"
    write_profile(profile)
    model = project / "models/asr/model.int8.onnx"
    model.parent.mkdir(parents=True)
    model.write_bytes(b"fixture")
    monkeypatch.setattr(application.sys, "frozen", True, raising=False)
    monkeypatch.setattr(application.sys, "executable", str(project / "dist/0.4.6-voice/FanyiJiVoiceNext/FanyiJiVoiceNext.exe"))
    probe_normal_startup(monkeypatch, [], lambda window: assert_model_path(window, model))


def assert_model_path(window, path):
    assert window.settings.asr_model == str(path)


def test_explicit_settings_file_wins_over_an_invalid_project_profile(monkeypatch, tmp_path):
    project, source = make_project(tmp_path)
    profile = project / "settings/voice.settings.json"
    profile.parent.mkdir()
    profile.write_text("{invalid project JSON", encoding="utf-8")
    explicit = tmp_path / "chosen/voice.json"
    saved = write_profile(explicit, recognition_backend="sensevoice", translation_backend="nllb")
    monkeypatch.setattr(application, "__file__", str(source))
    monkeypatch.setattr(application.sys, "frozen", False, raising=False)

    def check(window):
        assert window._settings_source == str(explicit.resolve())
        assert window.settings.recognition_backend == saved.recognition_backend
        assert window.settings.translation_backend == saved.translation_backend
    probe_normal_startup(monkeypatch, ["--settings-file", str(explicit)], check)


@pytest.mark.parametrize("problem", ["invalid-json", "invalid-setting"])
def test_independent_explicit_profile_without_require_flag_keeps_legacy_error_notice_and_file(monkeypatch, tmp_path, problem):
    project, source = make_project(tmp_path)
    target = tmp_path / "independent/settings.json"
    target.parent.mkdir()
    target.write_text("{" if problem == "invalid-json" else '{"schema_version":1,"font_size":100}', encoding="utf-8")
    before = target.read_bytes()
    monkeypatch.setattr(application, "__file__", str(source))
    monkeypatch.setattr(application.sys, "frozen", False, raising=False)
    monkeypatch.setattr(QMessageBox, "critical", forbidden)

    def check(window):
        assert window._settings_source == str(target.resolve())
        assert "设置读取失败" in window.notice.text()
        assert window.settings.recognition_backend == "sensevoice"
        assert window.settings.translation_backend == "hymt"
    probe_normal_startup(monkeypatch, ["--settings-file", str(target)], check)
    assert target.read_bytes() == before


def test_independent_portable_exe_keeps_legacy_appdata_without_cwd_profile_discovery(monkeypatch, tmp_path):
    project, _source = make_project(tmp_path)
    write_profile(project / "settings/voice.settings.json", recognition_backend="whisper", translation_backend="hymt")
    legacy = tmp_path / "local/FanyiJiVoiceNext/settings.json"
    write_profile(legacy, recognition_backend="sensevoice", translation_backend="m2m100")
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "local"))
    monkeypatch.setattr(application.sys, "frozen", True, raising=False)
    monkeypatch.setattr(application.sys, "executable", str(tmp_path / "independent/FanyiJiVoiceNext.exe"))
    monkeypatch.chdir(project)
    assert application._discover_project_root() is None

    def check(window):
        assert window._settings_source == str(legacy.resolve())
        assert window.settings.recognition_backend == "sensevoice"
        assert window.settings.translation_backend == "m2m100"
    probe_normal_startup(monkeypatch, [], check)


@pytest.mark.parametrize("marker", ["missing", "other-project", "invalid-toml", "invalid-project-table"])
def test_standard_dist_layout_requires_valid_project_identity_marker(monkeypatch, tmp_path, marker):
    project = tmp_path / "unrelated"
    project.mkdir()
    if marker != "missing":
        text = {"other-project": '[project]\nname="another-app"', "invalid-toml": "{",
                "invalid-project-table": 'project="unexpected"'}[marker]
        (project / "pyproject.toml").write_text(text, encoding="utf-8")
    monkeypatch.setattr(application.sys, "frozen", True, raising=False)
    monkeypatch.setattr(application.sys, "executable", str(project / "dist/0.4.6-voice/FanyiJiVoiceNext/FanyiJiVoiceNext.exe"))
    assert application._discover_project_root() is None


@pytest.mark.parametrize("selection", ["auto-project", "explicit-project", "required-explicit"])
@pytest.mark.parametrize("problem", ["missing", "invalid-json", "invalid-setting"])
def test_selected_profile_errors_are_actionable_and_never_create_window_or_load_models(monkeypatch, tmp_path, selection, problem, stable_version):
    qt_app = QApplication.instance() or QApplication([])
    project, source = make_project(tmp_path)
    target = tmp_path / "chosen.json" if selection == "required-explicit" else project / "settings/voice.settings.json"
    if problem != "missing":
        target.parent.mkdir(parents=True, exist_ok=True)
        text = "{" if problem == "invalid-json" else '{"schema_version":1,"recognition_backend":"invalid"}'
        target.write_text(text, encoding="utf-8")
    monkeypatch.setattr(application, "__file__", str(source))
    monkeypatch.setattr(application.sys, "frozen", False, raising=False)
    monkeypatch.setattr(studio, "StudioWindow", forbidden)
    monkeypatch.setattr(audio_runtime, "_LocalEngine", forbidden)
    monkeypatch.setattr(audio_runtime.VoiceRuntime, "start", forbidden)
    monkeypatch.setattr(settings, "save_settings", forbidden)
    monkeypatch.setattr(QApplication, "exec", forbidden)
    messages = []
    monkeypatch.setattr(QMessageBox, "critical", lambda parent, title, message: messages.append((title, message)))
    report = tmp_path / "startup-error.json"
    arguments = ["--report", str(report)]
    if selection != "auto-project":
        arguments.extend(("--settings-file", str(target)))
    if selection == "required-explicit":
        arguments.append("--require-settings")
    assert application.main(arguments) == 2
    assert len(messages) == 1
    assert str(target.resolve()) in messages[0][1]
    assert "--settings-file" in messages[0][1] and "恢复" in messages[0][1]
    evidence = json.loads(report.read_text(encoding="utf-8"))
    assert evidence["settings_file"] == str(target.resolve())
    assert evidence["audio_capture"] is evidence["models_loaded"] is evidence["settings_written"] is False
    qt_app.processEvents()


@pytest.mark.parametrize("required", [False, True])
@pytest.mark.parametrize("frozen", [False, True])
def test_dev_project_first_start_reads_stable_only_then_saves_and_reopens_preview(monkeypatch, tmp_path, required, frozen):
    project, source = make_project(tmp_path)
    formal = project / "settings/voice.settings.json"
    saved = write_profile(formal, recognition_backend="whisper", translation_backend="hymt",
                          system_audio_processing="mdx_gtcrn")
    legacy = json.loads(formal.read_text(encoding="utf-8"))
    legacy.pop("window_transparency_enabled")
    formal.write_text(json.dumps(legacy), encoding="utf-8")
    before = formal.read_bytes()
    preview = formal.with_name("voice.preview.settings.json")
    monkeypatch.setattr(application, "__version__", "0.5.2-dev")
    monkeypatch.setattr(application, "__file__", str(source))
    monkeypatch.setattr(application.sys, "frozen", frozen, raising=False)
    monkeypatch.setattr(application.sys, "executable", str(project / "dist/0.5.2-dev-voice/FanyiJiVoiceNext/FanyiJiVoiceNext.exe"))
    original_save = settings.save_settings
    arguments = ["--require-settings"] if required else []

    def change_transparency(window):
        assert formal.read_bytes() == before and not preview.exists()
        assert window._settings_source == str(preview.resolve())
        assert window._controller._settings_file == preview.resolve()
        assert window.settings.recognition_backend == saved.recognition_backend
        assert window.settings.translation_backend == saved.translation_backend
        assert window.settings.system_audio_processing == saved.system_audio_processing
        monkeypatch.setattr(settings, "save_settings", original_save)
        window.transparency_slider.setValue(45)
        assert window.settings.window_opacity == 55
        assert window._controller._accepted_settings == window.settings
        assert preview.is_file() and formal.read_bytes() == before

    probe_normal_startup(monkeypatch, arguments, change_transparency)
    preview_before = preview.read_bytes()

    def check_reopened(window):
        assert window._settings_source == str(preview.resolve())
        assert window._controller._settings_file == preview.resolve()
        assert window.settings.window_opacity == 55
        assert window.settings.window_transparency_enabled is True
        assert window.settings.recognition_backend == saved.recognition_backend
        assert window.settings.translation_backend == saved.translation_backend

    probe_normal_startup(monkeypatch, arguments, check_reopened)
    assert formal.read_bytes() == before and preview.read_bytes() == preview_before


@pytest.mark.parametrize("required", [False, True])
def test_dev_explicit_stable_profile_still_overrides_existing_preview(monkeypatch, tmp_path, required):
    project, source = make_project(tmp_path)
    formal = project / "settings/voice.settings.json"
    write_profile(formal, recognition_backend="sensevoice", translation_backend="m2m100", window_opacity=100)
    preview = formal.with_name("voice.preview.settings.json")
    write_profile(preview, recognition_backend="whisper", translation_backend="hymt", window_opacity=45)
    formal_before, preview_before = formal.read_bytes(), preview.read_bytes()
    monkeypatch.setattr(application, "__version__", "0.5.2-dev")
    monkeypatch.setattr(application, "__file__", str(source))
    monkeypatch.setattr(application.sys, "frozen", False, raising=False)

    def check(window):
        assert window._settings_source == str(formal.resolve())
        assert window._controller._settings_file == formal.resolve()
        assert window.settings.recognition_backend == "sensevoice"
        assert window.settings.translation_backend == "m2m100"
        assert window.settings.window_opacity == 100

    arguments = ["--settings-file", str(formal)] + (["--require-settings"] if required else [])
    probe_normal_startup(monkeypatch, arguments, check)
    assert formal.read_bytes() == formal_before and preview.read_bytes() == preview_before


@pytest.mark.parametrize("required", [False, True])
@pytest.mark.parametrize("problem", ["both-missing", "invalid-formal-json", "invalid-formal-setting",
                                     "invalid-preview-json", "invalid-preview-setting", "preview-directory"])
def test_dev_default_profile_error_reports_actual_read_source_without_falling_back(monkeypatch, tmp_path, required, problem):
    qt_app = QApplication.instance() or QApplication([])
    project, source = make_project(tmp_path)
    formal = project / "settings/voice.settings.json"
    preview = formal.with_name("voice.preview.settings.json")
    if problem != "both-missing":
        write_profile(formal, recognition_backend="whisper", translation_backend="hymt")
    selected = preview if "preview" in problem else formal
    if problem.endswith("json"):
        selected.write_text("{invalid fixture JSON", encoding="utf-8")
    elif problem.endswith("setting"):
        selected.write_text('{"schema_version":1,"recognition_backend":"invalid"}', encoding="utf-8")
    elif problem == "preview-directory":
        preview.mkdir()
    before = {path: path.read_bytes() for path in (formal, preview) if path.is_file()}
    monkeypatch.setattr(application, "__version__", "0.5.2-dev")
    monkeypatch.setattr(application, "__file__", str(source))
    monkeypatch.setattr(application.sys, "frozen", False, raising=False)
    monkeypatch.setattr(studio, "StudioWindow", forbidden)
    monkeypatch.setattr(audio_runtime, "_LocalEngine", forbidden)
    monkeypatch.setattr(audio_runtime.VoiceRuntime, "start", forbidden)
    monkeypatch.setattr(settings, "save_settings", forbidden)
    monkeypatch.setattr(QApplication, "exec", forbidden)
    messages = []
    monkeypatch.setattr(QMessageBox, "critical", lambda parent, title, message: messages.append(message))
    report = tmp_path / "startup-error.json"
    arguments = ["--report", str(report)] + (["--require-settings"] if required else [])
    assert application.main(arguments) == 2
    assert len(messages) == 1 and str(selected.resolve()) in messages[0]
    evidence = json.loads(report.read_text(encoding="utf-8"))
    assert evidence["settings_file"] == str(selected.resolve())
    assert evidence["audio_capture"] is evidence["models_loaded"] is evidence["settings_written"] is False
    assert all(path.read_bytes() == data for path, data in before.items())
    if "preview" not in problem:
        assert not preview.exists()
    qt_app.processEvents()


def test_root_launcher_uses_project_profile_and_fails_if_it_is_missing():
    launcher = (Path(__file__).resolve().parents[1] / "启动翻译姬.cmd").read_text(encoding="utf-8")
    assert 'set "profile=%~dp0settings\\voice.settings.json"' in launcher
    assert '--settings-file "%profile%"' in launcher
    assert "--require-settings" in launcher
    assert "%LOCALAPPDATA%" not in launcher
    assert launcher.index('if not exist "%profile%"') < launcher.index('start "" "%app%"')
    assert "exit /b 1" in launcher[launcher.index('if not exist "%profile%"'):]
