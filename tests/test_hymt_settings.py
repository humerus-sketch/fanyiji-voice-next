"""Hy-MT2 settings and offscreen UI contracts; no native engines or user files."""

import json
import os
from dataclasses import replace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtCore import QPoint, QRect
from PySide6.QtWidgets import QApplication, QFileDialog

from fanyiji_next import app as application
from fanyiji_next import settings as settings_store
from fanyiji_next import verification
from fanyiji_next.controller import AppController
from fanyiji_next.models import AppSettings, RuntimeEvent
from fanyiji_next.settings import SettingsError, load_settings, save_settings, validate_settings, with_local_models
from fanyiji_next.studio import RuntimeSettingsDialog, StudioWindow


def forbidden(*args, **kwargs):
    pytest.fail("must not access engines, devices, or user settings")


def settle(app):
    for _ in range(6):
        app.processEvents()


@pytest.fixture
def widgets(monkeypatch, tmp_path):
    app = QApplication.instance() or QApplication([])
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "isolated-user-data"))
    created = []

    def create(settings=None, *, window=False):
        widget = StudioWindow(settings, prototype=False) if window else RuntimeSettingsDialog(settings or AppSettings())
        created.append(widget)
        widget.show()
        settle(app)
        return widget, app

    yield create
    for widget in reversed(created):
        if isinstance(widget, StudioWindow):
            widget.finish_shutdown()
        else:
            widget.close()
        widget.deleteLater()
    settle(app)
    assert not (tmp_path / "isolated-user-data" / "FanyiJiVoiceNext" / "settings.json").exists()


def choose(dialog, field, value):
    combo = dialog.engine_combos[field]
    index = combo.findData(value)
    assert index >= 0
    combo.setCurrentIndex(index)


def test_new_default_is_hymt_but_missing_paths_are_not_invented():
    settings = AppSettings()
    assert settings.translation_backend == "hymt"
    assert settings.recognition_backend == "sensevoice"
    assert settings.hymt_model_file == settings.hymt_server_exe == ""
    validate_settings(settings)


@pytest.mark.parametrize("backend", [None, "m2m100", "nllb"])
def test_existing_schema1_keeps_its_backend_and_paths_without_rewriting(tmp_path, backend):
    target = tmp_path / "legacy.json"
    payload = {"schema_version": 1, "translation_model_dir": "C:/legacy/m2m",
               "nllb_model_dir": "C:/legacy/nllb", "translation_device": "cpu",
               "translation_compute_type": "int8", "font_size": 34}
    if backend is not None:
        payload["translation_backend"] = backend
    target.write_text(json.dumps(payload), encoding="utf-8")
    before = target.read_bytes()
    loaded = load_settings(target)
    assert loaded.translation_backend == (backend or "m2m100")
    assert loaded.translation_model_dir == "C:/legacy/m2m" and loaded.nllb_model_dir == "C:/legacy/nllb"
    assert loaded.hymt_model_file == loaded.hymt_server_exe == "" and loaded.font_size == 34
    assert target.read_bytes() == before


def test_hymt_roundtrip_preserves_independent_paths_and_inactive_ct2_preferences(tmp_path):
    target = tmp_path / "fixture.json"
    original = AppSettings(hymt_model_file="D:/模型/Hy-MT2-Q8_0.gguf", hymt_server_exe="D:/运行库/llama-server.exe",
                           translation_model_dir="C:/m2m", nllb_model_dir="C:/nllb",
                           nllb_tokenizer_dir="C:/nllb-tokenizer", translation_device="cpu",
                           translation_compute_type="float16")
    save_settings(original, target)
    assert load_settings(target) == original
    assert json.loads(target.read_text(encoding="utf-8"))["schema_version"] == 1
    with pytest.raises(SettingsError, match="CPU mode uses int8"):
        validate_settings(replace(original, translation_backend="nllb"))


@pytest.mark.parametrize("field,value", [("hymt_model_file", None), ("hymt_server_exe", None),
                                         ("hymt_model_file", "bad\0path"), ("hymt_server_exe", "bad\0path")])
def test_invalid_hymt_paths_are_not_saved(tmp_path, field, value):
    target = tmp_path / "invalid.json"
    with pytest.raises(SettingsError, match=field):
        save_settings(replace(AppSettings(), **{field: value}), target)
    assert not target.exists()


def test_model_root_finds_only_stable_hymt_paths_and_preserves_explicit_paths(tmp_path):
    root = tmp_path / "models"
    model = root / "hymt2-1.8b" / "Hy-MT2-1.8B-Q8_0.gguf"
    runtime = tmp_path / "runtime" / "llama-b11696-cuda12.4" / "llama-server.exe"
    for file in (model, runtime):
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_bytes(b"public fake file; never executed")
    result = with_local_models(AppSettings(), root)
    assert result.hymt_model_file == str(model) and result.hymt_server_exe == str(runtime)
    chosen = replace(result, hymt_model_file="D:/chosen/model.gguf", hymt_server_exe="D:/chosen/server.exe")
    assert with_local_models(chosen, root) == chosen
    assert with_local_models(AppSettings(), tmp_path / "missing-models").hymt_model_file == ""
    assert not (tmp_path / "settings.json").exists()


def test_hymt_dialog_shows_q8_cuda_and_preserves_hidden_ct2_values(widgets):
    original = AppSettings(translation_backend="nllb", translation_device="cuda", translation_compute_type="float16",
                           translation_model_dir="C:/m2m", nllb_model_dir="C:/nllb",
                           hymt_model_file="C:/hymt/model.gguf", hymt_server_exe="C:/runtime/server.exe")
    dialog, app = widgets(original)
    choose(dialog, "translation_backend", "hymt")
    settle(app)
    assert dialog.engine_combos["translation_backend"].currentText() == "Hy-MT2-1.8B Q8_0"
    for name in ("hymt_model_file", "hymt_server_exe"):
        assert dialog.path_groups[name].isVisible()
    for name in ("translation_model_dir", "translation_spm", "nllb_model_dir", "nllb_tokenizer_dir"):
        assert dialog.path_groups[name].isHidden()
    assert all(dialog.engine_rows[name].isHidden() for name in ("translation_device", "translation_compute_type"))
    assert "Q8_0" in dialog.engine_hint.text() and "CUDA" in dialog.engine_hint.text()
    assert "不自动回退" in dialog.engine_hint.text()
    choose(dialog, "translation_backend", "nllb")
    assert dialog.engine_combos["translation_device"].currentData() == "cuda"
    assert dialog.engine_combos["translation_compute_type"].currentData() == "float16"
    assert dialog.path_edits["hymt_model_file"].text() == original.hymt_model_file
    assert dialog.path_edits["nllb_model_dir"].text() == original.nllb_model_dir


@pytest.mark.parametrize("field,extension", [("hymt_model_file", "*.gguf"), ("hymt_server_exe", "*.exe")])
def test_hymt_browse_uses_file_picker(widgets, monkeypatch, field, extension):
    dialog, _app = widgets()
    calls = []
    def pick(parent, title, current, filters):
        calls.append(filters)
        return "C:/chosen/local-file", filters
    monkeypatch.setattr(QFileDialog, "getOpenFileName", pick)
    monkeypatch.setattr(QFileDialog, "getExistingDirectory", forbidden)
    dialog._browse(field)
    assert calls and extension in calls[0]
    assert dialog.path_edits[field].text() == "C:/chosen/local-file"


def test_hymt_apply_keeps_other_backend_paths_and_cancel_does_not_mutate_settings(widgets):
    original = AppSettings(translation_backend="nllb", nllb_model_dir="C:/nllb", nllb_tokenizer_dir="C:/tokenizer",
                           translation_model_dir="C:/m2m", translation_spm="C:/m2m/spm.model", font_size=34)
    window, app = widgets(original, window=True)
    accepted, starts = [], []
    window.settings_changed.connect(accepted.append)
    window.start_requested.connect(starts.append)
    dialog = window.open_runtime_settings()
    choose(dialog, "translation_backend", "hymt")
    dialog.path_edits["hymt_model_file"].setText(" C:/hymt/model.gguf ")
    dialog.path_edits["hymt_server_exe"].setText(" C:/runtime/llama-server.exe ")
    assert window.settings == original
    dialog.apply_button.click()
    settle(app)
    selected = window.settings
    assert selected.translation_backend == "hymt" and selected.hymt_model_file == "C:/hymt/model.gguf"
    assert selected.hymt_server_exe == "C:/runtime/llama-server.exe"
    for field in ("nllb_model_dir", "nllb_tokenizer_dir", "translation_model_dir", "translation_spm", "font_size"):
        assert getattr(selected, field) == getattr(original, field)
    assert accepted == [selected] and not starts
    dialog = window.open_runtime_settings()
    choose(dialog, "translation_backend", "m2m100")
    dialog.path_edits["hymt_model_file"].setText("C:/discarded.gguf")
    dialog.reject()
    assert window.settings == selected and len(accepted) == 1


def test_small_hymt_dialog_keeps_apply_and_both_file_fields_reachable(widgets):
    dialog, app = widgets()
    area = QRect(120, 40, 1280, 680)
    dialog.fit_to_screen(area)
    settle(app)
    assert area.contains(dialog.frameGeometry())
    for button in dialog.buttons.buttons():
        assert button.isVisible()
        assert dialog.rect().contains(QRect(button.mapTo(dialog, QPoint()), button.size()))
    viewport = dialog.scroll.viewport()
    for name in ("hymt_model_file", "hymt_server_exe"):
        edit = dialog.path_edits[name]
        dialog.scroll.ensureWidgetVisible(edit)
        settle(app)
        assert edit.isVisible()
        assert viewport.rect().contains(QRect(edit.mapTo(viewport, QPoint()), edit.size()))


def test_old_verifier_default_is_explicit_m2m100_while_hymt_paths_can_override(tmp_path):
    old = verification.create_verification_settings(tmp_path / "models")
    assert old.translation_backend == "m2m100" and old.translation_device == "cpu"
    profile = verification.create_verification_settings(tmp_path / "models", translation_backend="hymt",
                                                       hymt_model_file="C:/chosen.gguf", hymt_server_exe="C:/server.exe")
    assert profile.translation_backend == "hymt"
    assert profile.hymt_model_file == "C:/chosen.gguf" and profile.hymt_server_exe == "C:/server.exe"


def test_verify_cli_hymt_paths_never_read_or_write_user_settings(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(settings_store, "load_settings", forbidden)
    monkeypatch.setattr(settings_store, "save_settings", forbidden)
    monkeypatch.setattr(verification, "run_verification", lambda root, *, settings: calls.append(settings) or {"ok": True})
    args = ["--model-root", str(tmp_path / "models"), "--translation-backend", "hymt",
            "--hymt-model-file", "C:/chosen.gguf", "--hymt-server-exe", "C:/server.exe"]
    assert verification.main(args) == 0
    assert calls[0].translation_backend == "hymt" and calls[0].hymt_model_file == "C:/chosen.gguf"
    assert calls[0].hymt_server_exe == "C:/server.exe"
    forwarded = []
    monkeypatch.setattr(verification, "main", lambda argv: forwarded.append(argv) or 0)
    assert application.main(["--verify-models", *args]) == 0
    assert forwarded == [args]


@pytest.mark.parametrize("field", ["hymt_model_file", "hymt_server_exe"])
def test_active_session_rejects_hymt_path_changes_without_restarting_or_saving(widgets, monkeypatch, field):
    monkeypatch.setattr(settings_store, "save_settings", forbidden)
    window, app = widgets(AppSettings(), window=True)
    class FakeRuntime:
        def __init__(self, callback):
            self.callback = callback
            self.is_running = False
        def start(self, settings, session_id):
            self.session_id = session_id
            self.is_running = True
        def stop(self):
            self.is_running = False
            self.callback(RuntimeEvent("state", self.session_id, status="stopped"))
    controller = AppController(window, runtime_factory=FakeRuntime, persist_settings=False)
    original = window.settings
    try:
        controller.start_session(original)
        changed = replace(original, **{field: "C:/changed/local-file"})
        window.update_settings(changed)
        window.settings_changed.emit(changed)
        settle(app)
        assert window.settings == original
        assert "请先停止" in window.notice.text()
        assert controller.session_id == 1 and controller._runtime.is_running
    finally:
        controller.shutdown()
        settle(app)
