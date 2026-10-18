from dataclasses import replace
from pathlib import Path

import pytest
from PySide6.QtWidgets import QApplication

from fanyiji_next.models import AppSettings, ChannelSettings
from fanyiji_next.settings import load_settings, save_settings, validate_settings, with_model_combination
from fanyiji_next.studio import StudioWindow, RuntimeSettingsDialog


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


def test_pair_settings_roundtrip_preserves_user_preferences(tmp_path):
    initial = AppSettings(font_size=34, window_opacity=65,
                          system=ChannelSettings(direction="ja_to_zh"))
    selected = with_model_combination(initial, "B", tmp_path / "Reader_models")
    validate_settings(selected)
    assert selected.recognition_backend == "fun_asr"
    assert selected.translation_backend == "translategemma"
    assert selected.system_audio_processing == "raw"
    assert selected.font_size == 34 and selected.window_opacity == 65
    assert Path(selected.gemma_model_file).is_relative_to(tmp_path / "Reader_models")
    assert selected.whisper_model_dir == selected.hymt_model_file == ""
    path = tmp_path / "settings.json"
    save_settings(selected, path)
    assert load_settings(path) == selected


def test_AB_settings_only_offer_supported_pairs_and_original_audio(app, tmp_path):
    selected = with_model_combination(AppSettings(), "A", tmp_path / "Reader_models")
    window = StudioWindow(selected, prototype=False)
    dialog = RuntimeSettingsDialog(selected, parent=window)
    assert "Qwen" in window.mode_label.text()
    assert window.download_models_button.text() == "模型下载 · A/B"
    assert dialog.engine_combos["recognition_backend"].count() == 2
    assert dialog.engine_combos["translation_backend"].count() == 1
    assert dialog.engine_combos["system_audio_processing"].count() == 1
    assert dialog.channel_controls["system"][1].currentText() == "自动识别 → 中文"
    assert window.panels["system"].direction_label.text() == "自动识别 → 中文"
    combo = dialog.engine_combos["recognition_backend"]
    combo.setCurrentIndex(combo.findData("fun_asr"))
    dialog.accept()
    assert dialog.result_settings.model_combination == "B"
    assert dialog.result_settings.translation_backend == "translategemma"
    dialog.close()
    window.close()


def test_runtime_dialog_applies_all_b_paths_and_keeps_selection_on_reopen(app, tmp_path):
    selected = with_model_combination(AppSettings(font_size=34, window_opacity=65), "A", tmp_path / "Reader_models")
    window = StudioWindow(selected, prototype=False)
    saved = []
    window.settings_changed.connect(saved.append)
    try:
        dialog = window.open_runtime_settings()
        combo = dialog.engine_combos["recognition_backend"]
        combo.setCurrentIndex(combo.findData("fun_asr"))
        dialog.accept()
        assert window.settings.model_combination == "B"
        assert window.settings.recognition_backend == "fun_asr"
        assert not window.settings.qwen_asr_model_dir
        assert Path(window.settings.fun_asr_model_dir).parent == tmp_path / "Reader_models"
        assert window.settings.font_size == 34 and window.settings.window_opacity == 65
        assert saved == [window.settings]
        reopened = window.open_runtime_settings()
        assert reopened.engine_combos["recognition_backend"].currentData() == "fun_asr"
        assert reopened.path_edits["fun_asr_model_dir"].text() == window.settings.fun_asr_model_dir
        reopened.reject()
    finally:
        window.close()


def test_new_voice_window_has_production_a_defaults_and_only_ab_options(app):
    window = StudioWindow(prototype=False)
    try:
        assert window.settings.model_combination == "A"
        assert window.settings.recognition_backend == "qwen3_asr"
        assert window.settings.translation_backend == "translategemma"
        dialog = window.open_runtime_settings()
        assert [dialog.engine_combos["recognition_backend"].itemData(i) for i in range(2)] == ["qwen3_asr", "fun_asr"]
        assert dialog.engine_combos["recognition_backend"].currentData() == "qwen3_asr"
        assert set(dialog.path_edits) == {"qwen_asr_model_dir", "fun_asr_model_dir", "gemma_model_file", "vad_model"}
        dialog.reject()
    finally:
        window.close()


def test_saved_absolute_model_root_does_not_control_new_name(tmp_path):
    old = with_model_combination(AppSettings(), "A", tmp_path / "Old_models")
    new = with_model_combination(old, "A", tmp_path / "Renamed_models")
    for value in (new.qwen_asr_model_dir, new.gemma_model_file, new.vad_model):
        assert str(tmp_path / "Renamed_models") in value
        assert "Old_models" not in value
