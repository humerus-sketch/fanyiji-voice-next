"""Backend settings and real Qt dialog regressions; no models, devices, or user files."""
import json
import os
from dataclasses import FrozenInstanceError, asdict, replace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtWidgets import QApplication, QDialog, QDialogButtonBox

from fanyiji_next.models import AppSettings, ChannelSettings
from fanyiji_next.settings import SettingsError, load_settings, save_settings, validate_settings, with_local_models
from fanyiji_next.studio import RuntimeSettingsDialog, StudioWindow


PATHS = {
    "asr_model": "C:/fixtures/sensevoice/model.int8.onnx",
    "asr_tokens": "C:/fixtures/sensevoice/tokens.txt",
    "translation_model_dir": "C:/fixtures/m2m100",
    "translation_spm": "C:/fixtures/m2m100/sentencepiece.bpe.model",
    "whisper_model_dir": "C:/fixtures/whisper-large-v3-ct2",
    "nllb_model_dir": "C:/fixtures/nllb-ct2",
    "nllb_tokenizer_dir": "C:/fixtures/nllb-tokenizer",
    "vad_model": "C:/fixtures/vad/silero_vad.onnx",
}


@pytest.fixture(scope="module")
def qt_app():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def widgets(qt_app, monkeypatch, tmp_path):
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "isolated-user-data"))
    created = []

    def make(kind, settings):
        widget = StudioWindow(settings, prototype=False) if kind == "window" else RuntimeSettingsDialog(settings)
        created.append(widget)
        widget.show()
        settle(qt_app)
        return widget

    yield make
    for widget in reversed(created):
        widget.close()
        widget.deleteLater()
    settle(qt_app)
    assert not (tmp_path / "isolated-user-data" / "FanyiJiVoiceNext" / "settings.json").exists()


def settle(app):
    for _ in range(4):
        app.processEvents()


def choose(dialog, field, value):
    combo = dialog.engine_combos[field]
    index = combo.findData(value)
    assert index >= 0
    combo.setCurrentIndex(index)


def advanced_settings():
    return AppSettings(**PATHS, recognition_backend="whisper", recognition_device="cuda",
        recognition_compute_type="float16", translation_backend="nllb", translation_device="cuda",
        translation_compute_type="int8_float16", font_size=34, show_original=False,
        always_on_top=True, window_opacity=83,
        microphone=ChannelSettings(device_index=15),
        system=ChannelSettings(direction="ja_to_zh", device_index=17))


def test_new_fields_default_to_hymt_without_inventing_model_paths():
    settings = AppSettings()
    assert settings.recognition_backend == "sensevoice" and settings.translation_backend == "hymt"
    assert settings.recognition_device == settings.translation_device == "cpu"
    assert settings.recognition_compute_type == settings.translation_compute_type == "int8"
    assert settings.whisper_model_dir == settings.nllb_model_dir == settings.nllb_tokenizer_dir == settings.vad_model == ""
    with pytest.raises(FrozenInstanceError):
        settings.translation_backend = "nllb"


def test_schema1_configuration_without_new_fields_retains_original_values_and_defaults(tmp_path):
    target = tmp_path / "schema1.json"
    old = {"schema_version": 1, "microphone": {"enabled": False, "direction": "zh_to_ja", "device_index": 2},
           "system": {"enabled": True, "direction": "ja_to_zh", "device_index": 4},
           "asr_model": "C:/existing/asr.onnx", "asr_tokens": "C:/existing/tokens.txt",
           "translation_model_dir": "C:/existing/m2m", "translation_spm": "C:/existing/spm.model",
           "font_size": 30, "show_original": False, "always_on_top": True, "window_opacity": 88}
    target.write_text(json.dumps(old), encoding="utf-8")
    before = target.read_bytes()
    settings = load_settings(target)
    assert settings.microphone == ChannelSettings(enabled=False, device_index=2)
    assert settings.system == ChannelSettings(direction="ja_to_zh", device_index=4)
    for name in ("asr_model", "asr_tokens", "translation_model_dir", "translation_spm", "font_size", "show_original", "always_on_top", "window_opacity"):
        assert getattr(settings, name) == old[name]
    assert settings.recognition_backend == "sensevoice" and settings.translation_backend == "m2m100"
    assert settings.recognition_device == settings.translation_device == "cpu"
    assert settings.whisper_model_dir == settings.nllb_model_dir == settings.nllb_tokenizer_dir == settings.vad_model == ""
    assert target.read_bytes() == before


def test_new_backend_fields_roundtrip_in_schema1_with_all_paths_independent(tmp_path):
    original = advanced_settings()
    target = tmp_path / "settings.json"
    save_settings(original, target)
    assert load_settings(target) == original
    payload = json.loads(target.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 1
    assert all(payload[name] == value for name, value in PATHS.items())
    assert len({payload[name] for name in PATHS}) == len(PATHS)


def test_local_models_fill_both_backend_families_without_overwriting_selected_paths(tmp_path):
    root = tmp_path / "models"
    for name in ("asr", "translation", "vad", "whisper-large-v3-ct2", "nllb-200-distilled-1.3B-ct2", "nllb-200-distilled-1.3B-tokenizer"):
        (root / name).mkdir(parents=True)
    for name in ("asr/model.int8.onnx", "asr/tokens.txt", "translation/sentencepiece.bpe.model", "vad/silero_vad.onnx"):
        (root / name).write_bytes(b"synthetic placeholder, never loaded")
    original = AppSettings(recognition_backend="whisper", translation_backend="nllb",
                           asr_model=PATHS["asr_model"], nllb_model_dir=PATHS["nllb_model_dir"])
    result = with_local_models(original, root)
    assert result.asr_model == original.asr_model and result.nllb_model_dir == original.nllb_model_dir
    expected = {"asr_tokens": "asr/tokens.txt", "translation_model_dir": "translation",
                "translation_spm": "translation/sentencepiece.bpe.model", "whisper_model_dir": "whisper-large-v3-ct2",
                "nllb_tokenizer_dir": "nllb-200-distilled-1.3B-tokenizer", "vad_model": "vad/silero_vad.onnx"}
    assert all(getattr(result, name) == str(root / relative) for name, relative in expected.items())
    assert result.recognition_backend == "whisper" and result.translation_backend == "nllb"
    assert original.whisper_model_dir == "" and original.translation_model_dir == ""
    assert not (root / "settings.json").exists()


def test_legacy_model_directory_is_not_reused_as_whisper_or_nllb(tmp_path):
    (tmp_path / "asr").mkdir()
    (tmp_path / "translation").mkdir()
    result = with_local_models(AppSettings(), tmp_path)
    assert result.translation_model_dir == str(tmp_path / "translation")
    assert result.whisper_model_dir == result.nllb_model_dir == result.nllb_tokenizer_dir == result.vad_model == ""


@pytest.mark.parametrize("role,backend", [("recognition", "whisper"), ("translation", "m2m100"), ("translation", "nllb")])
@pytest.mark.parametrize("precision", ["int8_float16", "float16"])
def test_active_cpu_backend_rejects_cuda_precision(role, backend, precision):
    settings = replace(AppSettings(), **{role + "_backend": backend, role + "_device": "cpu", role + "_compute_type": precision})
    with pytest.raises(SettingsError, match="CPU mode uses int8"):
        validate_settings(settings)


@pytest.mark.parametrize("role,backend", [("recognition", "whisper"), ("translation", "m2m100"), ("translation", "nllb")])
@pytest.mark.parametrize("precision", ["int8", "int8_float16", "float16"])
def test_cuda_choices_validate_without_probing_gpu_or_models(role, backend, precision):
    settings = replace(AppSettings(), **{role + "_backend": backend, role + "_device": "cuda", role + "_compute_type": precision})
    validate_settings(settings)


@pytest.mark.parametrize("field,value", [
    ("recognition_backend", "archived"), ("translation_backend", "remote"),
    ("recognition_device", "gpu"), ("translation_device", "gpu"),
    ("recognition_compute_type", "float32"), ("translation_compute_type", True),
    ("whisper_model_dir", None), ("nllb_model_dir", "invalid\0path"),
    ("nllb_tokenizer_dir", None), ("vad_model", "invalid\0path"),
])
def test_invalid_new_settings_are_not_written(tmp_path, field, value):
    target = tmp_path / "invalid.json"
    with pytest.raises(SettingsError):
        save_settings(replace(AppSettings(), **{field: value}), target)
    assert not target.exists()


def test_dialog_switches_backend_path_groups_independently(widgets, qt_app):
    dialog = widgets("dialog", AppSettings(**PATHS, translation_backend="m2m100"))
    assert not dialog.path_groups["asr_model"].isHidden()
    assert dialog.path_groups["whisper_model_dir"].isHidden()
    assert not dialog.path_groups["translation_model_dir"].isHidden()
    assert dialog.path_groups["nllb_model_dir"].isHidden()
    choose(dialog, "recognition_backend", "whisper")
    settle(qt_app)
    assert dialog.path_groups["asr_model"].isHidden() and dialog.path_groups["asr_tokens"].isHidden()
    assert not dialog.path_groups["whisper_model_dir"].isHidden()
    assert not dialog.path_groups["translation_model_dir"].isHidden()
    choose(dialog, "translation_backend", "nllb")
    settle(qt_app)
    assert dialog.path_groups["translation_model_dir"].isHidden() and dialog.path_groups["translation_spm"].isHidden()
    assert not dialog.path_groups["nllb_model_dir"].isHidden() and not dialog.path_groups["nllb_tokenizer_dir"].isHidden()
    assert not dialog.path_groups["vad_model"].isHidden()
    choose(dialog, "recognition_backend", "sensevoice")
    choose(dialog, "translation_backend", "m2m100")
    assert not dialog.path_groups["asr_model"].isHidden() and not dialog.path_groups["translation_spm"].isHidden()
    assert all(dialog.path_edits[name].text() == value for name, value in PATHS.items())


@pytest.mark.parametrize("role", ["recognition", "translation"])
def test_dialog_cpu_selection_resets_and_disables_fp16_without_loading_runtime(widgets, role):
    dialog = widgets("dialog", advanced_settings())
    choose(dialog, role + "_device", "cpu")
    precision = dialog.engine_combos[role + "_compute_type"]
    assert precision.currentData() == "int8"
    for index in range(precision.count()):
        assert precision.model().item(index).isEnabled() == (precision.itemData(index) == "int8")
    choose(dialog, role + "_device", "cuda")
    assert all(precision.model().item(index).isEnabled() for index in range(precision.count()))
    choose(dialog, role + "_compute_type", "float16")
    assert precision.currentData() == "float16"


@pytest.mark.parametrize("recognition,translation", [("whisper", "nllb"), ("sensevoice", "m2m100")])
def test_window_apply_updates_new_fields_and_keeps_inactive_backend_paths(widgets, qt_app, recognition, translation):
    original = advanced_settings()
    window = widgets("window", original)
    accepted, started, refreshed = [], [], []
    window.settings_changed.connect(accepted.append)
    window.start_requested.connect(started.append)
    window.refresh_devices_requested.connect(lambda: refreshed.append(True))
    dialog = window.open_runtime_settings()
    settle(qt_app)
    changed_paths = {name: value.replace("C:/fixtures/", "C:/staged/") for name, value in PATHS.items()}
    for name, value in changed_paths.items():
        dialog.path_edits[name].setText(" " + value + " ")
    choose(dialog, "recognition_backend", recognition)
    choose(dialog, "translation_backend", translation)
    choose(dialog, "recognition_device", "cuda")
    choose(dialog, "recognition_compute_type", "int8_float16")
    choose(dialog, "translation_device", "cuda")
    choose(dialog, "translation_compute_type", "float16")
    assert window.settings == original and not accepted
    dialog.apply_button.click()
    settle(qt_app)
    assert dialog.result() == QDialog.DialogCode.Accepted and len(accepted) == 1
    selected = window.settings
    assert all(getattr(selected, name) == value for name, value in changed_paths.items())
    assert selected.recognition_backend == recognition and selected.translation_backend == translation
    assert selected.recognition_device == selected.translation_device == "cuda"
    assert selected.recognition_compute_type == "int8_float16" and selected.translation_compute_type == "float16"
    assert selected.font_size == original.font_size and selected.window_opacity == original.window_opacity
    assert selected.show_original == original.show_original and selected.always_on_top == original.always_on_top
    assert selected.microphone == original.microphone and selected.system == original.system
    assert accepted[0] == selected and not started and not refreshed


def test_cancel_discards_engine_precision_and_all_path_edits(widgets, qt_app):
    original = advanced_settings()
    before = asdict(original)
    window = widgets("window", original)
    accepted, started = [], []
    window.settings_changed.connect(accepted.append)
    window.start_requested.connect(started.append)
    dialog = window.open_runtime_settings()
    settle(qt_app)
    for name in PATHS:
        dialog.path_edits[name].setText("C:/discarded/" + name)
    choose(dialog, "recognition_backend", "sensevoice")
    choose(dialog, "translation_backend", "m2m100")
    choose(dialog, "recognition_device", "cpu")
    choose(dialog, "translation_device", "cpu")
    dialog.buttons.button(QDialogButtonBox.StandardButton.Cancel).click()
    settle(qt_app)
    assert dialog.result() == QDialog.DialogCode.Rejected and dialog.result_settings is None
    assert asdict(window.settings) == before and asdict(original) == before
    assert not accepted and not started
