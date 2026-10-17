"""Configuration checks use only pytest temporary directories."""

import json
from dataclasses import replace

import pytest

from fanyiji_next.models import AppSettings, ChannelSettings
from fanyiji_next.settings import SettingsError, load_settings, save_settings, settings_path, with_local_models


def test_settings_path_is_new_project_only(monkeypatch, tmp_path):
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    assert settings_path() == tmp_path / "FanyiJiVoiceNext" / "settings.json"


def test_missing_settings_returns_defaults_without_writing(tmp_path):
    target = tmp_path / "missing" / "settings.json"
    assert load_settings(target) == AppSettings()
    assert not target.exists()


def test_roundtrip_preserves_unicode_paths_and_device_indexes(tmp_path):
    settings = AppSettings(
        microphone=ChannelSettings(device_index=2),
        system=ChannelSettings(enabled=False, direction="ja_to_zh", device_index=4),
        asr_model="D:/模型/asr.onnx",
        asr_tokens="D:/模型/tokens.txt",
        translation_model_dir="D:/模型/翻译",
        translation_spm="D:/模型/spm.model",
        font_size=32,
        show_original=False,
        always_on_top=True,
        window_opacity=85,
        system_audio_processing="mdx_gtcrn",
        mdx_model_file="D:/模型/人声/Kim_Vocal_2.onnx",
        gtcrn_model_file="D:/模型/降噪/gtcrn_simple.onnx",
    )
    target = tmp_path / "settings" / "settings.json"
    save_settings(settings, target)
    assert load_settings(target) == settings
    assert json.loads(target.read_text(encoding="utf-8"))["schema_version"] == 1
    assert list(target.parent.glob("*.tmp")) == []


def test_partial_settings_get_defaults_for_missing_fields(tmp_path):
    target = tmp_path / "settings.json"
    target.write_text('{"schema_version": 1, "system": {"enabled": false}}', encoding="utf-8")
    settings = load_settings(target)
    assert settings.system == ChannelSettings(enabled=False, direction="ja_to_zh")
    assert settings.font_size == 22
    assert settings.system_audio_processing == "raw"
    assert settings.mdx_model_file == settings.gtcrn_model_file == ""


@pytest.mark.parametrize("source", ["{", "[]", "null", "{}", '{"schema_version": true}', '{"schema_version": 2}'])
def test_malformed_or_unsupported_settings_raise(tmp_path, source):
    target = tmp_path / "settings.json"
    target.write_text(source, encoding="utf-8")
    with pytest.raises(SettingsError):
        load_settings(target)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("font_size", 15), ("font_size", 41), ("font_size", True),
        ("window_opacity", 69), ("window_opacity", 101), ("window_opacity", 90.0),
        ("show_original", 1), ("always_on_top", "true"),
        ("asr_model", None), ("translation_spm", "bad\x00path"),
        ("mdx_model_file", None), ("mdx_model_file", "bad\x00path"),
        ("gtcrn_model_file", 1), ("gtcrn_model_file", "bad\x00path"),
    ],
)
def test_invalid_app_values_are_not_saved(tmp_path, field, value):
    target = tmp_path / "settings.json"
    with pytest.raises(SettingsError):
        save_settings(replace(AppSettings(), **{field: value}), target)
    assert not target.exists()


@pytest.mark.parametrize(
    "channel",
    [
        ChannelSettings(enabled=1), ChannelSettings(direction="auto"),
        ChannelSettings(device_index=-1), ChannelSettings(device_index=True),
        ChannelSettings(device_index=1.5),
    ],
)
def test_invalid_channel_values_are_rejected(tmp_path, channel):
    with pytest.raises(SettingsError):
        save_settings(replace(AppSettings(), microphone=channel), tmp_path / "settings.json")


@pytest.mark.parametrize(
    "payload",
    [
        {"unknown": 1}, {"microphone": []}, {"system": {"unknown": True}},
        {"font_size": "22"}, {"microphone": {"enabled": 1}},
    ],
)
def test_invalid_loaded_values_are_rejected(tmp_path, payload):
    target = tmp_path / "settings.json"
    target.write_text(json.dumps({"schema_version": 1, **payload}), encoding="utf-8")
    with pytest.raises(SettingsError):
        load_settings(target)


def test_failed_atomic_replace_preserves_previous_settings(tmp_path, monkeypatch):
    target = tmp_path / "settings.json"
    save_settings(AppSettings(), target)
    previous = target.read_bytes()

    def fail_replace(source, destination):
        raise OSError("simulated replacement failure")

    monkeypatch.setattr("fanyiji_next.settings.os.replace", fail_replace)
    with pytest.raises(SettingsError, match="Unable to save"):
        save_settings(replace(AppSettings(), font_size=30), target)
    assert target.read_bytes() == previous
    assert list(tmp_path.glob("*.tmp")) == []


def test_local_model_defaults_only_fill_blank_fields_and_do_not_write_settings(tmp_path):
    (tmp_path / "asr").mkdir()
    (tmp_path / "translation").mkdir()
    for name in ("asr/model.int8.onnx", "asr/tokens.txt", "translation/sentencepiece.bpe.model"):
        (tmp_path / name).write_bytes(b"fixture")
    existing = AppSettings(asr_model="D:/chosen/custom.onnx")
    result = with_local_models(existing, tmp_path)
    assert result.asr_model == existing.asr_model
    assert result.asr_tokens == str(tmp_path / "asr/tokens.txt")
    assert result.translation_model_dir == str(tmp_path / "translation")
    assert result.translation_spm == str(tmp_path / "translation/sentencepiece.bpe.model")
    assert not (tmp_path / "settings.json").exists()
    assert with_local_models(AppSettings(), tmp_path / "missing") == AppSettings()


@pytest.mark.parametrize("mode", ["raw", "gtcrn", "gtcrn_preserve", "mdx", "mdx_gtcrn"])
def test_audio_processing_selection_roundtrips_without_requiring_models_for_inactive_modes(tmp_path, mode):
    target = tmp_path / "settings.json"
    selected = AppSettings(system_audio_processing=mode)
    save_settings(selected, target)
    assert load_settings(target) == selected
    assert selected.microphone == AppSettings().microphone


@pytest.mark.parametrize("value", ["MDX", "GTCRN", "gtcrn_only", "disabled", None, True, 1, [], {}])
def test_invalid_audio_processing_mode_is_rejected_before_saving_or_loading(tmp_path, value):
    target = tmp_path / "settings.json"
    with pytest.raises(SettingsError, match="system_audio_processing"):
        save_settings(replace(AppSettings(), system_audio_processing=value), target)
    assert not target.exists()
    target.write_text(json.dumps({"schema_version": 1, "system_audio_processing": value}), encoding="utf-8")
    with pytest.raises(SettingsError, match="system_audio_processing"):
        load_settings(target)


def test_audio_model_discovery_fills_paths_without_enabling_processing_or_overwriting_choices(tmp_path):
    audio = tmp_path / "audio-processing"
    audio.mkdir()
    (audio / "Kim_Vocal_2.onnx").write_bytes(b"fixture")
    (audio / "gtcrn_simple.onnx").write_bytes(b"fixture")
    discovered = with_local_models(AppSettings(), tmp_path)
    assert discovered.mdx_model_file == str(audio / "Kim_Vocal_2.onnx")
    assert discovered.gtcrn_model_file == str(audio / "gtcrn_simple.onnx")
    assert discovered.system_audio_processing == "raw"
    existing = AppSettings(system_audio_processing="mdx", mdx_model_file="D:/chosen/custom-mdx.onnx",
                           gtcrn_model_file="D:/chosen/custom-gtcrn.onnx")
    assert with_local_models(existing, tmp_path) == existing
    assert not (tmp_path / "settings.json").exists()


def test_audio_model_discovery_does_not_accept_a_directory_as_a_model_file(tmp_path):
    (tmp_path / "audio-processing/Kim_Vocal_2.onnx").mkdir(parents=True)
    (tmp_path / "audio-processing/gtcrn_simple.onnx").mkdir()
    assert with_local_models(AppSettings(), tmp_path) == AppSettings()
