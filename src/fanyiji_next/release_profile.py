"""Portable release defaults, independent of the developer's user settings."""
from __future__ import annotations

from dataclasses import asdict, replace
import json
from pathlib import Path

from .models import AppSettings, ChannelSettings
from .settings import SCHEMA_VERSION, SettingsError, load_settings, validate_settings


RELEASE_DEFAULTS_NAME = "release-defaults.json"


def new_application_settings() -> AppSettings:
    """Use the same A defaults for a fresh application and a packaged release."""
    settings = replace(
        AppSettings(font_size=30),
        microphone=ChannelSettings(enabled=False, direction="zh_to_ja"),
        system=ChannelSettings(enabled=True, direction="ja_to_zh"),
        model_combination="A", recognition_backend="qwen3_asr", recognition_device="cuda",
        recognition_compute_type="bfloat16", translation_backend="translategemma",
        translation_device="cuda", translation_compute_type="int8",
        system_audio_processing="raw",
    )
    validate_settings(settings)
    return settings


def release_defaults_payload() -> dict:
    """Return a public first-run profile; model paths are resolved after launch."""
    settings = new_application_settings()
    return {"schema_version": SCHEMA_VERSION, **asdict(settings)}


def write_release_defaults(path: Path) -> None:
    """Write only the generated public profile; never read a user's profile."""
    path = Path(path)
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"发布默认配置已存在：{path}")
    path.write_text(json.dumps(release_defaults_payload(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_release_defaults(path: Path) -> AppSettings:
    """Accept our exact generated defaults, without loading arbitrary file paths."""
    path = Path(path)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise SettingsError(f"无法读取发布默认配置：{path.name}；{exc}") from exc
    if isinstance(payload, dict) and "window_transparency_enabled" not in payload:
        # Published 0.5.x defaults predate this flag; migrate only in memory.
        payload = {**payload, "window_transparency_enabled": True}
    # Accept exactly the historical public profile, so a program-only update
    # can preserve an old release-defaults.json. User profiles are loaded by
    # settings.py and never replaced by this generated first-run profile.
    legacy = {
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
        "window_transparency_enabled": True,
    }
    normalized = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    if normalized not in {json.dumps(value, sort_keys=True, ensure_ascii=False)
                          for value in (release_defaults_payload(), legacy)}:
        raise SettingsError("发布默认配置与当前版本不匹配；不接受设备号、用户路径或额外配置。")
    return load_settings(path)
