"""Portable release defaults, independent of the developer's user settings."""
from __future__ import annotations

from dataclasses import asdict, replace
import json
from pathlib import Path

from .models import AppSettings, ChannelSettings
from .settings import SCHEMA_VERSION, SettingsError, load_settings, validate_settings


RELEASE_DEFAULTS_NAME = "release-defaults.json"


def release_defaults_payload() -> dict:
    """Return a public first-run profile; model paths are resolved after launch."""
    settings = replace(
        AppSettings(font_size=30),
        microphone=ChannelSettings(enabled=False, direction="zh_to_ja"),
        system=ChannelSettings(enabled=True, direction="ja_to_zh"),
        recognition_backend="whisper", recognition_device="cuda",
        recognition_compute_type="int8_float16", translation_backend="hymt",
        translation_device="cuda", translation_compute_type="int8_float16",
        system_audio_processing="mdx_gtcrn",
    )
    validate_settings(settings)
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
    if payload != release_defaults_payload():
        raise SettingsError("发布默认配置与当前版本不匹配；不接受设备号、用户路径或额外配置。")
    return load_settings(path)
