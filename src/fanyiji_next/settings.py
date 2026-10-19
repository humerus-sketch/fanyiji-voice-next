"""Validated configuration for this project only, saved with atomic replacement."""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, fields, replace
from pathlib import Path
from typing import Any

from .models import AppSettings, ChannelSettings


SCHEMA_VERSION = 1
_DIRECTIONS = {"zh_to_ja", "ja_to_zh"}
_PATH_FIELDS = ("asr_model", "asr_tokens", "translation_model_dir", "translation_spm",
                "whisper_model_dir", "nllb_model_dir", "nllb_tokenizer_dir",
                "hymt_model_file", "hymt_server_exe", "vad_model", "mdx_model_file", "gtcrn_model_file",
                "external_model_root", "qwen_asr_model_dir", "fun_asr_model_dir", "gemma_model_file")
_BACKENDS = {"recognition_backend": {"sensevoice", "whisper", "qwen3_asr", "fun_asr"},
             "translation_backend": {"m2m100", "nllb", "hymt", "translategemma"},
             "system_audio_processing": {"raw", "gtcrn", "gtcrn_preserve", "mdx", "mdx_gtcrn"}}


class SettingsError(ValueError):
    """The configuration is unreadable or contains an invalid value."""


def selected_model_combination(settings: AppSettings) -> str:
    """Recover older UI saves that changed the backend but omitted the pair."""
    return {"qwen3_asr": "A", "fun_asr": "B"}.get(settings.recognition_backend, settings.model_combination)


def with_model_combination(settings: AppSettings, combination: str, root: Path) -> AppSettings:
    """Bind one supported pair to the caller's explicit external weight root."""
    from .model_catalog import MODEL_CATALOG, MODEL_COMBINATIONS
    from .model_paths import bundled_llama_server

    if combination not in MODEL_COMBINATIONS:
        raise SettingsError("模型方案必须是 A 或 B。")
    folder = Path(root).absolute()
    models = [MODEL_CATALOG[key] for key in MODEL_COMBINATIONS[combination]]
    asr, gemma = models[:2]
    server = bundled_llama_server()
    # A source checkout can keep the already installed, matching llama runtime.
    if not server.is_file() and settings.hymt_server_exe:
        server = Path(settings.hymt_server_exe)
    return replace(settings, model_combination=combination, external_model_root=str(folder),
                   recognition_backend="qwen3_asr" if combination == "A" else "fun_asr",
                   translation_backend="translategemma", system_audio_processing="raw",
                   qwen_asr_model_dir=str(folder / asr.directory_name) if combination == "A" else "",
                   fun_asr_model_dir=str(folder / asr.directory_name) if combination == "B" else "",
                   gemma_model_file=str(folder / gemma.directory_name / gemma.files[0].relative_path),
                   vad_model=str(folder / "silero-vad-onnx" / "silero_vad.onnx"),
                   hymt_server_exe=str(server), asr_model="", asr_tokens="",
                   translation_model_dir="", translation_spm="", whisper_model_dir="",
                   nllb_model_dir="", nllb_tokenizer_dir="", hymt_model_file="",
                   mdx_model_file="", gtcrn_model_file="",
                   recognition_compute_type="bfloat16", translation_compute_type="int8")


def settings_path() -> Path:
    local_app_data = os.environ.get("LOCALAPPDATA")
    base = Path(local_app_data) if local_app_data else Path.home() / "AppData" / "Local"
    return base / "FanyiJiVoiceNext" / "settings.json"


def with_local_models(settings: AppSettings, root: Path) -> AppSettings:
    """Suggest existing files under an explicit local directory, without saving."""
    directory = Path(root).resolve()
    candidates = {
        "asr_model": directory / "asr" / "model.int8.onnx",
        "asr_tokens": directory / "asr" / "tokens.txt",
        "translation_model_dir": directory / "translation",
        "translation_spm": directory / "translation" / "sentencepiece.bpe.model",
        "whisper_model_dir": directory / "whisper-large-v3-ct2",
        "nllb_model_dir": directory / "nllb-200-distilled-1.3B-ct2",
        "nllb_tokenizer_dir": directory / "nllb-200-distilled-1.3B-tokenizer",
        "hymt_model_file": directory / "hymt2-1.8b" / "Hy-MT2-1.8B-Q8_0.gguf",
        "hymt_server_exe": directory.parent / "runtime" / "llama-b11696-cuda12.4" / "llama-server.exe",
        "vad_model": directory / "vad" / "silero_vad.onnx",
        "mdx_model_file": directory / "audio-processing" / "Kim_Vocal_2.onnx",
        "gtcrn_model_file": directory / "audio-processing" / "gtcrn_simple.onnx",
    }
    values = {}
    for name, path in candidates.items():
        present = path.is_dir() if name in {"translation_model_dir", "whisper_model_dir", "nllb_model_dir", "nllb_tokenizer_dir"} else path.is_file()
        if not getattr(settings, name) and present:
            values[name] = str(path)
    return replace(settings, **values)


def _expect_bool(value: Any, name: str) -> None:
    if type(value) is not bool:
        raise SettingsError(f"{name} must be a boolean")


def _expect_int(value: Any, name: str, minimum: int, maximum: int) -> None:
    if type(value) is not int or not minimum <= value <= maximum:
        raise SettingsError(f"{name} must be an integer between {minimum} and {maximum}")


def _validate_channel(value: ChannelSettings, name: str) -> None:
    if not isinstance(value, ChannelSettings):
        raise SettingsError(f"{name} must be ChannelSettings")
    _expect_bool(value.enabled, f"{name}.enabled")
    if not isinstance(value.direction, str) or value.direction not in _DIRECTIONS:
        raise SettingsError(f"{name}.direction must be zh_to_ja or ja_to_zh")
    if value.device_index is not None and (
        type(value.device_index) is not int or value.device_index < 0
    ):
        raise SettingsError(f"{name}.device_index must be null or a non-negative integer")


def validate_settings(settings: AppSettings) -> None:
    if not isinstance(settings, AppSettings):
        raise SettingsError("settings must be AppSettings")
    _validate_channel(settings.microphone, "microphone")
    _validate_channel(settings.system, "system")
    if settings.model_combination not in {"A", "B"}:
        raise SettingsError("model_combination must be A or B")
    for name in _PATH_FIELDS:
        value = getattr(settings, name)
        if not isinstance(value, str) or "\x00" in value:
            raise SettingsError(f"{name} must be a path string without null characters")
    for name, allowed in _BACKENDS.items():
        value = getattr(settings, name)
        if not isinstance(value, str) or value not in allowed:
            raise SettingsError(f"{name} must be one of {', '.join(sorted(allowed))}")
    for role in ("recognition", "translation"):
        device = getattr(settings, role + "_device")
        compute = getattr(settings, role + "_compute_type")
        if not isinstance(device, str) or device not in {"cpu", "cuda"}:
            raise SettingsError(f"{role}_device must be cpu or cuda")
        if not isinstance(compute, str) or compute not in {"int8", "int8_float16", "float16", "bfloat16"}:
            raise SettingsError(f"{role}_compute_type must be int8, int8_float16, float16 or bfloat16")
        # The shared translation device/precision fields configure CTranslate2.
        # Hy-MT2 uses its GGUF quantization and a CUDA-only local runtime.
        active = (settings.translation_backend in {"m2m100", "nllb"} if role == "translation"
                  else settings.recognition_backend == "whisper")
        if active and device == "cpu" and compute != "int8":
            raise SettingsError(f"{role}: CPU mode uses int8; float16 modes require CUDA")
    _expect_int(settings.font_size, "font_size", 16, 40)
    _expect_bool(settings.show_original, "show_original")
    _expect_bool(settings.always_on_top, "always_on_top")
    _expect_bool(settings.window_transparency_enabled, "window_transparency_enabled")
    _expect_int(settings.window_opacity, "window_opacity", 30, 100)


def _read_channel(data: Any, name: str, default: ChannelSettings) -> ChannelSettings:
    if not isinstance(data, dict):
        raise SettingsError(f"{name} must be an object")
    unknown = set(data) - {item.name for item in fields(ChannelSettings)}
    if unknown:
        raise SettingsError(f"{name} contains unknown fields: {', '.join(sorted(unknown))}")
    values = asdict(default)
    values.update(data)
    return ChannelSettings(**values)


def load_settings(path: Path | None = None) -> AppSettings:
    target = Path(path) if path is not None else settings_path()
    try:
        source = target.read_text(encoding="utf-8")
    except FileNotFoundError:
        return AppSettings()
    except OSError as exc:
        raise SettingsError(f"Unable to read settings: {exc}") from exc
    except UnicodeError as exc:
        raise SettingsError("Settings must be UTF-8 JSON") from exc

    try:
        data = json.loads(source)
    except json.JSONDecodeError as exc:
        raise SettingsError(f"Invalid settings JSON at line {exc.lineno}, column {exc.colno}") from exc
    if not isinstance(data, dict):
        raise SettingsError("Settings must be a JSON object")
    version = data.get("schema_version")
    if type(version) is not int or version != SCHEMA_VERSION:
        raise SettingsError(f"schema_version must be {SCHEMA_VERSION}")

    names = {item.name for item in fields(AppSettings)}
    unknown = set(data) - names - {"schema_version"}
    if unknown:
        raise SettingsError(f"Settings contain unknown fields: {', '.join(sorted(unknown))}")
    defaults = AppSettings()
    values = {name: value for name, value in data.items() if name in names}
    if "translation_backend" not in data:
        # Schema 1 predates backend selection; loading it must keep its original
        # engine rather than silently adopting the new-install default.
        values["translation_backend"] = "m2m100"
    for name in ("microphone", "system"):
        if name in values:
            values[name] = _read_channel(values[name], name, getattr(defaults, name))
    result = AppSettings(**values)
    validate_settings(result)
    return result


def save_settings(settings: AppSettings, path: Path | None = None) -> None:
    validate_settings(settings)
    target = Path(path) if path is not None else settings_path()
    payload = {"schema_version": SCHEMA_VERSION, **asdict(settings)}
    temporary: Path | None = None
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=target.parent, prefix=f".{target.name}.",
            suffix=".tmp", delete=False,
        ) as stream:
            temporary = Path(stream.name)
            json.dump(payload, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
        temporary = None
    except OSError as exc:
        raise SettingsError(f"Unable to save settings: {exc}") from exc
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
