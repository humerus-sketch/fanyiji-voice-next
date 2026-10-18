"""Local recognizers with serialized native lifetimes and no model downloads.

The caller supplies normalized mono samples at 16 kHz. Construction loads the
chosen model; importing this module never imports either recognition runtime.
"""

from __future__ import annotations

import json
import math
from numbers import Real
import os
from pathlib import Path
import re
import threading
from typing import Sequence

from .models import AppSettings


SAMPLE_RATE = 16000
WHISPER_REQUIRED_FILES = (
    "model.bin", "config.json", "tokenizer.json", "preprocessor_config.json", "vocabulary.json",
)
WHISPER_TRANSCRIBE_OPTIONS = {
    "task": "transcribe",
    "beam_size": 5,
    # Only suspicious decodes retry once; nonzero sampling gets one candidate.
    "temperature": (0.0, 0.2),
    "best_of": 1,
    "compression_ratio_threshold": 2.4,
    "log_prob_threshold": -1.0,
    "no_speech_threshold": 0.6,
    "condition_on_previous_text": False,
    "vad_filter": False,
    "without_timestamps": True,
    "word_timestamps": False,
}


def _whisper_segment_diagnostics(segment, text):
    """Expose score heuristics, not accuracy, and never retain subtitle text.

    In faster-whisper 1.2.1 temperature can describe the last fallback attempt
    even when the library selects an earlier candidate after all attempts fail.
    """
    values = {}
    for name in ("avg_logprob", "compression_ratio", "no_speech_prob", "temperature"):
        value = getattr(segment, name, None)
        # Check the real type, not a Mock's spoofed __class__ or float coercion.
        if isinstance(value, bool) or not issubclass(type(value), Real):
            values[name] = None
            continue
        try:
            number = float(value)
        except (ValueError, TypeError, OverflowError):
            values[name] = None
            continue
        values[name] = number if math.isfinite(number) else None
    values["char_count"] = len(text.strip())
    logprob, compression = values["avg_logprob"], values["compression_ratio"]
    suspect = ((logprob is not None and logprob < WHISPER_TRANSCRIBE_OPTIONS["log_prob_threshold"])
               or (compression is not None and compression > WHISPER_TRANSCRIBE_OPTIONS["compression_ratio_threshold"]))
    values["suspect"] = True if suspect else False if logprob is not None and compression is not None else None
    return values


class RecognitionCancelled(RuntimeError):
    """A stop request reached a boundary outside an in-flight native call."""


def _local_path(value: str | os.PathLike[str]) -> Path:
    return Path(os.path.expandvars(os.fspath(value))).expanduser().resolve()


def validate_whisper_model_directory(value: str | os.PathLike[str]) -> list[str]:
    """Validate local assets without imports, downloads, or large-file hashing.

128-mel preprocessing is required for large-v3. This shape check is not a model
identity proof: a pinned source/hash manifest belongs to application validation.
"""
    if not os.fspath(value).strip():
        return ["尚未选择 Whisper large-v3 本地模型目录。"]
    folder = _local_path(value)
    if not folder.is_dir():
        return [f"Whisper 本地模型目录不存在：{folder}"]
    issues = []
    for name in WHISPER_REQUIRED_FILES:
        if not (folder / name).is_file():
            issues.append(f"Whisper 模型目录缺少 {name}，不能进行完整离线加载。")
    for name in ("config.json", "preprocessor_config.json"):
        path = folder / name
        if not path.is_file():
            continue
        try:
            config = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            issues.append(f"无法读取 Whisper {name}：{exc}")
            continue
        if not isinstance(config, dict):
            issues.append(f"Whisper {name} 必须是 JSON 对象。")
        elif name == "preprocessor_config.json":
            if config.get("feature_size") != 128:
                issues.append("Whisper large-v3 需要 feature_size=128 的配套预处理配置。")
            if config.get("sampling_rate") != SAMPLE_RATE:
                issues.append("Whisper 预处理配置的 sampling_rate 必须为 16000。")
    return issues


def _check_language(language: str | None) -> None:
    if language not in {None, "zh", "ja"}:
        raise ValueError(f"不支持的识别语言：{language}；请选择 zh 或 ja。")


class _RecognizerBase:
    def __init__(self, stop_event: threading.Event | None) -> None:
        self._stop_event = stop_event
        self._lock = threading.RLock()
        self._closed = False
        self.backend_info: dict[str, object] = {}

    def _check_cancelled(self) -> None:
        if self._stop_event is not None and self._stop_event.is_set():
            raise RecognitionCancelled("语音识别已取消。")

    def _check_open(self) -> None:
        if self._closed:
            raise RuntimeError("识别引擎已关闭，请重新创建会话。")
        self._check_cancelled()


class SenseVoiceRecognitionEngine(_RecognizerBase):
    """Preserve the existing CPU SenseVoice model and automatic language mode."""

    def __init__(self, settings: AppSettings, *, stop_event: threading.Event | None = None) -> None:
        super().__init__(stop_event)
        self._recognizer = None
        try:
            self._check_cancelled()
            model, tokens = _local_path(settings.asr_model), _local_path(settings.asr_tokens)
            if not settings.asr_model.strip() or not model.is_file():
                raise RuntimeError(f"SenseVoice ONNX 模型文件不存在：{settings.asr_model}")
            if not settings.asr_tokens.strip() or not tokens.is_file():
                raise RuntimeError(f"SenseVoice tokens 文件不存在：{settings.asr_tokens}")
            import sherpa_onnx

            self._check_cancelled()
            threads = max(1, min(4, os.cpu_count() or 1))
            self._recognizer = sherpa_onnx.OfflineRecognizer.from_sense_voice(
                model=str(model), tokens=str(tokens), num_threads=threads,
                sample_rate=SAMPLE_RATE, provider="cpu", language="auto", use_itn=True,
            )
            self._check_cancelled()
            self.backend_info = {
                "backend": "sensevoice", "device": "cpu", "compute_type": "model",
                "sample_rate": SAMPLE_RATE, "language_mode": "auto", "num_threads": threads,
                "use_itn": True, "local_files_only": True,
            }
        except BaseException:
            self.close()
            raise

    def recognize(self, samples: Sequence[float], language: str | None = None) -> str:
        _check_language(language)
        with self._lock:
            self._check_open()
            if not len(samples):
                return ""
            stream = self._recognizer.create_stream()
            try:
                stream.accept_waveform(SAMPLE_RATE, samples)
                self._check_cancelled()
                self._recognizer.decode_stream(stream)
                self._check_cancelled()
                return re.sub(r"<\|[^>]*\|>", "", stream.result.text).strip()
            finally:
                stream = None

    def close(self) -> None:
        with self._lock:
            self._closed = True
            self._recognizer = None


class WhisperRecognitionEngine(_RecognizerBase):
    """faster-whisper over an explicit local large-v3 conversion directory."""

    def __init__(self, settings: AppSettings, *, stop_event: threading.Event | None = None) -> None:
        super().__init__(stop_event)
        self._model = None
        self.last_diagnostics = {"segments": [], "suspect": None}
        try:
            self._check_cancelled()
            folder_value = getattr(settings, "whisper_model_dir", "")
            issues = validate_whisper_model_directory(folder_value)
            if issues:
                raise RuntimeError("\n".join(issues))
            device = getattr(settings, "recognition_device", "cpu")
            compute_type = getattr(settings, "recognition_compute_type", "int8")
            if device not in {"cpu", "cuda"}:
                raise ValueError("Whisper recognition_device 必须是 cpu 或 cuda。")
            if compute_type not in {"int8", "int8_float16", "float16"}:
                raise ValueError("Whisper recognition_compute_type 必须是 int8、int8_float16 或 float16。")
            if device == "cpu" and compute_type != "int8":
                raise ValueError("Whisper CPU 模式请选择 int8；float16 与 int8_float16 用于 CUDA。")
            if device == "cuda":
                from .native_runtime import configure_cuda_runtime
                configure_cuda_runtime("cuda")
            # A missing local tokenizer otherwise triggers a from_pretrained
            # network fallback inside faster-whisper, even with this flag true.
            import faster_whisper

            self._check_cancelled()
            threads = max(1, min(4, os.cpu_count() or 1))
            self._model = faster_whisper.WhisperModel(
                str(_local_path(folder_value)), device=device, compute_type=compute_type,
                local_files_only=True, cpu_threads=threads, num_workers=1,
            )
            self._check_cancelled()
            native = self._model.model
            actual_device = getattr(native, "device", None)
            actual_compute_type = getattr(native, "compute_type", None)
            self.backend_info = {
                "backend": "whisper", "model": "large-v3",
                "model_identity": "requested_profile; assets verified by preparation manifest",
                "device": actual_device if isinstance(actual_device, str) else device,
                "compute_type": actual_compute_type if isinstance(actual_compute_type, str) else compute_type,
                "requested_device": device, "requested_compute_type": compute_type,
                "sample_rate": SAMPLE_RATE,
                "num_threads": threads, "local_files_only": True,
                "transcribe_options": dict(WHISPER_TRANSCRIBE_OPTIONS),
            }
        except BaseException:
            try:
                self.close()
            except Exception:
                pass
            raise

    def recognize(self, samples: Sequence[float], language: str | None = None) -> str:
        with self._lock:
            self.last_diagnostics = {"segments": [], "suspect": None}
            _check_language(language)
            self._check_open()
            if not len(samples):
                return ""
            import numpy as np

            waveform = np.asarray(samples, dtype=np.float32)
            if waveform.ndim != 1 or not np.isfinite(waveform).all():
                raise ValueError("Whisper 输入必须为有限数值的 16kHz 单声道波形。")
            self._check_cancelled()
            segments, _info = self._model.transcribe(
                waveform, language=language, **WHISPER_TRANSCRIBE_OPTIONS,
            )
            iterator = iter(segments)
            parts = []
            diagnostics = []
            try:
                while True:
                    self._check_cancelled()
                    try:
                        segment = next(iterator)
                    except StopIteration:
                        break
                    self._check_cancelled()
                    text = segment.text
                    if not isinstance(text, str):
                        raise RuntimeError("Whisper 返回了无效的分段文本。")
                    diagnostics.append(_whisper_segment_diagnostics(segment, text))
                    if text.strip():
                        parts.append(text)
                self._check_cancelled()
                flags = [item["suspect"] for item in diagnostics]
                self.last_diagnostics = {"segments": diagnostics,
                                         "suspect": True if any(flag is True for flag in flags) else
                                         False if flags and all(flag is False for flag in flags) else None}
                return "".join(parts).strip()
            finally:
                # Do not retain a lazy native generator beyond the lock lifetime.
                closer = getattr(iterator, "close", None)
                if closer is not None:
                    try:
                        closer()
                    except Exception:
                        pass

    def close(self) -> None:
        with self._lock:
            self._closed = True
            model = self._model
            self._model = None
            try:
                if model is not None:
                    model.model.unload_model()
            finally:
                model = None


def create_recognition_engine(settings: AppSettings, *, stop_event: threading.Event | None = None):
    backend = getattr(settings, "recognition_backend", "sensevoice")
    if backend in {"qwen3_asr", "fun_asr"}:
        from .ab_recognition import ABRecognitionEngine
        return ABRecognitionEngine(settings, stop_event=stop_event)
    if backend == "sensevoice":
        return SenseVoiceRecognitionEngine(settings, stop_event=stop_event)
    if backend == "whisper":
        return WhisperRecognitionEngine(settings, stop_event=stop_event)
    raise ValueError(f"不支持的识别后端：{backend}")
