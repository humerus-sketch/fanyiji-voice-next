"""Local audio preprocessing, ASR and translation; no UI or heavy imports here.

Only ``start`` opens audio devices. Device discovery opens no recording stream.
The worker owns model and stream lifetimes; UI callers never wait for decoding.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import json
import os
import math
from pathlib import Path
import queue
import re
import threading
import time
from typing import Callable, Sequence

from .models import AppSettings, ChannelSettings, RuntimeEvent
from .speech_gate import (SILENCE_SECONDS, SYSTEM_LEADING_SECONDS, SYSTEM_SILENCE_SECONDS,
                          SYSTEM_TRAILING_SECONDS, SileroSpeechGate)
from .recognition_engines import RecognitionCancelled, create_recognition_engine
from .translation_engines import TranslationCancelled, create_translation_engine
from .hymt_runtime import HyRuntimeError


SAMPLE_RATE = 16000
SEGMENT_QUEUE_SIZE = 6
AUDIO_QUEUE_SIZE = 25  # At 40 ms per callback, keep at most one second of input.
PROCESSING_BUFFER_SECONDS = 12
LEVEL_INTERVAL = 0.1
# PortAudio owns process-global host API/stream state and WASAPI COM objects.
# Hold this for one complete manager lifetime, not just individual API calls.
_PORTAUDIO_LIFETIME_LOCK = threading.Lock()


@dataclass(frozen=True)
class AudioDevice:
    index: int
    name: str
    channel: str


def list_audio_devices() -> list[AudioDevice]:
    """Enumerate input microphones and WASAPI loopbacks; never start recording."""
    import pyaudiowpatch as audio

    if not _PORTAUDIO_LIFETIME_LOCK.acquire(blocking=False):
        raise RuntimeError("声音设备正在使用或枚举，请先停止声音处理再刷新设备。")
    try:
        pa = audio.PyAudio()
        try:
            devices = []
            for index in range(pa.get_device_count()):
                info = pa.get_device_info_by_index(index)
                if int(info.get("maxInputChannels", 0)) <= 0:
                    continue
                channel = "system" if info.get("isLoopbackDevice", False) else "mic"
                devices.append(AudioDevice(index, str(info["name"]), channel))
            return devices
        finally:
            pa.terminate()
    finally:
        _PORTAUDIO_LIFETIME_LOCK.release()


def _path(value: str) -> Path:
    return Path(os.path.expandvars(value)).expanduser()


def _milliseconds(started_at: float, finished_at: float) -> int:
    return max(0, round((finished_at - started_at) * 1000))


def _microphone_vad_model(settings: AppSettings) -> Path | None:
    # Historical helper name retained for callers; both sources share the
    # local model file, while each capture owns an independent detector state.
    if settings.vad_model.strip():
        return _path(settings.vad_model)
    if not settings.asr_model.strip():
        return None
    candidate = _path(settings.asr_model).parent.parent / "vad" / "silero_vad.onnx"
    # A present but invalid model must fail explicitly, including a directory
    # accidentally given the model filename. Only absence permits fallback.
    return candidate if candidate.exists() else None


def validate_models(settings: AppSettings) -> list[str]:
    """Return actionable local-file errors, without importing or loading engines."""
    issues = []
    if settings.recognition_backend in {"qwen3_asr", "fun_asr"}:
        from .model_catalog import MODEL_COMBINATIONS
        from .model_download import ModelDownloadManager, ModelDownloadError
        try:
            manager = ModelDownloadManager(Path(settings.external_model_root))
            for key in MODEL_COMBINATIONS[settings.model_combination]:
                if not manager.inspect(key).ready:
                    issues.append("方案模型未完整下载或未通过校验，请点击“模型下载 · A/B”。")
                    break
        except (ModelDownloadError, ValueError, OSError) as exc:
            issues.append(f"模型目录不可用：{exc}")
        if not settings.vad_model or not _path(settings.vad_model).is_file():
            issues.append("请下载 A/B 共用的 Silero VAD 辅助模型。")
        if not settings.hymt_server_exe or not _path(settings.hymt_server_exe).is_file():
            issues.append("缺少随程序提供的 llama-server 翻译运行库，请使用完整程序包。")
        return issues
    if settings.recognition_backend == "whisper":
        from .recognition_engines import validate_whisper_model_directory
        issues.extend(validate_whisper_model_directory(settings.whisper_model_dir))
    elif settings.recognition_backend != "sensevoice":
        issues.append(f"不支持的识别后端：{settings.recognition_backend}")
    recognition_files = () if settings.recognition_backend != "sensevoice" else (
        ("SenseVoice ONNX 模型", settings.asr_model),
        ("SenseVoice tokens.txt", settings.asr_tokens),
    )
    translation_files = (("M2M100 SentencePiece 模型", settings.translation_spm),) if settings.translation_backend == "m2m100" else ()
    for label, value in (*recognition_files, *translation_files):
        if not value.strip():
            issues.append(f"尚未选择{label}。")
        elif not _path(value).is_file():
            issues.append(f"{label}文件不存在：{value}")

    if settings.vad_model.strip() and not _path(settings.vad_model).is_file():
        issues.append(f"语音 VAD 模型文件不存在：{settings.vad_model}")
    if settings.system.enabled and settings.system_audio_processing != "raw":
        if settings.system_audio_processing not in {"gtcrn", "gtcrn_preserve", "mdx", "mdx_gtcrn"}:
            issues.append(f"不支持的电脑声音处理模式：{settings.system_audio_processing}")
        processing_files = []
        if settings.system_audio_processing in {"mdx", "mdx_gtcrn"}:
            processing_files.append(("MDX 人声模型", settings.mdx_model_file))
        if settings.system_audio_processing in {"gtcrn", "gtcrn_preserve", "mdx_gtcrn"}:
            processing_files.append(("GTCRN 降噪模型", settings.gtcrn_model_file))
        for label, value in processing_files:
            if not value.strip() or not _path(value).is_file():
                issues.append(f"{label}文件不存在：{value or '尚未选择'}")
        vad = _microphone_vad_model(settings)
        if vad is None or not vad.is_file():
            issues.append("人声处理模式需要有效的 Silero VAD 模型。")
    if settings.translation_backend == "hymt":
        for label, value, suffix in (
            ("Hy-MT2 Q8_0 GGUF 模型", settings.hymt_model_file, ".gguf"),
            ("llama-server 运行库", settings.hymt_server_exe, ".exe"),
        ):
            if not value.strip():
                issues.append(f"尚未选择{label}。")
            elif not _path(value).is_file() or _path(value).suffix.lower() != suffix:
                issues.append(f"{label}文件不存在或类型无效：{value}")
        return issues
    if settings.translation_backend == "nllb":
        if not settings.nllb_tokenizer_dir.strip():
            issues.append("尚未选择 NLLB tokenizer 目录。")
        elif not (_path(settings.nllb_tokenizer_dir) / "tokenizer.json").is_file():
            issues.append("NLLB tokenizer 目录缺少 tokenizer.json。")
        folder_value = settings.nllb_model_dir
        translation_label = "NLLB"
    elif settings.translation_backend == "m2m100":
        folder_value = settings.translation_model_dir
        translation_label = "M2M100"
    else:
        issues.append(f"不支持的翻译后端：{settings.translation_backend}")
        return issues
    if not folder_value.strip():
        issues.append(f"尚未选择 {translation_label} CTranslate2 模型目录。")
    else:
        folder = _path(folder_value)
        if not folder.is_dir():
            issues.append(f"{translation_label} CTranslate2 模型目录不存在：{folder_value}")
        else:
            for filename in ("model.bin", "config.json"):
                if not (folder / filename).is_file():
                    issues.append(f"{translation_label} 模型目录缺少 {filename}。")
            shared = (folder / "shared_vocabulary.json").is_file()
            separate = all(
                (folder / name).is_file()
                for name in ("source_vocabulary.json", "target_vocabulary.json")
            )
            if not (shared or separate):
                issues.append(
                    f"{translation_label} 模型目录缺少 shared_vocabulary.json，"
                    "或 source_vocabulary.json 与 target_vocabulary.json。"
                )
    return issues


class _Segmenter:
    """Energy gate with a short pre-roll and bounded utterance duration."""

    def __init__(
        self,
        sample_rate: int = 16000,
        threshold: float = 0.012,
        silence_seconds: float = 0.55,
        max_seconds: float = 6.0,
        min_speech_seconds: float = 0.18,
    ) -> None:
        self.threshold = threshold
        self.silence_limit = int(sample_rate * silence_seconds)
        self.max_samples = int(sample_rate * max_seconds)
        self.min_speech = int(sample_rate * min_speech_seconds)
        self.pre_roll: deque[float] = deque(maxlen=int(sample_rate * 0.16))
        self.samples: list[float] = []
        self.quiet_samples = 0
        self.speech_samples = 0

    def feed(self, samples: Sequence[float], rms: float) -> list[float] | None:
        loud = rms >= self.threshold
        if not self.samples:
            if not loud:
                self.pre_roll.extend(samples)
                return None
            self.samples = list(self.pre_roll)
            self.pre_roll.clear()
        self.samples.extend(samples)
        if loud:
            self.speech_samples += len(samples)
            self.quiet_samples = 0
        else:
            self.quiet_samples += len(samples)
        if self.quiet_samples >= self.silence_limit or len(self.samples) >= self.max_samples:
            return self.finish()
        return None

    def finish(self) -> list[float] | None:
        result = self.samples if self.speech_samples >= self.min_speech else None
        self.samples = []
        self.speech_samples = 0
        self.quiet_samples = 0
        return result


def _clean_transcript(text: str) -> str:
    # SenseVoice includes language/emotion/event tags in some exported models.
    return re.sub(r"<\|[^>]*\|>", "", text).strip()


class _LoadingCancelled(Exception):
    """A stop request reached a boundary between native model constructors."""


def _translation_config(folder: str) -> dict:
    try:
        config = json.loads((_path(folder) / "config.json").read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise RuntimeError(f"无法读取 CTranslate2 config.json：{exc}") from exc
    if not isinstance(config, dict) or type(config.get("add_source_eos", False)) is not bool:
        raise RuntimeError("CTranslate2 config.json 的 add_source_eos 必须为布尔值。")
    if config.get("eos_token", "</s>") != "</s>" or config.get("decoder_start_token", "</s>") not in (None, "</s>"):
        raise RuntimeError("翻译模型的特殊 token 与 M2M100 不匹配，请选择配套的 M2M100 CTranslate2 模型。")
    return config


class _LocalEngine:
    """Compose independently selected local ASR and translation adapters."""

    def __init__(self, settings: AppSettings, *, stop_event: threading.Event | None = None) -> None:
        self.recognizer = None
        self.translator = None
        self._translation_backend = settings.translation_backend
        self._stop_event = stop_event
        try:
            self._check_cancelled()
            self.recognizer = create_recognition_engine(settings, stop_event=stop_event)
            self._check_cancelled()
            self.translator = create_translation_engine(settings, stop_event=stop_event)
            self._check_cancelled()
        except BaseException as exc:
            try:
                self.close()
            except Exception:
                pass
            if isinstance(exc, (RecognitionCancelled, TranslationCancelled)):
                raise _LoadingCancelled() from exc
            raise

    def _check_cancelled(self) -> None:
        stop = getattr(self, "_stop_event", None)
        if stop is not None and stop.is_set():
            raise _LoadingCancelled()

    def recognize(self, samples: Sequence[float], language: str | None = None) -> str:
        self._check_cancelled()
        try:
            return self.recognizer.recognize(samples, language=language)
        except RecognitionCancelled as exc:
            raise _LoadingCancelled() from exc

    def recognize_result(self, samples):
        self._check_cancelled()
        try:
            return self.recognizer.recognize_result(samples)
        except RecognitionCancelled as exc:
            raise _LoadingCancelled() from exc

    def translate(self, text: str, direction: str, *, context_id: str = "default", source_language=None) -> str:
        self._check_cancelled()
        try:
            if self._translation_backend == "translategemma":
                return self.translator.translate(text, direction, context_id=context_id,
                                                 source_language=source_language)
            if self._translation_backend == "hymt":
                return self.translator.translate(text, direction, context_id=context_id)
            return self.translator.translate(text, direction)
        except TranslationCancelled as exc:
            raise _LoadingCancelled() from exc

    def close(self) -> None:
        error = None
        # Release both engines even if one native destructor reports a failure.
        for name in ("translator", "recognizer"):
            engine = getattr(self, name, None)
            if engine is not None:
                try:
                    engine.close()
                except Exception as exc:
                    error = error or exc
                else:
                    # A failed child-process close must remain reachable for a
                    # later cleanup attempt instead of losing its owner.
                    setattr(self, name, None)
        if error is not None:
            raise error


@dataclass
class _Segment:
    channel: str
    segment_id: str
    samples: list[float]
    direction: str
    captured_at: float
    # Callback-arrival estimates, not PortAudio/OS acoustic timestamps. Keep the
    # original five positional fields for file tools and existing fake sources.
    speech_start_at: float = 0.0
    speech_end_at: float = 0.0
    segment_submitted_at: float = 0.0
    dequeued_at: float = 0.0
    end_reason: str = ""
    speech_detector: str = ""


@dataclass
class _CaptureSource:
    """Plain callback data shared with a consumer; contains no native objects."""

    sample_rate: int
    channels: int
    chunks: queue.Queue[tuple[bytes, float]] = field(default_factory=lambda: queue.Queue(maxsize=AUDIO_QUEUE_SIZE))
    discontinuity: threading.Event = field(default_factory=threading.Event)


@dataclass
class _OpenedInput:
    channel: str
    settings: ChannelSettings
    source: _CaptureSource
    stream: object


@dataclass
class _Session:
    settings: AppSettings
    id: int
    stop: threading.Event = field(default_factory=threading.Event)
    segments: queue.Queue[_Segment] = field(default_factory=lambda: queue.Queue(maxsize=SEGMENT_QUEUE_SIZE))
    captures: list[threading.Thread] = field(default_factory=list)
    worker: threading.Thread | None = None


class VoiceRuntime:
    """Nonblocking lifecycle; callback runs on workers and must marshal to the UI."""

    def __init__(self, callback: Callable[[RuntimeEvent], None]) -> None:
        self._callback = callback
        self._lock = threading.RLock()
        self._session: _Session | None = None

    @property
    def is_running(self) -> bool:
        with self._lock:
            return self._session is not None and not self._session.stop.is_set()

    @property
    def is_busy(self) -> bool:
        """True until native loading/inference and device cleanup have finished."""
        with self._lock:
            return self._session is not None

    def start(self, settings: AppSettings, session_id: int) -> None:
        with self._lock:
            if self._session is not None:
                raise RuntimeError("上一轮声音处理尚未释放，请等待 stopped 后再开始。")
            session = _Session(settings, session_id)
            self._session = session
            session.worker = threading.Thread(
                target=self._run,
                args=(session,),
                name=f"fanyiji-session-{session_id}",
                daemon=True,
            )
            try:
                session.worker.start()
            except BaseException:
                self._session = None
                raise

    def stop(self) -> None:
        with self._lock:
            if self._session is not None:
                self._session.stop.set()

    def _emit(self, session: _Session, kind: str, **values: object) -> None:
        # Identity is checked as well as the public ID: reused IDs cannot revive old work.
        with self._lock:
            if self._session is not session:
                return
            stopping_result = ((kind == "state" and values.get("status") == "stopped")
                               or (kind == "error" and values.get("status") == "cleanup_failed"))
            if session.stop.is_set() and not stopping_result:
                return
            try:
                emitted_at = time.perf_counter()
                values["emitted_at"] = emitted_at
                speech_end_at = float(values.get("speech_end_at", 0.0))
                if speech_end_at > 0:
                    values["elapsed_ms"] = _milliseconds(speech_end_at, emitted_at)
                self._callback(RuntimeEvent(kind=kind, session_id=session.id, **values))
            except Exception:
                # A destroyed UI must never prevent stream cleanup in a worker finally.
                pass

    def _run(self, session: _Session) -> None:
        engine = None
        pa = None
        opened: list[_OpenedInput] = []
        owns_audio_lock = False
        try:
            self._emit(session, "state", status="loading")
            issues = validate_models(session.settings)
            if issues:
                raise RuntimeError("\n".join(issues))
            channels = [
                (name, settings)
                for name, settings in (
                    ("mic", session.settings.microphone),
                    ("system", session.settings.system),
                )
                if settings.enabled
            ]
            if not channels:
                raise RuntimeError("请至少启用麦克风或系统声音中的一个声道。")
            for _, settings in channels:
                if settings.direction not in {"zh_to_ja", "ja_to_zh"}:
                    raise RuntimeError(f"不支持的翻译方向：{settings.direction}")
            if session.stop.is_set():
                return
            engine = _LocalEngine(session.settings, stop_event=session.stop)
            if session.stop.is_set():
                return
            import pyaudiowpatch as audio

            while not _PORTAUDIO_LIFETIME_LOCK.acquire(timeout=0.05):
                if session.stop.is_set():
                    return
            owns_audio_lock = True
            if session.stop.is_set():
                return
            # This worker is the sole PortAudio owner: initialization, device
            # lookup, both streams and final termination stay on this thread.
            pa = audio.PyAudio()
            for channel, settings in channels:
                if session.stop.is_set():
                    return
                try:
                    opened.append(self._open_input(session, pa, audio, channel, settings))
                except Exception as exc:
                    name = "系统声音" if channel == "system" else "麦克风"
                    self._emit(session, "error", channel=channel, message=f"{name}设备打开失败：{exc}")
                    return
            ready_events = []
            for item in opened:
                if session.stop.is_set():
                    return
                ready = threading.Event()
                ready_events.append(ready)
                capture = threading.Thread(
                    target=self._capture,
                    args=(session, item.channel, item.settings, ready, item.source),
                    name=f"fanyiji-{item.channel}-{session.id}",
                    daemon=True,
                )
                capture.start()
                session.captures.append(capture)
            deadline = time.perf_counter() + 10
            while not all(ready.is_set() for ready in ready_events):
                if session.stop.wait(0.05):
                    return
                if time.perf_counter() > deadline:
                    raise RuntimeError("音频设备打开超时，请检查设备是否可用或被其他程序独占。")
            if session.stop.is_set():
                return
            for item in opened:
                if session.stop.is_set():
                    return
                item.stream.start_stream()
            self._check_inputs(opened)
            message = ""
            if _microphone_vad_model(session.settings) is None:
                message = "未发现 Silero VAD 模型，当前使用基础能量断句。"
            self._emit(session, "state", status="running", message=message)
            self._emit_queue(session)
            while not session.stop.is_set():
                self._check_inputs(opened)
                try:
                    segment = session.segments.get(timeout=0.1)
                except queue.Empty:
                    continue
                segment.dequeued_at = time.perf_counter()
                self._emit_queue(session)
                self._process_segment(session, engine, segment)
        except _LoadingCancelled:
            pass
        except ImportError as exc:
            self._emit(session, "error", message=f"缺少离线运行依赖：{exc.name}。源码环境请安装本项目 .[voice]；便携版请保留完整程序目录。")
        except Exception as exc:
            self._emit(session, "error", message=f"语音运行失败：{exc}")
        finally:
            session.stop.set()
            # Callbacks may still be running. Never hold _lock while native stop
            # waits for them; they only touch stop/data queues, not PortAudio.
            for item in reversed(opened):
                try:
                    item.stream.stop_stream()
                except Exception:
                    pass
                try:
                    item.stream.close()
                except Exception:
                    pass
            for capture in session.captures:
                # Only the background worker waits. "stopped" means streams have
                # actually closed, so a fresh session cannot race a retiring device.
                capture.join()
            if pa is not None:
                try:
                    pa.terminate()
                except Exception:
                    pass
            if owns_audio_lock:
                _PORTAUDIO_LIFETIME_LOCK.release()
            if engine is not None:
                try:
                    engine.close()
                except Exception as exc:
                    self._emit(session, "error", status="cleanup_failed",
                               message=f"本地模型释放失败：{type(exc).__name__}: {exc}")
            while not session.segments.empty():
                try:
                    session.segments.get_nowait()
                except queue.Empty:
                    break
            with self._lock:
                if self._session is session:
                    self._session = None
                    try:
                        self._callback(RuntimeEvent(
                            kind="state", session_id=session.id, status="stopped",
                            emitted_at=time.perf_counter(),
                        ))
                    except Exception:
                        pass

    @staticmethod
    def _check_inputs(opened: list[_OpenedInput]) -> None:
        for item in opened:
            if not item.stream.is_active():
                name = "系统声音" if item.channel == "system" else "麦克风"
                raise RuntimeError(f"{name}音频流已意外停止，请重新检查设备。")

    @staticmethod
    def _open_input(session: _Session, pa, audio, channel: str, settings: ChannelSettings) -> _OpenedInput:
        if settings.device_index is None:
            info = pa.get_default_wasapi_loopback() if channel == "system" else pa.get_default_input_device_info()
        else:
            info = pa.get_device_info_by_index(settings.device_index)
        is_loopback = bool(info.get("isLoopbackDevice", False))
        if (channel == "system") != is_loopback:
            raise RuntimeError("设备类型不匹配，请重新选择该声道的设备。")
        channels = int(info.get("maxInputChannels", 0))
        if channels < 1:
            raise RuntimeError("所选设备不支持录音输入。")
        sample_rate = int(info["defaultSampleRate"])
        if sample_rate <= 0:
            raise RuntimeError("音频设备返回了无效采样率。")
        source = _CaptureSource(sample_rate, channels)

        def on_audio(in_data, _frame_count, _time_info, status):
            if session.stop.is_set():
                return (None, audio.paComplete)
            if status:
                source.discontinuity.set()
            if in_data:
                try:
                    source.chunks.put_nowait((in_data, time.perf_counter()))
                except queue.Full:
                    source.discontinuity.set()
            return (None, audio.paContinue)

        stream = pa.open(
            format=audio.paInt16, channels=channels, rate=sample_rate,
            input=True, input_device_index=int(info["index"]),
            frames_per_buffer=max(1, int(sample_rate * 0.04)),
            stream_callback=on_audio, start=False,
        )
        return _OpenedInput(channel, settings, source, stream)

    def _emit_queue(self, session: _Session, *, channel: str = "", message: str = "") -> None:
        self._emit(session, "queue", channel=channel, queue_depth=session.segments.qsize(), message=message)

    def _enqueue_segment(self, session: _Session, segment: _Segment, *, wait: bool = False) -> None:
        if session.stop.is_set():
            return
        if wait:
            # A separation batch may finalize several short utterances at once.
            # Backpressure this worker, while the independent input drainer
            # retains its finite limit and the stop request remains cancellable.
            while not session.stop.is_set():
                segment.segment_submitted_at = time.perf_counter()
                try:
                    session.segments.put(segment, timeout=0.05)
                except queue.Full:
                    continue
                self._emit_queue(session, channel=segment.channel)
                return
            return
        segment.segment_submitted_at = time.perf_counter()
        try:
            session.segments.put_nowait(segment)
        except queue.Full:
            message = "识别队列已满，已跳过最新一段音频；可停用一个声道减轻负载。"
            self._emit_queue(session, channel=segment.channel, message=message)
            self._emit(session, "error", channel=segment.channel, status="recoverable", message=message)
        else:
            self._emit_queue(session, channel=segment.channel)

    def _process_segment(self, session: _Session, engine: _LocalEngine, segment: _Segment) -> None:
        if session.stop.is_set():
            return
        taken_at = segment.dequeued_at or time.perf_counter()
        submitted_at = segment.segment_submitted_at or segment.captured_at
        # Older five-field _Segment sources have no detected speech boundary;
        # retain their captured_at fallback without inventing a speech onset.
        speech_end_at = segment.speech_end_at or segment.captured_at
        timing = {
            "speech_start_at": segment.speech_start_at,
            "speech_end_at": speech_end_at,
            "segment_submitted_at": submitted_at,
            "endpoint_ms": _milliseconds(speech_end_at, submitted_at),
            "queue_ms": _milliseconds(submitted_at, taken_at),
            "end_reason": segment.end_reason,
            "speech_detector": segment.speech_detector,
        }
        asr_started_at = time.perf_counter()
        source_language = None
        try:
            if session.settings.recognition_backend in {"qwen3_asr", "fun_asr"}:
                result = engine.recognize_result(segment.samples)
                text, source_language = result.text, result.language
            elif session.settings.recognition_backend == "whisper":
                language = "zh" if segment.direction == "zh_to_ja" else "ja"
                text = engine.recognize(segment.samples, language=language)
            else:
                text = engine.recognize(segment.samples)
        except _LoadingCancelled:
            return
        except Exception as exc:
            self._emit(
                session, "error", channel=segment.channel,
                segment_id=segment.segment_id, status="recoverable",
                message=f"本地语音识别失败：{exc}",
                asr_ms=_milliseconds(asr_started_at, time.perf_counter()), **timing,
            )
            return
        asr_ms = _milliseconds(asr_started_at, time.perf_counter())
        if session.settings.recognition_backend in {"qwen3_asr", "fun_asr"}:
            timing.update(source_language=source_language or "",
                          target_language="ja" if segment.direction == "zh_to_ja" else "zh")
        if session.stop.is_set() or not text:
            return
        self._emit(
            session, "original", channel=segment.channel, segment_id=segment.segment_id,
            text=text, asr_ms=asr_ms, **timing,
        )
        if session.stop.is_set():
            return
        translation_started_at = time.perf_counter()
        try:
            if session.settings.translation_backend == "translategemma":
                translated = engine.translate(text, segment.direction, context_id=segment.channel,
                                               source_language=source_language)
            elif session.settings.translation_backend == "hymt":
                translated = engine.translate(text, segment.direction, context_id=segment.channel)
            else:
                translated = engine.translate(text, segment.direction)
            self._emit(
                session, "translation", channel=segment.channel,
                segment_id=segment.segment_id, text=translated,
                asr_ms=asr_ms,
                translation_ms=_milliseconds(translation_started_at, time.perf_counter()),
                **timing,
            )
        except _LoadingCancelled:
            return
        except HyRuntimeError:
            # A timed-out or disconnected private server is retired. Let the
            # owning session stop and close streams instead of retrying every
            # following utterance against a dead child.
            raise
        except Exception as exc:
            self._emit(
                session, "error", channel=segment.channel,
                segment_id=segment.segment_id, status="recoverable",
                message=f"本地翻译失败：{exc}",
                asr_ms=asr_ms,
                translation_ms=_milliseconds(translation_started_at, time.perf_counter()),
                **timing,
            )

    def _capture(
        self, session: _Session, channel: str, settings: ChannelSettings,
        ready: threading.Event, source: _CaptureSource,
    ) -> None:
        speech_gate = None
        try:
            if session.stop.is_set():
                return
            import numpy as np
            import soxr

            vad_model = _microphone_vad_model(session.settings)
            if vad_model is not None:
                try:
                    options = ({"silence_seconds": SYSTEM_SILENCE_SECONDS,
                                "leading_seconds": SYSTEM_LEADING_SECONDS,
                                "trailing_seconds": SYSTEM_TRAILING_SECONDS} if channel == "system" else {})
                    speech_gate = SileroSpeechGate(vad_model, **options)
                except Exception as exc:
                    label = "系统声音" if channel == "system" else "麦克风"
                    raise RuntimeError(f"{label} Silero VAD 模型加载失败：{exc}") from exc
                if not session.stop.is_set():
                    if channel == "system" and session.settings.system_audio_processing != "raw":
                        self._capture_processed(session, settings, ready, source, speech_gate, np)
                    else:
                        self._capture_vad(session, settings, ready, source, speech_gate, np, soxr, channel=channel)
                return

            if channel == "system" and session.settings.system_audio_processing != "raw":
                raise RuntimeError("人声处理模式需要 Silero VAD，请先选择本地模型。")

            channels = source.channels
            resampler = soxr.ResampleStream(source.sample_rate, SAMPLE_RATE, 1, dtype="float32")
            chunks = source.chunks
            discontinuity = source.discontinuity
            ready.set()
            segmenter = _Segmenter()
            number = 0
            last_overflow = 0.0
            last_level = -math.inf
            gap_level_cleared = False
            last_audio_at: float | None = None
            speech_start_at: float | None = None
            speech_end_at: float | None = None
            while not session.stop.is_set():
                try:
                    raw, captured_at = chunks.get(timeout=0.05)
                except queue.Empty:
                    if session.stop.is_set():
                        continue
                    if (
                        last_audio_at is not None and not gap_level_cleared
                        and time.perf_counter() - last_audio_at >= segmenter.silence_limit / SAMPLE_RATE
                    ):
                        self._emit(session, "level", channel=channel, level=0.0)
                        gap_level_cleared = True
                    if (
                        segmenter.samples and last_audio_at is not None
                        and time.perf_counter() - last_audio_at >= segmenter.silence_limit / SAMPLE_RATE
                    ):
                        # WASAPI may omit callbacks for digital silence. A live
                        # but quiet source must still finish its last sentence.
                        # Flush filter delay before resetting this continuous
                        # sequence, so a resumed source cannot bridge the gap.
                        tail = resampler.resample_chunk(np.empty(0, dtype=np.float32), last=True)
                        completed = None
                        if len(tail):
                            rms = float(np.sqrt(np.mean(tail * tail)))
                            completed = segmenter.feed(tail.tolist(), rms)
                        if completed is None:
                            completed = segmenter.finish()
                        if completed:
                            number += 1
                            self._enqueue_segment(session, _Segment(
                                channel, f"{channel}-{session.id}-{number}", completed,
                                settings.direction, last_audio_at,
                                speech_start_at=speech_start_at or 0.0,
                                speech_end_at=speech_end_at or last_audio_at,
                                end_reason="input_gap", speech_detector="energy",
                            ))
                        segmenter = _Segmenter()
                        resampler.clear()
                        last_audio_at = None
                        speech_start_at = None
                        speech_end_at = None
                    continue
                if discontinuity.is_set():
                    discontinuity.clear()
                    # Do not join speech from opposite sides of an audio gap.
                    segmenter = _Segmenter()
                    resampler.clear()
                    last_audio_at = None
                    speech_start_at = None
                    speech_end_at = None
                    while not chunks.empty():
                        try:
                            chunks.get_nowait()
                        except queue.Empty:
                            break
                    now = time.perf_counter()
                    if now - last_overflow >= 5:
                        last_overflow = now
                        message = "音频缓冲溢出，已丢弃不连续片段；请降低后台负载。"
                        self._emit(session, "error", channel=channel, status="recoverable", message=message)
                        self._emit_queue(session, channel=channel, message=message)
                    continue
                last_audio_at = captured_at
                gap_level_cleared = False
                samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
                if channels > 1:
                    samples = samples.reshape(-1, channels).mean(axis=1)
                # Use raw mono block energy for boundary estimates so SoXR's
                # internal filter delay does not move a loud block's timestamp
                # to a later callback. This is still a callback-arrival estimate:
                # device buffering, block duration and energy gating add error.
                raw_rms = float(np.sqrt(np.mean(samples * samples)))
                if raw_rms >= segmenter.threshold:
                    if speech_start_at is None:
                        speech_start_at = captured_at
                    speech_end_at = captured_at
                samples = resampler.resample_chunk(samples, last=False)
                if not len(samples):
                    continue
                rms = float(np.sqrt(np.mean(samples * samples)))
                now = time.perf_counter()
                if now - last_level >= LEVEL_INTERVAL:
                    last_level = now
                    self._emit(session, "level", channel=channel, level=min(1.0, rms * 10))
                had_utterance = bool(segmenter.samples)
                completed = segmenter.feed(samples.tolist(), rms)
                if completed is None:
                    if had_utterance and not segmenter.samples:
                        # A short noise spike can be discarded by the gate.
                        speech_start_at = None
                        speech_end_at = None
                    continue
                number += 1
                segment = _Segment(
                    channel, f"{channel}-{session.id}-{number}", completed, settings.direction, captured_at,
                    speech_start_at=speech_start_at or 0.0,
                    speech_end_at=speech_end_at or captured_at,
                    end_reason="max_duration" if len(completed) >= segmenter.max_samples else "energy_silence",
                    speech_detector="energy",
                )
                self._enqueue_segment(session, segment)
                speech_start_at = None
                speech_end_at = None
        except Exception as exc:
            if not session.stop.is_set():
                label = "系统声音" if channel == "system" else "麦克风"
                self._emit(session, "error", channel=channel, message=f"{label}采集失败：{exc}。请检查设备、Windows 录音权限和独占占用。")
                session.stop.set()
        finally:
            if speech_gate is not None:
                try:
                    speech_gate.reset()
                except Exception:
                    pass
            ready.set()

    def _capture_processed(self, session, settings, ready, source, gate, np) -> None:
        """Drain native input independently of the bounded processing worker.

        Only real source gaps finalize speech. Inference pauses never count as
        silence, and sample positions retain their original capture clock.
        """
        from .audio_processing import AudioProcessingCancelled, create_system_audio_processor

        processor = None
        drainer = None
        inbox = queue.Queue(maxsize=PROCESSING_BUFFER_SECONDS * 25 + 1)
        pending_frames = 0
        pending_lock = threading.Lock()
        origin = None
        captured_end = 0.0
        number = 0
        uses_mdx = session.settings.system_audio_processing in {"mdx", "mdx_gtcrn"}
        preserves_asr = session.settings.system_audio_processing == "gtcrn_preserve"
        preserve_next_sample = 0

        def offer(raw, captured_at):
            nonlocal pending_frames
            frames = len(raw) // (source.channels * 2) if raw is not None else 0
            with pending_lock:
                if pending_frames + frames > PROCESSING_BUFFER_SECONDS * source.sample_rate:
                    raise RuntimeError("人声处理待处理输入超过 12 秒，已停止采集；请切换原声直通或减轻负载。")
                try:
                    inbox.put_nowait((raw, captured_at, frames))
                except queue.Full as exc:
                    raise RuntimeError("人声处理缓冲已满，已停止采集；请切换原声直通。") from exc
                pending_frames += frames

        def drain():
            last_audio_at = None
            last_warning = -math.inf
            last_level = -math.inf
            try:
                while not session.stop.is_set():
                    try:
                        raw, captured_at = source.chunks.get(timeout=0.05)
                    except queue.Empty:
                        if last_audio_at is not None and time.perf_counter() - last_audio_at >= SYSTEM_SILENCE_SECONDS:
                            offer(None, last_audio_at)
                            last_audio_at = None
                            self._emit(session, "level", channel="system", level=0.0)
                        continue
                    if source.discontinuity.is_set():
                        raise RuntimeError("电脑声音输入缓冲溢出，已停止采集，避免跨越缺失音频拼接字幕。")
                    frames, remainder = divmod(len(raw), source.channels * 2)
                    if remainder or frames > source.sample_rate:
                        raise RuntimeError("电脑声音输入块格式无效或超过一秒。")
                    if not frames:
                        continue
                    if last_audio_at is not None and captured_at - last_audio_at - frames / source.sample_rate >= SYSTEM_SILENCE_SECONDS:
                        offer(None, last_audio_at)
                    offer(raw, captured_at)
                    last_audio_at = captured_at
                    now = time.perf_counter()
                    if now - last_level >= LEVEL_INTERVAL:
                        last_level = now
                        samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
                        rms = float(np.sqrt(np.mean(samples * samples)))
                        self._emit(session, "level", channel="system", level=min(1.0, rms * 10))
                    with pending_lock:
                        waiting_seconds = pending_frames / source.sample_rate
                    if waiting_seconds >= 2 and now - last_warning >= 5:
                        last_warning = now
                        detail = ("MDX 批处理会增加字幕延迟。" if uses_mdx else
                                  "流式降噪处理积压，请检查计算负载。")
                        self._emit_queue(session, channel="system",
                                         message=f"人声处理待处理 {waiting_seconds:.1f} 秒音频；{detail}")
            except Exception as exc:
                if not session.stop.is_set():
                    self._emit(session, "error", channel="system", message=str(exc))
                    session.stop.set()

        def submit(slices, *, reason=None):
            nonlocal number
            for item in slices:
                if session.stop.is_set():
                    return
                number += 1
                onset = item.speech_start_sample if item.speech_start_sample is not None else item.start_sample
                endpoint = item.speech_end_sample if item.speech_end_sample is not None else item.end_sample
                self._enqueue_segment(session, _Segment(
                    "system", f"system-{session.id}-{number}", item.samples, settings.direction, captured_end,
                    speech_start_at=origin + onset / SAMPLE_RATE,
                    speech_end_at=origin + endpoint / SAMPLE_RATE,
                    end_reason=reason or item.end_reason, speech_detector="silero",
                ), wait=True)

        def feed(outputs, *, reason=None):
            nonlocal preserve_next_sample
            for output in outputs:
                if session.stop.is_set():
                    return
                if preserves_asr:
                    if output.raw_samples is None or output.denoised_samples is None:
                        raise RuntimeError("原声识别模式缺少对应区间的检测辅路，已停止以免错配音频。")
                    if (output.start_sample != preserve_next_sample
                            or output.end_sample < output.start_sample
                            or len(output.samples) != output.end_sample - output.start_sample
                            or len(output.denoised_samples) != len(output.samples)
                            or not np.array_equal(output.samples, output.raw_samples)):
                        raise RuntimeError("原声识别模式的源样本不一致，已停止以免改写识别输入。")
                    submit(gate.feed(output.raw_samples.tolist(), output.denoised_samples.tolist(),
                                     start_sample=output.start_sample), reason=reason)
                    preserve_next_sample = output.end_sample
                else:
                    submit(gate.feed(output.samples.tolist()), reason=reason)

        try:
            if preserves_asr:
                from .dual_speech_gate import DualSpeechGate
                gate = DualSpeechGate(_microphone_vad_model(session.settings), raw_gate=gate,
                                      silence_seconds=SYSTEM_SILENCE_SECONDS,
                                      leading_seconds=SYSTEM_LEADING_SECONDS,
                                      trailing_seconds=SYSTEM_TRAILING_SECONDS)
            processor = create_system_audio_processor(
                session.settings, sample_rate=source.sample_rate, channels=source.channels, stop_event=session.stop,
            )
            if session.stop.is_set():
                return
            drainer = threading.Thread(target=drain, name=f"fanyiji-system-input-{session.id}", daemon=True)
            drainer.start()
            ready.set()
            message = ("已启用 MDX 人声处理；首批需要约 5.8 秒原声，另加推理时间。" if uses_mdx else
                       "试验：原声断句和识别；GTCRN仅作并行诊断，背景声也会保留。" if preserves_asr else
                       "已启用 GTCRN 流式降噪；保留原有静音断句和逐句识别翻译。")
            self._emit_queue(session, channel="system", message=message)
            while not session.stop.is_set():
                try:
                    raw, captured_at, frames = inbox.get(timeout=0.05)
                except queue.Empty:
                    continue
                with pending_lock:
                    pending_frames -= frames
                if raw is None:
                    if origin is not None:
                        feed(processor.finish(), reason="input_gap")
                        submit(gate.finish(), reason="input_gap")
                        processor.reset()
                        origin = None
                        preserve_next_sample = 0
                    continue
                if origin is None:
                    origin = captured_at - frames / source.sample_rate
                captured_end = captured_at
                samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32).reshape(-1, source.channels) / 32768.0
                feed(processor.process_block(samples))
        except AudioProcessingCancelled:
            if not session.stop.is_set():
                self._emit(session, "error", channel="system", message="电脑声音人声处理意外取消，请重新开始。")
        except Exception as exc:
            if not session.stop.is_set():
                self._emit(session, "error", channel="system", message=f"电脑声音人声处理失败：{exc}")
        finally:
            # On a user stop discard the delayed tail. Never submit old audio
            # after the UI has stopped or moved to another processing mode.
            session.stop.set()
            if drainer is not None:
                drainer.join()
            if processor is not None:
                try:
                    processor.close()
                except Exception as exc:
                    self._emit(session, "error", channel="system", status="cleanup_failed",
                               message=f"电脑声音人声处理器释放失败：{type(exc).__name__}: {exc}")
            if preserves_asr:
                try:
                    gate.reset()
                except Exception as exc:
                    self._emit(session, "error", channel="system", status="cleanup_failed",
                               message=f"电脑声音双路检测器释放失败：{type(exc).__name__}: {exc}")

    def _capture_vad(self, session, settings, ready, source, gate, np, soxr, *, channel="mic") -> None:
        """Per-source VAD consumer. No native audio lifecycle calls here."""
        silence_seconds = SYSTEM_SILENCE_SECONDS if channel == "system" else SILENCE_SECONDS
        resampler = soxr.ResampleStream(source.sample_rate, SAMPLE_RATE, 1, dtype="float32")
        raw_frames = 0
        number = 0
        last_audio_at = None
        last_level = -math.inf
        last_overflow = 0.0

        def submit(slices, captured_at, *, reason=None):
            nonlocal number
            # Callback arrival approximates the end of the latest raw block.
            # Map VAD's *trimmed* 16 kHz sample positions against that block,
            # retaining device buffering/callback jitter as timing uncertainty.
            # is_speech_detected includes hangover and cannot timestamp speech end.
            origin = captured_at - raw_frames / source.sample_rate
            for item in slices:
                if session.stop.is_set():
                    return
                number += 1
                onset = item.speech_start_sample if item.speech_start_sample is not None else item.start_sample
                endpoint = item.speech_end_sample if item.speech_end_sample is not None else item.end_sample
                self._enqueue_segment(session, _Segment(
                    channel, f"{channel}-{session.id}-{number}", item.samples, settings.direction, captured_at,
                    speech_start_at=origin + onset / SAMPLE_RATE,
                    speech_end_at=origin + endpoint / SAMPLE_RATE,
                    end_reason=reason or item.end_reason, speech_detector="silero",
                ))

        ready.set()
        while not session.stop.is_set():
            try:
                raw, captured_at = source.chunks.get(timeout=0.05)
            except queue.Empty:
                if session.stop.is_set():
                    continue
                if last_audio_at is not None and time.perf_counter() - last_audio_at >= silence_seconds:
                    self._emit(session, "level", channel=channel, level=0.0)
                    tail = resampler.resample_chunk(np.empty(0, dtype=np.float32), last=True)
                    if len(tail):
                        submit(gate.feed(tail.tolist()), last_audio_at, reason="input_gap")
                    submit(gate.finish(), last_audio_at, reason="input_gap")
                    resampler.clear()
                    raw_frames = 0
                    last_audio_at = None
                continue
            if source.discontinuity.is_set():
                source.discontinuity.clear()
                gate.reset()
                resampler.clear()
                raw_frames = 0
                last_audio_at = None
                while not source.chunks.empty():
                    try:
                        source.chunks.get_nowait()
                    except queue.Empty:
                        break
                now = time.perf_counter()
                if now - last_overflow >= 5:
                    last_overflow = now
                    message = "音频缓冲溢出，已丢弃不连续片段；请降低后台负载。"
                    self._emit(session, "error", channel=channel, status="recoverable", message=message)
                    self._emit_queue(session, channel=channel, message=message)
                continue
            samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
            if source.channels > 1:
                samples = samples.reshape(-1, source.channels).mean(axis=1)
            raw_frames += len(samples)
            last_audio_at = captured_at
            samples = resampler.resample_chunk(samples, last=False)
            if not len(samples):
                continue
            rms = float(np.sqrt(np.mean(samples * samples)))
            now = time.perf_counter()
            if now - last_level >= LEVEL_INTERVAL:
                last_level = now
                self._emit(session, "level", channel=channel, level=min(1.0, rms * 10))
            submit(gate.feed(samples.tolist()), captured_at)
