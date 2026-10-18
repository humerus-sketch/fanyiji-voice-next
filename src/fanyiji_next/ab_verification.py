"""Explicit file-only A/B inference check; never capture audio or save settings."""
from __future__ import annotations

import hashlib
from pathlib import Path
import time


def verify_audio(audio_path: Path, settings_path: Path) -> dict:
    import numpy as np
    import soundfile as sf
    import soxr
    from .ab_recognition import ABRecognitionEngine
    from .settings import load_settings

    audio_path, settings_path = Path(audio_path), Path(settings_path)
    before = hashlib.sha256(settings_path.read_bytes()).hexdigest()
    settings = load_settings(settings_path)
    if settings.recognition_backend not in {"qwen3_asr", "fun_asr"}:
        raise ValueError("文件检查需要显式的 A/B 识别配置。")
    with sf.SoundFile(audio_path) as stream:
        if not 8000 <= stream.samplerate <= 192000 or not 1 <= stream.channels <= 8:
            raise ValueError("录音格式不支持。")
        # Bounded input avoids accidentally transcribing a whole long recording.
        rate = stream.samplerate
        audio = stream.read(min(len(stream), 8 * rate), dtype="float32", always_2d=True)
    mono = audio.mean(axis=1)
    if rate != 16000:
        mono = soxr.resample(mono, rate, 16000)
    mono = np.asarray(mono, dtype=np.float32)
    engine = None
    started = time.monotonic()
    try:
        engine = ABRecognitionEngine(settings)
        loaded = time.monotonic()
        result = engine.recognize_result(mono)
        return {"backend": settings.recognition_backend, "model_combination": settings.model_combination,
                "text": result.text, "language": result.language,
                "nonempty_transcription": bool(result.text),
                "load_seconds": round(loaded - started, 3),
                "inference_seconds": round(time.monotonic() - loaded, 3),
                "audio_seconds": len(mono) / 16000, "audio_capture": False,
                "settings_written": False, "models_downloaded": False,
                "settings_unchanged": hashlib.sha256(settings_path.read_bytes()).hexdigest() == before,
                "accuracy_evaluated": False}
    finally:
        if engine is not None:
            engine.close()
