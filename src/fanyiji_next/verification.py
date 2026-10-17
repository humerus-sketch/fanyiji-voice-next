"""Verify local ASR/translation and optional VAD using pinned public samples."""
from __future__ import annotations

import argparse
from array import array
import hashlib
import json
from pathlib import Path
import sys
import time
import wave

SOURCE_REPO = "csukuangfj/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17"
SOURCE_REVISION = "2365baeacb507f821a0c8120fcee3d484dba7a07"
OFFICIAL_SAMPLES = {
    "zh": "b77f1794fe374a0ba1ee1dc458bfaf9349496cbbfc32780c50ba3c5a7ad8e373",
    "ja": "460bd8dccb0d2a5f4e29c628f837be4082d13defc64c3fc21dd1b6bb0e119095",
}
BACKEND_OPTIONS = (
    ("recognition_backend", ("sensevoice", "whisper"), "sensevoice"),
    ("translation_backend", ("m2m100", "nllb", "hymt"), "m2m100"),
    ("recognition_device", ("cpu", "cuda"), "cpu"),
    ("translation_device", ("cpu", "cuda"), "cpu"),
    ("recognition_compute_type", ("int8", "int8_float16", "float16"), "int8"),
    ("translation_compute_type", ("int8", "int8_float16", "float16"), "int8"),
    ("system_audio_processing", ("raw", "gtcrn", "gtcrn_preserve", "mdx", "mdx_gtcrn"), "raw"),
)
BACKEND_PATH_OPTIONS = ("hymt_model_file", "hymt_server_exe", "mdx_model_file", "gtcrn_model_file")


def add_backend_arguments(parser, *, defaults=True):
    """Keep executable and standalone verifier flags identical."""
    for name, choices, default in BACKEND_OPTIONS:
        help_text = ("仅电脑声音：raw 原声直通；gtcrn 流式降噪；gtcrn_preserve 原声断句识别、GTCRN并行诊断（试验）；mdx 提取人声；mdx_gtcrn 提取后降噪。固定 WAV 验证不测试输入增强链。"
                     if name == "system_audio_processing" else None)
        parser.add_argument("--" + name.replace("_", "-"), choices=choices,
                            default=default if defaults else None, help=help_text)
    path_help = {"hymt_model_file": "Hy-MT2 Q8_0 本地 GGUF 文件，不下载",
                 "hymt_server_exe": "Hy-MT2 本地 llama-server.exe，不下载",
                 "mdx_model_file": "电脑声音 MDX 本地 Kim_Vocal_2.onnx，不下载",
                 "gtcrn_model_file": "电脑声音 GTCRN 本地 gtcrn_simple.onnx，不下载"}
    for name in BACKEND_PATH_OPTIONS:
        parser.add_argument("--" + name.replace("_", "-"), help=path_help[name])


def create_verification_settings(model_root, **backend_options):
    """Construct an explicit local profile without accessing user settings."""
    from .models import AppSettings

    root = Path(model_root).resolve()
    values = {name: default for name, _choices, default in BACKEND_OPTIONS}
    values.update(
        asr_model=str(root / "asr" / "model.int8.onnx"),
        asr_tokens=str(root / "asr" / "tokens.txt"),
        translation_model_dir=str(root / "translation"),
        translation_spm=str(root / "translation" / "sentencepiece.bpe.model"),
        whisper_model_dir=str(root / "whisper-large-v3-ct2"),
        nllb_model_dir=str(root / "nllb-200-distilled-1.3B-ct2"),
        nllb_tokenizer_dir=str(root / "nllb-200-distilled-1.3B-tokenizer"),
        hymt_model_file=str(root / "hymt2-1.8b" / "Hy-MT2-1.8B-Q8_0.gguf"),
        hymt_server_exe=str(root.parent / "runtime" / "llama-b11696-cuda12.4" / "llama-server.exe"),
        mdx_model_file=str(root / "audio-processing" / "Kim_Vocal_2.onnx"),
        gtcrn_model_file=str(root / "audio-processing" / "gtcrn_simple.onnx"),
    )
    values.update(backend_options)
    return AppSettings(**values)


def read_official_samples(model_root):
    result = []
    for language, expected_hash in OFFICIAL_SAMPLES.items():
        path = model_root / "self_test" / f"{language}.wav"
        with path.open("rb") as stream:
            actual_hash = hashlib.file_digest(stream, "sha256").hexdigest()
        if actual_hash != expected_hash:
            raise ValueError(f"{language}.wav 与固定官方样例 SHA-256 不符，未解码该文件。")
        with wave.open(str(path), "rb") as stream:
            if (stream.getnchannels(), stream.getsampwidth(), stream.getframerate(), stream.getcomptype()) != (1, 2, 16000, "NONE"):
                raise ValueError(f"{language}.wav 必须是官方 16kHz 单声道 PCM16 样例。")
            frames = stream.getnframes()
            pcm = array("h", stream.readframes(frames))
        if len(pcm) != frames or not frames:
            raise ValueError(f"{language}.wav 样例为空或已截断。")
        if sys.byteorder != "little":
            pcm.byteswap()
        result.append((language, [sample / 32768.0 for sample in pcm], actual_hash))
    return result


def _verify_optional_vad(model_root, official_samples, report):
    """Exercise the production detector without opening devices or forcing a tail."""
    result = report["vad"]
    model = model_root / "vad" / "silero_vad.onnx"
    if not model.exists():
        result["status"] = "not_available"
        return
    result["model_present"] = True
    gate = None
    try:
        report["phase"] = "load_vad"
        with model.open("rb") as stream:
            result["sha256"] = hashlib.file_digest(stream, "sha256").hexdigest()
        from .speech_gate import SileroSpeechGate

        gate = SileroSpeechGate(model)
        result["model_loaded"] = True
        language, waveform, digest = next(
            (sample for sample in official_samples if sample[0] == "zh"), official_samples[0],
        )
        result.update(sample=f"self_test/{language}.wav", sample_sha256=digest,
                      sample_rate=16000, padding_samples=8000, silence_input_samples=8000)
        report["phase"] = "verify_vad:" + language

        def feed(samples):
            completed = []
            for offset in range(0, len(samples), 640):
                completed.extend(gate.feed(samples[offset:offset + 640]))
            return completed

        # Feed an actual half-second of zeros through the native model. Do not
        # call finish()/flush(): a forced input-gap segment could mask a broken
        # natural endpoint and make a frozen build appear to pass.
        completed = feed(waveform)
        completed.extend(feed([0.0] * 8000))
        result["completed_segments"] = len(completed)
        result["segments"] = [
            {"speech_detector": "silero", "end_reason": item.end_reason,
             "start_sample": item.start_sample, "end_sample": item.end_sample,
             "sample_count": len(item.samples)}
            for item in completed
        ]
        if not any(item.end_reason == "vad_silence" for item in completed):
            raise RuntimeError("Silero VAD 未从固定官方语音及 0.5 秒零输入中产生自然结束段。")
        gate.reset()
        report["phase"] = "verify_vad:silence"
        silence = feed([0.0] * 8000)
        result["silence_segments"] = len(silence)
        if silence:
            raise RuntimeError("Silero VAD 将纯零输入误判为已完成语音段。")
        result["status"] = "passed"
    except Exception as exc:
        result.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        if gate is not None:
            try:
                gate.reset()
            except Exception as exc:
                result.update(status="failed", cleanup_error=f"{type(exc).__name__}: {exc}")
                raise
            finally:
                gate = None
            result["model_released"] = True


def run_verification(model_root: Path, *, engine_factory=None, settings=None):
    from .audio_runtime import _LocalEngine, validate_models
    from .settings import validate_settings

    model_root = Path(model_root).resolve()
    settings = settings if settings is not None else create_verification_settings(model_root)
    report = {"ok": False, "audio_capture": False, "settings_written": False, "network_used": False,
              "models_loaded": False, "models_released": False, "quality_assessment": "not_scored",
              "audio_processing_applied": False,
              "audio_processing_scope": "Fixed official WAVs go directly to ASR/translation; system capture MDX/GTCRN is not exercised.",
              "sample_source": f"https://huggingface.co/{SOURCE_REPO}/tree/{SOURCE_REVISION}/test_wavs",
              "results": [], "phase": "validate_models",
              "vad": {"model": "vad/silero_vad.onnx", "status": "not_checked",
                      "model_present": False, "model_loaded": False, "model_released": False,
                      "speech_detector": "silero", "completed_segments": 0,
                      "silence_segments": 0, "segments": []}}
    report.update({name: getattr(settings, name) for name, _choices, _default in BACKEND_OPTIONS})
    if settings.translation_backend == "hymt":
        report.update(translation_device="cuda", translation_compute_type="not_applicable",
                      translation_quantization="Q8_0")
    engine = None
    try:
        validate_settings(settings)
        issues = validate_models(settings)
        if issues:
            raise ValueError("\n".join(issues))
        report["phase"] = "verify_official_samples"
        official_samples = read_official_samples(model_root)
        _verify_optional_vad(model_root, official_samples, report)
        report["phase"] = "load_models"
        started = time.perf_counter()
        engine = (engine_factory or _LocalEngine)(settings)
        report["models_loaded"] = True
        report["model_load_seconds"] = round(time.perf_counter() - started, 3)
        for language, waveform, digest in official_samples:
            direction = "zh_to_ja" if language == "zh" else "ja_to_zh"
            report["phase"] = "recognize:" + language
            started = time.perf_counter()
            if settings.recognition_backend == "whisper":
                original = engine.recognize(waveform, language=language).strip()
            else:
                original = engine.recognize(waveform).strip()
            asr_ms = round((time.perf_counter() - started) * 1000, 1)
            if not original:
                raise RuntimeError(f"{language}.wav 识别结果为空。")
            report["phase"] = "translate:" + direction
            started = time.perf_counter()
            translated = engine.translate(original, direction).strip()
            if not translated:
                raise RuntimeError(f"{direction} 翻译结果为空。")
            report["results"].append({"sample": f"self_test/{language}.wav", "sha256": digest,
                                      "direction": direction, "sample_rate": 16000,
                                      "recognition_language": language if settings.recognition_backend == "whisper" else "auto",
                                      "duration_seconds": round(len(waveform) / 16000, 3),
                                      "original": original, "translated": translated, "asr_ms": asr_ms,
                                      "translation_ms": round((time.perf_counter() - started) * 1000, 1)})
        report.update(ok=True, phase="complete")
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        if engine is not None:
            try:
                engine.close()
                report["models_released"] = True
            except Exception as exc:
                report.update(ok=False, cleanup_error=f"{type(exc).__name__}: {exc}")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description="仅用固定官方 WAV 样例做离线文件推理，不枚举设备或采集声音。")
    parser.add_argument("--model-root", required=True, type=Path)
    parser.add_argument("--report", type=Path)
    add_backend_arguments(parser)
    args = parser.parse_args(argv)
    if args.report is not None and args.report.resolve().is_relative_to(args.model_root.resolve()):
        parser.error("请将验证报告保存到模型目录之外，模型目录只读。")
    options = {name: getattr(args, name) for name, _choices, _default in BACKEND_OPTIONS}
    paths = {name: getattr(args, name) for name in BACKEND_PATH_OPTIONS if getattr(args, name) is not None}
    # Retain the original one-argument default call for existing verifier users.
    if not paths and all(options[name] == default for name, _choices, default in BACKEND_OPTIONS):
        report = run_verification(args.model_root)
    else:
        report = run_verification(args.model_root, settings=create_verification_settings(args.model_root, **options, **paths))
    content = json.dumps(report, ensure_ascii=False, indent=2)
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(content + "\n", encoding="utf-8")
    if sys.stdout is not None:
        print(content)
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
