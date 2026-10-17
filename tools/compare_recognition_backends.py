"""Opt-in SenseVoice/Whisper A/B on two hash-pinned public WAV files only."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

TOOLS = Path(__file__).resolve().parent
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))
from compare_translation_backends import (
    COMPUTE_TYPES, add_benchmark_arguments, report_header, run_engine_benchmark,
    timing_summary, validate_report_path, write_report,
)
from fanyiji_next.verification import SOURCE_REPO, SOURCE_REVISION, read_official_samples


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="固定官方两段WAV：SenseVoice/Whisper本地A/B；不采音、不评准确率。")
    add_benchmark_arguments(parser)
    for name, device, precision in (("sensevoice", "cpu", "int8"), ("whisper", "cuda", "int8_float16")):
        parser.add_argument("--" + name + "-device", choices=("cpu", "cuda"), default=device)
        parser.add_argument("--" + name + "-compute-type", choices=COMPUTE_TYPES, default=precision)
    parser.add_argument("--sensevoice-model", type=Path)
    parser.add_argument("--sensevoice-tokens", type=Path)
    parser.add_argument("--whisper-model-dir", type=Path)
    return validate_report_path(parser, parser.parse_args(argv))


def run_comparison(args, *, engine_factory=None, settings_type=None, sample_loader=None, clock=None):
    # Hash verification happens before even constructing an inference adapter.
    official = (read_official_samples if sample_loader is None else sample_loader)(args.model_root)
    cases = [{"id": "official_" + language, "language": language, "samples": tuple(samples),
              "sample_sha256": digest, "duration_seconds": len(samples) / 16000}
             for language, samples, digest in official]
    if len(cases) != 2 or {case["language"] for case in cases} != {"zh", "ja"}:
        raise ValueError("The comparison requires exactly the pinned zh.wav and ja.wav")
    if any(case["duration_seconds"] <= 0 for case in cases):
        raise ValueError("Official samples must contain audio frames")
    if engine_factory is None:
        from fanyiji_next.recognition_engines import create_recognition_engine
        engine_factory = create_recognition_engine
    if settings_type is None:
        from fanyiji_next.models import AppSettings
        settings_type = AppSettings
    report = report_header("recognition_ab", args)
    report.update(case_count=len(cases), sample_rate=16000,
        sample_source=f"https://huggingface.co/{SOURCE_REPO}/tree/{SOURCE_REVISION}/test_wavs",
        cases=[{key: value for key, value in case.items() if key != "samples"} for case in cases],
        language_hints="known zh/ja passed to both adapters; SenseVoice factory retains automatic language recognition, Whisper explicitly uses the hint; transcription, not translation",
        quality_limitation="only two fixed public files, no verified reference transcript; no WER/CER or quality ranking")
    paths = {"asr_model": str(args.sensevoice_model or args.model_root / "asr" / "model.int8.onnx"),
             "asr_tokens": str(args.sensevoice_tokens or args.model_root / "asr" / "tokens.txt"),
             "whisper_model_dir": str(args.whisper_model_dir or args.model_root / "whisper-large-v3-ct2")}
    for backend in ("sensevoice", "whisper"):
        device, precision = getattr(args, backend + "_device"), getattr(args, backend + "_compute_type")
        settings = settings_type(**paths, recognition_backend=backend,
                                 recognition_device=device, recognition_compute_type=precision)
        result = run_engine_benchmark(settings, cases, engine_factory,
            lambda engine, case: engine.recognize(case["samples"], language=case["language"]),
            repeats=args.repeats, warmup=args.warmup, clock=clock, backend=backend)
        result["requested_configuration"] = {"backend": backend, "device": device, "compute_type": precision, **paths}
        durations = {case["id"]: case["duration_seconds"] for case in cases}
        factors = []
        for row in result["measurements"]:
            if row["ok"]:
                row["real_time_factor"] = round(row["elapsed_ms"] / 1000 / durations[row["case_id"]], 6)
                factors.append(row["real_time_factor"])
        result["formal_real_time_factor"] = timing_summary(factors)
        report["backends"].append(result)
        if result.get("cancelled") or not result["cleanup_completed"]:
            report["execution_stopped"] = "cancelled_or_cleanup_failed"
            break
    report["ok"] = len(report["backends"]) == 2 and all(item["ok"] for item in report["backends"])
    report["side_by_side"] = [{"case_id": case["id"], "language": case["language"],
        "outputs": {item["requested_configuration"]["backend"]:
        [{"iteration": row["iteration"], "output": row.get("output"), "elapsed_ms": row["elapsed_ms"], "ok": row["ok"]}
         for row in item["measurements"] if row["case_id"] == case["id"]] for item in report["backends"]}}
        for case in cases]
    return report


def main(argv=None):
    import json
    args = parse_args(argv)
    report = run_comparison(args)
    write_report(args.report, report)
    print(json.dumps({"ok": report["ok"], "report": str(args.report), "quality_assessment": "not_scored"}))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
