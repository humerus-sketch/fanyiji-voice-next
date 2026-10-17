"""Opt-in local translation A/B. Public fictional inputs; no audio or user settings."""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys
from threading import Event
import time

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "src"
if str(SOURCE) not in sys.path:
    sys.path.insert(0, str(SOURCE))
DEFAULT_FIXTURES = Path(__file__).resolve().parent / "fixtures" / "bilingual_ab.json"
COMPUTE_TYPES = ("int8", "int8_float16", "float16")


def timing_summary(values):
    values = sorted(values)
    if not values:
        return {"count": 0, "p50": None, "p95": None, "mean": None, "min": None, "max": None}
    return {"count": len(values), "p50": round(statistics.median(values), 3),
            "p95": round(values[math.ceil(len(values) * .95) - 1], 3),
            "mean": round(statistics.mean(values), 3), "min": round(values[0], 3), "max": round(values[-1], 3)}


def load_translation_cases(path=DEFAULT_FIXTURES, *, limit=None):
    raw = Path(path).read_bytes()
    payload = json.loads(raw.decode("utf-8"))
    cases = payload.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("Fixture cases must be a nonempty list")
    ids = set()
    for case in cases:
        if not isinstance(case, dict) or not isinstance(case.get("id"), str) or not case["id"] or case["id"] in ids:
            raise ValueError("Fixture ids must be unique nonempty strings")
        if case.get("direction") not in {"zh_to_ja", "ja_to_zh"} or not isinstance(case.get("text"), str) or not case["text"].strip():
            raise ValueError("Each fixture needs direction and nonempty source text")
        if not isinstance(case.get("review_checks", []), list) or not all(isinstance(item, str) for item in case.get("review_checks", [])):
            raise ValueError("Review checks must be prose, not automatic scores")
        ids.add(case["id"])
    return cases[:limit] if limit is not None else cases, hashlib.sha256(raw).hexdigest()


def run_engine_benchmark(settings, cases, factory, invoke, *, repeats=1, warmup=1, clock=None, backend=None):
    """One engine at a time; factory eagerly loads, first inference may still warm kernels."""
    clock = time.perf_counter if clock is None else clock
    result = {"backend": backend or getattr(settings, "translation_backend", "unknown"),
              "ok": False, "quality_assessment": "not_scored", "cold_load_ms": None,
              "first_inference_ms": None, "close_ms": None, "cleanup_completed": False,
              "warmup": [], "measurements": [], "errors": [], "backend_info": {}}
    engine = None
    stop = Event()
    load_started = clock()
    try:
        engine = factory(settings, stop_event=stop)
        result["cold_load_ms"] = round((clock() - load_started) * 1000, 3)
        info = getattr(engine, "backend_info", {})
        if isinstance(info, dict):
            allowed = ("backend", "device", "compute_type", "model_dir", "tokenizer_dir", "source_layout",
                       "requested_device", "requested_compute_type", "model_identity",
                       "model", "sample_rate", "num_threads", "local_files_only", "language_mode", "use_itn",
                       "beam_size", "temperature", "condition_on_previous_text", "vad_filter", "without_timestamps", "task")
            result["backend_info"] = {key: info[key] for key in allowed if key in info}
            options = info.get("transcribe_options")
            if isinstance(options, dict):
                keys = ("beam_size", "temperature", "condition_on_previous_text", "vad_filter", "without_timestamps", "word_timestamps", "task")
                result["backend_info"]["transcribe_options"] = {key: options[key] for key in keys if key in options}
        for phase, rounds in (("warmup", warmup), ("measurements", repeats)):
            for iteration in range(1, rounds + 1):
                for case in cases:
                    started = clock()
                    row = {"case_id": case["id"], "iteration": iteration}
                    try:
                        output = invoke(engine, case)
                        if not isinstance(output, str):
                            raise TypeError("Engine output must be a string")
                        row.update(output=output, ok=True)
                    except (Exception, KeyboardInterrupt) as exc:
                        row.update(ok=False, error_type=type(exc).__name__, error=str(exc))
                        result["errors"].append({"phase": phase, "case_id": case["id"], "error_type": type(exc).__name__})
                        if isinstance(exc, KeyboardInterrupt):
                            result["cancelled"] = True
                    row["elapsed_ms"] = round((clock() - started) * 1000, 3)
                    if result["first_inference_ms"] is None:
                        result["first_inference_ms"] = row["elapsed_ms"]
                    result[phase].append(row)
                    if not row["ok"]:
                        return result
        result["ok"] = True
    except (Exception, KeyboardInterrupt) as exc:
        result["cold_load_ms"] = round((clock() - load_started) * 1000, 3)
        result["errors"].append({"phase": "load", "error_type": type(exc).__name__, "error": str(exc)})
        result["cancelled"] = isinstance(exc, KeyboardInterrupt)
    finally:
        stop.set()
        if engine is not None:
            started = clock()
            try:
                engine.close()
                result["cleanup_completed"] = True
            except Exception as exc:
                result["ok"] = False
                result["errors"].append({"phase": "close", "error_type": type(exc).__name__, "error": str(exc)})
            result["close_ms"] = round((clock() - started) * 1000, 3)
        else:
            result["cleanup_completed"] = True  # Factory is responsible for its failed construction.
        for phase in ("warmup", "measurements"):
            rows = result[phase]
            result[phase + "_timing_ms"] = timing_summary([row["elapsed_ms"] for row in rows if row["ok"]])
        result["empty_output_count"] = sum(row.get("output") == "" for row in result["measurements"] if row["ok"])
    return result


def report_header(kind, args):
    info = time.get_clock_info("perf_counter")
    return {"kind": kind, "created_at": datetime.now(timezone(timedelta(hours=8))).isoformat(),
            "ok": False, "quality_assessment": "not_scored", "recommendation": None,
            "audio_capture": False, "new_audio_saved": False, "settings_written": False,
            "network_used": False, "model_root": str(args.model_root),
            "repeats": args.repeats, "warmup_rounds": args.warmup,
            "warmup_scope": "each round runs the same complete input set; excluded from formal P50/P95",
            "cold_load_scope": "engine factory includes local dependencies/config/tokenizer/native model loading; first inference reported separately",
            "comparison_boundary": "different model/device/precision configurations; elapsed differences are not isolated model effects or quality scores",
            "clock": {"function": "time.perf_counter", "implementation": info.implementation,
                      "resolution_seconds": info.resolution}, "backends": []}


def write_report(path, report):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(report, ensure_ascii=False, allow_nan=False, indent=2) + "\n")


def add_benchmark_arguments(parser):
    parser.add_argument("--model-root", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--repeats", choices=range(1, 6), type=int, default=1, metavar="1..5")
    parser.add_argument("--warmup", choices=range(0, 4), type=int, default=1, metavar="0..3")


def validate_report_path(parser, args):
    if args.report.exists():
        parser.error("报告已存在；请选择新文件，历史报告不会被覆盖。")
    if args.report.resolve().is_relative_to(args.model_root.resolve()):
        parser.error("报告请保存到模型目录之外。")
    return args


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="固定非私人示例：M2M100与NLLB本地A/B，不给出自动质量排名。")
    add_benchmark_arguments(parser)
    parser.add_argument("--fixtures", type=Path, default=DEFAULT_FIXTURES)
    parser.add_argument("--limit", type=int, choices=range(1, 51), metavar="1..50")
    for name, device, precision in (("m2m", "cpu", "int8"), ("nllb", "cuda", "int8_float16")):
        parser.add_argument("--" + name + "-device", choices=("cpu", "cuda"), default=device)
        parser.add_argument("--" + name + "-compute-type", choices=COMPUTE_TYPES, default=precision)
    parser.add_argument("--m2m-model-dir", type=Path)
    parser.add_argument("--m2m-spm", type=Path)
    parser.add_argument("--nllb-model-dir", type=Path)
    parser.add_argument("--nllb-tokenizer-dir", type=Path)
    return validate_report_path(parser, parser.parse_args(argv))


def run_comparison(args, *, engine_factory=None, settings_type=None, clock=None):
    if engine_factory is None:
        from fanyiji_next.translation_engines import create_translation_engine
        engine_factory = create_translation_engine
    if settings_type is None:
        from fanyiji_next.models import AppSettings
        settings_type = AppSettings
    cases, digest = load_translation_cases(args.fixtures, limit=args.limit)
    report = report_header("translation_ab", args)
    report.update(fixture_sha256=digest, fixture_path=str(args.fixtures), cases=cases,
                  case_count=len(cases), reference_status="synthetic non-private examples; review checks are not validated references")
    paths = {"translation_model_dir": str(args.m2m_model_dir or args.model_root / "translation"),
             "translation_spm": str(args.m2m_spm or args.model_root / "translation" / "sentencepiece.bpe.model"),
             "nllb_model_dir": str(args.nllb_model_dir or args.model_root / "nllb-200-distilled-1.3B-ct2"),
             "nllb_tokenizer_dir": str(args.nllb_tokenizer_dir or args.model_root / "nllb-200-distilled-1.3B-tokenizer")}
    for backend, prefix in (("m2m100", "m2m"), ("nllb", "nllb")):
        device, precision = getattr(args, prefix + "_device"), getattr(args, prefix + "_compute_type")
        settings = settings_type(**paths, translation_backend=backend,
                                 translation_device=device, translation_compute_type=precision)
        result = run_engine_benchmark(settings, cases, engine_factory,
            lambda engine, case: engine.translate(case["text"], case["direction"]),
            repeats=args.repeats, warmup=args.warmup, clock=clock, backend=backend)
        result["requested_configuration"] = {"backend": backend, "device": device, "compute_type": precision, **paths}
        report["backends"].append(result)
        if result.get("cancelled") or not result["cleanup_completed"]:
            report["execution_stopped"] = "cancelled_or_cleanup_failed"
            break
    report["ok"] = len(report["backends"]) == 2 and all(item["ok"] for item in report["backends"])
    report["side_by_side"] = [{"case_id": case["id"], "direction": case["direction"], "source": case["text"],
        "review_checks": case.get("review_checks", []), "outputs": {item["requested_configuration"]["backend"]:
        [{"iteration": row["iteration"], "output": row.get("output"), "elapsed_ms": row["elapsed_ms"], "ok": row["ok"]}
         for row in item["measurements"] if row["case_id"] == case["id"]] for item in report["backends"]}}
        for case in cases]
    return report


def main(argv=None):
    args = parse_args(argv)
    report = run_comparison(args)
    write_report(args.report, report)
    print(json.dumps({"ok": report["ok"], "report": str(args.report), "quality_assessment": "not_scored"}))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
