"""A/B harness checks use fake engines and synthetic inputs, never native inference."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]


def load_tool(name):
    path = ROOT / "tools" / (name + ".py")
    spec = importlib.util.spec_from_file_location("next_" + name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def translation_tool():
    return load_tool("compare_translation_backends")


class FakeClock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


def translation_args(tool, tmp_path, *extra):
    return tool.parse_args(["--model-root", str(tmp_path / "models"), "--report", str(tmp_path / "translation.json"),
                           "--limit", "2", "--warmup", "1", "--repeats", "2", *extra])


def test_fixture_has_balanced_unique_non_private_cases_and_no_reference_scores(translation_tool):
    cases, digest = translation_tool.load_translation_cases()
    assert len(cases) == 50 and len({case["id"] for case in cases}) == 50
    assert sum(case["direction"] == "zh_to_ja" for case in cases) == 25
    assert sum(case["direction"] == "ja_to_zh" for case in cases) == 25
    assert all(case["text"] and case["review_checks"] for case in cases)
    assert len(digest) == 64
    tags = {tag for case in cases for tag in case["tags"]}
    assert {"numbers", "name", "spoken", "negation", "long"} <= tags
    assert all("reference" not in case and "score" not in case for case in cases)


def test_translation_uses_identical_inputs_sequential_engines_and_separate_timings(translation_tool, tmp_path):
    clock, active, calls = FakeClock(), [], []

    class Engine:
        def __init__(self, settings):
            self.backend = settings.translation_backend
            self.backend_info = {"backend": self.backend, "device": settings.translation_device,
                                 "compute_type": settings.translation_compute_type}

        def translate(self, text, direction):
            calls.append((self.backend, text, direction))
            clock.now += .01
            return self.backend + ":" + text

        def close(self):
            clock.now += .02
            active.remove(self.backend)

    def factory(settings, *, stop_event):
        assert not active and not stop_event.is_set()
        active.append(settings.translation_backend)
        clock.now += .1
        return Engine(settings)

    args = translation_args(translation_tool, tmp_path)
    report = translation_tool.run_comparison(args, engine_factory=factory, settings_type=SimpleNamespace, clock=clock)
    assert report["ok"] and not active
    m2m, nllb = report["backends"]
    assert m2m["cold_load_ms"] == nllb["cold_load_ms"] == 100
    assert m2m["first_inference_ms"] == nllb["first_inference_ms"] == 10
    assert m2m["close_ms"] == nllb["close_ms"] == 20
    assert m2m["warmup_timing_ms"]["count"] == 2
    assert m2m["measurements_timing_ms"] == {"count": 4, "p50": 10, "p95": 10, "mean": 10, "min": 10, "max": 10}
    assert [call[1:] for call in calls if call[0] == "m2m100"] == [call[1:] for call in calls if call[0] == "nllb"]
    assert m2m["requested_configuration"]["device"] == "cpu" and nllb["requested_configuration"]["device"] == "cuda"
    assert nllb["requested_configuration"]["compute_type"] == "int8_float16"
    assert all(len(row["outputs"]["m2m100"]) == len(row["outputs"]["nllb"]) == 2 for row in report["side_by_side"])
    assert report["quality_assessment"] == "not_scored" and report["recommendation"] is None
    assert not report["audio_capture"] and not report["settings_written"] and not report["network_used"]


def test_inference_failure_always_closes_and_keeps_partial_error_report(translation_tool, tmp_path):
    closed = []

    class Engine:
        def __init__(self, settings):
            self.name = settings.translation_backend

        def translate(self, text, direction):
            raise RuntimeError("synthetic inference failure")

        def close(self):
            closed.append(self.name)

    report = translation_tool.run_comparison(translation_args(translation_tool, tmp_path),
            engine_factory=lambda settings, **kw: Engine(settings), settings_type=SimpleNamespace)
    assert not report["ok"] and closed == ["m2m100", "nllb"]
    assert all(item["cleanup_completed"] for item in report["backends"])
    assert report["backends"][0]["warmup"][0]["error_type"] == "RuntimeError"
    assert report["backends"][0]["measurements_timing_ms"]["count"] == 0


def test_close_failure_prevents_loading_another_engine(translation_tool, tmp_path):
    loaded = []

    class Engine:
        def translate(self, text, direction):
            return "fixture output"

        def close(self):
            raise RuntimeError("synthetic close failure")

    def factory(settings, **kw):
        loaded.append(settings.translation_backend)
        return Engine()

    report = translation_tool.run_comparison(translation_args(translation_tool, tmp_path),
                                             engine_factory=factory, settings_type=SimpleNamespace)
    assert not report["ok"] and loaded == ["m2m100"]
    assert report["execution_stopped"] == "cancelled_or_cleanup_failed"
    assert report["backends"][0]["cleanup_completed"] is False


def test_empty_output_is_not_automatically_ranked_as_quality(translation_tool, tmp_path):
    class Engine:
        def translate(self, text, direction):
            return ""

        def close(self):
            pass

    report = translation_tool.run_comparison(translation_args(translation_tool, tmp_path),
                     engine_factory=lambda *a, **kw: Engine(), settings_type=SimpleNamespace)
    assert report["ok"] and report["quality_assessment"] == "not_scored" and report["recommendation"] is None
    assert report["backends"][0]["empty_output_count"] == 4


def test_existing_report_is_preserved_and_repeats_are_bounded(translation_tool, tmp_path):
    path = tmp_path / "translation.json"
    path.write_text("keep original", encoding="utf-8")
    with pytest.raises(SystemExit):
        translation_args(translation_tool, tmp_path)
    assert path.read_text(encoding="utf-8") == "keep original"
    with pytest.raises(FileExistsError):
        translation_tool.write_report(path, {"ok": True})
    with pytest.raises(SystemExit):
        translation_tool.parse_args(["--model-root", str(tmp_path / "models"), "--report", str(tmp_path / "new.json"), "--repeats", "6"])


def test_translation_cli_writes_only_explicit_report_with_fake_engine(translation_tool, tmp_path, monkeypatch):
    target = tmp_path / "report.json"
    monkeypatch.setattr(translation_tool, "run_comparison", lambda args: {"ok": True, "quality_assessment": "not_scored"})
    assert translation_tool.main(["--model-root", str(tmp_path / "models"), "--report", str(target)]) == 0
    assert list(tmp_path.iterdir()) == [target]
    assert json.loads(target.read_text(encoding="utf-8"))["quality_assessment"] == "not_scored"


@pytest.fixture
def recognition_tool():
    return load_tool("compare_recognition_backends")


def recognition_args(tool, tmp_path, *extra):
    return tool.parse_args(["--model-root", str(tmp_path / "models"), "--report", str(tmp_path / "recognition.json"),
                           "--warmup", "1", "--repeats", "2", *extra])


def public_samples(_):
    return [("zh", [.123456, -.12] * 800, "a" * 64), ("ja", [.234567, -.23] * 1600, "b" * 64)]


def test_recognition_uses_hash_verified_public_inputs_and_language_hints_sequentially(recognition_tool, tmp_path, monkeypatch):
    clock, active, calls, loaders = FakeClock(), [], [], []

    class Engine:
        def __init__(self, settings):
            self.name = settings.recognition_backend
            self.backend_info = {"backend": self.name, "device": settings.recognition_device,
                                 "compute_type": "model" if self.name == "sensevoice" else settings.recognition_compute_type,
                                 "requested_device": settings.recognition_device,
                                 "requested_compute_type": settings.recognition_compute_type,
                                 "model_identity": "synthetic-public-fixture", "vad_filter": False,
                                 "transcribe_options": {"beam_size": 5, "task": "transcribe", "initial_prompt": "excluded"}}

        def recognize(self, samples, language=None):
            calls.append((self.name, samples, language))
            clock.now += .02
            return "public " + language + " " + self.name

        def close(self):
            active.remove(self.name)

    def factory(settings, *, stop_event):
        assert loaders and not active and not stop_event.is_set()
        active.append(settings.recognition_backend)
        clock.now += .05
        return Engine(settings)

    def loader(root):
        loaders.append(root)
        return public_samples(root)

    # Exercise the default verification loader hook without reading real model/sample files.
    monkeypatch.setattr(recognition_tool, "read_official_samples", loader)
    args = recognition_args(recognition_tool, tmp_path)
    report = recognition_tool.run_comparison(args, engine_factory=factory, settings_type=SimpleNamespace, clock=clock)
    assert report["ok"] and not active and loaders == [args.model_root]
    sensevoice, whisper = report["backends"]
    assert [item["backend"] for item in report["backends"]] == ["sensevoice", "whisper"]
    assert sensevoice["cold_load_ms"] == whisper["cold_load_ms"] == 50
    assert sensevoice["warmup_timing_ms"]["count"] == 2
    assert sensevoice["measurements_timing_ms"]["p50"] == 20
    assert sensevoice["formal_real_time_factor"]["p50"] == pytest.approx(.15)
    assert [call[1:] for call in calls if call[0] == "sensevoice"] == [call[1:] for call in calls if call[0] == "whisper"]
    assert {call[2] for call in calls} == {"zh", "ja"}
    assert all(isinstance(call[1], tuple) for call in calls)
    assert whisper["requested_configuration"]["device"] == "cuda"
    assert whisper["backend_info"]["vad_filter"] is False
    assert whisper["backend_info"]["transcribe_options"] == {"beam_size": 5, "task": "transcribe"}
    assert sensevoice["backend_info"]["compute_type"] == "model"
    assert sensevoice["backend_info"]["requested_compute_type"] == "int8"
    assert whisper["backend_info"]["device"] == whisper["backend_info"]["requested_device"] == "cuda"
    assert whisper["backend_info"]["model_identity"] == "synthetic-public-fixture"
    assert "SenseVoice factory retains automatic language recognition" in report["language_hints"]
    assert "Whisper explicitly uses the hint" in report["language_hints"]
    assert report["cases"][0]["sample_sha256"] == "a" * 64
    encoded = json.dumps(report)
    assert "0.123456" not in encoded and "0.234567" not in encoded
    assert all("samples" not in case for case in report["cases"])
    assert report["quality_assessment"] == "not_scored" and report["recommendation"] is None
    assert report["audio_capture"] is False and report["new_audio_saved"] is False


def test_recognition_hash_failure_prevents_engine_creation(recognition_tool, tmp_path):
    def reject(root):
        raise ValueError("fixed official SHA mismatch")

    with pytest.raises(ValueError, match="SHA"):
        recognition_tool.run_comparison(recognition_args(recognition_tool, tmp_path),
            sample_loader=reject, engine_factory=lambda *a, **kw: pytest.fail("hash failed but model loaded"),
            settings_type=SimpleNamespace)


def test_recognition_closes_after_decode_error_and_persists_error_in_memory_report(recognition_tool, tmp_path):
    closed = []

    class Engine:
        def __init__(self, settings):
            self.name = settings.recognition_backend

        def recognize(self, samples, language=None):
            raise RuntimeError("synthetic decode failure")

        def close(self):
            closed.append(self.name)

    report = recognition_tool.run_comparison(recognition_args(recognition_tool, tmp_path),
        sample_loader=public_samples, engine_factory=lambda settings, **kw: Engine(settings), settings_type=SimpleNamespace)
    assert closed == ["sensevoice", "whisper"] and not report["ok"]
    assert all(item["cleanup_completed"] for item in report["backends"])
    assert report["backends"][0]["warmup"][0]["error_type"] == "RuntimeError"


def test_cancelled_benchmark_closes_and_does_not_start_second_engine(translation_tool, tmp_path):
    operations = []

    class Engine:
        def translate(self, text, direction):
            raise KeyboardInterrupt()

        def close(self):
            operations.append("close")

    def factory(settings, **kw):
        operations.append("load:" + settings.translation_backend)
        return Engine()

    report = translation_tool.run_comparison(translation_args(translation_tool, tmp_path),
                                             engine_factory=factory, settings_type=SimpleNamespace)
    assert operations == ["load:m2m100", "close"]
    assert report["execution_stopped"] == "cancelled_or_cleanup_failed" and not report["ok"]


def test_custom_device_precision_and_local_paths_are_explicit(recognition_tool, translation_tool, tmp_path):
    recognition = recognition_args(recognition_tool, tmp_path, "--whisper-device", "cpu",
        "--whisper-compute-type", "int8", "--whisper-model-dir", str(tmp_path / "custom-whisper"))
    assert recognition.whisper_device == "cpu" and recognition.whisper_compute_type == "int8"
    assert recognition.whisper_model_dir == tmp_path / "custom-whisper"
    translation = translation_args(translation_tool, tmp_path, "--nllb-device", "cpu", "--nllb-compute-type", "int8")
    assert translation.nllb_device == "cpu" and translation.nllb_compute_type == "int8"
