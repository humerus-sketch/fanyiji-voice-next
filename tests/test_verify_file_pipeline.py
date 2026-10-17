"""The file verifier only decodes pinned samples and never starts live audio."""
import hashlib
import importlib.util
import json
from pathlib import Path
import struct
from types import SimpleNamespace
import wave

import pytest

from fanyiji_next import audio_runtime, speech_gate, verification
from fanyiji_next.models import AppSettings


@pytest.fixture
def verifier(tmp_path, monkeypatch):
    module = verification
    root = tmp_path / "models"
    folder = root / "self_test"
    folder.mkdir(parents=True)
    hashes = {}
    for language in ("zh", "ja"):
        sample = folder / f"{language}.wav"
        with wave.open(str(sample), "wb") as stream:
            stream.setnchannels(1)
            stream.setsampwidth(2)
            stream.setframerate(16000)
            stream.writeframes(struct.pack("<4h", 0, 8192, -8192, 0))
        hashes[language] = hashlib.sha256(sample.read_bytes()).hexdigest()
    # Synthetic fixtures have their own explicit allowlist; actual CLI pins the
    # two published hashes and has no option for arbitrary user recordings.
    monkeypatch.setattr(module, "OFFICIAL_SAMPLES", hashes)
    monkeypatch.setattr(audio_runtime, "validate_models", lambda settings: [])
    monkeypatch.setattr(audio_runtime, "list_audio_devices", lambda: pytest.fail("must not enumerate devices"))
    monkeypatch.setattr(audio_runtime.VoiceRuntime, "start", lambda *args: pytest.fail("must not start audio"))
    return module, root


class FakeEngine:
    def __init__(self, settings):
        self.settings = settings
        self.closed = False
        self.directions = []

    def recognize(self, samples):
        assert samples == [0.0, .25, -.25, 0.0]
        return "fixture transcript"

    def translate(self, text, direction):
        assert text == "fixture transcript"
        self.directions.append(direction)
        return "fixture translation"

    def close(self):
        self.closed = True


class FakeSpeechGate:
    def __init__(self, model):
        self.model = model
        self.calls = []
        self.reset_count = 0
        self.speaking = False
        self.quiet = 0

    def feed(self, samples):
        self.calls.append((len(samples), any(samples)))
        if any(samples):
            self.speaking = True
        elif self.speaking:
            self.quiet += len(samples)
            if self.quiet >= 8000:
                self.speaking = False
                return [SimpleNamespace(samples=[.25, -.25], start_sample=1, end_sample=3,
                                        end_reason="vad_silence")]
        return []

    def reset(self):
        self.reset_count += 1
        self.speaking = False
        self.quiet = 0

    def finish(self):
        pytest.fail("verification must not force a VAD endpoint")


def add_vad_model(root):
    model = root / "vad" / "silero_vad.onnx"
    model.parent.mkdir()
    model.write_bytes(b"fixture VAD model")
    return model


def test_verifier_runs_both_directions_and_releases_models_without_live_audio(verifier):
    module, root = verifier
    engines = []

    def factory(settings):
        engine = FakeEngine(settings)
        engines.append(engine)
        return engine

    report = module.run_verification(root, engine_factory=factory)
    assert report["ok"] and report["models_loaded"] and report["models_released"]
    assert not report["audio_capture"] and not report["network_used"] and not report["settings_written"]
    assert report["quality_assessment"] == "not_scored"
    assert [row["direction"] for row in report["results"]] == ["zh_to_ja", "ja_to_zh"]
    assert engines[0].directions == ["zh_to_ja", "ja_to_zh"]
    assert engines[0].closed
    assert Path(engines[0].settings.asr_model).is_relative_to(root)
    assert Path(engines[0].settings.translation_model_dir).is_relative_to(root)
    assert report["vad"]["status"] == "not_available"
    assert not report["vad"]["model_present"]
    assert report["recognition_backend"] == "sensevoice"
    assert report["translation_backend"] == "m2m100"
    assert report["recognition_device"] == report["translation_device"] == "cpu"
    assert report["recognition_compute_type"] == report["translation_compute_type"] == "int8"


def test_optional_vad_uses_production_gate_with_public_sample_and_half_second_zero_inputs(verifier, monkeypatch):
    module, root = verifier
    model = add_vad_model(root)
    gate = FakeSpeechGate(model)
    paths = []

    def factory(path):
        paths.append(path)
        return gate

    monkeypatch.setattr(speech_gate, "SileroSpeechGate", factory)
    report = module.run_verification(root, engine_factory=FakeEngine)
    assert report["ok"]
    result = report["vad"]
    assert result["status"] == "passed" and result["model_loaded"] and result["model_released"]
    assert result["sha256"] == hashlib.sha256(model.read_bytes()).hexdigest()
    assert result["sample"] == "self_test/zh.wav"
    assert result["completed_segments"] == 1 and result["silence_segments"] == 0
    assert result["padding_samples"] == result["silence_input_samples"] == 8000
    assert result["segments"] == [{"speech_detector": "silero", "end_reason": "vad_silence",
                                    "start_sample": 1, "end_sample": 3, "sample_count": 2}]
    assert paths == [model]
    assert gate.calls[0] == (4, True)
    assert all(not speaking and count <= 640 for count, speaking in gate.calls[1:])
    assert sum(count for count, _ in gate.calls[1:]) == 16000
    assert gate.reset_count == 2
    assert not report["audio_capture"] and not report["network_used"]


def test_present_vad_load_failure_fails_verification_before_loading_asr(verifier, monkeypatch):
    module, root = verifier
    add_vad_model(root)

    def fail_load(path):
        raise RuntimeError("fixture VAD DLL load failed")

    monkeypatch.setattr(speech_gate, "SileroSpeechGate", fail_load)
    report = module.run_verification(root, engine_factory=lambda settings: pytest.fail("must not load ASR"))
    assert not report["ok"] and not report["models_loaded"]
    assert report["phase"] == "load_vad"
    assert report["vad"]["status"] == "failed" and report["vad"]["model_present"]
    assert not report["vad"]["model_loaded"]
    assert "fixture VAD DLL load failed" in report["error"]


def test_vad_that_never_completes_naturally_cannot_pass_by_forced_flush(verifier, monkeypatch):
    module, root = verifier
    model = add_vad_model(root)

    class NoEndpointGate(FakeSpeechGate):
        def feed(self, samples):
            return []

    gate = NoEndpointGate(model)
    monkeypatch.setattr(speech_gate, "SileroSpeechGate", lambda path: gate)
    report = module.run_verification(root, engine_factory=lambda settings: pytest.fail("must not load ASR"))
    assert not report["ok"] and report["phase"] == "verify_vad:zh"
    assert report["vad"]["status"] == "failed" and report["vad"]["model_released"]
    assert report["vad"]["completed_segments"] == 0
    assert "自然结束段" in report["error"]
    assert gate.reset_count == 1


def test_vad_pure_zero_false_positive_fails_and_releases_model(verifier, monkeypatch):
    module, root = verifier
    model = add_vad_model(root)

    class FalsePositiveGate(FakeSpeechGate):
        def feed(self, samples):
            if self.reset_count:
                return [SimpleNamespace(samples=[0.0], start_sample=0, end_sample=1,
                                        end_reason="vad_silence")]
            return super().feed(samples)

    gate = FalsePositiveGate(model)
    monkeypatch.setattr(speech_gate, "SileroSpeechGate", lambda path: gate)
    report = module.run_verification(root, engine_factory=lambda settings: pytest.fail("must not load ASR"))
    assert not report["ok"] and report["phase"] == "verify_vad:silence"
    assert report["vad"]["status"] == "failed" and report["vad"]["model_released"]
    assert report["vad"]["silence_segments"] > 0
    assert "纯零输入" in report["error"]
    assert gate.reset_count == 2


def test_unknown_wave_is_rejected_before_loading_models(verifier):
    module, root = verifier
    (root / "self_test" / "ja.wav").write_bytes(b"unapproved audio fixture")
    report = module.run_verification(root, engine_factory=lambda settings: pytest.fail("must not load models"))
    assert not report["ok"] and not report["models_loaded"]
    assert report["phase"] == "verify_official_samples"
    assert "SHA-256" in report["error"]
    assert report["results"] == []


def test_inference_failure_is_not_reported_as_success_and_still_releases_models(verifier):
    module, root = verifier

    class FailingEngine(FakeEngine):
        def translate(self, text, direction):
            raise RuntimeError("fixture inference failure")

    engine = FailingEngine(None)
    report = module.run_verification(root, engine_factory=lambda settings: engine)
    assert not report["ok"] and report["models_released"]
    assert report["phase"] == "translate:zh_to_ja"
    assert "fixture inference failure" in report["error"]
    assert engine.closed


def test_missing_models_prevent_sample_reads_and_engine_loading(verifier, monkeypatch):
    module, root = verifier
    monkeypatch.setattr(audio_runtime, "validate_models", lambda settings: ["fixture missing model"])
    monkeypatch.setattr(module, "read_official_samples", lambda path: pytest.fail("must not read audio"))
    report = module.run_verification(root, engine_factory=lambda settings: pytest.fail("must not load models"))
    assert not report["ok"] and report["phase"] == "validate_models"
    assert "fixture missing model" in report["error"]


def test_report_cannot_overwrite_any_file_in_read_only_model_root(verifier):
    module, root = verifier
    target = root / "config.json"
    target.write_text("keep model metadata", encoding="utf-8")
    with pytest.raises(SystemExit) as error:
        module.main(["--model-root", str(root), "--report", str(target)])
    assert error.value.code == 2
    assert target.read_text(encoding="utf-8") == "keep model metadata"


def test_cli_writes_only_the_explicit_external_report(verifier, tmp_path, monkeypatch):
    module, root = verifier
    expected = {"ok": True, "audio_capture": False, "settings_written": False}
    monkeypatch.setattr(module, "run_verification", lambda model_root: expected)
    path = Path(__file__).resolve().parents[1] / "tools/verify_file_pipeline.py"
    spec = importlib.util.spec_from_file_location("next_file_verifier_cli", path)
    wrapper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(wrapper)
    assert wrapper.main is verification.main
    report = tmp_path / "checks" / "report.json"
    assert wrapper.main(["--model-root", str(root), "--report", str(report)]) == 0
    assert json.loads(report.read_text(encoding="utf-8")) == expected


def test_whisper_nllb_profile_uses_each_official_sample_language_and_exact_injected_settings(verifier):
    module, root = verifier
    settings = module.create_verification_settings(
        root, recognition_backend="whisper", translation_backend="nllb",
        recognition_device="cuda", translation_device="cuda",
        recognition_compute_type="int8_float16", translation_compute_type="float16",
    )
    engines = []
    class WhisperFakeEngine(FakeEngine):
        def __init__(self, profile):
            super().__init__(profile)
            self.languages = []
        def recognize(self, samples, language=None):
            self.languages.append(language)
            return super().recognize(samples)
    def factory(profile):
        engine = WhisperFakeEngine(profile)
        engines.append(engine)
        return engine
    report = module.run_verification(root, settings=settings, engine_factory=factory)
    assert report["ok"] and report["models_released"]
    assert engines[0].settings is settings
    assert engines[0].languages == ["zh", "ja"]
    assert engines[0].directions == ["zh_to_ja", "ja_to_zh"]
    assert report["recognition_backend"] == "whisper" and report["translation_backend"] == "nllb"
    assert report["recognition_device"] == report["translation_device"] == "cuda"
    assert report["recognition_compute_type"] == "int8_float16"
    assert report["translation_compute_type"] == "float16"
    assert [row["recognition_language"] for row in report["results"]] == ["zh", "ja"]


def test_invalid_cpu_precision_is_rejected_before_sample_or_model_reads(verifier, monkeypatch):
    module, root = verifier
    monkeypatch.setattr(module, "read_official_samples", lambda path: pytest.fail("must not read audio"))
    profile = AppSettings(recognition_backend="whisper", recognition_compute_type="float16")
    report = module.run_verification(root, settings=profile,
                                     engine_factory=lambda settings: pytest.fail("must not load models"))
    assert not report["ok"] and report["phase"] == "validate_models"
    assert "CPU mode uses int8" in report["error"]


def test_backend_settings_use_only_the_documented_model_root_layout(tmp_path):
    profile = verification.create_verification_settings(tmp_path)
    assert profile.asr_model == str(tmp_path / "asr" / "model.int8.onnx")
    assert profile.translation_model_dir == str(tmp_path / "translation")
    assert profile.whisper_model_dir == str(tmp_path / "whisper-large-v3-ct2")
    assert profile.nllb_model_dir == str(tmp_path / "nllb-200-distilled-1.3B-ct2")
    assert profile.nllb_tokenizer_dir == str(tmp_path / "nllb-200-distilled-1.3B-tokenizer")
    assert profile.system_audio_processing == "raw"
    assert profile.mdx_model_file == str(tmp_path / "audio-processing/Kim_Vocal_2.onnx")
    assert profile.gtcrn_model_file == str(tmp_path / "audio-processing/gtcrn_simple.onnx")
    assert not tmp_path.joinpath("settings.json").exists()


def test_cli_selects_independent_profile_without_reading_user_settings(verifier, monkeypatch):
    from fanyiji_next import settings

    module, root = verifier
    calls = []
    def fake_run(model_root, *, settings):
        calls.append((model_root, settings))
        return {"ok": True}
    monkeypatch.setattr(module, "run_verification", fake_run)
    monkeypatch.setattr(settings, "load_settings", lambda *args: pytest.fail("must not read settings"))
    monkeypatch.setattr(settings, "save_settings", lambda *args: pytest.fail("must not write settings"))
    assert module.main(["--model-root", str(root), "--recognition-backend", "whisper",
                        "--translation-backend", "nllb", "--recognition-device", "cuda",
                        "--translation-device", "cuda", "--recognition-compute-type", "int8_float16",
                        "--translation-compute-type", "float16"]) == 0
    assert len(calls) == 1 and calls[0][0] == root
    profile = calls[0][1]
    assert profile.recognition_backend == "whisper" and profile.translation_backend == "nllb"
    assert profile.recognition_device == profile.translation_device == "cuda"
    assert profile.recognition_compute_type == "int8_float16" and profile.translation_compute_type == "float16"


@pytest.mark.parametrize("mode", ["raw", "gtcrn", "mdx", "mdx_gtcrn"])
def test_audio_processing_cli_uses_explicit_profile_paths_without_user_settings(verifier, monkeypatch, mode):
    from fanyiji_next import settings as settings_store
    module, root = verifier
    profiles = []
    monkeypatch.setattr(settings_store, "load_settings", lambda *_args: pytest.fail("must not read user settings"))
    monkeypatch.setattr(settings_store, "save_settings", lambda *_args: pytest.fail("must not write user settings"))
    monkeypatch.setattr(module, "run_verification", lambda _root, *, settings: profiles.append(settings) or {"ok": True})
    assert module.main(["--model-root", str(root), "--system-audio-processing", mode,
                        "--mdx-model-file", "D:/模型/selected-mdx.onnx",
                        "--gtcrn-model-file", "D:/模型/selected-gtcrn.onnx"]) == 0
    assert len(profiles) == 1
    assert profiles[0].system_audio_processing == mode
    assert profiles[0].mdx_model_file == "D:/模型/selected-mdx.onnx"
    assert profiles[0].gtcrn_model_file == "D:/模型/selected-gtcrn.onnx"


def test_file_verifier_marks_source_processing_as_unexercised_even_when_selected(verifier):
    module, root = verifier
    selected = module.create_verification_settings(root, system_audio_processing="mdx_gtcrn")
    report = module.run_verification(root, settings=selected, engine_factory=FakeEngine)
    assert report["ok"] and report["system_audio_processing"] == "mdx_gtcrn"
    assert report["audio_processing_applied"] is False
    assert "not exercised" in report["audio_processing_scope"]


def test_audio_cli_help_names_each_model_and_gui_defaults_preserve_omitted_options():
    import argparse
    parser = argparse.ArgumentParser()
    verification.add_backend_arguments(parser, defaults=False)
    actions = {action.dest: action for action in parser._actions}
    assert "MDX" in actions["mdx_model_file"].help and "Hy-MT2" not in actions["mdx_model_file"].help
    assert "GTCRN" in actions["gtcrn_model_file"].help and "Hy-MT2" not in actions["gtcrn_model_file"].help
    assert "原声直通" in actions["system_audio_processing"].help
    assert "gtcrn 流式降噪" in actions["system_audio_processing"].help
    omitted = parser.parse_args([])
    assert omitted.system_audio_processing is None and omitted.mdx_model_file is None and omitted.gtcrn_model_file is None
    assert parser.parse_args(["--system-audio-processing", "gtcrn"]).system_audio_processing == "gtcrn"
    with pytest.raises(SystemExit):
        parser.parse_args(["--system-audio-processing", "GTCRN"])
