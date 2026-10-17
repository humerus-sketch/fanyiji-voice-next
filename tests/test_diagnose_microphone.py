"""Numeric and fake-PortAudio checks only: never opens a physical device."""
import importlib.util
import json
from pathlib import Path
from threading import get_ident
from types import SimpleNamespace

import numpy as np
import pytest


@pytest.fixture
def diagnostic():
    path = Path(__file__).resolve().parents[1] / "tools/diagnose_microphone.py"
    spec = importlib.util.spec_from_file_location("next_microphone_diagnostic", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_dc_is_reported_separately_and_excluded_from_frequency_bands(diagnostic):
    rate = 16000
    tone = .3 + .2 * np.sin(2 * np.pi * 1000 * np.arange(rate) / rate)
    metrics = diagnostic.energy_metrics(tone, rate)
    assert metrics["mono_dc"] == pytest.approx(.3)
    assert metrics["mono_rms"] == pytest.approx(np.sqrt(.3 ** 2 + .2 ** 2 / 2))
    assert metrics["mono_dc_removed_band_power_fraction"]["300_3400_hz"] > .999
    assert metrics["mono_spectral_flatness"] < .001
    assert metrics["clip_fraction"] == 0


def test_stereo_phase_cancellation_and_clipping_are_visible(diagnostic):
    tone = .1 * np.sin(2 * np.pi * 300 * np.arange(16000) / 16000)
    stereo = np.column_stack((tone, -tone))
    metrics = diagnostic.energy_metrics(stereo, 16000)
    assert metrics["raw_rms"] > .07 and metrics["mono_rms"] < 1e-8
    assert metrics["stereo_correlation"] == pytest.approx(-1)
    assert metrics["channel_rms"][0] == metrics["channel_rms"][1]
    assert metrics["mono_dc_removed_band_power_fraction"] is None
    clipped = diagnostic.energy_metrics(np.array([[1, -1], [.1, .1]]), 16000)
    assert clipped["clip_fraction"] == .5


def test_fft_normalized_correlation_finds_lag_and_sign_without_mutating_inputs(diagnostic):
    random = np.random.default_rng(12)
    reference = random.normal(0, .1, 16000)
    delayed = np.r_[np.zeros(640), -reference[:-640]]
    original = reference.copy()
    valid = np.ones(len(reference), dtype=bool)
    result = diagnostic.normalized_cross_correlation(reference, delayed, valid, valid)
    assert result["lag_ms"] == 40
    assert result["best_correlation"] == pytest.approx(-1)
    assert result["effective_seconds"] == .96
    assert result["source_attribution"] is False
    np.testing.assert_array_equal(reference, original)


def test_near_silence_and_missing_audio_do_not_claim_correlation(diagnostic):
    quiet = np.ones(16000) * .0001
    valid = np.ones(16000, dtype=bool)
    result = diagnostic.normalized_cross_correlation(quiet, quiet, valid, valid)
    assert result["best_correlation"] is None and result["reason"] == "near_silence_or_dc_only"
    valid[1000:] = False
    result = diagnostic.normalized_cross_correlation(quiet, quiet, valid, valid)
    assert result["best_correlation"] is None and result["reason"] == "insufficient_common_contiguous_audio"


def test_low_energy_runs_break_at_missing_audio_and_use_explicit_thresholds(diagnostic):
    samples = np.zeros(16000, dtype=np.float32)
    valid = np.ones(16000, dtype=bool)
    valid[9600:9920] = False
    stats = diagnostic.low_energy_statistics(samples, valid)
    assert stats["threshold_rms"] == .012 and stats["silence_seconds"] == .55
    assert stats["longest_low_energy_seconds"] == .6 and stats["silence_runs"] == 1
    assert stats["valid_windows"] == 49


def test_resampling_and_qpc_start_offsets_limit_shared_audio(diagnostic):
    analysis = diagnostic.ChannelAnalysis(8000, 1, 100.0, 10, "float32")
    tone = (.1 * np.sin(2 * np.pi * 200 * np.arange(8000) / 8000)).astype(np.float32)
    analysis.consume(tone.tobytes(), 101.2, len(tone), 0, 0)
    calls = []

    def resample(values, source, target):
        calls.append((source, target))
        return np.repeat(values, 2)

    report, values, valid = analysis.finish(resample)
    assert calls == [(8000, 16000)]
    assert report["estimated_first_sample_at"] == pytest.approx(100.2)
    assert np.flatnonzero(valid)[0] == 3200 and valid.sum() == 16000
    assert values[:3200].sum() == 0
    assert [row["second"] for row in report["per_second"]] == [0, 1]


class FakeClock:
    def __init__(self):
        self.value = 0.0

    def __call__(self):
        return self.value


class FakeStream:
    def __init__(self, pa, index, options):
        self.pa, self.index, self.options = pa, index, options
        self.activity_checks = 0

    def start_stream(self):
        self.pa.record("start", self.index)
        self.pa.clock.value = 1.0
        channels = self.options["channels"]
        tone = np.random.default_rng(17).normal(0, .05, 16000).astype(np.float32)
        samples = np.repeat(tone[:, None], channels, axis=1)
        if self.options["format"] == 8:
            samples = np.rint(samples * 32768).astype("<i2")
        if self.index not in self.pa.silent_devices:
            assert self.options["stream_callback"](samples.tobytes(), len(tone), {}, 0) == (None, 0)
        if self.index == 17:
            self.pa.clock.value = 10.0

    def is_active(self):
        self.pa.record("is_active", self.index)
        self.activity_checks += 1
        return not (self.pa.inactive_device == self.index and self.activity_checks > 1)

    def stop_stream(self):
        self.pa.record("stop", self.index)
        if self.pa.fail_stop == self.index:
            raise OSError("private stop detail")

    def close(self):
        self.pa.record("close", self.index)


class FakePA:
    def __init__(self, clock, *, fail_open=None, fail_stop=None, silent_devices=(), inactive_device=None):
        self.clock, self.fail_open, self.fail_stop = clock, fail_open, fail_stop
        self.silent_devices, self.inactive_device = silent_devices, inactive_device
        self.events = []
        self.record("initialize")

    def record(self, operation, index=None):
        self.events.append((operation, index, get_ident()))

    def get_device_info_by_index(self, index):
        self.record("query", index)
        return {"index": index, "name": "fixture device", "maxInputChannels": 1 if index == 14 else 2,
                "defaultSampleRate": 16000.0, "hostApi": 2, "isLoopbackDevice": index == 17}

    def get_default_wasapi_loopback(self):
        self.record("resolve_default_loopback")
        return self.get_device_info_by_index(17)

    def open(self, **options):
        index = options["input_device_index"]
        self.record("open", index)
        assert options["input"] is True and options["start"] is False
        assert "output" not in options
        if self.fail_open == index:
            raise OSError("private open detail")
        return FakeStream(self, index, options)

    def terminate(self):
        self.record("terminate")


def fake_run(diagnostic, *, fail_open=None, fail_stop=None, queue_size=256, loopback=None,
             sample_format="int16", silent_devices=(), inactive_device=None):
    clock = FakeClock()
    managers = []

    def factory():
        manager = FakePA(clock, fail_open=fail_open, fail_stop=fail_stop,
                         silent_devices=silent_devices, inactive_device=inactive_device)
        managers.append(manager)
        return manager

    module = SimpleNamespace(PyAudio=factory, paFloat32=1, paInt16=8, paContinue=0)
    args = SimpleNamespace(mic_device=15, compare_mic=14, loopback_device=loopback, duration=10, sample_format=sample_format)
    result = diagnostic.run_diagnosis(args, audio_module=module, clock=clock,
        resampler=lambda *a: pytest.fail("all fixtures already 16k"), queue_size=queue_size)
    return result, managers


def test_single_manager_owner_lifecycle_and_default_loopback_are_read_only(diagnostic):
    report, managers = fake_run(diagnostic)
    assert len(managers) == 1
    events = managers[0].events
    assert {thread for operation, index, thread in events} == {get_ident()}
    assert [index for operation, index, thread in events if operation == "start"] == [15, 14, 17]
    assert [index for operation, index, thread in events if operation == "close"] == [17, 14, 15]
    assert events[-1][0] == "terminate" and report["cleanup_completed"]
    assert any(operation == "resolve_default_loopback" for operation, index, thread in events)
    assert report["channels"]["loopback"]["device"]["index"] == 17
    assert report["channels"]["compare_mic"]["channels"] == 1
    assert report["sample_format"] == "int16"
    assert report["channels"]["mic"]["bytes_per_frame"] == 4
    assert report["channels"]["mic"]["bytes_frame_valid"]
    assert report["channels"]["loopback"]["device"]["activity_checks"] >= 2
    assert report["ok"] and report["cross_correlation"]["mic_to_compare_mic"]["best_correlation"] > .99
    assert not any(report[key] for key in ("raw_audio_saved", "transcripts_saved", "settings_written",
                                          "screenshots_taken", "model_loading", "device_or_volume_changes"))
    json.dumps(report, allow_nan=False)


def test_partial_open_failure_still_closes_every_opened_stream_and_terminates(diagnostic):
    report, managers = fake_run(diagnostic, fail_open=14, loopback=17)
    operations = [operation for operation, index, thread in managers[0].events]
    assert not report["ok"] and report["cleanup_completed"]
    assert operations[-3:] == ["stop", "close", "terminate"]
    assert not any(operation == "start" for operation in operations)
    assert report["errors"] == [{"stage": "open_compare_mic", "error_type": "OSError"}]
    assert "private" not in json.dumps(report)


def test_cleanup_continues_after_stop_failure_and_queue_is_bounded(diagnostic):
    report, managers = fake_run(diagnostic, fail_stop=14, queue_size=1, loopback=17)
    assert report["queue_dropped_blocks"] == {"mic": 0, "compare_mic": 1, "loopback": 1}
    assert not report["ok"] and not report["cleanup_completed"]
    assert [index for operation, index, thread in managers[0].events if operation == "close"] == [17, 14, 15]
    assert managers[0].events[-1][0] == "terminate"
    assert "private" not in json.dumps(report)


@pytest.mark.parametrize("duration", [9, 31])
def test_cli_bounds_duration_and_requires_explicit_microphones(diagnostic, tmp_path, duration):
    with pytest.raises(SystemExit):
        diagnostic.parse_args(["--mic-device", "15", "--compare-mic", "14", "--duration", str(duration),
                               "--report", str(tmp_path / "numeric.json")])
    with pytest.raises(SystemExit):
        diagnostic.parse_args(["--report", str(tmp_path / "numeric.json")])


def test_cli_defaults_to_current_loopback_and_writes_only_requested_json(diagnostic, tmp_path, monkeypatch):
    report_path = tmp_path / "numeric.json"
    argv = ["--mic-device", "15", "--compare-mic", "14", "--report", str(report_path)]
    args = diagnostic.parse_args(argv)
    assert args.loopback_device is None and args.sample_format == "int16"
    monkeypatch.setattr(diagnostic.sys, "platform", "win32")
    monkeypatch.setattr(diagnostic, "run_diagnosis", lambda args: {"ok": True, "raw_audio_saved": False})
    assert diagnostic.main(argv) == 0
    assert list(tmp_path.iterdir()) == [report_path]
    assert json.loads(report_path.read_text(encoding="utf-8"))["raw_audio_saved"] is False


def test_pcm16_and_float32_decode_have_equal_normalized_amplitude(diagnostic):
    pcm = np.tile(np.array([-16384, -8192, 8192, 16384], dtype="<i2"), 4000)
    floating = pcm.astype(np.float32) / 32768.0
    reports = []
    for kind, data in (("int16", pcm), ("float32", floating)):
        analysis = diagnostic.ChannelAnalysis(16000, 1, 0, 10, kind)
        analysis.consume(data.tobytes(), 1.0, len(data), 0, 0)
        report, signal, valid = analysis.finish(lambda *a: pytest.fail("unexpected resample"))
        assert report["sample_format"] == kind and report["bytes_frame_valid"]
        assert report["received_bytes"] == len(data.tobytes())
        assert report["bytes_per_frame"] == (2 if kind == "int16" else 4)
        reports.append(report["per_second"][0])
    assert reports[0]["raw_rms"] == reports[1]["raw_rms"]
    assert reports[0]["mono_rms"] == reports[1]["mono_rms"]


@pytest.mark.parametrize("kind,data", [("int16", b"\0" * 3), ("float32", b"\0" * 6)])
def test_frame_byte_mismatch_is_rejected_without_energy_values(diagnostic, kind, data):
    analysis = diagnostic.ChannelAnalysis(16000, 1, 0, 10, kind)
    analysis.consume(data, 1.0, 2, 0, 0)
    report, _, valid = analysis.finish(lambda *a: pytest.fail("invalid data resampled"))
    assert report["received_callback_blocks"] == 1 and report["callbacks"] == 0
    assert not report["bytes_frame_valid"] and report["invalid_blocks"] == 1
    assert report["per_second"] == [] and not valid.any()


def test_float32_opt_in_selects_float32_stream_and_reports_format(diagnostic):
    report, _ = fake_run(diagnostic, sample_format="float32")
    assert report["ok"] and report["sample_format"] == "float32"
    assert report["channels"]["mic"]["bytes_per_frame"] == 8
    assert report["channels"]["compare_mic"]["bytes_per_frame"] == 4


def test_no_callback_loopback_is_observed_silence_only_when_activity_was_checked(diagnostic):
    report, managers = fake_run(diagnostic, silent_devices=(17,))
    assert report["ok"] and report["cleanup_completed"]
    assert report["loopback_observation"] == "no_callbacks_while_active"
    loopback = report["channels"]["loopback"]
    assert loopback["device"]["all_activity_checks_active"] and loopback["device"]["activity_checks"] >= 2
    assert loopback["received_callback_blocks"] == 0 and loopback["per_second"] == []
    correlation = report["cross_correlation"]["mic_to_loopback"]
    assert correlation["best_correlation"] is None and correlation["lag_ms"] is None
    assert correlation["reason"] == "insufficient_common_contiguous_audio"
    assert any(operation == "is_active" and index == 17 for operation, index, thread in managers[0].events)


def test_inactive_loopback_cannot_be_accepted_as_no_callback_silence(diagnostic):
    report, managers = fake_run(diagnostic, silent_devices=(17,), inactive_device=17)
    assert not report["ok"] and report["cleanup_completed"]
    assert report["loopback_observation"] == "no_callbacks_unverified"
    assert report["channels"]["loopback"]["device"]["all_activity_checks_active"] is False
    assert any(error["stage"] == "active_loopback" for error in report["errors"])
    assert managers[0].events[-1][0] == "terminate"


def test_microphone_without_callbacks_remains_a_failed_diagnostic(diagnostic):
    report, _ = fake_run(diagnostic, silent_devices=(15, 17))
    assert not report["ok"]
    assert {"stage": "callbacks_mic", "error_type": "NoCallbacks"} in report["errors"]
