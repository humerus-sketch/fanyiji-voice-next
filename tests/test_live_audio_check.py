"""Live-check reporting and playback helpers tested without any real audio device."""
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import struct
import subprocess
from threading import Event
from types import SimpleNamespace
import wave

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtCore import QObject, Signal
from PySide6.QtWidgets import QApplication, QLabel

from fanyiji_next import verification


@pytest.fixture
def live_tool():
    path = Path(__file__).resolve().parents[1] / "tools/live_audio_check.py"
    spec = importlib.util.spec_from_file_location("next_live_audio_check", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def qt_app():
    return QApplication.instance() or QApplication([])


def event(kind, *, text="private fixture phrase", **values):
    return SimpleNamespace(kind=kind, session_id=1, channel="mic", segment_id="segment-1",
                           text=text, level=0.0, message="", status="", elapsed_ms=0, **values)


def test_private_text_is_absent_and_pairs_require_matching_ui_and_caption_data(live_tool):
    now = [100.0]
    metrics = live_tool.EventMetrics(clock=lambda: now[0])
    original, translated = event("original"), event("translation")
    metrics.received(original)
    now[0] += .01
    sequence = metrics.applied(original, ui_text_matches=True, paired_data_matches=True)
    now[0] += .02
    metrics.painted(sequence)
    metrics.received(translated)
    metrics.applied(translated, ui_text_matches=False, paired_data_matches=True)
    report = metrics.snapshot()
    assert "private fixture phrase" not in json.dumps(report)
    assert report["updates"][0]["char_count"] == len(original.text)
    assert report["updates"][0]["callback_to_apply_ms"] == 10
    assert report["updates"][0]["apply_to_paint_proxy_ms"] == 20
    assert not report["segments"][0]["paired"]


def test_only_fixture_is_an_explicit_transcript_opt_in(live_tool):
    metrics = live_tool.EventMetrics(only_fixture=True)
    item = event("original", text="operator-declared fixture")
    metrics.received(item)
    assert metrics.snapshot()["updates"][0]["fixture_text"] == item.text


def test_review_text_opt_in_does_not_mislabel_live_speech_as_a_fixture(live_tool):
    metrics = live_tool.EventMetrics(review_text=True)
    item = event("original", text="本轮人工核查字幕")
    metrics.received(item)
    row = metrics.snapshot()["updates"][0]
    assert row["review_text"] == item.text
    assert "fixture_text" not in row


def test_segment_diagnostics_belong_to_each_call_and_are_copied(live_tool):
    engine = SimpleNamespace(recognizer=SimpleNamespace(last_diagnostics={"suspect": True}),
                             translator=SimpleNamespace(last_diagnostics={"calls": 9}))
    captured = []

    def process(session, current_engine, segment):
        assert session == "fixture-session" and current_engine is engine
        engine.recognizer.last_diagnostics = {"segments": [{"char_count": len(segment.segment_id)}], "suspect": False}
        engine.translator.last_diagnostics = {"calls": len(segment.segment_id), "issues": [],
                                              "not_reported_private_text": "private phrase"}
        return "fixture-return"

    for segment_id in ("first", "second"):
        segment = SimpleNamespace(channel="system", segment_id=segment_id)
        assert live_tool.observe_segment_diagnostics(process, "fixture-session", engine, segment, captured) == "fixture-return"
    assert [row["translation"]["calls"] for row in captured] == [5, 6]
    assert [row["recognition"]["segments"][0]["char_count"] for row in captured] == [5, 6]
    engine.recognizer.last_diagnostics["segments"][0]["char_count"] = 999
    engine.translator.last_diagnostics["issues"].append("later-change")
    assert captured[1]["recognition"]["segments"][0]["char_count"] == 6
    assert captured[1]["translation"]["issues"] == []
    assert "private phrase" not in json.dumps(captured)


@pytest.mark.parametrize("asr_failure", [False, True])
def test_empty_or_failed_asr_cannot_reuse_previous_translation_diagnostics(live_tool, asr_failure):
    engine = SimpleNamespace(recognizer=SimpleNamespace(last_diagnostics={"suspect": True}),
                             translator=SimpleNamespace(last_diagnostics={"calls": 2, "issues": ["previous"]}))
    captured = []

    def process(_session, _engine, _segment):
        engine.recognizer.last_diagnostics = {"segments": [], "suspect": None}
        if asr_failure:
            raise RuntimeError("fixture ASR failure")

    segment = SimpleNamespace(channel="mic", segment_id="empty-or-failed")
    if asr_failure:
        with pytest.raises(RuntimeError, match="fixture ASR failure"):
            live_tool.observe_segment_diagnostics(process, None, engine, segment, captured)
    else:
        assert live_tool.observe_segment_diagnostics(process, None, engine, segment, captured) is None
    assert captured == [{"channel": "mic", "segment_id": "empty-or-failed",
                         "recognition": {"segments": [], "suspect": None}, "translation": {}}]


def test_early_stop_records_no_stale_recognition_or_translation(live_tool):
    engine = SimpleNamespace(recognizer=SimpleNamespace(last_diagnostics={"suspect": True}),
                             translator=SimpleNamespace(last_diagnostics={"calls": 2, "history_used": 2}))
    captured = []
    segment = SimpleNamespace(channel="system", segment_id="stopped-before-ASR")
    live_tool.observe_segment_diagnostics(lambda *_: None, None, engine, segment, captured)
    assert captured == [{"channel": "system", "segment_id": "stopped-before-ASR",
                         "recognition": {}, "translation": {}}]


def test_shared_engines_do_not_transfer_diagnostics_between_channels(live_tool):
    engine = SimpleNamespace(recognizer=SimpleNamespace(), translator=SimpleNamespace())
    captured = []

    def process(_session, _engine, segment):
        engine.recognizer.last_diagnostics = {"segments": [], "suspect": segment.channel == "mic"}
        if segment.channel == "system":
            engine.translator.last_diagnostics = {"calls": 1, "history_available": 2, "history_used": 2}

    for channel in ("system", "mic", "system"):
        segment = SimpleNamespace(channel=channel, segment_id=f"{channel}-{len(captured)}")
        live_tool.observe_segment_diagnostics(process, None, engine, segment, captured)
    assert captured[0]["translation"] == {"calls": 1, "history_available": 2, "history_used": 2}
    assert captured[1]["recognition"]["suspect"] is True and captured[1]["translation"] == {}
    assert captured[2]["recognition"]["suspect"] is False
    assert captured[2]["translation"] == captured[0]["translation"]


def test_known_prompt_matches_only_verified_mic_pairs_without_saving_any_text(live_tool):
    metrics = live_tool.EventMetrics(expected_phrases=["你好，今天测试字幕延迟。"])
    original = event("original", text="你好 今天测试字幕延迟！", end_reason="vad_silence", speech_detector="silero")
    metrics.received(original)
    metrics.applied(original, ui_text_matches=True, paired_data_matches=True)
    report = metrics.snapshot()
    assert not report["expected_prompt_check"]["all_matched_in_paired_segments"]
    translated = event("translation", text="known translation")
    metrics.received(translated)
    metrics.applied(translated, ui_text_matches=True, paired_data_matches=True)
    report = metrics.snapshot()
    assert report["expected_prompt_check"]["all_matched_in_paired_segments"]
    assert report["expected_prompt_check"]["matched_indices"] == [0]
    assert report["updates"][0]["end_reason"] == "vad_silence"
    assert report["updates"][0]["speech_detector"] == "silero"
    serialized = json.dumps(report, ensure_ascii=False)
    assert "你好" not in serialized and "known translation" not in serialized


def test_unrelated_speech_cannot_pass_expected_prompt_validation(live_tool, qt_app, tmp_path):
    metrics = live_tool.EventMetrics(expected_phrases=["请检查字幕更新速度"])
    for kind in ("original", "translation"):
        item = event(kind, text="完全无关的音频内容")
        metrics.received(item)
        metrics.applied(item, ui_text_matches=True, paired_data_matches=True)
    window = FakeWindow()
    args = SimpleNamespace(report=tmp_path / "mic.json", channel="mic", duration=20, play_sample=None,
                           only_fixture=False, model_root=tmp_path / "models", output_device=None)
    session = live_tool.LiveSession(SimpleNamespace(), window, SimpleNamespace(), [], metrics, args)
    session.completed = True
    report = session.report()
    assert report["all_enabled_channels_paired"]
    assert not report["expected_prompt_check"]["all_matched_in_paired_segments"]
    assert report["ok"] is False
    assert "完全无关的音频内容" not in json.dumps(report, ensure_ascii=False)
    session.monitor.stop()
    window.deleteLater()
    qt_app.processEvents()


def test_explicit_silence_expectation_rejects_noise_generated_text(live_tool, qt_app, tmp_path):
    window = FakeWindow()
    args = SimpleNamespace(report=tmp_path / "silent.json", channel="mic", duration=20, play_sample=None,
                           only_fixture=False, model_root=tmp_path / "models", output_device=None,
                           expect_silence=["mic"])
    metrics = live_tool.EventMetrics()
    session = live_tool.LiveSession(SimpleNamespace(), window, SimpleNamespace(), [], metrics, args)
    try:
        session.completed = True
        assert session.report()["ok"]
        for kind in ("original", "translation"):
            item = event(kind)
            metrics.received(item)
            metrics.applied(item, ui_text_matches=True, paired_data_matches=True)
        report = session.report()
        assert report["all_enabled_channels_paired"]
        assert not report["ok"] and not report["silence_expectation"]["passed"]
        assert report["silence_expectation"]["unexpected_text_updates"] == 2
    finally:
        session.monitor.stop()
        window.deleteLater()
        qt_app.processEvents()


def test_default_clock_and_speech_start_latency_use_one_high_resolution_domain(live_tool, monkeypatch):
    now = [100.0]
    monkeypatch.setattr(live_tool.time, "perf_counter", lambda: now[0])
    monkeypatch.setattr(live_tool.time, "get_clock_info", lambda name: SimpleNamespace(
        implementation="QueryPerformanceCounter()", resolution=1e-7) if name == "perf_counter" else pytest.fail(name))
    metrics = live_tool.EventMetrics()
    item = event("original", speech_start_at=94.0, speech_end_at=99.0, emitted_at=100.0)
    now[0] = 100.0002
    metrics.received(item)
    now[0] = 100.0007
    metrics.applied(item, ui_text_matches=True, paired_data_matches=True)
    report = metrics.snapshot()
    row = report["updates"][0]
    assert row["callback_ms"] == pytest.approx(.2)
    assert row["callback_to_apply_ms"] == pytest.approx(.5)
    assert row["speech_start_to_apply_ms"] == pytest.approx(6000.7)
    assert report["performance_ms"]["original"]["speech_start_to_apply_ms"]["max"] == pytest.approx(6000.7)
    assert report["clock"] == {"function": "time.perf_counter", "implementation": "QueryPerformanceCounter()",
                               "resolution_seconds": 1e-7, "query_performance_counter": True}


def test_recoverable_error_details_are_redacted_and_meter_is_not_called_raw_rms(live_tool):
    metrics = live_tool.EventMetrics()
    error = event("error")
    error.message = "本地翻译失败：private fixture phrase -9999"
    error.status = "recoverable"
    metrics.received(error)
    level = event("level")
    level.level = .6
    metrics.received(level)
    report = metrics.snapshot()
    assert "private fixture phrase" not in json.dumps(report)
    assert report["errors"][0]["category"] == "translation"
    assert report["errors"][0]["numeric_codes"] == ["-9999"]
    assert report["level_meter"]["mic"]["meter_peak"] == .6
    assert "rms_peak" not in json.dumps(report)


def test_visible_label_paint_is_observed_after_apply_without_pixel_capture(live_tool, qt_app):
    metrics = live_tool.EventMetrics()
    item = event("translation")
    metrics.received(item)
    sequence = metrics.applied(item, ui_text_matches=True, paired_data_matches=True)
    label = QLabel(item.text)
    probe = live_tool.PaintProxy(metrics, label)
    label.show()
    probe.watch(label, sequence)
    for _ in range(8):
        qt_app.processEvents()
    report = metrics.snapshot()
    assert report["updates"][0]["paint_proxy_ms"] is not None
    assert report["updates"][0]["paint_proxy_ms"] >= report["updates"][0]["ui_applied_ms"]
    label.close()
    label.deleteLater()
    qt_app.processEvents()


@pytest.fixture
def official_fixture(tmp_path, monkeypatch):
    root = tmp_path / "models"
    folder = root / "self_test"
    folder.mkdir(parents=True)
    path = folder / "zh.wav"
    with wave.open(str(path), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(16000)
        stream.writeframes(struct.pack("<4h", 0, 8192, -8192, 0))
    monkeypatch.setattr(verification, "OFFICIAL_SAMPLES", {"zh": hashlib.sha256(path.read_bytes()).hexdigest()})
    return root


class FakeOutput:
    def __init__(self, *, fail_stop=False):
        self.writes = []
        self.closed = False
        self.stopped = False
        self.fail_stop = fail_stop

    def write(self, data):
        self.writes.append(data)

    def stop_stream(self):
        self.stopped = True
        if self.fail_stop:
            raise RuntimeError("fixture stop error")

    def close(self):
        self.closed = True


class FakeAudio:
    def __init__(self, *, fail_stop=False):
        self.stream = FakeOutput(fail_stop=fail_stop)
        self.terminated = False
        self.open_options = None

    def get_default_output_device_info(self):
        return {"index": 3, "defaultSampleRate": 44100, "maxOutputChannels": 2}

    def get_device_info_by_index(self, index):
        return {"index": index, "defaultSampleRate": 48000, "maxOutputChannels": 2}

    def open(self, **options):
        self.open_options = options
        return self.stream

    def terminate(self):
        self.terminated = True


@pytest.mark.parametrize("device,rate", [(None, 44100), (8, 48000)])
def test_sample_uses_real_output_api_shape_resampling_and_channel_duplication(live_tool, official_fixture, device, rate):
    audio = FakeAudio()
    module = SimpleNamespace(PyAudio=lambda: audio, paInt16=8)
    resampled = []

    def resample(samples, input_rate, output_rate):
        resampled.append((input_rate, output_rate))
        return samples

    result = live_tool.play_official_sample(official_fixture, "zh", device, Event(), audio_module=module, resampler=resample)
    assert result["played"] and result["sample_rate"] == rate
    assert resampled == [(16000, rate)]
    assert audio.open_options["output"] is True
    assert "input" not in audio.open_options
    assert audio.open_options["output_device_index"] == (3 if device is None else device)
    samples = struct.unpack("<8h", b"".join(audio.stream.writes))
    assert samples[::2] == samples[1::2]
    assert audio.stream.closed and audio.stream.stopped and audio.terminated


def test_unknown_sample_never_creates_an_audio_object(live_tool, official_fixture):
    (official_fixture / "self_test" / "zh.wav").write_bytes(b"unapproved fixture")
    module = SimpleNamespace(PyAudio=lambda: pytest.fail("must not open audio"), paInt16=8)
    with pytest.raises(ValueError, match="SHA-256"):
        live_tool.play_official_sample(official_fixture, "zh", None, Event(), audio_module=module)


def test_output_termination_runs_even_if_stop_stream_fails(live_tool, official_fixture):
    audio = FakeAudio(fail_stop=True)
    module = SimpleNamespace(PyAudio=lambda: audio, paInt16=8)
    with pytest.raises(RuntimeError, match="fixture stop"):
        live_tool.play_official_sample(official_fixture, "zh", None, Event(), audio_module=module,
                                      resampler=lambda samples, source, target: samples)
    assert audio.stream.closed and audio.terminated


class FakeWindow(QObject):
    closing = Signal()

    def __init__(self):
        super().__init__()
        self.states = []
        self.close_count = 0

    def set_session_state(self, state, message=""):
        self.states.append((state, message))

    def close(self):
        self.close_count += 1


def test_ready_marks_running_and_capture_duration_excludes_model_startup(live_tool, qt_app, tmp_path, monkeypatch):
    now = [100.0]
    monkeypatch.setattr(live_tool.time, "perf_counter", lambda: now[0])
    window = FakeWindow()
    calls = []
    controller = SimpleNamespace(state="loading", stop_session=lambda: calls.append("stop"), shutdown=lambda: calls.append("shutdown"))
    runtime = SimpleNamespace(is_busy=True, is_running=True)
    args = SimpleNamespace(report=tmp_path / "mic.json", channel="mic", duration=20, play_sample=None,
                           only_fixture=False, model_root=tmp_path / "models", output_device=None)
    session = live_tool.LiveSession(SimpleNamespace(quit=lambda: calls.append("quit")), window, controller,
                                    [runtime], live_tool.EventMetrics(clock=lambda: now[0]), args)
    now[0] = 129
    session.poll()
    assert calls == [] and not session.ready_file.exists()
    controller.state = "running"
    session.poll()
    ready = json.loads(session.ready_file.read_text(encoding="utf-8"))
    assert ready["state"] == "running" and ready["perf_counter"] == 129
    assert set(ready) == {"state", "ready_at", "perf_counter"}
    assert "麦克风" in window.states[-1][1]
    now[0] = 148
    session.poll()
    assert calls == []
    now[0] = 149
    session.poll()
    assert calls == ["stop"]
    assert session.running_at == 129
    session.monitor.stop()
    window.deleteLater()
    qt_app.processEvents()


@pytest.mark.parametrize("duration", [19, 61])
def test_duration_is_bounded_without_starting_any_backend(live_tool, tmp_path, duration):
    with pytest.raises(SystemExit):
        live_tool.parse_args(["--model-root", str(tmp_path / "models"), "--channel", "mic",
                              "--duration", str(duration), "--report", str(tmp_path / "report.json")])


def test_manual_window_waits_without_capture_or_idle_timeout_then_bounds_model_startup(live_tool, qt_app, tmp_path, monkeypatch):
    now, calls = [100.0], []
    monkeypatch.setattr(live_tool.time, "perf_counter", lambda: now[0])
    window = FakeWindow()
    window.settings = object()
    controller = SimpleNamespace(state="stopped", start_session=lambda _settings: calls.append("start"),
                                 stop_session=lambda: calls.append("stop"))
    args = SimpleNamespace(report=tmp_path / "manual.json", channel="mic", duration=60, play_sample=None,
                           only_fixture=False, model_root=tmp_path / "models", output_device=None, manual_start=True)
    session = live_tool.LiveSession(SimpleNamespace(), window, controller, [], live_tool.EventMetrics(), args)
    try:
        session.begin()
        now[0] = 300.0
        session.poll()
        assert not calls and not session.ready_file.exists()
        assert session.timeout is None and session.running_at is None
        controller.state = "loading"
        session.poll()
        assert session.loading_at == 300
        now[0] = 329.0
        session.poll()
        assert not calls
        now[0] = 330.0
        session.poll()
        assert calls == ["stop"] and session.timeout == "startup_30_seconds"
    finally:
        session.monitor.stop()
        window.deleteLater()
        qt_app.processEvents()


def test_acoustic_microphone_sample_and_fixture_privacy_are_explicit(live_tool, tmp_path):
    base = ["--model-root", str(tmp_path / "models"), "--channel", "mic", "--report", str(tmp_path / "report.json")]
    args = live_tool.parse_args(base + ["--play-sample", "zh"])
    assert args.channel == "mic" and not args.only_fixture
    with pytest.raises(SystemExit):
        live_tool.parse_args(base + ["--only-fixture"])


@pytest.mark.parametrize("mode", ["raw", "gtcrn", "gtcrn_preserve", "mdx", "mdx_gtcrn"])
def test_live_cli_accepts_audio_processing_mode_and_explicit_model_paths_without_capture(live_tool, tmp_path, mode):
    base = ["--model-root", str(tmp_path / "models"), "--channel", "system", "--report", str(tmp_path / "live.json")]
    args = live_tool.parse_args(base + ["--system-audio-processing", mode, "--mdx-model-file", "D:/mdx.onnx",
                                       "--gtcrn-model-file", "D:/gtcrn.onnx"])
    assert args.system_audio_processing == mode
    assert args.mdx_model_file == "D:/mdx.onnx" and args.gtcrn_model_file == "D:/gtcrn.onnx"
    assert live_tool.parse_args(base).system_audio_processing == "raw"
    with pytest.raises(SystemExit):
        live_tool.parse_args(base + ["--system-audio-processing", "GTCRN"])
    assert not (tmp_path / "live.json").exists()


def test_live_report_records_runtime_profile_after_manual_start_settings_change(live_tool, qt_app, tmp_path):
    from fanyiji_next.models import AppSettings
    window = FakeWindow()
    args = SimpleNamespace(report=tmp_path / "manual.json", channel="system", duration=20, play_sample=None,
                           only_fixture=False, model_root=tmp_path / "models", output_device=None,
                           system_audio_processing="raw")
    selected = AppSettings(system_audio_processing="mdx_gtcrn", recognition_backend="whisper", translation_backend="hymt")
    runtime = SimpleNamespace(_audit_settings=selected)
    session = live_tool.LiveSession(SimpleNamespace(), window, SimpleNamespace(), [runtime], live_tool.EventMetrics(), args)
    try:
        report = session.report()
        assert report["selected_backends"]["system_audio_processing"] == "mdx_gtcrn"
        assert report["selected_backends"]["recognition_backend"] == "whisper"
        assert report["selected_backends_source"] == "runtime_start"
        session.runtimes.clear()
        report = session.report()
        assert report["selected_backends"]["system_audio_processing"] == "raw"
        assert report["selected_backends_source"] == "arguments"
    finally:
        session.monitor.stop()
        window.deleteLater()
        qt_app.processEvents()


class FakePipe:
    def __init__(self, content=b""):
        self.data = bytearray()
        self.closed = False
        self.content = io.BytesIO(content)

    def write(self, data):
        self.data.extend(data)

    def flush(self):
        pass

    def read(self, count):
        return self.content.read(count)

    def close(self):
        self.closed = True


class FakeProcess:
    def __init__(self, payload=None, *, returncode=0, running=False, timeouts=0,
                 stdout_prefix=b"", stderr=b""):
        self.pid = 4321
        self.returncode = None if running else returncode
        self.exit_code = returncode
        self.timeouts = timeouts
        self.output = stdout_prefix + (json.dumps(payload).encode() + b"\n" if payload is not None else b"")
        self.errors = stderr
        self.stdin, self.stdout, self.stderr = FakePipe(), FakePipe(self.output), FakePipe(stderr)
        self.terminated = self.killed = False
        self.wait_count = 0

    def poll(self):
        return self.returncode

    def wait(self, *, timeout):
        self.wait_count += 1
        assert timeout == 2
        if self.timeouts:
            self.timeouts -= 1
            raise subprocess.TimeoutExpired("fake-output-worker", timeout)
        self.returncode = self.exit_code
        return self.returncode

    def terminate(self):
        self.terminated = True

    def kill(self):
        self.killed = True

class CancelOnWait:
    """Request cancellation after process launch without sleeps or real subprocesses."""
    def __init__(self):
        self.cancelled = False

    def is_set(self):
        return self.cancelled

    def wait(self, timeout):
        assert timeout > 0
        self.cancelled = True
        return True


def playback_payload(**changes):
    return {"protocol": "fanyiji-output-v1", "ok": True, "played": True, "cancelled": False,
            "device_index": 8, "sample_rate": 48000, "channels": 2,
            "sample_sha256": "fixed-hash", "started_at": 100.0, "finished_at": 107.2, **changes}


@pytest.mark.parametrize("device,platform", [(None, "linux"), (8, "win32")])
def test_playback_parent_only_launches_hidden_worker_and_preserves_device_and_timing(
        live_tool, tmp_path, monkeypatch, device, platform):
    process = FakeProcess(playback_payload(private_text="must not enter report"),
                          stdout_prefix=b"discarded library output\n")
    calls = []
    monkeypatch.setattr(live_tool.sys, "platform", platform)
    monkeypatch.setattr(live_tool.subprocess, "CREATE_NO_WINDOW", 0x08000000, raising=False)
    monkeypatch.setattr(live_tool, "play_official_sample", lambda *a, **kw: pytest.fail("parent opened audio"))

    def popen(command, **options):
        calls.append((command, options))
        return process

    result = live_tool.play_official_sample_process(tmp_path, "ja", device, Event(), popen_factory=popen)
    command, options = calls[0]
    assert command[:3] == [live_tool.sys.executable, str(Path(live_tool.__file__).resolve()), "--_playback-worker"]
    assert command[3:7] == ["--model-root", str(tmp_path), "--play-sample", "ja"]
    assert command[7:] == ([] if device is None else ["--output-device", "8"])
    assert options["stdin"] == options["stdout"] == options["stderr"] == subprocess.PIPE
    assert options.get("creationflags", 0) == (0x08000000 if platform == "win32" else 0)
    assert "shell" not in options
    assert result["isolated_output_process"] and result["worker_pid"] == 4321
    assert result["started_at"] == 100.0 and result["finished_at"] == 107.2
    assert result["sample_rate"] == 48000 and result["sample_sha256"] == "fixed-hash"
    assert "must not enter report" not in json.dumps(result) and "discarded library output" not in json.dumps(result)
    assert process.stdin.closed and process.stdout.closed and process.stderr.closed


def test_already_cancelled_output_does_not_launch_a_worker(live_tool, tmp_path):
    stop = Event()
    stop.set()
    result = live_tool.play_official_sample_process(tmp_path, "zh", None, stop,
                    popen_factory=lambda *a, **kw: pytest.fail("cancelled playback launched"))
    assert result["cancelled"] and result["played"] is False


def test_cancel_output_sends_stop_then_reaps_and_closes_pipes(live_tool, tmp_path):
    process = FakeProcess(playback_payload(cancelled=True), running=True)
    result = live_tool.play_official_sample_process(tmp_path, "zh", 8, CancelOnWait(),
                                                   popen_factory=lambda *a, **kw: process)
    assert result["cancelled"] and process.stdin.data == b"stop\n"
    assert process.returncode == 0 and process.wait_count == 1
    assert not process.terminated and not process.killed
    assert process.stdin.closed and process.stdout.closed and process.stderr.closed


def test_unresponsive_worker_is_terminated_killed_and_reaped(live_tool, tmp_path):
    stage = {"protocol": "fanyiji-output-v1", "kind": "stage", "stage": "pyaudio_init_start", "at": 100.0}
    process = FakeProcess(returncode=-9, running=True, timeouts=2, stdout_prefix=json.dumps(stage).encode() + b"\n")
    result = live_tool.play_official_sample_process(tmp_path, "zh", None, CancelOnWait(),
                                                   popen_factory=lambda *a, **kw: process)
    assert process.terminated and process.killed and process.wait_count == 3
    assert process.returncode == -9 and result["worker_exit_code"] == -9
    assert result["forced_cleanup"] and result["error_type"] == "PlaybackCleanupTimeout"
    assert result["played"] is None and result["cancelled"]
    assert result["worker_stages"] == [{"stage": "pyaudio_init_start", "at": 100.0}]
    assert process.stdin.closed and process.stdout.closed and process.stderr.closed


@pytest.mark.parametrize("payload,exit_code", [(None, -1073741819),
    ({"protocol": "fanyiji-output-v1", "ok": False, "error_type": "OSError"}, 2), (None, 0)])
def test_worker_failure_is_explicit_and_native_output_is_redacted(live_tool, tmp_path, payload, exit_code):
    process = FakeProcess(payload, returncode=exit_code, stdout_prefix=b"private native stdout\n",
                          stderr=b"private native stderr")
    result = live_tool.play_official_sample_process(tmp_path, "zh", None, Event(),
                                                   popen_factory=lambda *a, **kw: process)
    assert result["played"] is False and result["worker_exit_code"] == exit_code
    assert result["error_type"] == ("OSError" if payload else "PlaybackProcessError")
    assert result["stderr_bytes"] == len(b"private native stderr")
    assert "private native" not in json.dumps(result)
    assert process.stdin.closed and process.stdout.closed and process.stderr.closed


@pytest.mark.parametrize("control_input", [b"", b"stop\n"])
def test_worker_cancels_for_parent_exit_or_stop_byte(live_tool, control_input):
    stop = Event()
    live_tool._watch_parent_input(io.BytesIO(control_input), stop)
    assert stop.is_set()


def test_hidden_worker_dispatch_does_not_enter_capture_main(live_tool, monkeypatch):
    calls = []
    monkeypatch.setattr(live_tool, "_playback_worker_main", lambda argv: calls.append(argv) or 7)
    monkeypatch.setattr(live_tool, "parse_args", lambda argv: pytest.fail("capture main entered"))
    assert live_tool.main(["--_playback-worker", "--play-sample", "ja"]) == 7
    assert calls == [["--play-sample", "ja"]]


def test_requested_playback_failure_cannot_report_successful_live_check(live_tool, qt_app, tmp_path):
    window = FakeWindow()
    args = SimpleNamespace(report=tmp_path / "system.json", channel="mic", duration=20, play_sample="zh",
                           only_fixture=False, model_root=tmp_path / "models", output_device=None)
    metrics = live_tool.EventMetrics()
    for kind in ("original", "translation"):
        item = event(kind)
        metrics.received(item)
        metrics.applied(item, ui_text_matches=True, paired_data_matches=True)
    session = live_tool.LiveSession(SimpleNamespace(), window, SimpleNamespace(), [], metrics, args)
    session.completed = True
    session.playback.update(played=False, error_type="PlaybackProcessError")
    report = session.report()
    assert report["all_enabled_channels_paired"] and not report["ok"]
    session.monitor.stop()
    window.deleteLater()
    qt_app.processEvents()


def test_worker_imports_native_dependencies_before_starting_stdin_reader(live_tool, monkeypatch, capsys, tmp_path):
    operations = []
    audio_module, resampler = object(), object()

    def load(stage):
        operations.append("imports")
        stage("numpy_import_start")
        stage("numpy_import_done")
        return audio_module, resampler

    class Listener:
        def __init__(self, *, target, args, name, daemon):
            assert operations == ["imports"]
            assert args[0] == 23 and args[1].is_set() is False
            assert target is live_tool._watch_parent_input
            operations.append("reader_constructed")

        def start(self):
            operations.append("reader_started")

    def play(root, language, device, stop, **options):
        assert operations == ["imports", "reader_constructed", "reader_started"]
        assert options["audio_module"] is audio_module and options["resampler"] is resampler
        options["stage"]("first_write_done")
        return {"played": True, "cancelled": False}

    monkeypatch.setattr(live_tool, "_load_playback_dependencies", load)
    monkeypatch.setattr(live_tool, "Thread", Listener)
    monkeypatch.setattr(live_tool.sys, "stdin", SimpleNamespace(fileno=lambda: 23))
    monkeypatch.setattr(live_tool, "play_official_sample", play)
    assert live_tool._playback_worker_main(["--model-root", str(tmp_path), "--play-sample", "zh"]) == 0
    rows = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert [row["stage"] for row in rows[:-1]] == ["worker_enter", "numpy_import_start", "numpy_import_done",
                                                                "control_reader_started", "first_write_done"]
    assert rows[-1]["reported_worker_pid"] == os.getpid()
    assert rows[-1]["reported_parent_pid"] == os.getppid()
    assert all(row["protocol"] == "fanyiji-output-v1" for row in rows)


@pytest.mark.parametrize("data,reason", [(b"", "parent_eof"), (b"s", "stop_byte")])
def test_worker_control_os_read_avoids_buffered_stdin_and_reports_reason(live_tool, monkeypatch, data, reason):
    stop, observed = Event(), []
    reads = []
    monkeypatch.setattr(live_tool.os, "read", lambda fd, count: reads.append((fd, count)) or data)
    live_tool._watch_parent_input(23, stop, lambda kind, error: observed.append((kind, error)))
    assert stop.is_set() and reads == [(23, 1)] and observed == [(reason, None)]


def test_worker_control_read_error_is_distinct_from_eof_without_saving_message(live_tool, monkeypatch):
    stop, observed = Event(), []

    def fail(fd, count):
        raise OSError("private local detail")

    monkeypatch.setattr(live_tool.os, "read", fail)
    live_tool._watch_parent_input(23, stop, lambda kind, error: observed.append((kind, error)))
    assert stop.is_set() and observed == [("input_error", "OSError")]
    assert "private" not in json.dumps(observed)


def test_parent_diagnostics_keep_only_fixed_stages_control_reasons_and_numeric_timestamps(live_tool, tmp_path, monkeypatch):
    rows = [
        {"kind": "stage", "stage": "numpy_import_start", "at": 100.1, "text": "private stage detail"},
        {"kind": "stage", "stage": "private unknown stage", "at": 100.2},
        {"kind": "stage", "stage": "numpy_import_done", "at": "private time"},
        {"kind": "control", "reason": "stop_byte", "at": 119.5, "error_type": "private error"},
    ]
    prefix = b"".join(json.dumps({"protocol": "fanyiji-output-v1", **row}).encode() + b"\n" for row in rows)
    process = FakeProcess(playback_payload(cancelled=True, reported_worker_pid=44, reported_parent_pid=33),
                          running=True, stdout_prefix=prefix)
    monkeypatch.setattr(live_tool.time, "perf_counter", lambda: 120.0)
    result = live_tool.play_official_sample_process(tmp_path, "zh", None, CancelOnWait(),
                                                   popen_factory=lambda *a, **kw: process)
    assert result["worker_stages"] == [{"stage": "numpy_import_start", "at": 100.1}]
    assert result["worker_control_reason"] == "stop_byte" and result["worker_control_at"] == 119.5
    assert result["reported_worker_pid"] == 44 and result["reported_parent_pid"] == 33
    assert result["parent_launch_at"] == result["parent_spawned_at"] == result["parent_cancel_observed_at"] == 120.0
    assert result["parent_collect_started_at"] == result["parent_collected_at"] == 120.0
    assert "private" not in json.dumps(result)


def test_both_output_pipes_are_drained_while_control_stdin_remains_open(live_tool, tmp_path):
    output_read, error_read = Event(), Event()
    process = FakeProcess(playback_payload(), stdout_prefix=b"discarded native output" * 700 + b"\n",
                          stderr=b"discarded native error" * 700)

    class ObservedPipe(FakePipe):
        def __init__(self, content, observed):
            super().__init__(content)
            self.observed = observed

        def read(self, count):
            self.observed.set()
            return super().read(count)

    process.stdout = ObservedPipe(process.output, output_read)
    process.stderr = ObservedPipe(process.errors, error_read)

    def poll():
        assert output_read.wait(1) and error_read.wait(1)
        assert not process.stdin.closed
        return process.returncode

    process.poll = poll
    result = live_tool.play_official_sample_process(tmp_path, "zh", None, Event(),
                                                   popen_factory=lambda *a, **kw: process)
    assert result["played"] is True and process.stdin.closed
    assert "discarded" not in json.dumps(result)
