"""Native-device-free contracts for the processed system capture worker."""

from dataclasses import replace
import itertools
import queue
import threading
import time
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from fanyiji_next import audio_processing, audio_runtime
from fanyiji_next.audio_processing import ProcessedAudio
from fanyiji_next.audio_runtime import VoiceRuntime, _CaptureSource, _Segment, _Session
from fanyiji_next.models import AppSettings
from fanyiji_next.speech_gate import SpeechSlice


_IDS = itertools.count(5200)


def wait_for(predicate, timeout=1.5):
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() >= deadline:
            pytest.fail("fake processing worker did not reach the expected boundary")
        threading.Event().wait(.002)


class FakeProcessor:
    def __init__(self, trace, rate):
        self.trace, self.rate = trace, rate
        self.inputs = []
        self.output_samples = 0
        self.on_process = None
        self.on_finish = None
        self.finish_calls = self.reset_calls = self.close_calls = 0
        self.entered = threading.Event()

    def output(self, native):
        # Selecting the left channel deliberately exposes premature downmix:
        # antiphase stereo must reach this fake separator without cancellation.
        count = round(len(native) * 16000 / self.rate)
        values = np.full(count, native[0, 0], dtype=np.float32)
        first = self.output_samples
        self.output_samples += count
        return [ProcessedAudio(values, first, self.output_samples, len(native))]

    def process_block(self, samples):
        self.inputs.append(samples.copy())
        self.trace.append("processor_block")
        self.entered.set()
        return self.on_process(samples) if self.on_process else self.output(samples)

    def finish(self):
        self.trace.append("processor_finish")
        self.finish_calls += 1
        return self.on_finish() if self.on_finish else []

    def reset(self):
        self.trace.append("processor_reset")
        self.reset_calls += 1
        self.output_samples = 0

    def close(self):
        self.trace.append("processor_close")
        self.close_calls += 1


class FakeGate:
    def __init__(self, trace):
        self.trace = trace
        self.inputs = []
        self.cursor = 0
        self.finish_calls = 0

    def feed(self, samples):
        self.trace.append("gate_feed")
        self.inputs.append(samples[:])
        first = self.cursor
        self.cursor += len(samples)
        return [SpeechSlice(samples[:], first, self.cursor, "vad_silence")]

    def finish(self):
        self.trace.append("gate_finish")
        self.finish_calls += 1
        self.cursor = 0  # Real SileroSpeechGate.finish() resets its sample axis.
        return []


class Harness:
    def __init__(self, monkeypatch, *, rate=16000, channels=2):
        self.trace, self.events, self.thread_errors = [], [], []
        self.settings = replace(AppSettings(), system_audio_processing="mdx")
        self.session = _Session(self.settings, next(_IDS))
        self.runtime = VoiceRuntime(self.events.append)
        self.runtime._session = self.session
        self.source = _CaptureSource(rate, channels, chunks=queue.Queue())
        self.ready = threading.Event()
        self.release = threading.Event()
        self.processor = FakeProcessor(self.trace, rate)
        self.gate = FakeGate(self.trace)
        self.factory = Mock(return_value=self.processor)
        monkeypatch.setattr(audio_processing, "create_system_audio_processor", self.factory)

        def run():
            try:
                self.runtime._capture_processed(self.session, self.settings.system, self.ready,
                                                self.source, self.gate, np)
            except BaseException as exc:
                self.thread_errors.append(exc)

        self.thread = threading.Thread(target=run, name=f"test-capture-{self.session.id}", daemon=True)

    def start(self):
        self.thread.start()
        assert self.ready.wait(1)
        return self

    def offer(self, values, captured_at=None):
        self.source.chunks.put((np.asarray(values, dtype="<i2").tobytes(),
                                time.perf_counter() if captured_at is None else captured_at))

    def stop(self):
        self.session.stop.set()
        self.release.set()
        self.thread.join(2)
        assert not self.thread.is_alive(), "capture worker leaked"
        assert not self.thread_errors
        assert not any(t.name == f"fanyiji-system-input-{self.session.id}" and t.is_alive()
                       for t in threading.enumerate()), "native-input drainer leaked"


@pytest.fixture
def harness(monkeypatch):
    owned = []

    def make(**kwargs):
        item = Harness(monkeypatch, **kwargs)
        owned.append(item)
        return item

    yield make
    for item in owned:
        if item.thread.ident is not None:
            item.stop()


def test_native_antiphase_stereo_reaches_processor_before_mono_gate(harness):
    h = harness(rate=48000).start()
    source = np.tile(np.array([8192, -8192], dtype=np.int16), (480, 1))
    h.offer(source)
    wait_for(lambda: h.session.segments.qsize() == 1)
    h.stop()
    h.factory.assert_called_once_with(h.settings, sample_rate=48000, channels=2,
                                     stop_event=h.session.stop)
    assert h.processor.inputs[0].shape == (480, 2)
    np.testing.assert_array_equal(h.processor.inputs[0], source.astype(np.float32) / 32768)
    assert len(h.gate.inputs[0]) == 160
    assert all(value == .25 for value in h.gate.inputs[0])
    assert h.trace.index("processor_block") < h.trace.index("gate_feed")
    assert h.processor.close_calls == 1


def test_delayed_processing_keeps_original_source_clock(harness, monkeypatch):
    # A deliberately old native timestamp and a later processing clock prove
    # that metadata uses source sample positions, not completion wall time.
    monkeypatch.setattr(audio_runtime, "time", SimpleNamespace(perf_counter=lambda: 5000.0))
    monkeypatch.setattr(audio_runtime, "SYSTEM_SILENCE_SECONDS", 10000)
    h = harness(rate=48000).start()
    h.offer(np.full((480, 2), 4000, dtype=np.int16), captured_at=100.01)
    wait_for(lambda: h.session.segments.qsize() == 1)
    h.stop()
    segment = h.session.segments.get_nowait()
    assert segment.speech_start_at == pytest.approx(100.0)
    assert segment.speech_end_at == pytest.approx(100.01)
    assert segment.captured_at == 100.01
    assert segment.segment_submitted_at == 5000.0
    assert segment.speech_detector == "silero"


def test_inference_pause_does_not_finish_a_source_that_keeps_producing(harness, monkeypatch):
    clock = SimpleNamespace(now=100.04)
    monkeypatch.setattr(audio_runtime, "time", SimpleNamespace(perf_counter=lambda: clock.now))
    h = harness()
    # This test isolates preprocessing from ASR backpressure. Queue behaviour
    # at the real six-segment limit is exercised separately below.
    h.session.segments = queue.Queue(maxsize=32)

    def slow_first(samples):
        if len(h.processor.inputs) == 1:
            assert h.release.wait(1.5)
        return h.processor.output(samples)

    h.processor.on_process = slow_first
    h.start()
    block = np.full((640, 2), 4000, dtype=np.int16)
    h.offer(block, captured_at=100.04)
    assert h.processor.entered.wait(1)
    for index in range(1, 20):
        clock.now = 100 + (index + 1) * .04
        h.offer(block, captured_at=clock.now)
        wait_for(h.source.chunks.empty)
    clock.now += .001
    h.release.set()
    wait_for(lambda: len(h.processor.inputs) == 20)
    h.stop()
    assert h.processor.finish_calls == 0
    assert h.gate.finish_calls == 0
    assert not any(event.kind == "error" and "缓冲" in event.message for event in h.events)


def test_real_timestamp_gap_finishes_tail_then_resets_both_timelines(harness):
    h = harness(rate=48000)
    buffered = []

    def process(samples):
        if len(h.processor.inputs) == 1:
            buffered.append(samples.copy())
            return []
        return h.processor.output(samples)

    h.processor.on_process = process
    h.processor.on_finish = lambda: h.processor.output(buffered.pop()) if buffered else []
    now = time.perf_counter()
    first_at, second_at = now - 1.0, now - .01
    h.offer(np.full((480, 2), 4096, dtype=np.int16), captured_at=first_at)
    h.offer(np.full((480, 2), 8192, dtype=np.int16), captured_at=second_at)
    h.start()
    wait_for(lambda: h.session.segments.qsize() == 2)
    h.stop()
    first, second = h.session.segments.get_nowait(), h.session.segments.get_nowait()
    assert first.end_reason == "input_gap"
    assert first.speech_start_at == pytest.approx(first_at - .01)
    assert first.speech_end_at == pytest.approx(first_at)
    assert second.speech_start_at == pytest.approx(second_at - .01)
    assert second.speech_end_at == pytest.approx(second_at)
    assert first.samples == [.125] * 160 and second.samples == [.25] * 160
    assert h.processor.finish_calls == h.processor.reset_calls == h.gate.finish_calls == 1
    assert h.trace.index("processor_finish") < h.trace.index("gate_finish") < h.trace.index("processor_reset")


def test_user_stop_during_native_call_discards_late_results_without_flush(harness):
    h = harness()

    def in_flight(samples):
        assert h.release.wait(1.5)
        return h.processor.output(samples)

    h.processor.on_process = in_flight
    h.start()
    h.offer(np.full((640, 2), 5000, dtype=np.int16))
    assert h.processor.entered.wait(1)
    h.stop()
    assert h.gate.inputs == []
    assert h.session.segments.empty()
    assert h.processor.finish_calls == h.gate.finish_calls == 0
    assert h.processor.close_calls == 1
    assert not any(event.kind == "error" for event in h.events)


def test_processor_failure_is_explicit_without_raw_fallback_or_history_clear(harness):
    h = harness()
    old = _Segment("system", "already-committed", [.1], "ja_to_zh", time.perf_counter())
    h.session.segments.put(old)
    h.runtime._emit(h.session, "original", channel="system", text="existing source")
    h.runtime._emit(h.session, "translation", channel="system", text="existing translation")

    def fail(samples):
        raise RuntimeError("fixture model failure")

    h.processor.on_process = fail
    h.start()
    h.offer(np.full((640, 2), 5000, dtype=np.int16))
    assert h.session.stop.wait(1)
    h.stop()
    assert h.session.segments.get_nowait() is old
    assert [e.text for e in h.events if e.kind in {"original", "translation"}] == [
        "existing source", "existing translation"]
    errors = [e for e in h.events if e.kind == "error"]
    assert len(errors) == 1 and "fixture model failure" in errors[0].message
    h.factory.assert_called_once()
    assert h.gate.inputs == []
    assert h.processor.finish_calls == 0 and h.processor.close_calls == 1


def test_processing_backlog_stops_explicitly_and_retires_drainer(harness):
    h = harness()
    h.runtime._emit(h.session, "translation", channel="system", text="already visible")

    def blocked(samples):
        assert h.session.stop.wait(1.5)
        return h.processor.output(samples)

    h.processor.on_process = blocked
    h.start()
    block = np.full((16000, 2), 5000, dtype=np.int16)
    h.offer(block)
    assert h.processor.entered.wait(1)
    for _ in range(13):
        h.offer(block)
    assert h.session.stop.wait(1)
    h.stop()
    errors = [e for e in h.events if e.kind == "error"]
    assert len(errors) == 1 and "12 秒" in errors[0].message and "停止采集" in errors[0].message
    assert [e.text for e in h.events if e.kind == "translation"] == ["already visible"]
    assert h.gate.inputs == [] and h.session.segments.empty()
    h.factory.assert_called_once()
    assert h.processor.finish_calls == 0 and h.processor.close_calls == 1


def test_native_capture_discontinuity_stops_instead_of_joining_missing_audio(harness):
    h = harness().start()
    h.source.discontinuity.set()
    h.offer(np.full((640, 2), 5000, dtype=np.int16))
    assert h.session.stop.wait(1)
    h.stop()
    assert h.processor.inputs == []
    assert h.processor.finish_calls == h.gate.finish_calls == 0
    assert any(e.kind == "error" and "缺失音频" in e.message for e in h.events)


@pytest.mark.parametrize("raw", [b"\x00", b"\x00" * (16001 * 2 * 2)],
                         ids=["unaligned-frame", "over-one-second"])
def test_invalid_or_oversized_native_block_is_rejected_before_processing(harness, raw):
    h = harness().start()
    h.source.chunks.put((raw, time.perf_counter()))
    assert h.session.stop.wait(1)
    h.stop()
    assert h.processor.inputs == [] and h.gate.inputs == []
    assert any(e.kind == "error" and "输入块格式" in e.message for e in h.events)


def fill_asr_queue(h):
    old = [_Segment("system", f"older-{index}", [.1], "ja_to_zh", time.perf_counter())
           for index in range(audio_runtime.SEGMENT_QUEUE_SIZE)]
    for item in old:
        h.session.segments.put_nowait(item)
    return old


def test_processed_batch_waits_for_asr_space_instead_of_dropping_new_speech(harness):
    h = harness()
    old = fill_asr_queue(h)
    h.start()
    h.offer(np.full((640, 2), 5000, dtype=np.int16))
    wait_for(lambda: "gate_feed" in h.trace)
    assert not h.session.stop.is_set()
    assert not any("跳过" in e.message for e in h.events)
    assert h.session.segments.get_nowait() is old[0]
    wait_for(lambda: h.session.segments.qsize() == audio_runtime.SEGMENT_QUEUE_SIZE)
    h.stop()
    retained = [h.session.segments.get_nowait() for _ in range(audio_runtime.SEGMENT_QUEUE_SIZE)]
    assert retained[:-1] == old[1:]
    assert retained[-1].segment_id == f"system-{h.session.id}-1"
    assert retained[-1].samples == [5000 / 32768] * 640
    assert not any(e.kind == "error" for e in h.events)
    assert h.processor.finish_calls == 0 and h.processor.close_calls == 1


def test_stop_cancels_processed_batch_waiting_for_full_asr_queue(harness):
    h = harness()
    old = fill_asr_queue(h)
    h.start()
    h.offer(np.full((640, 2), 5000, dtype=np.int16))
    wait_for(lambda: "gate_feed" in h.trace)
    h.stop()
    retained = [h.session.segments.get_nowait() for _ in range(audio_runtime.SEGMENT_QUEUE_SIZE)]
    assert retained == old and h.session.segments.empty()
    assert not any(e.kind == "error" for e in h.events)
    assert h.processor.finish_calls == h.gate.finish_calls == 0


@pytest.mark.parametrize("mode", ["gtcrn", "gtcrn_preserve", "mdx", "mdx_gtcrn"])
def test_disabled_system_does_not_require_unused_enhancement_files(mode):
    base = AppSettings()
    disabled = replace(base, system=replace(base.system, enabled=False))
    selected = replace(disabled, system_audio_processing=mode,
                       mdx_model_file="missing-mdx.onnx", gtcrn_model_file="missing-gtcrn.onnx")
    assert audio_runtime.validate_models(selected) == audio_runtime.validate_models(disabled)


@pytest.mark.parametrize("mode,needs_gtcrn", [("mdx", False), ("mdx_gtcrn", True)])
def test_enabled_system_requires_only_selected_enhancement_models(tmp_path, mode, needs_gtcrn):
    vad = tmp_path / "vad.onnx"
    mdx = tmp_path / "Kim_Vocal_2.onnx"
    vad.write_bytes(b"validation only; never loaded")
    mdx.write_bytes(b"validation only; never loaded")
    base = AppSettings()
    selected = replace(base, system=replace(base.system, enabled=True), system_audio_processing=mode,
                       mdx_model_file=str(mdx), gtcrn_model_file=str(tmp_path / "absent-gtcrn.onnx"),
                       vad_model=str(vad))
    issues = audio_runtime.validate_models(selected)
    assert any("GTCRN 降噪模型" in issue for issue in issues) is needs_gtcrn
    assert not any("MDX 人声模型" in issue or "人声处理模式需要" in issue for issue in issues)
    missing_mdx = replace(selected, mdx_model_file=str(tmp_path / "absent-mdx.onnx"))
    assert any("MDX 人声模型" in issue for issue in audio_runtime.validate_models(missing_mdx))
    missing_vad = replace(selected, vad_model="", asr_model="")
    assert any("人声处理模式需要" in issue for issue in audio_runtime.validate_models(missing_vad))


@pytest.mark.parametrize("mode", ["gtcrn", "gtcrn_preserve"])
def test_gtcrn_only_requires_denoiser_and_vad_but_not_mdx(tmp_path, mode):
    vad, gtcrn = tmp_path / "vad.onnx", tmp_path / "gtcrn.onnx"
    vad.write_bytes(b"validation only")
    gtcrn.write_bytes(b"validation only")
    base = AppSettings()
    selected = replace(base, system=replace(base.system, enabled=True), system_audio_processing=mode,
                       mdx_model_file="", gtcrn_model_file=str(gtcrn), vad_model=str(vad))
    assert audio_runtime.validate_models(selected) == audio_runtime.validate_models(base)
    missing = audio_runtime.validate_models(replace(selected, gtcrn_model_file=""))
    assert any("GTCRN 降噪模型" in issue for issue in missing)
    assert not any("MDX" in issue for issue in missing)
    assert any("人声处理模式需要" in issue for issue in audio_runtime.validate_models(
        replace(selected, vad_model="", asr_model="")))


def test_gtcrn_processed_worker_submits_source_aligned_audio_without_mdx_wait_message(harness):
    h = harness(rate=48000, channels=2)
    h.settings = replace(h.settings, system_audio_processing="gtcrn")
    h.session.settings = h.settings
    h.start()
    h.offer(np.full((1920, 2), 4096, dtype=np.int16))
    wait_for(lambda: h.session.segments.qsize() == 1)
    h.stop()
    segment = h.session.segments.get_nowait()
    assert segment.samples == [.125] * 640
    assert any("GTCRN 流式降噪" in event.message for event in h.events)
    assert not any("5.8" in event.message or "MDX" in event.message for event in h.events)
    h.factory.assert_called_once_with(h.settings, sample_rate=48000, channels=2,
                                     stop_event=h.session.stop)


def preserve_harness(harness, monkeypatch, *, rate=48000):
    from fanyiji_next import dual_speech_gate
    h = harness(rate=rate)
    h.settings = replace(h.settings, system_audio_processing="gtcrn_preserve")
    h.session.settings = h.settings
    h.dual_inputs = []

    class SpyDual(FakeGate):
        def __init__(self, model, *, raw_gate, **options):
            super().__init__(h.trace)
            assert raw_gate is h.gate
            assert options == dict(silence_seconds=audio_runtime.SYSTEM_SILENCE_SECONDS,
                                   leading_seconds=audio_runtime.SYSTEM_LEADING_SECONDS,
                                   trailing_seconds=audio_runtime.SYSTEM_TRAILING_SECONDS)
            self.reset_calls = 0

        def feed(self, raw, denoised, *, start_sample):
            assert start_sample == self.cursor
            h.dual_inputs.append((raw[:], denoised[:], start_sample))
            return super().feed(raw)

        def reset(self):
            self.reset_calls += 1
            self.cursor = 0

    factory = Mock(side_effect=SpyDual)
    monkeypatch.setattr(dual_speech_gate, "DualSpeechGate", factory)
    h.dual_factory = factory

    def process(samples):
        output = h.processor.output(samples)[0]
        raw = output.samples
        return [replace(output, raw_samples=raw, denoised_samples=raw * .1)]

    h.processor.on_process = process
    return h


def test_preserve_worker_routes_only_exact_raw_to_asr_with_original_clock(harness, monkeypatch):
    h = preserve_harness(harness, monkeypatch)
    now = time.perf_counter()
    h.offer(np.full((1920, 2), 4096, dtype=np.int16), captured_at=now)
    h.start()
    wait_for(lambda: h.session.segments.qsize() == 1)
    h.stop()
    segment = h.session.segments.get_nowait()
    assert segment.samples == [.125] * 640
    assert segment.speech_start_at == pytest.approx(now - .04)
    assert segment.speech_end_at == pytest.approx(now)
    raw, denoised, start = h.dual_inputs[0]
    assert raw == segment.samples and start == 0
    assert denoised == [float(np.float32(.0125))] * 640
    assert any("试验" in event.message and "原声" in event.message for event in h.events)
    assert not any("5.8" in event.message or "MDX" in event.message for event in h.events)
    h.dual_factory.assert_called_once()
    assert not h.gate.inputs


def test_preserve_worker_real_gap_resets_both_source_axes(harness, monkeypatch):
    h = preserve_harness(harness, monkeypatch)
    now = time.perf_counter()
    h.offer(np.full((480, 2), 4096, dtype=np.int16), captured_at=now - 1)
    h.offer(np.full((480, 2), 8192, dtype=np.int16), captured_at=now - .01)
    h.start()
    wait_for(lambda: h.session.segments.qsize() == 2)
    h.stop()
    assert [item[2] for item in h.dual_inputs] == [0, 0]
    assert h.processor.finish_calls == h.processor.reset_calls == 1
    first, second = h.session.segments.get_nowait(), h.session.segments.get_nowait()
    assert first.samples == [.125] * 160 and second.samples == [.25] * 160
    assert second.speech_start_at == pytest.approx(now - .02)


@pytest.mark.parametrize("failure", ["missing", "different_raw", "wrong_length", "noncontiguous"])
def test_preserve_worker_rejects_misaligned_branches_without_fallback(harness, monkeypatch, failure):
    h = preserve_harness(harness, monkeypatch)
    original = h.processor.on_process

    def malformed(samples):
        output = original(samples)[0]
        changes = {"missing": {"raw_samples": None},
                   "different_raw": {"raw_samples": output.samples * 2},
                   "wrong_length": {"denoised_samples": output.denoised_samples[:-1]},
                   "noncontiguous": {"start_sample": 1, "end_sample": output.end_sample + 1}}
        return [replace(output, **changes[failure])]

    h.processor.on_process = malformed
    h.start()
    h.offer(np.full((480, 2), 4096, dtype=np.int16))
    assert h.session.stop.wait(1)
    h.stop()
    assert not h.dual_inputs and h.session.segments.empty()
    assert any(event.kind == "error" and "原声识别模式" in event.message for event in h.events)
    assert h.processor.finish_calls == 0 and h.processor.close_calls == 1


def test_preserve_worker_stop_discards_pending_output_without_finish(harness, monkeypatch):
    h = preserve_harness(harness, monkeypatch)
    original = h.processor.on_process

    def blocked(samples):
        assert h.release.wait(1.5)
        return original(samples)

    h.processor.on_process = blocked
    h.start()
    h.offer(np.full((480, 2), 4096, dtype=np.int16))
    assert h.processor.entered.wait(1)
    h.stop()
    assert not h.dual_inputs and h.session.segments.empty()
    assert h.processor.finish_calls == 0 and h.processor.close_calls == 1
