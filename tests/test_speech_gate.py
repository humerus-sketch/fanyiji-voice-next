"""VAD adapter contracts without devices, native engines, or model weights."""

from collections import deque
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from fanyiji_next.speech_gate import SileroSpeechGate, SOURCE_LOOKBACK_SAMPLES
from fanyiji_next.dual_speech_gate import DualSpeechGate


class FakeDetector:
    """Model fixture: quiet blocks may have energy, with a 0.35 s hangover."""

    def __init__(self):
        self.calls = []
        self.reset_count = 0
        self.reset()

    def reset(self):
        self.reset_count += 1
        self.samples = []
        self.start = None
        self.speech_end = 0
        self.quiet = 0
        self.segments = deque()

    def accept_waveform(self, block):
        self.calls.append(len(block))
        offset = len(self.samples)
        self.samples.extend(block)
        if block[0] >= 0.2:
            if self.start is None:
                self.start = offset
            self.speech_end = len(self.samples)
            self.quiet = 0
        elif self.start is not None:
            self.quiet += len(block)
            if self.quiet >= 5600:
                self._complete(self.speech_end)

    def _complete(self, end):
        self.segments.append(SimpleNamespace(start=self.start, samples=self.samples[self.start:end]))
        self.start = None
        self.quiet = 0

    def is_speech_detected(self):
        return self.start is not None

    @property
    def current_segment(self):
        return SimpleNamespace(start=self.start)

    def empty(self):
        return not self.segments

    @property
    def front(self):
        return self.segments[0]

    def pop(self):
        self.segments.popleft()

    def flush(self):
        if self.start is not None:
            self._complete(len(self.samples))


class NativeStateDetector(FakeDetector):
    """Flush clears an output start, not model hysteresis/minimum-speech state."""

    def reset(self):
        super().reset()
        self.triggered = False
        self.candidate = None

    def accept_waveform(self, block):
        self.calls.append(len(block))
        offset = len(self.samples)
        self.samples.extend(block)
        # The weak marker represents a positive model decision, regardless of
        # amplitude. It is not an energy threshold in the production adapter.
        voiced = any(value >= .2 or value == .00001 for value in block)
        if voiced:
            if self.candidate is None:
                self.candidate = offset
            if len(self.samples) - self.candidate >= 2880:
                self.triggered = True
            if self.triggered and self.start is None:
                self.start = self.candidate
            self.speech_end = len(self.samples)
            self.quiet = 0
        elif self.triggered:
            if self.start is None:
                self.start = offset
            self.quiet += len(block)
            if self.quiet >= 5600:
                self._natural_complete()
                self.triggered = False
                self.candidate = None
        else:
            self.candidate = None

    def _natural_complete(self):
        self._complete(self.speech_end)

    def flush(self):
        if self.start is not None:
            self.segments.append(SimpleNamespace(start=self.start, samples=self.samples[self.start:]))
            self.start = None
            # triggered/quiet/candidate intentionally survive, as in the
            # native implementation whose forced-cut failure is reproduced.


class NativeOverhangDetector(NativeStateDetector):
    """A normal detector endpoint can include a few real post-speech zeros."""

    def _natural_complete(self):
        self._complete(min(self.speech_end + 32, len(self.samples)))


class SpeechGateTests(unittest.TestCase):
    def gate(self):
        detector = FakeDetector()
        return SileroSpeechGate(Path("unused.onnx"), detector=detector), detector

    def test_native_configuration_uses_existing_cpu_engine_and_model(self):
        config = SimpleNamespace(silero_vad=SimpleNamespace())
        factory = Mock(return_value=FakeDetector())
        module = SimpleNamespace(VadModelConfig=lambda: config, VoiceActivityDetector=factory)
        with patch.dict(sys.modules, {"sherpa_onnx": module}):
            SileroSpeechGate(Path("models/vad/silero_vad.onnx"))
        self.assertEqual("models\\vad\\silero_vad.onnx" if sys.platform == "win32" else "models/vad/silero_vad.onnx", config.silero_vad.model)
        self.assertEqual((0.5, 0.35, 0.18, 512), (
            config.silero_vad.threshold, config.silero_vad.min_silence_duration,
            config.silero_vad.min_speech_duration, config.silero_vad.window_size,
        ))
        self.assertEqual((16000, 1, "cpu"), (config.sample_rate, config.num_threads, config.provider))
        factory.assert_called_once_with(config, buffer_size_in_seconds=15)

    def test_high_energy_non_speech_uses_model_decision(self):
        gate, _ = self.gate()
        self.assertEqual([], gate.feed([0.10] * (7 * 16000)))
        self.assertEqual([], gate.finish())

    def test_natural_end_excludes_native_silence_wait_and_has_sample_positions(self):
        gate, _ = self.gate()
        self.assertEqual([], gate.feed([0.3] * 8192))
        completed = gate.feed([0.03] * 6144)
        self.assertEqual(1, len(completed))
        segment = completed[0]
        self.assertEqual((0, 8192, "vad_silence"), (segment.start_sample, segment.end_sample, segment.end_reason))
        self.assertEqual([0.3] * 8192, segment.samples)
        self.assertEqual([], gate.finish())

    def test_hard_limit_preserves_continuous_speech_without_overlap_or_loss(self):
        gate, detector = self.gate()
        original = [0.3] * 200000
        completed = []
        for offset in range(0, len(original), 640):
            completed.extend(gate.feed(original[offset:offset + 640]))
        completed.extend(gate.finish())
        self.assertEqual(["max_duration", "max_duration", "input_gap"], [s.end_reason for s in completed])
        self.assertTrue(all(0 < len(s.samples) <= 96000 for s in completed))
        self.assertEqual(original, [value for s in completed for value in s.samples])
        self.assertEqual(0, completed[0].start_sample)
        self.assertEqual(len(original), completed[-1].end_sample)
        self.assertTrue(all(a.end_sample == b.start_sample for a, b in zip(completed, completed[1:])))
        self.assertEqual({512}, set(detector.calls))

    def test_native_reset_replay_preserves_global_clock_and_pending_in_one_large_feed(self):
        for with_context in (False, True):
            with self.subTest(with_context=with_context):
                detector = NativeStateDetector()
                options = {"leading_seconds": .12, "trailing_seconds": .16} if with_context else {}
                gate = SileroSpeechGate(Path("unused"), detector=detector, **options)
                source = [.01] * 4096 + [.3] * 200173
                completed = gate.feed(source)
                self.assertEqual(len(source), gate._received)
                self.assertEqual(len(source) // 512 * 512, gate._processed)
                self.assertEqual(len(source) % 512, len(gate._pending))
                self.assertGreater(gate._native_offset, 0)
                self.assertGreaterEqual(detector.reset_count, 3)
                completed.extend(gate.finish())
                self.assertTrue(all(0 < len(item.samples) <= 96000 for item in completed))
                self.assertTrue(all(first.end_sample == second.start_sample
                                    for first, second in zip(completed, completed[1:])))
                self.assertEqual(len(source), completed[-1].end_sample)
                for item in completed:
                    self.assertEqual(source[item.start_sample:item.end_sample], item.samples)
                recovered = [value for item in completed for value in item.samples]
                self.assertEqual(200173, recovered.count(.3))
                self.assertEqual({512}, set(detector.calls))

    def test_forced_cut_in_native_hangover_resets_stale_model_end_state(self):
        for with_context in (False, True):
            with self.subTest(with_context=with_context):
                detector = NativeStateDetector()
                options = {"leading_seconds": .12, "trailing_seconds": .16} if with_context else {}
                gate = SileroSpeechGate(Path("unused"), detector=detector, **options)
                # The last four windows are already in native hangover when
                # the application forces its six-second cap.
                source = [.3] * (95744 - 2048) + [.01] * (2048 + 6144)
                completed = gate.feed(source)
                completed.extend(gate.finish())
                self.assertEqual(1, len(completed))
                self.assertEqual("max_duration", completed[0].end_reason)
                self.assertEqual((0, 95744), (completed[0].start_sample, completed[0].end_sample))
                self.assertEqual(source[:95744], completed[0].samples)
                self.assertGreaterEqual(detector.reset_count, 3)

    def test_forced_cut_preserves_50_100_170ms_model_positive_tails_including_weak_voice(self):
        for with_context in (False, True):
            for duration_ms in (50, 100, 170):
                for amplitude in (.4, .00001):
                    with self.subTest(with_context=with_context, duration_ms=duration_ms, amplitude=amplitude):
                        detector = NativeStateDetector()
                        options = {"leading_seconds": .12, "trailing_seconds": .16} if with_context else {}
                        gate = SileroSpeechGate(Path("unused"), detector=detector, **options)
                        cut = 95744
                        tail = [amplitude] * (duration_ms * 16)
                        source = [.3] * cut + tail + [.0] * 6144
                        completed = []
                        for first in range(0, len(source), 641):
                            completed.extend(gate.feed(source[first:first + 641]))
                        completed.extend(gate.finish())
                        self.assertEqual(2, len(completed))
                        first, last = completed
                        self.assertEqual("max_duration", first.end_reason)
                        self.assertEqual("vad_silence", last.end_reason)
                        self.assertEqual(cut, last.start_sample)
                        self.assertGreaterEqual(last.end_sample, cut + len(tail))
                        self.assertLessEqual(len(last.samples), 96000)
                        self.assertEqual(tail, last.samples[:len(tail)])
                        self.assertEqual(source[last.start_sample:last.end_sample], last.samples)
                        if with_context:
                            self.assertEqual(cut, last.speech_start_sample)
                            self.assertGreaterEqual(last.speech_end_sample, cut + len(tail))

    def test_native_zero_overhang_after_cut_never_becomes_an_extra_utterance(self):
        for with_context in (False, True):
            with self.subTest(with_context=with_context):
                detector = NativeOverhangDetector()
                options = {"leading_seconds": .12, "trailing_seconds": .16} if with_context else {}
                gate = SileroSpeechGate(Path("unused"), detector=detector, **options)
                source = [.3] * 95744 + [.0] * 6144
                completed = gate.feed(source)
                completed.extend(gate.finish())
                self.assertEqual(1, len(completed))
                self.assertEqual((0, 95744), (completed[0].start_sample, completed[0].end_sample))

    def test_zero_overhang_dedup_has_no_nonzero_energy_floor(self):
        detector = NativeOverhangDetector()
        gate = SileroSpeechGate(Path("unused"), detector=detector)
        source = [.3] * 95744 + [.00001] * 800 + [.0] * 6144
        completed = gate.feed(source)
        completed.extend(gate.finish())
        self.assertEqual(2, len(completed))
        self.assertEqual([.00001] * 800, completed[1].samples[:800])
        self.assertEqual(95744, completed[1].start_sample)

    def test_tiny_nonzero_native_continuation_is_retained_without_a_length_or_volume_gate(self):
        detector = NativeOverhangDetector()
        gate = SileroSpeechGate(Path("unused"), detector=detector)
        completed = gate.feed([.3] * 95744 + [1e-12] * 6144)
        completed.extend(gate.finish())
        self.assertEqual(2, len(completed))
        self.assertEqual((95744, 95776), (completed[1].start_sample, completed[1].end_sample))
        self.assertEqual([1e-12] * 32, completed[1].samples)

    def test_eos_after_cut_preserves_real_partial_window_tail_without_returning_padding(self):
        for with_context in (False, True):
            for count in (1, 123, 511):
                with self.subTest(with_context=with_context, count=count):
                    options = {"leading_seconds": .12, "trailing_seconds": .16} if with_context else {}
                    gate = SileroSpeechGate(Path("unused"), detector=NativeStateDetector(), **options)
                    source = [.3] * 95744 + [.00001] * count
                    completed = gate.feed(source)
                    completed.extend(gate.finish())
                    self.assertEqual(2, len(completed))
                    self.assertEqual((95744, len(source)), (completed[1].start_sample, completed[1].end_sample))
                    self.assertEqual([.00001] * count, completed[1].samples)
                    self.assertEqual(source, [value for item in completed for value in item.samples])

    def test_natural_segments_keep_original_boundaries_without_native_reset_or_replay(self):
        for with_context in (False, True):
            with self.subTest(with_context=with_context):
                detector = NativeStateDetector()
                options = {"leading_seconds": .12, "trailing_seconds": .16} if with_context else {}
                gate = SileroSpeechGate(Path("unused"), detector=detector, **options)
                source = [.01] * 6144 + [.3] * 8192 + [.01] * 8192 + [.4] * 4096 + [.01] * 8192
                completed = gate.feed(source)
                expected = [(4224, 16896), (20608, 29184)] if with_context else [(6144, 14336), (22528, 26624)]
                self.assertEqual(expected, [(item.start_sample, item.end_sample) for item in completed])
                self.assertEqual(1, detector.reset_count)
                self.assertEqual(gate._processed // 512, len(detector.calls))
                self.assertEqual(0, gate._native_offset)
                for item in completed:
                    self.assertEqual(source[item.start_sample:item.end_sample], item.samples)

    def test_normal_native_short_segment_is_unchanged_by_cut_only_zero_dedup(self):
        detector = FakeDetector()
        gate = SileroSpeechGate(Path("unused"), detector=detector)
        # Explicit detector-positive fixture; the adapter does not introduce
        # an amplitude gate on ordinary segments, even for exact zeros.
        detector.segments.append(SimpleNamespace(start=0, samples=[0.] * 128))
        completed = gate.feed([0.] * 512)
        self.assertEqual(1, len(completed))
        self.assertEqual([0.] * 128, completed[0].samples)
        self.assertEqual((0, 128), (completed[0].start_sample, completed[0].end_sample))

    def test_forced_cut_warm_history_budget_tracks_selected_silence_profile(self):
        default = SileroSpeechGate(Path("unused"), detector=FakeDetector())
        longer = SileroSpeechGate(Path("unused"), detector=FakeDetector(), silence_seconds=.45)
        self.assertEqual(9728, default._warm_samples)
        self.assertEqual(11264, longer._warm_samples)

    def test_native_empty_real_segment_and_negative_start_raise_visible_errors(self):
        for start, samples, message in ((0, [], "空语音段"), (-1, [.3] * 512, "负数语音起点")):
            with self.subTest(start=start, sample_count=len(samples)):
                detector = FakeDetector()
                gate = SileroSpeechGate(Path("unused"), detector=detector)
                detector.segments.append(SimpleNamespace(start=start, samples=samples))
                with self.assertRaisesRegex(RuntimeError, message):
                    gate.feed([0.] * 512)

    def test_eos_empty_native_segment_outside_real_source_is_ignored(self):
        detector = FakeDetector()
        gate = SileroSpeechGate(Path("unused"), detector=detector)
        detector.segments.append(SimpleNamespace(start=1024, samples=[]))
        self.assertEqual([], gate.finish())

    def test_gap_padding_is_never_returned_as_captured_audio(self):
        gate, _ = self.gate()
        self.assertEqual([], gate.feed([0.3] * 4296))
        segment, = gate.finish()
        self.assertEqual((0, 4296, "input_gap"), (segment.start_sample, segment.end_sample, segment.end_reason))
        self.assertEqual([0.3] * 4296, segment.samples)
        self.assertEqual([], gate.finish())

    def test_reset_discards_partial_speech_and_restarts_sample_positions(self):
        gate, _ = self.gate()
        gate.feed([0.4] * 4097)
        gate.reset()
        self.assertEqual([], gate.finish())
        gate.feed([0.3] * 8192)
        segment, = gate.feed([0.03] * 6144)
        self.assertEqual(0, segment.start_sample)
        self.assertEqual([0.3] * 8192, segment.samples)

    def test_consumers_do_not_share_state(self):
        first, _ = self.gate()
        second, _ = self.gate()
        first.feed([0.3] * 8192)
        self.assertEqual([], second.feed([0.03] * 16384))
        self.assertEqual([], second.finish())
        segment, = first.finish()
        self.assertEqual(8192, len(segment.samples))

    def test_optional_system_padding_keeps_weak_tail_samples_and_original_speech_timestamps(self):
        detector = FakeDetector()
        gate = SileroSpeechGate(Path("unused"), detector=detector, leading_seconds=.12, trailing_seconds=.16)
        prefix = [0.01] * 4096
        speech = [0.3] * 8192
        weak_tail = [0.04] * 2560
        silence = [0.01] * 6144
        source = prefix + speech + weak_tail + silence
        completed = gate.feed(source)
        segment, = completed
        self.assertEqual((2176, 14848), (segment.start_sample, segment.end_sample))
        self.assertEqual((4096, 12288), (segment.speech_start_sample, segment.speech_end_sample))
        self.assertEqual(source[segment.start_sample:segment.end_sample], segment.samples)
        self.assertEqual(weak_tail, segment.samples[-2560:])
        self.assertEqual("vad_silence", segment.end_reason)
        self.assertEqual([], gate.finish())

    def test_system_padding_never_exceeds_hard_limit_or_duplicates_continuous_audio(self):
        gate = SileroSpeechGate(Path("unused"), detector=FakeDetector(), leading_seconds=.12, trailing_seconds=.16)
        source = [0.3] * 200000
        completed = []
        for offset in range(0, len(source), 640):
            completed.extend(gate.feed(source[offset:offset + 640]))
        completed.extend(gate.finish())
        self.assertTrue(all(0 < len(item.samples) <= 96000 for item in completed))
        self.assertEqual(source, [sample for item in completed for sample in item.samples])
        self.assertEqual(0, completed[0].start_sample)
        self.assertEqual(len(source), completed[-1].end_sample)
        self.assertTrue(all(first.end_sample == second.start_sample for first, second in zip(completed, completed[1:])))
        self.assertEqual(["max_duration", "max_duration", "input_gap"], [item.end_reason for item in completed])

    def test_system_padding_preserves_source_order_for_two_utterances_without_overlap(self):
        gate = SileroSpeechGate(Path("unused"), detector=FakeDetector(), leading_seconds=.12, trailing_seconds=.16)
        source = [0.01] * 4096 + [0.3] * 8192 + [0.04] * 8192 + [0.4] * 8192 + [0.02] * 8192
        completed = gate.feed(source)
        self.assertEqual(2, len(completed))
        self.assertLessEqual(completed[0].end_sample, completed[1].start_sample)
        for item in completed:
            self.assertEqual(source[item.start_sample:item.end_sample], item.samples)
            self.assertLessEqual(item.start_sample, item.speech_start_sample)
            self.assertGreaterEqual(item.end_sample, item.speech_end_sample)
        self.assertEqual([], gate.finish())

    def test_system_gap_and_reset_never_return_artificial_padding_or_previous_audio(self):
        gate = SileroSpeechGate(Path("unused"), detector=FakeDetector(), leading_seconds=.12, trailing_seconds=.16)
        source = [0.3] * 4296
        gate.feed(source)
        segment, = gate.finish()
        self.assertEqual(source, segment.samples)
        self.assertEqual((0, len(source)), (segment.start_sample, segment.end_sample))
        self.assertEqual((0, len(source)), (segment.speech_start_sample, segment.speech_end_sample))
        gate.feed([0.4] * 4097)
        gate.reset()
        self.assertEqual([], gate.finish())
        gate.feed([0.3] * 8192)
        segment, = gate.feed([0.03] * 6144)
        self.assertEqual(0, segment.start_sample)
        self.assertEqual([0.3] * 8192, segment.samples[:8192])

    def test_silence_profile_configures_native_detector_and_finish_padding_budget(self):
        config = SimpleNamespace(silero_vad=SimpleNamespace())
        detector = FakeDetector()
        module = SimpleNamespace(VadModelConfig=lambda: config, VoiceActivityDetector=Mock(return_value=detector))
        with patch.dict(sys.modules, {"sherpa_onnx": module}):
            gate = SileroSpeechGate(Path("unused"), silence_seconds=.45, leading_seconds=.12, trailing_seconds=.16)
        self.assertEqual(.45, config.silero_vad.min_silence_duration)
        self.assertEqual(.5, config.silero_vad.threshold)
        self.assertEqual(.45, gate.silence_seconds)

    def test_invalid_profile_durations_fail_before_native_loading(self):
        for name in ("silence_seconds", "leading_seconds", "trailing_seconds"):
            for value in (-1, 6, float("nan"), float("inf"), "0.1"):
                with self.subTest(name=name, value=value), self.assertRaises(ValueError):
                    SileroSpeechGate(Path("unused"), **{name: value})
        with self.assertRaises(ValueError):
            SileroSpeechGate(Path("unused"), silence_seconds=0)

    def test_readonly_source_frontiers_include_native_v5_onset_lookback(self):
        gate, _ = self.gate()
        self.assertEqual(4544, SOURCE_LOOKBACK_SAMPLES)
        self.assertEqual((0, None, 0), (gate.processed_samples, gate.active_start_sample,
                                       gate.safe_frontier_sample))
        gate.feed([.01] * 8193)
        self.assertEqual((8192, None, 8192 - SOURCE_LOOKBACK_SAMPLES),
                         (gate.processed_samples, gate.active_start_sample, gate.safe_frontier_sample))
        gate.feed([.3] * 2048)
        self.assertEqual(gate.active_start_sample, gate.safe_frontier_sample)
        self.assertGreaterEqual(gate.processed_samples, gate.active_start_sample)
        gate.reset()
        self.assertEqual((0, None, 0), (gate.processed_samples, gate.active_start_sample,
                                       gate.safe_frontier_sample))

    def test_readonly_frontier_preserves_global_offset_after_native_warm_cut(self):
        gate = SileroSpeechGate(Path("unused"), detector=NativeStateDetector())
        gate.feed([.3] * 96000)
        self.assertEqual(95744, gate.active_start_sample)
        self.assertEqual(95744, gate.safe_frontier_sample)
        gate.feed([.00001] * 1600)
        self.assertEqual(95744, gate.active_start_sample)
        self.assertGreater(gate.processed_samples, gate.safe_frontier_sample)
        gate.finish()
        self.assertEqual((0, None, 0), (gate.processed_samples, gate.active_start_sample,
                                       gate.safe_frontier_sample))

    def test_dual_native_adapters_keep_exact_raw_across_warm_caps_and_weak_tail(self):
        for block_size in (31, 512, 640, 10000):
            with self.subTest(block_size=block_size):
                raw_gate = SileroSpeechGate(Path("unused"), detector=NativeStateDetector(),
                                           leading_seconds=.12, trailing_seconds=.16)
                den_gate = SileroSpeechGate(Path("unused"), detector=NativeStateDetector())
                gate = DualSpeechGate(Path("unused"), raw_gate=raw_gate, denoised_gate=den_gate)
                raw = [.3] * 192000 + [.00001] * 1700 + [.01] * 8192
                denoised = [.3] * 90000 + [.01] * (len(raw) - 90000)
                output = []
                for first in range(0, len(raw), block_size):
                    output.extend(gate.feed(raw[first:first + block_size], denoised[first:first + block_size]))
                    self.assertFalse(hasattr(gate, "_raw"))
                output.extend(gate.finish())
                self.assertGreaterEqual(len(output), 3)
                self.assertEqual(0, output[0].start_sample)
                self.assertGreaterEqual(output[-1].end_sample, 193700)
                self.assertTrue(all(0 < len(item.samples) <= 96000 for item in output))
                self.assertTrue(all(first.end_sample == second.start_sample
                                    for first, second in zip(output, output[1:])))
                self.assertEqual(raw[:output[-1].end_sample], [sample for item in output for sample in item.samples])
                self.assertEqual(raw[192000:193700],
                                 [sample for item in output for sample in item.samples][192000:193700])

    def test_dual_native_adapters_keep_denoised_only_detection_diagnostic(self):
        raw_gate = SileroSpeechGate(Path("unused"), detector=FakeDetector(),
                                   leading_seconds=.12, trailing_seconds=.16)
        den_gate = SileroSpeechGate(Path("unused"), detector=FakeDetector())
        gate = DualSpeechGate(Path("unused"), raw_gate=raw_gate, denoised_gate=den_gate)
        raw = [.01] * 16384
        denoised = [.01] * 4096 + [.3] * 4096 + [.01] * 8192
        output = gate.feed(raw, denoised)
        output.extend(gate.finish())
        self.assertEqual([], output)
        diagnostic, = gate.take_diagnostics()
        self.assertEqual((4096, 8192), (diagnostic.speech_start_sample, diagnostic.speech_end_sample))
        self.assertEqual([], gate.finish())


if __name__ == "__main__":
    unittest.main()
