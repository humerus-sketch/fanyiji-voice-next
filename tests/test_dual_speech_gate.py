"""Canonical raw delivery and bounded denoised diagnostics; no models/devices."""
from pathlib import Path
import unittest
from unittest.mock import patch

from fanyiji_next.dual_speech_gate import DualSpeechGate, DIAGNOSTIC_HISTORY_LIMIT
from fanyiji_next.speech_gate import SpeechSlice


class RecordingGate:
    def __init__(self, intervals=()):
        self.intervals = list(intervals)
        self.reset_count = 0
        self.reset()

    def reset(self):
        self.reset_count += 1
        self.values = []
        self.processed_samples = 0
        self.emitted = set()
        self.last_output = None

    def _slice(self, start, end, reason):
        return SpeechSlice(self.values[start:end], start, end, reason, start, end)

    def feed(self, values):
        self.values.extend(values)
        self.processed_samples = len(self.values) // 512 * 512
        result = []
        for index, (start, end, ready) in enumerate(self.intervals):
            if index not in self.emitted and ready <= self.processed_samples:
                result.append(self._slice(start, end, 'vad_silence'))
                self.emitted.add(index)
        self.last_output = result
        return result

    def finish(self):
        result = [self._slice(start, min(end, len(self.values)), 'input_gap')
                  for index, (start, end, _) in enumerate(self.intervals)
                  if index not in self.emitted and start < len(self.values)]
        self.reset()
        self.last_output = result
        return result


class DualSpeechGateTests(unittest.TestCase):
    def gate(self, raw=(), denoised=()):
        a, b = RecordingGate(raw), RecordingGate(denoised)
        gate = DualSpeechGate(Path('not-loaded.onnx'), raw_gate=a, denoised_gate=b)
        return gate, a, b

    def test_returns_exact_raw_list_and_slice_objects_without_boundary_changes(self):
        gate, raw, _ = self.gate([(512, 1536, 2048)])
        values = [index / 10000 for index in range(4096)]
        output = gate.feed(values, [0.] * len(values))
        self.assertIs(output, raw.last_output)
        self.assertIs(output[0], raw.last_output[0])
        self.assertEqual(values[512:1536], output[0].samples)
        self.assertEqual((512, 1536, 'vad_silence'),
                         (output[0].start_sample, output[0].end_sample, output[0].end_reason))
        self.assertEqual('canonical_raw', gate.recognition_policy)

    def test_denoised_always_active_does_not_hold_or_merge_raw_natural_sentences(self):
        gate, raw, den = self.gate([(0, 1024, 1536), (2048, 3072, 3584)],
                                   [(0, 100000, 101000)])
        den.active_start_sample = 0
        first = gate.feed([.1] * 1536, [.8] * 1536)
        self.assertEqual([(0, 1024)], [(s.start_sample, s.end_sample) for s in first])
        second = gate.feed([.1] * 2048, [.8] * 2048)
        self.assertEqual([(2048, 3072)], [(s.start_sample, s.end_sample) for s in second])
        self.assertEqual(0, gate.diagnostic_counts['pending'])
        self.assertFalse(hasattr(gate, '_raw'))

    def test_denoised_only_trigger_is_diagnostic_and_never_creates_asr_sentence(self):
        gate, _, _ = self.gate(denoised=[(512, 2048, 4096)])
        self.assertEqual([], gate.feed([.001] * 8192, [.5] * 8192))
        diagnostic, = gate.take_diagnostics()
        self.assertEqual((512, 2048, 'vad_silence'),
                         (diagnostic.speech_start_sample, diagnostic.speech_end_sample, diagnostic.end_reason))
        self.assertFalse(hasattr(diagnostic, 'samples'))
        self.assertEqual([], gate.take_diagnostics())
        self.assertEqual({'detected_total': 1, 'discarded_before_drain': 0, 'pending': 0}, gate.diagnostic_counts)
        self.assertEqual([], gate.finish())

    def test_diagnostic_history_is_bounded_with_explicit_discard_counter(self):
        count = DIAGNOSTIC_HISTORY_LIMIT + 17
        gate, _, _ = self.gate(denoised=[(i*512, (i+1)*512, (i+1)*512) for i in range(count)])
        self.assertEqual([], gate.feed([.01] * (512*count), [.3] * (512*count)))
        self.assertEqual({'detected_total': count, 'discarded_before_drain': 17,
                          'pending': DIAGNOSTIC_HISTORY_LIMIT}, gate.diagnostic_counts)
        retained = gate.take_diagnostics()
        self.assertEqual(DIAGNOSTIC_HISTORY_LIMIT, len(retained))
        self.assertEqual(17*512, retained[0].start_sample)
        self.assertEqual(0, gate.diagnostic_counts['pending'])

    def test_finish_keeps_real_partial_raw_tail_and_eos_diagnostics_until_drain(self):
        gate, raw, _ = self.gate([(0, 9999, 12000)], [(0, 9999, 12000)])
        values = [.123] * 1237
        self.assertEqual([], gate.feed(values, [.8] * len(values)))
        output = gate.finish()
        self.assertIs(output, raw.last_output)
        self.assertEqual(values, output[0].samples)
        self.assertEqual((0, 1237, 'input_gap'), (output[0].start_sample, output[0].end_sample, output[0].end_reason))
        diagnostic, = gate.take_diagnostics()
        self.assertEqual((0, 1237, 'input_gap'), (diagnostic.start_sample, diagnostic.end_sample, diagnostic.end_reason))
        self.assertEqual(0, gate.received_samples)
        self.assertEqual([], gate.finish())
        self.assertEqual([], gate.take_diagnostics())

    def test_eos_then_new_epoch_has_unambiguous_diagnostic_source_origin(self):
        gate, _, _ = self.gate(denoised=[(0, 1024, 1536)])
        gate.feed([.1] * 2048, [.3] * 2048)
        gate.finish()
        gate.feed([.1] * 2048, [.3] * 2048, start_sample=0)
        records = gate.take_diagnostics()
        self.assertEqual([0, 1], [item.epoch for item in records])
        self.assertEqual([0, 0], [item.start_sample for item in records])

    def test_stop_reset_discards_pending_speech_and_diagnostic_history(self):
        gate, a, b = self.gate([(0, 8192, 10000)], [(0, 1024, 1536)])
        gate.feed([.1] * 2048, [.8] * 2048)
        gate.reset()
        self.assertEqual(0, gate.received_samples)
        self.assertEqual([], gate.take_diagnostics())
        self.assertEqual({'detected_total': 0, 'discarded_before_drain': 0, 'pending': 0}, gate.diagnostic_counts)
        self.assertEqual([], gate.finish())
        self.assertGreater(a.reset_count, 1)
        self.assertGreater(b.reset_count, 1)

    def test_validation_rejects_length_nonfinite_or_gap_before_advancing_either_gate(self):
        gate, a, b = self.gate()
        for raw, den, options in [([.1], [], {}), ([.1], [float('nan')], {}),
                                  ([float('inf')], [.1], {}), ([.1], [.1], {'start_sample': 2})]:
            with self.subTest(raw=raw, den=den), self.assertRaises(ValueError):
                gate.feed(raw, den, **options)
            self.assertEqual((0, [], []), (gate.received_samples, a.values, b.values))

    def test_diagnostic_failure_does_not_return_partially_successful_raw_delivery(self):
        gate, a, b = self.gate([(0, 512, 512)])
        original = b.feed
        def fail(values):
            raise RuntimeError('native diagnostic failure')
        b.feed = fail
        with self.assertRaisesRegex(RuntimeError, 'diagnostic failure'):
            gate.feed([.1] * 512, [.8] * 512)
        with self.assertRaisesRegex(RuntimeError, 'reset'):
            gate.feed([], [])
        b.feed = original
        gate.reset()
        self.assertEqual([], gate.feed([], []))

    def test_invalid_or_overlapping_raw_source_intervals_fail_observably(self):
        gate, _, _ = self.gate([(0, 1024, 2048), (512, 1536, 2048)])
        with self.assertRaisesRegex(RuntimeError, 'overlapping'):
            gate.feed([.1] * 4096, [.3] * 4096)

    def test_failed_reset_still_attempts_other_gate_and_requires_successful_reset(self):
        gate, a, b = self.gate()
        original = a.reset
        before = b.reset_count
        def fail():
            raise RuntimeError('native reset failed')
        a.reset = fail
        with self.assertRaisesRegex(RuntimeError, 'native reset'):
            gate.reset()
        self.assertEqual(before + 1, b.reset_count)
        with self.assertRaisesRegex(RuntimeError, 'reset'):
            gate.feed([], [])
        a.reset = original
        gate.reset()
        self.assertEqual([], gate.feed([], []))

    def test_default_constructor_gives_raw_gate_requested_context_and_den_no_extra_context(self):
        with patch('fanyiji_next.dual_speech_gate.SileroSpeechGate',
                   side_effect=[RecordingGate(), RecordingGate()]) as factory:
            DualSpeechGate(Path('unused'), leading_seconds=.1, trailing_seconds=.2)
        self.assertEqual({'silence_seconds': .35, 'leading_seconds': .1, 'trailing_seconds': .2},
                         factory.call_args_list[0].kwargs)
        self.assertEqual({'silence_seconds': .35}, factory.call_args_list[1].kwargs)

    def test_same_detector_state_is_rejected_and_invalid_settings_do_not_load_models(self):
        same = RecordingGate()
        with self.assertRaises(ValueError):
            DualSpeechGate(Path('unused'), raw_gate=same, denoised_gate=same)
        with patch('fanyiji_next.dual_speech_gate.SileroSpeechGate') as factory:
            for name in ('silence_seconds', 'leading_seconds', 'trailing_seconds'):
                for value in (-1, 6, float('nan'), float('inf'), '0.1'):
                    with self.subTest(name=name, value=value), self.assertRaises(ValueError):
                        DualSpeechGate(Path('unused'), **{name:value})
            factory.assert_not_called()


if __name__ == '__main__':
    unittest.main()
