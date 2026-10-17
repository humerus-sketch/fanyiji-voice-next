"""Source alignment, bounded buffering and native lifecycle without real models."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
from threading import Event
from types import SimpleNamespace

import numpy as np
import pytest
import soxr

import fanyiji_next.audio_processing as audio_processing
from fanyiji_next.audio_processing import (
    AudioProcessingCancelled, AudioProcessingError, MDX_SAMPLE_RATE,
    MDX_STEP_SAMPLES, MDX_TRIM_SAMPLES, MDX_WINDOW_SAMPLES,
    GTCRN_MAX_PENDING_SAMPLES,
    ProcessedAudio, SystemAudioProcessor, _istft, _stft, create_system_audio_processor,
)


class IdentityMdx:
    def __init__(self, stop=None):
        self.calls = 0
        self.closed = False
        self.stop = stop
        self.first_window = None

    def process_window(self, samples):
        self.calls += 1
        if self.first_window is None:
            self.first_window = samples.copy()
        if self.stop is not None:
            self.stop.set()
        return samples.copy()

    def close(self):
        self.closed = True


class DelayedIdentityGtcrn:
    """Native contract: one hop buffered, compensated content origin, flush tail."""

    def __init__(self, stop=None):
        self.stop = stop
        self.closed = False
        self.reset_count = 0
        self.flush_count = 0
        self.pending = np.empty(0, dtype=np.float32)

    def process(self, samples):
        self.pending = np.concatenate((self.pending, samples))
        count = max(0, (len(self.pending) // 256 - 1) * 256)
        result, self.pending = self.pending[:count].copy(), self.pending[count:].copy()
        if self.stop is not None:
            self.stop.set()
        return result

    def finish(self):
        self.flush_count += 1
        result, self.pending = self.pending, np.empty(0, dtype=np.float32)
        return result

    def reset(self):
        self.reset_count += 1
        self.pending = np.empty(0, dtype=np.float32)

    def close(self):
        self.closed = True
        self.pending = np.empty(0, dtype=np.float32)


class DelayedScaleGtcrn(DelayedIdentityGtcrn):
    def __init__(self, factor=.25, stop=None):
        super().__init__(stop)
        self.factor = factor

    def process(self, samples):
        return super().process(samples) * self.factor

    def finish(self):
        return super().finish() * self.factor


def signal(count, rate, channels=2):
    t = np.arange(count, dtype=np.float32) / rate
    left = .1 * np.sin(2 * np.pi * 311 * t)
    right = .07 * np.cos(2 * np.pi * 593 * t)
    if channels == 1:
        return left[:, None]
    return np.stack((left, right), axis=1).astype(np.float32)


def run_blocks(processor, values, block_size, *, finish=True):
    outputs = []
    for first in range(0, len(values), block_size):
        blocks = processor.process_block(values[first:first + block_size])
        assert processor.pending_mdx_samples < MDX_WINDOW_SAMPLES
        outputs.extend(blocks)
    if finish:
        outputs.extend(processor.finish())
    cursor = 0
    for block in outputs:
        assert isinstance(block, ProcessedAudio)
        assert block.start_sample == cursor
        assert block.end_sample == cursor + len(block.samples)
        assert block.samples.dtype == np.float32 and block.samples.ndim == 1
        assert block.input_end_frame <= len(values)
        cursor = block.end_sample
    return np.concatenate([block.samples for block in outputs]) if outputs else np.empty(0, dtype=np.float32), outputs


@pytest.mark.parametrize("rate", [8000, 16000, 44100, 48000, 96000])
@pytest.mark.parametrize("count", [0, 1, 513, 16037])
def test_raw_stream_exact_length_and_source_origin(rate, count):
    values = signal(count, rate)
    processor = SystemAudioProcessor("raw", sample_rate=rate, channels=2)
    try:
        output, blocks = run_blocks(processor, values, max(1, rate // 25))
        assert len(output) == round(count * 16000 / rate)
        expected = values.mean(axis=1)
        if rate != 16000 and len(expected):
            expected = soxr.resample(expected, rate, 16000)
        assert abs(len(expected) - len(output)) <= 1
        np.testing.assert_allclose(output[:min(len(output), len(expected))], expected[:len(output)], atol=2e-6)
        assert processor.finish() == []
        assert processor.output_end_sample == len(output)
        with pytest.raises(AudioProcessingError):
            processor.process_block(np.zeros((1, 2), dtype=np.float32))
    finally:
        processor.close()


@pytest.mark.parametrize("rate", [8000, 16000, 44100, 48000, 96000])
@pytest.mark.parametrize("count", [0, 1, 513, 16037])
def test_gtcrn_direct_resampling_preserves_every_source_sample_and_eos_length(rate, count):
    values = signal(count, rate)
    gtcrn = DelayedIdentityGtcrn()
    processor = SystemAudioProcessor("gtcrn", sample_rate=rate, channels=2, _gtcrn_model=gtcrn)
    try:
        output, blocks = run_blocks(processor, values, max(1, rate // 25))
        expected = values.mean(axis=1)
        if rate != 16000 and len(expected):
            expected = soxr.resample(expected, rate, 16000)
        assert len(output) == round(count * 16000 / rate)
        np.testing.assert_allclose(output, expected, atol=2e-6)
        assert processor.pending_mdx_samples == 0
        assert processor._mdx is None and processor._output_resampler is None
        assert gtcrn.flush_count == 1
        assert processor.finish() == [] and gtcrn.flush_count == 1
        if count and len(output):
            assert blocks[0].start_sample == 0
            assert blocks[-1].input_end_frame == count
    finally:
        processor.close()
    assert gtcrn.closed


@pytest.mark.parametrize("channels", [1, 2, 6, 32])
def test_gtcrn_native_multichannel_uses_raw_channel_mean(channels):
    values = np.random.default_rng(601).uniform(-.3, .3, (1237, channels)).astype(np.float32)
    processor = SystemAudioProcessor("gtcrn", sample_rate=16000, channels=channels,
        _gtcrn_model=DelayedIdentityGtcrn())
    try:
        output, _ = run_blocks(processor, values, 159)
        np.testing.assert_array_equal(output, values.mean(axis=1))
    finally:
        processor.close()


def test_gtcrn_first_output_needs_only_two_native_hops_not_mdx_window():
    values = signal(768, 16000)
    processor = SystemAudioProcessor("gtcrn", sample_rate=16000, channels=2,
        _gtcrn_model=DelayedIdentityGtcrn())
    try:
        assert processor.process_block(values[:256]) == []
        blocks = processor.process_block(values[256:512])
        assert len(blocks) == 1
        assert (blocks[0].start_sample, blocks[0].end_sample, blocks[0].input_end_frame) == (0, 256, 512)
        np.testing.assert_array_equal(blocks[0].samples, values[:256].mean(axis=1))
        remaining = processor.process_block(values[512:]) + processor.finish()
        output = np.concatenate([block.samples for block in blocks + remaining])
        np.testing.assert_array_equal(output, values.mean(axis=1))
    finally:
        processor.close()


def test_gtcrn_48k_stereo_emits_before_a_quarter_second_of_audio():
    values = signal(48000, 48000)
    processor = SystemAudioProcessor("gtcrn", sample_rate=48000, channels=2,
        _gtcrn_model=DelayedIdentityGtcrn())
    try:
        output, blocks = run_blocks(processor, values, 960)
        assert blocks[0].input_end_frame < 12000
        np.testing.assert_allclose(output, soxr.resample(values.mean(axis=1), 48000, 16000), atol=2e-6)
    finally:
        processor.close()


@pytest.mark.parametrize("rate", [8000, 16000, 44100, 48000, 96000])
@pytest.mark.parametrize("factor", [0.0, .25, -1.0])
def test_gtcrn_residual_pairs_exact_source_samples_across_callbacks_and_flush(rate, factor):
    values = np.random.default_rng(765).uniform(-.2, .2, (rate + 173, 2)).astype(np.float32)
    gtcrn = DelayedScaleGtcrn(factor)
    processor = SystemAudioProcessor("gtcrn", sample_rate=rate, channels=2, _gtcrn_model=gtcrn)
    try:
        blocks = []
        # Input and denoiser hop boundaries deliberately do not coincide.
        for start in range(0, len(values), 479):
            blocks.extend(processor.process_block(values[start:start + 479]))
            assert 0 <= processor.pending_gtcrn_raw_samples <= GTCRN_MAX_PENDING_SAMPLES
        pending_before_flush = processor.pending_gtcrn_raw_samples
        assert pending_before_flush > 0
        blocks.extend(processor.finish())
        output = np.concatenate([block.samples for block in blocks])
        raw = values.mean(axis=1)
        if rate != 16000:
            raw = soxr.resample(raw, rate, 16000)
        expected = .5 * raw + .5 * (factor * raw)
        np.testing.assert_allclose(output, expected, atol=2e-6)
        assert len(output) == round(len(values) * 16000 / rate)
        assert processor.pending_gtcrn_raw_samples == 0 and gtcrn.flush_count == 1
        assert processor.finish() == [] and gtcrn.flush_count == 1
        cursor = 0
        for block in blocks:
            assert block.start_sample == cursor
            cursor = block.end_sample
        assert cursor == len(output)
    finally:
        processor.close()
    assert processor.pending_gtcrn_raw_samples == 0


def test_gtcrn_residual_retains_an_impulse_in_denoiser_buffered_tail():
    values = np.zeros((103, 1), dtype=np.float32)
    values[-1] = .8
    processor = SystemAudioProcessor("gtcrn", sample_rate=16000, channels=1,
        _gtcrn_model=DelayedScaleGtcrn(0.0))
    try:
        assert processor.process_block(values) == []
        assert processor.pending_gtcrn_raw_samples == 103
        blocks = processor.finish()
        assert len(blocks) == 1 and blocks[0].start_sample == 0 and blocks[0].end_sample == 103
        np.testing.assert_array_equal(blocks[0].samples, values[:, 0] * .5)
        assert blocks[0].samples[-1] == np.float32(.4)
        assert processor.pending_gtcrn_raw_samples == 0
    finally:
        processor.close()


def test_gtcrn_residual_does_not_wait_for_any_extra_input_beyond_denoiser_output():
    values = np.arange(1024, dtype=np.float32)[:, None] / 2048
    processor = SystemAudioProcessor("gtcrn", sample_rate=16000, channels=1,
        _gtcrn_model=DelayedScaleGtcrn(0.0))
    try:
        assert processor.process_block(values[:511]) == []
        assert processor.pending_gtcrn_raw_samples == 511
        first = processor.process_block(values[511:512])
        assert len(first) == 1 and first[0].input_end_frame == 512
        np.testing.assert_array_equal(first[0].samples, values[:256, 0] * .5)
        second = processor.process_block(values[512:])
        assert len(second) == 1
        assert (second[0].start_sample, second[0].end_sample, second[0].input_end_frame) == (256, 768, 1024)
        np.testing.assert_array_equal(second[0].samples, values[256:768, 0] * .5)
        tail = processor.finish()
        np.testing.assert_array_equal(tail[0].samples, values[768:, 0] * .5)
    finally:
        processor.close()


@pytest.mark.parametrize("mode", ["raw", "mdx", "mdx_gtcrn"])
def test_gtcrn_residual_does_not_change_other_modes_or_their_metadata(mode):
    values = signal(MDX_STEP_SAMPLES + 173, MDX_SAMPLE_RATE)
    processor = SystemAudioProcessor(mode, sample_rate=MDX_SAMPLE_RATE, channels=2,
        _mdx_model=IdentityMdx(), _gtcrn_model=DelayedScaleGtcrn(0.0))
    try:
        output, _ = run_blocks(processor, values, 4093)
        expected = soxr.resample(values.mean(axis=1), MDX_SAMPLE_RATE, 16000)
        if mode == "mdx_gtcrn":
            expected = np.zeros_like(expected)
        np.testing.assert_allclose(output, expected, atol=2e-6)
        assert not any(key.startswith("gtcrn_raw_") or key == "gtcrn_residual_alignment" for key in processor.backend_info)
        assert processor.pending_gtcrn_raw_samples == 0
    finally:
        processor.close()


@pytest.mark.parametrize("mode", ["gtcrn", "gtcrn_preserve"])
@pytest.mark.parametrize("bad", ["too_many", "stalled", "shape", "nonfinite"])
def test_gtcrn_bad_output_does_not_invent_or_mispair_raw_samples(mode, bad):
    class BadGtcrn(DelayedIdentityGtcrn):
        def process(self, samples):
            if bad == "too_many":
                return np.zeros(len(samples) + 1, dtype=np.float32)
            if bad == "stalled":
                return np.empty(0, dtype=np.float32)
            if bad == "shape":
                return np.zeros((1, 1024), dtype=np.float32)
            return np.full(1024, np.nan, dtype=np.float32)

    processor = SystemAudioProcessor(mode, sample_rate=16000, channels=1, _gtcrn_model=BadGtcrn())
    try:
        with pytest.raises(AudioProcessingError):
            processor.process_block(np.full((1024, 1), .2, dtype=np.float32))
        assert processor.output_end_sample == 0 and processor.pending_gtcrn_raw_samples == 0
        with pytest.raises(AudioProcessingError, match="状态无效"):
            processor.process_block(np.zeros((1, 1), dtype=np.float32))
    finally:
        processor.close()


def test_gtcrn_residual_reset_clears_old_source_before_restarting():
    gtcrn = DelayedScaleGtcrn(0.0)
    processor = SystemAudioProcessor("gtcrn", sample_rate=16000, channels=1, _gtcrn_model=gtcrn)
    try:
        assert processor.process_block(np.full((103, 1), .8, dtype=np.float32)) == []
        assert processor.pending_gtcrn_raw_samples == 103
        processor.reset()
        assert processor.pending_gtcrn_raw_samples == 0
        output, blocks = run_blocks(processor, np.full((1027, 1), -.2, dtype=np.float32), 477)
        np.testing.assert_array_equal(output, np.full(1027, -.1, dtype=np.float32))
        assert blocks[0].start_sample == 0 and processor.pending_gtcrn_raw_samples == 0
    finally:
        processor.close()


@pytest.mark.parametrize("mode", ["gtcrn", "gtcrn_preserve"])
@pytest.mark.parametrize("action", ["process", "finish", "reset"])
def test_gtcrn_cancellation_clears_buffered_raw_before_any_further_operation(mode, action):
    stop = Event()
    processor = SystemAudioProcessor(mode, sample_rate=16000, channels=1,
        _gtcrn_model=DelayedScaleGtcrn(0.0), stop_event=stop)
    try:
        assert processor.process_block(np.full((103, 1), .8, dtype=np.float32)) == []
        assert processor.pending_gtcrn_raw_samples == 103
        stop.set()
        with pytest.raises(AudioProcessingCancelled):
            if action == "process":
                processor.process_block(np.zeros((1, 1), dtype=np.float32))
            else:
                getattr(processor, action)()
        assert processor.pending_gtcrn_raw_samples == 0 and processor.output_end_sample == 0
    finally:
        processor.close()


@pytest.mark.parametrize("mode", ["gtcrn", "gtcrn_preserve"])
def test_gtcrn_factory_never_loads_mdx_or_resamples_via_44100(monkeypatch, mode):
    loaded = []
    resamplers = []
    gtcrn = DelayedIdentityGtcrn()

    def load_gtcrn(path):
        loaded.append(path)
        return gtcrn

    def unexpected_mdx(path):
        raise AssertionError("GTCRN-only must not load MDX")

    original_factory = audio_processing._make_resampler

    def direct_resampler(source, target, channels):
        resamplers.append((source, target, channels))
        return original_factory(source, target, channels)

    monkeypatch.setattr(audio_processing, "_MdxModel", unexpected_mdx)
    monkeypatch.setattr(audio_processing, "_GtcrnModel", load_gtcrn)
    monkeypatch.setattr(audio_processing, "_make_resampler", direct_resampler)
    settings = SimpleNamespace(system_audio_processing=mode, gtcrn_model_file="gtcrn-test.onnx",
        mdx_model_file="missing-mdx.onnx")
    processor = create_system_audio_processor(settings, sample_rate=48000, channels=6)
    try:
        assert loaded == ["gtcrn-test.onnx"] and resamplers == [(48000, 16000, 1)]
        info = processor.backend_info
        assert all(value == 0 for key, value in info.items() if key.startswith("mdx_"))
        assert info["input_channels"] == 6 and info["output_channels"] == 1
        assert info["gtcrn_frame_shift_samples"] == 256
        assert info["gtcrn_first_output_input_samples"] == 512
        assert info["gtcrn_steady_pending_samples"] == 256
        if mode == "gtcrn":
            assert info["gtcrn_raw_residual_fraction"] == .5
        else:
            assert "gtcrn_raw_residual_fraction" not in info
            assert info["asr_audio_source"] == "hq_resampled_raw"
            assert info["vad_audio_sources"] == ["hq_resampled_raw"]
            assert info["diagnostic_audio_sources"] == ["pure_gtcrn"]
        assert info["gtcrn_raw_pending_limit_samples"] == 511
        assert info["resampler_quality"] == "HQ"
        json.dumps(info)
    finally:
        processor.close()
    assert gtcrn.closed


@pytest.mark.parametrize("mode", ["mdx", "mdx_gtcrn"])
@pytest.mark.parametrize("count", [0, 1, MDX_STEP_SAMPLES - 1, MDX_STEP_SAMPLES, MDX_STEP_SAMPLES * 2 + 147])
def test_mdx_identity_window_trim_is_contiguous_without_duplicate_or_lost_tail(mode, count):
    values = signal(count, MDX_SAMPLE_RATE)
    mdx = IdentityMdx()
    gtcrn = DelayedIdentityGtcrn() if mode == "mdx_gtcrn" else None
    processor = SystemAudioProcessor(mode, sample_rate=MDX_SAMPLE_RATE, channels=2,
        _mdx_model=mdx, _gtcrn_model=gtcrn)
    try:
        output, blocks = run_blocks(processor, values, 4093)
        expected = soxr.resample(values.mean(axis=1), MDX_SAMPLE_RATE, 16000) if count else np.empty(0)
        assert len(output) == round(count * 16000 / MDX_SAMPLE_RATE)
        np.testing.assert_allclose(output, expected, atol=2e-6)
        assert mdx.calls == (count + MDX_STEP_SAMPLES - 1) // MDX_STEP_SAMPLES
        if count:
            assert np.count_nonzero(mdx.first_window[:, :MDX_TRIM_SAMPLES]) == 0
        if gtcrn is not None:
            assert gtcrn.flush_count == 1
    finally:
        processor.close()
    assert mdx.closed and (gtcrn is None or gtcrn.closed)


def test_real_stft_istft_full_spectrum_preserves_phase_and_scale():
    values = np.random.default_rng(741).normal(0, .02, (2, MDX_WINDOW_SAMPLES)).astype(np.float32)
    values[0, 0], values[1, -1] = .5, -.4
    reconstructed = _istft(_stft(values))
    np.testing.assert_allclose(reconstructed, values, atol=2e-6)


@pytest.mark.parametrize("rate", [16000, 48000])
def test_cascaded_streaming_resampling_preserves_origin_and_terminal_length(rate):
    values = signal(rate * 7 + 173, rate)
    processor = SystemAudioProcessor("mdx", sample_rate=rate, channels=2, _mdx_model=IdentityMdx())
    try:
        output, blocks = run_blocks(processor, values, rate // 25)
        expected = soxr.resample(soxr.resample(values, rate, MDX_SAMPLE_RATE).mean(axis=1), MDX_SAMPLE_RATE, 16000)
        assert len(output) == round(len(values) * 16000 / rate)
        common = min(len(expected), len(output))
        np.testing.assert_allclose(output[:common], expected[:common], atol=2e-6)
        assert blocks[0].start_sample == 0 and blocks[-1].input_end_frame == len(values)
    finally:
        processor.close()


def test_mdx_preserves_stereo_until_inference_instead_of_early_mono_cancellation():
    left = np.linspace(-.1, .1, MDX_WINDOW_SAMPLES, dtype=np.float32)
    values = np.stack((left, -left), axis=1)
    mdx = IdentityMdx()
    processor = SystemAudioProcessor("mdx", sample_rate=MDX_SAMPLE_RATE, channels=2, _mdx_model=mdx)
    try:
        output, _ = run_blocks(processor, values, 16000)
        assert np.max(np.abs(mdx.first_window[0])) > .05
        np.testing.assert_allclose(mdx.first_window[0], -mdx.first_window[1])
        np.testing.assert_allclose(output, 0, atol=1e-7)
    finally:
        processor.close()


def test_mono_input_is_duplicated_for_mdx_without_changing_amplitude():
    values = signal(1203, MDX_SAMPLE_RATE, channels=1)
    mdx = IdentityMdx()
    processor = SystemAudioProcessor("mdx", sample_rate=MDX_SAMPLE_RATE, channels=1, _mdx_model=mdx)
    try:
        output, _ = run_blocks(processor, values, 127)
        np.testing.assert_allclose(mdx.first_window[0], mdx.first_window[1])
        np.testing.assert_allclose(output, soxr.resample(values[:, 0], MDX_SAMPLE_RATE, 16000), atol=2e-6)
    finally:
        processor.close()


@pytest.mark.parametrize("mode", ["raw", "gtcrn", "gtcrn_preserve", "mdx", "mdx_gtcrn"])
def test_reset_discards_old_audio_and_restarts_source_timeline(mode):
    gtcrn = DelayedIdentityGtcrn()
    processor = SystemAudioProcessor(mode, sample_rate=16000, channels=2,
        _mdx_model=IdentityMdx(), _gtcrn_model=gtcrn)
    try:
        processor.process_block(np.full((1001, 2), .4, dtype=np.float32))
        processor.reset()
        assert processor.input_end_frame == 0 and processor.output_end_sample == 0
        output, blocks = run_blocks(processor, np.zeros((3197, 2), dtype=np.float32), 479)
        assert len(output) == 3197 and np.count_nonzero(output) == 0
        assert blocks[0].start_sample == 0
        processor.reset()
        assert processor.finish() == []
    finally:
        processor.close()


@pytest.mark.parametrize("mode", ["gtcrn", "gtcrn_preserve", "mdx_gtcrn"])
def test_close_drops_pending_tail_without_a_model_run_or_flush(mode):
    mdx, gtcrn = IdentityMdx(), DelayedIdentityGtcrn()
    processor = SystemAudioProcessor(mode, sample_rate=16000, channels=1,
        _mdx_model=mdx, _gtcrn_model=gtcrn)
    processor.process_block(np.full((1000, 1), .1, dtype=np.float32))
    processor.close()
    processor.close()
    assert mdx.calls == 0 and gtcrn.flush_count == 0
    assert (mdx.closed if mode == "mdx_gtcrn" else not mdx.closed)
    assert gtcrn.closed and processor.pending_mdx_samples == 0
    with pytest.raises(AudioProcessingError):
        processor.finish()


@pytest.mark.parametrize("where", ["before", "mdx", "gtcrn"])
def test_cancellation_around_native_calls_never_returns_late_audio(where):
    stop = Event()
    mdx = IdentityMdx(stop if where == "mdx" else None)
    gtcrn = DelayedIdentityGtcrn(stop if where == "gtcrn" else None)
    processor = SystemAudioProcessor("mdx_gtcrn", sample_rate=16000, channels=1,
        _mdx_model=mdx, _gtcrn_model=gtcrn, stop_event=stop)
    try:
        if where == "before":
            stop.set()
        with pytest.raises(AudioProcessingCancelled):
            for _ in range(7):
                assert processor.process_block(np.zeros((16000, 1), dtype=np.float32)) == []
        assert processor.output_end_sample == 0
    finally:
        processor.close()
    assert mdx.closed and gtcrn.closed


@pytest.mark.parametrize("mode", ["gtcrn", "gtcrn_preserve"])
@pytest.mark.parametrize("where", ["before", "gtcrn"])
def test_gtcrn_only_cancellation_never_delivers_late_audio(mode, where):
    stop = Event()
    gtcrn = DelayedIdentityGtcrn(stop if where == "gtcrn" else None)
    processor = SystemAudioProcessor(mode, sample_rate=16000, channels=1,
        _gtcrn_model=gtcrn, stop_event=stop)
    try:
        if where == "before":
            stop.set()
        with pytest.raises(AudioProcessingCancelled):
            processor.process_block(np.ones((1024, 1), dtype=np.float32))
        assert processor.output_end_sample == 0
    finally:
        processor.close()
    assert gtcrn.closed and gtcrn.flush_count == 0


@pytest.mark.parametrize("values", [
    np.zeros((16001, 2), dtype=np.float32), np.zeros((1, 1), dtype=np.float32),
    np.zeros(2, dtype=np.float32), np.zeros((2, 2), dtype=np.int16),
    np.full((1, 2), np.nan, dtype=np.float32), np.full((1, 2), np.inf, dtype=np.float32),
    np.full((1, 2), 1e300, dtype=np.float64),
])
def test_invalid_or_unbounded_input_is_rejected_before_consumption(values):
    processor = SystemAudioProcessor("raw", sample_rate=16000, channels=2)
    try:
        with pytest.raises(ValueError):
            processor.process_block(values)
        assert processor.input_end_frame == 0 and processor.output_end_sample == 0
        output, _ = run_blocks(processor, np.zeros((1, 2), dtype=np.float32), 1)
        assert len(output) == 1
    finally:
        processor.close()


@pytest.mark.parametrize("kwargs", [
    {"mode": "other", "sample_rate": 16000, "channels": 1},
    {"mode": "raw", "sample_rate": True, "channels": 1},
    {"mode": "raw", "sample_rate": 7999, "channels": 1},
    {"mode": "raw", "sample_rate": 16000, "channels": 0},
    {"mode": "mdx", "sample_rate": 48000, "channels": 6},
])
def test_invalid_format_does_not_attempt_model_loading(kwargs):
    with pytest.raises(ValueError):
        SystemAudioProcessor(**kwargs)


def test_factory_raw_default_is_backwards_compatible_and_metadata_is_json():
    processor = create_system_audio_processor(SimpleNamespace(), sample_rate=16000, channels=2)
    try:
        assert processor.mode == "raw"
        json.dumps(processor.backend_info)
    finally:
        processor.close()


def test_module_import_does_not_import_native_audio_or_inference_packages():
    project = Path(__file__).resolve().parents[1]
    code = ("import sys;sys.path.insert(0,'src');import fanyiji_next.audio_processing;"
            "assert not {'numpy','soxr','onnxruntime','sherpa_onnx','faster_whisper'} & set(sys.modules)")
    completed = subprocess.run([sys.executable, "-B", "-c", code], cwd=project, capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr


@pytest.mark.parametrize("bad", ["shape", "nonfinite"])
def test_bad_native_mdx_output_poisoned_until_reset_and_never_delivered(bad):
    class BadMdx(IdentityMdx):
        def process_window(self, samples):
            return np.zeros((2, 1), dtype=np.float32) if bad == "shape" else np.full_like(samples, np.nan)

    mdx = BadMdx()
    processor = SystemAudioProcessor("mdx", sample_rate=16000, channels=1, _mdx_model=mdx)
    try:
        processor.process_block(np.zeros((1000, 1), dtype=np.float32))
        with pytest.raises(AudioProcessingError):
            processor.finish()
        assert processor.output_end_sample == 0
        with pytest.raises(AudioProcessingError):
            processor.process_block(np.zeros((1, 1), dtype=np.float32))
    finally:
        processor.close()
    assert mdx.closed


@pytest.mark.parametrize("mode", ["gtcrn", "gtcrn_preserve", "mdx_gtcrn"])
def test_bad_native_gtcrn_tail_length_is_reported_instead_of_silent_alignment_shift(mode):
    class BadGtcrn(DelayedIdentityGtcrn):
        def finish(self):
            return np.r_[super().finish(), np.float32(.01)]

    processor = SystemAudioProcessor(mode, sample_rate=16000, channels=1,
        _mdx_model=IdentityMdx(), _gtcrn_model=BadGtcrn())
    try:
        processor.process_block(np.zeros((1000, 1), dtype=np.float32))
        with pytest.raises(AudioProcessingError, match="样本总数"):
            processor.finish()
    finally:
        processor.close()


def test_gtcrn_close_failure_still_releases_mdx_and_buffers():
    class BadCloseGtcrn(DelayedIdentityGtcrn):
        def close(self):
            self.closed = True
            raise RuntimeError("native close failed")

    mdx, gtcrn = IdentityMdx(), BadCloseGtcrn()
    processor = SystemAudioProcessor("mdx_gtcrn", sample_rate=16000, channels=1,
        _mdx_model=mdx, _gtcrn_model=gtcrn)
    with pytest.raises(RuntimeError, match="native close"):
        processor.close()
    assert mdx.closed and gtcrn.closed and processor.pending_mdx_samples == 0
    processor.close()


@pytest.mark.parametrize("mode", ["gtcrn", "gtcrn_preserve", "mdx_gtcrn"])
def test_cancel_during_gtcrn_flush_does_not_deliver_tail(mode):
    stop = Event()

    class CancelFlushGtcrn(DelayedIdentityGtcrn):
        def finish(self):
            stop.set()
            return super().finish()

    processor = SystemAudioProcessor(mode, sample_rate=16000, channels=1,
        _mdx_model=IdentityMdx(), _gtcrn_model=CancelFlushGtcrn(), stop_event=stop)
    try:
        processor.process_block(np.zeros((103, 1), dtype=np.float32))
        with pytest.raises(AudioProcessingCancelled):
            processor.finish()
        assert processor.output_end_sample == 0
    finally:
        processor.close()


@pytest.mark.parametrize("rate", [8000, 16000, 44100, 48000, 96000])
@pytest.mark.parametrize("count", [0, 1, 255, 256, 511, 513, 16037])
@pytest.mark.parametrize("factor", [0.0, -.25])
def test_preserve_mode_asr_bytes_equal_raw_mode_and_auxiliary_is_unmixed(rate, count, factor):
    values = signal(count, rate)
    raw = SystemAudioProcessor("raw", sample_rate=rate, channels=2)
    gtcrn = DelayedScaleGtcrn(factor)
    preserved = SystemAudioProcessor("gtcrn_preserve", sample_rate=rate, channels=2, _gtcrn_model=gtcrn)
    try:
        expected, _ = run_blocks(raw, values, 479)
        blocks = []
        for first in range(0, count, 479):
            blocks.extend(preserved.process_block(values[first:first+479]))
            assert preserved.pending_gtcrn_raw_samples <= GTCRN_MAX_PENDING_SAMPLES
        blocks.extend(preserved.finish())
        actual = np.concatenate([block.samples for block in blocks]) if blocks else np.empty(0, np.float32)
        denoised = np.concatenate([block.denoised_samples for block in blocks]) if blocks else np.empty(0, np.float32)
        assert actual.tobytes() == expected.tobytes()
        assert denoised.tobytes() == (expected * factor).tobytes()
        cursor = 0
        for block in blocks:
            assert block.start_sample == cursor
            assert block.end_sample == cursor + len(block.samples)
            assert block.raw_samples is block.samples
            assert block.raw_samples.dtype == block.denoised_samples.dtype == np.float32
            assert block.raw_samples.flags.c_contiguous and block.denoised_samples.flags.c_contiguous
            assert len(block.raw_samples) == len(block.denoised_samples)
            assert not np.shares_memory(block.raw_samples, block.denoised_samples)
            cursor = block.end_sample
        assert cursor == round(count * 16000 / rate)
        assert preserved.pending_gtcrn_raw_samples == 0
        assert preserved.finish() == [] and gtcrn.flush_count == 1
    finally:
        raw.close()
        preserved.close()


def test_preserve_mode_short_weak_utterance_survives_total_auxiliary_suppression():
    # A suppressed detector view must never erase/scale the separate ASR view.
    values = np.array([[-1e-8], [3e-8], [-0.0], [.2], [-.7]], np.float32)
    processor = SystemAudioProcessor("gtcrn_preserve", sample_rate=16000, channels=1,
        _gtcrn_model=DelayedScaleGtcrn(0.0))
    try:
        assert processor.process_block(values) == []
        assert processor.pending_gtcrn_raw_samples == len(values)
        blocks = processor.finish()
        assert len(blocks) == 1 and blocks[0].input_end_frame == len(values)
        assert blocks[0].samples.tobytes() == values.mean(axis=1).tobytes()
        assert np.count_nonzero(blocks[0].denoised_samples) == 0
    finally:
        processor.close()


def test_preserve_mode_missing_native_tail_is_explicit_failure_not_padded_success():
    class MissingTailGtcrn(DelayedIdentityGtcrn):
        def finish(self):
            return super().finish()[:-1]

    processor = SystemAudioProcessor("gtcrn_preserve", sample_rate=16000, channels=1,
        _gtcrn_model=MissingTailGtcrn())
    try:
        assert processor.process_block(np.full((103, 1), .01, np.float32)) == []
        with pytest.raises(AudioProcessingError, match="样本总数"):
            processor.finish()
        assert processor.pending_gtcrn_raw_samples == 0
        with pytest.raises(AudioProcessingError, match="状态无效"):
            processor.process_block(np.zeros((1, 1), np.float32))
    finally:
        processor.close()


@pytest.mark.parametrize("mode", ["raw", "gtcrn", "mdx", "mdx_gtcrn"])
def test_legacy_audio_views_remain_unset_and_four_position_constructor_is_valid(mode):
    values = signal(1031, 16000)
    processor = SystemAudioProcessor(mode, sample_rate=16000, channels=2,
        _mdx_model=IdentityMdx(), _gtcrn_model=DelayedIdentityGtcrn())
    try:
        _values, blocks = run_blocks(processor, values, 479)
        assert blocks and all(block.raw_samples is None and block.denoised_samples is None for block in blocks)
        old_constructor = ProcessedAudio(values[:, 0], 0, len(values), len(values))
        assert old_constructor.raw_samples is None and old_constructor.denoised_samples is None
    finally:
        processor.close()
