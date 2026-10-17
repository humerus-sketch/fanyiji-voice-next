"""Small Silero adapter; only construction imports the optional native engine."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math
from pathlib import Path
from typing import Sequence


SAMPLE_RATE = 16000
WINDOW_SIZE = 512
SILENCE_SECONDS = 0.35
SYSTEM_SILENCE_SECONDS = SILENCE_SECONDS
SYSTEM_LEADING_SECONDS = 0.12
SYSTEM_TRAILING_SECONDS = 0.16
# Native v1.13.8 starts at Tail - 2*WindowSize - MinSpeechSamples.
# V5 has a 64-sample overlap and can retain one unprocessed 512-sample
# shift. Keep that complete bound while an onset is still unconfirmed.
SOURCE_LOOKBACK_SAMPLES = 2 * (WINDOW_SIZE + 64) + round(.18 * SAMPLE_RATE) + WINDOW_SIZE


@dataclass(frozen=True)
class SpeechSlice:
    samples: list[float]
    start_sample: int
    end_sample: int
    end_reason: str
    speech_start_sample: int | None = None
    speech_end_sample: int | None = None


class SileroSpeechGate:
    """Return finalized speech with positions in one continuous 16 kHz stream.

    The detector already waits for silence and removes that waiting period from
    its completed segment. Never add another energy-gate hangover afterwards.
    Instances belong to one DSP consumer and must not be shared across sources.
    """

    def __init__(self, model: Path, *, detector=None, silence_seconds=SILENCE_SECONDS,
                 leading_seconds=0.0, trailing_seconds=0.0) -> None:
        for name, value in (("silence_seconds", silence_seconds), ("leading_seconds", leading_seconds),
                            ("trailing_seconds", trailing_seconds)):
            if not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value < 6:
                raise ValueError(f"{name} must be finite and in [0, 6)")
        if silence_seconds == 0:
            raise ValueError("silence_seconds must be positive")
        self.silence_seconds = silence_seconds
        self._leading_samples = round(leading_seconds * SAMPLE_RATE)
        self._trailing_samples = round(trailing_seconds * SAMPLE_RATE)
        self._with_context = bool(self._leading_samples or self._trailing_samples)
        if detector is None:
            import sherpa_onnx

            config = sherpa_onnx.VadModelConfig()
            config.silero_vad.model = str(model)
            config.silero_vad.threshold = 0.5
            config.silero_vad.min_silence_duration = silence_seconds
            config.silero_vad.min_speech_duration = 0.18
            config.silero_vad.window_size = WINDOW_SIZE
            # This native setting changes its threshold for long utterances;
            # it is not a hard length limit. Enforce our own six-second cap.
            config.silero_vad.max_speech_duration = 20.0
            config.sample_rate = SAMPLE_RATE
            config.num_threads = 1
            config.provider = "cpu"
            detector = sherpa_onnx.VoiceActivityDetector(config, buffer_size_in_seconds=15)
        self._detector = detector
        self.max_samples = 6 * SAMPLE_RATE
        self._pending: list[float] = []
        self._received = 0
        self._processed = 0
        self._native_offset = 0
        self._active_start: int | None = None
        # A forced cut needs real pre-cut speech to warm the reset detector;
        # otherwise its minimum-speech check can discard a short continuation.
        self._warm_samples = math.ceil((silence_seconds * SAMPLE_RATE + .18 * SAMPLE_RATE
                                       + 2 * WINDOW_SIZE) / WINDOW_SIZE) * WINDOW_SIZE
        history_limit = (self.max_samples + self._leading_samples + self._trailing_samples
                         + math.ceil(silence_seconds * SAMPLE_RATE) + WINDOW_SIZE)
        self._history = deque(maxlen=max(history_limit if self._with_context else 0, self._warm_samples))
        self._emitted_end = 0

    @property
    def processed_samples(self) -> int:
        """Real source samples processed; excludes finish()'s VAD-only zeros."""
        return min(self._received, self._processed)

    @property
    def active_start_sample(self) -> int | None:
        """Current source onset, including the forced-cut native offset."""
        return self._active_start

    @property
    def safe_frontier_sample(self) -> int:
        """No future completed segment can begin before this source position."""
        if self._active_start is not None:
            return self._active_start
        return max(self._emitted_end, 0, self.processed_samples - SOURCE_LOOKBACK_SAMPLES)

    def reset(self) -> None:
        """Discard unfinished speech and all sample positions after an input gap."""
        self._detector.reset()
        self._pending.clear()
        self._received = 0
        self._processed = 0
        self._native_offset = 0
        self._active_start = None
        self._history.clear()
        self._emitted_end = 0

    def feed(self, samples: Sequence[float]) -> list[SpeechSlice]:
        self._received += len(samples)
        self._pending.extend(samples)
        completed = []
        while len(self._pending) >= WINDOW_SIZE:
            block = self._pending[:WINDOW_SIZE]
            del self._pending[:WINDOW_SIZE]
            completed.extend(self._accept(block))
        return completed

    def finish(self) -> list[SpeechSlice]:
        """Finalize a source that stopped producing callbacks, then reset.

        Missing digital-silence callbacks are represented only inside the VAD.
        Clip emitted slices to real received samples; never send padding to ASR.
        A user stop deliberately calls reset(), not this method.
        """
        completed = []
        if self._pending:
            block = self._pending + [0.0] * (WINDOW_SIZE - len(self._pending))
            self._pending.clear()
            completed.extend(self._accept(block, "input_gap"))
        for _ in range(math.ceil(self.silence_seconds * SAMPLE_RATE / WINDOW_SIZE) + 2):
            completed.extend(self._accept([0.0] * WINDOW_SIZE, "input_gap"))
        self._detector.flush()
        completed.extend(self._drain("input_gap"))
        self.reset()
        return completed

    def _accept(self, block: Sequence[float], reason: str = "vad_silence") -> list[SpeechSlice]:
        real_count = max(0, min(len(block), self._received - self._processed))
        self._history.extend(block[:real_count])
        self._detector.accept_waveform(block)
        self._processed += len(block)
        completed = self._drain(reason)
        if not self._detector.is_speech_detected():
            self._active_start = None
            return completed
        if self._active_start is None:
            # Read the growing segment once at onset, not on every callback.
            self._active_start = max(self._emitted_end,
                self._native_offset + int(self._detector.current_segment.start))
        if self._processed - self._active_start >= self.max_samples - WINDOW_SIZE:
            # Cut at a window boundary just before six seconds (at most 32 ms
            # early), so the native segment can never exceed the application cap.
            self._detector.flush()
            completed.extend(self._drain("max_duration" if reason == "vad_silence" else reason))
            self._reset_native_after_cut(warm=reason == "vad_silence")
        return completed

    def _reset_native_after_cut(self, *, warm: bool) -> None:
        # Native flush clears its segment queue, but leaves Silero's triggered
        # and temporary-end state alive. Reset only native state, keeping the
        # source clock, pending input and real history owned by this adapter.
        self._detector.reset()
        self._active_start = None
        self._native_offset = self._processed
        if not warm:
            return
        count = min(self._warm_samples, len(self._history))
        count -= count % WINDOW_SIZE
        history = list(self._history)[-count:] if count else []
        self._native_offset = self._processed - count
        for first in range(0, count, WINDOW_SIZE):
            # Do not call _accept: these already-received source samples must
            # not advance counters or reach ASR a second time.
            self._detector.accept_waveform(history[first:first + WINDOW_SIZE])
            if self._drain("max_duration"):
                raise RuntimeError("VAD 重放历史意外产生了未交付的源音频。")
        if self._detector.is_speech_detected():
            self._active_start = max(self._emitted_end,
                self._native_offset + int(self._detector.current_segment.start))

    def _drain(self, reason: str) -> list[SpeechSlice]:
        completed = []
        while not self._detector.empty():
            segment = self._detector.front
            native_start = int(segment.start)
            start = self._native_offset + native_start
            # front is invalidated by pop/accept/reset, so copy before pop.
            samples = list(segment.samples)
            self._detector.pop()
            if native_start < 0 or start < 0:
                raise RuntimeError("VAD 返回了负数语音起点，不能继续使用失效的源时间轴。")
            if not samples:
                if reason == "input_gap" and start >= self._received:
                    continue
                raise RuntimeError("VAD 返回了空语音段，可能存在原生断句状态异常，已停止以免静默漏句。")
            speech_end = min(start + len(samples), self._received)
            if start >= self._received:
                if reason == "input_gap":
                    continue
                raise RuntimeError("VAD 语音段超出了实际收到的源音频。")
            if speech_end <= self._emitted_end:
                continue
            clipped_start = max(start, self._emitted_end)
            count = speech_end - clipped_start
            samples = samples[clipped_start - start:speech_end - start]
            continuation = clipped_start > start
            start = clipped_start
            # Native endpoints can contain a window of silence after a forced
            # cut. Exact digital zero alone must not become a new utterance;
            # there is deliberately no energy floor that could erase weak tails.
            if continuation and self._native_offset and not any(samples):
                continue
            if count > self.max_samples:
                raise RuntimeError("VAD 返回的语音段超过六秒保护上限。")
            if self._with_context:
                real_end = min(self._received, self._processed)
                history_start = real_end - len(self._history)
                if history_start > start:
                    raise RuntimeError("VAD 原声缓冲不足，不能无声丢弃语音起头。")
                # Retain a small amount of real source around system speech,
                # including weak endings. Padding never replaces original
                # speech, crosses the six-second cap, or repeats emitted audio.
                padded_start = max(history_start, self._emitted_end,
                                   start - self._leading_samples, speech_end - self.max_samples)
                padded_end = min(real_end, speech_end + self._trailing_samples,
                                 padded_start + self.max_samples)
                if padded_end <= padded_start:
                    continue
                samples = list(self._history)[padded_start - history_start:padded_end - history_start]
                completed.append(SpeechSlice(samples, padded_start, padded_end, reason, start, speech_end))
                self._emitted_end = padded_end
            else:
                completed.append(SpeechSlice(samples[:count], start, speech_end, reason))
                self._emitted_end = speech_end
        return completed
