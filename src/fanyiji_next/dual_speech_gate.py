"""Source-aligned raw speech delivery with GTCRN VAD diagnostics only."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math
from pathlib import Path
from typing import Sequence

from .speech_gate import (SAMPLE_RATE, SILENCE_SECONDS, SYSTEM_LEADING_SECONDS,
                          SYSTEM_TRAILING_SECONDS, SileroSpeechGate, SpeechSlice)

DIAGNOSTIC_HISTORY_LIMIT = 64


@dataclass(frozen=True)
class SpeechDiagnostic:
    epoch: int
    start_sample: int
    end_sample: int
    speech_start_sample: int
    speech_end_sample: int
    end_reason: str


class DualSpeechGate:
    """Return the raw gate's original slices; denoised VAD never submits ASR.

    GTCRN decisions are bounded diagnostics, not extra speech, a sentence
    boundary veto, an accuracy score or evidence that raw VAD misses no speech.
    The raw gate owns its existing source history and context. This adapter
    neither mixes audio nor retains a second raw ring nor merges segments.
    """

    recognition_policy = "canonical_raw"

    def __init__(self, model: Path, *, raw_gate=None, denoised_gate=None,
                 silence_seconds=SILENCE_SECONDS,
                 leading_seconds=SYSTEM_LEADING_SECONDS,
                 trailing_seconds=SYSTEM_TRAILING_SECONDS) -> None:
        for name, value in (("silence_seconds", silence_seconds),
                            ("leading_seconds", leading_seconds),
                            ("trailing_seconds", trailing_seconds)):
            if not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value < 6:
                raise ValueError(f"{name} must be finite and in [0, 6)")
        if silence_seconds == 0:
            raise ValueError("silence_seconds must be positive")
        self.silence_seconds = silence_seconds
        self.max_samples = 6 * SAMPLE_RATE
        self._gates = (raw_gate if raw_gate is not None else SileroSpeechGate(
            model, silence_seconds=silence_seconds, leading_seconds=leading_seconds,
            trailing_seconds=trailing_seconds),
            denoised_gate if denoised_gate is not None else SileroSpeechGate(
            model, silence_seconds=silence_seconds))
        if self._gates[0] is self._gates[1]:
            raise ValueError("Raw and denoised VAD must use separate state")
        self._received = self._delivered_end = self._epoch = 0
        self._failed = False
        self._diagnostics: deque[SpeechDiagnostic] = deque()
        self._diagnostic_total = self._diagnostic_discarded = 0

    @property
    def received_samples(self) -> int:
        return self._received

    @property
    def diagnostic_counts(self) -> dict[str, int]:
        return {"detected_total": self._diagnostic_total,
                "discarded_before_drain": self._diagnostic_discarded,
                "pending": len(self._diagnostics)}

    def take_diagnostics(self) -> list[SpeechDiagnostic]:
        """Drain bounded interval metadata. No denoised audio is retained."""
        output = list(self._diagnostics)
        self._diagnostics.clear()
        return output

    def reset(self) -> None:
        """User Stop or a source gap discards pending speech and diagnostics."""
        error = None
        for gate in self._gates:
            try:
                gate.reset()
            except Exception as exc:
                if error is None:
                    error = exc
        self._received = self._delivered_end = self._epoch = 0
        self._diagnostics.clear()
        self._diagnostic_total = self._diagnostic_discarded = 0
        self._failed = error is not None
        if error is not None:
            raise error

    def feed(self, raw_samples: Sequence[float], denoised_samples: Sequence[float],
             *, start_sample: int | None = None) -> list[SpeechSlice]:
        if self._failed:
            raise RuntimeError("Dual VAD failed; reset is required before reuse")
        if len(raw_samples) != len(denoised_samples):
            raise ValueError("Raw/denoised source intervals must have equal lengths")
        if start_sample is not None and start_sample != self._received:
            raise ValueError("Non-contiguous source interval; reset before an input gap")
        if any(not math.isfinite(value) for values in (raw_samples, denoised_samples) for value in values):
            raise ValueError("Dual VAD requires finite mono samples")
        try:
            self._received += len(raw_samples)
            raw_completed = self._gates[0].feed(raw_samples)
            self._record_diagnostics(self._gates[1].feed(denoised_samples))
            if self._gates[0].processed_samples != self._gates[1].processed_samples:
                raise RuntimeError("Raw and denoised VAD source clocks diverged")
            self._validate_raw(raw_completed)
            return raw_completed
        except Exception:
            self._failed = True
            raise

    def finish(self) -> list[SpeechSlice]:
        """Finish real source once; keep EOS diagnostics available to drain."""
        if self._failed:
            raise RuntimeError("Dual VAD failed; reset is required before reuse")
        try:
            raw_completed = self._gates[0].finish()
            self._record_diagnostics(self._gates[1].finish())
            self._validate_raw(raw_completed)
            # Inner finish resets its own clocks; do not clear final diagnostics
            # before their consumer can take them. Epoch disambiguates new input.
            self._received = self._delivered_end = 0
            self._epoch += 1
            return raw_completed
        except Exception:
            self._failed = True
            raise

    def _validate_raw(self, slices: Sequence[SpeechSlice]) -> None:
        cursor = self._delivered_end
        for segment in slices:
            if not (cursor <= segment.start_sample < segment.end_sample <= self._received
                    and len(segment.samples) == segment.end_sample - segment.start_sample
                    and len(segment.samples) <= self.max_samples):
                raise RuntimeError("Raw VAD returned invalid or overlapping source coordinates")
            cursor = segment.end_sample
        self._delivered_end = cursor

    def _record_diagnostics(self, slices: Sequence[SpeechSlice]) -> None:
        for segment in slices:
            start = segment.start_sample if segment.speech_start_sample is None else segment.speech_start_sample
            end = segment.end_sample if segment.speech_end_sample is None else segment.speech_end_sample
            if not 0 <= segment.start_sample <= start < end <= segment.end_sample <= self._received:
                raise RuntimeError("Denoised VAD returned invalid diagnostic source coordinates")
            self._diagnostic_total += 1
            if len(self._diagnostics) == DIAGNOSTIC_HISTORY_LIMIT:
                self._diagnostics.popleft()
                self._diagnostic_discarded += 1
            self._diagnostics.append(SpeechDiagnostic(self._epoch, segment.start_sample,
                segment.end_sample, start, end, segment.end_reason))
