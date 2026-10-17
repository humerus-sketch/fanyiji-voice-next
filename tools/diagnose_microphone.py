"""Opt-in numeric-only capture diagnostics. Never saves audio, text, or settings."""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import json
import math
from pathlib import Path
from queue import Empty, Full, Queue
import sys
from threading import Event, get_ident
import time

import numpy as np


ENERGY_THRESHOLD = .012
SILENCE_SECONDS = .55
ANALYSIS_RATE = 16000
MAX_LAG_MS = 250
NEAR_SILENCE_RMS = .001


def _rms(values):
    return float(np.sqrt(np.mean(np.square(values, dtype=np.float64)))) if len(values) else 0.0


def _number(value):
    return round(float(value), 8) if math.isfinite(float(value)) else None


def energy_metrics(samples, rate):
    """Normalized sample metrics; DC is separate from the Hann-windowed spectrum."""
    samples = np.asarray(samples, dtype=np.float64)
    if samples.ndim == 1:
        samples = samples[:, None]
    mono = samples.mean(axis=1)
    channel_rms = np.sqrt(np.mean(samples * samples, axis=0))
    channel_dc = samples.mean(axis=0)
    correlation = None
    if samples.shape[1] == 2:
        centered = samples - channel_dc
        denominator = np.sqrt(np.sum(centered[:, 0] ** 2) * np.sum(centered[:, 1] ** 2))
        if denominator > 1e-12:
            correlation = _number(np.clip(np.sum(centered[:, 0] * centered[:, 1]) / denominator, -1, 1))
    bands, flatness = None, None
    if len(mono) > 3:
        centered = mono - mono.mean()
        power = np.abs(np.fft.rfft(centered * np.hanning(len(mono)))) ** 2
        frequencies = np.fft.rfftfreq(len(mono), 1 / rate)
        positive = frequencies > 0
        total = float(power[positive].sum())
        if _rms(centered) > 1e-10 and total > 1e-20:
            edges = ((0, 80), (80, 300), (300, 3400), (3400, float("inf")))
            names = ("below_80_hz", "80_300_hz", "300_3400_hz", "3400_nyquist_hz")
            bands = {name: _number(power[positive & (frequencies >= lo) & (frequencies < hi)].sum() / total)
                     for name, (lo, hi) in zip(names, edges)}
            spectrum = np.maximum(power[positive], total * 1e-20)
            flatness = _number(np.exp(np.mean(np.log(spectrum))) / np.mean(spectrum))
    return {"frames": len(mono), "raw_rms": _number(_rms(samples)), "mono_rms": _number(_rms(mono)),
            "raw_dc": _number(samples.mean()), "mono_dc": _number(mono.mean()),
            "raw_peak": _number(np.max(np.abs(samples))),
            "clip_fraction": _number(np.mean(np.abs(samples) >= .999)),
            "channel_rms": [_number(value) for value in channel_rms],
            "channel_dc": [_number(value) for value in channel_dc], "stereo_correlation": correlation,
            "mono_dc_removed_band_power_fraction": bands, "mono_spectral_flatness": flatness}


def low_energy_statistics(samples, valid):
    window = ANALYSIS_RATE // 50
    count = len(samples) // window
    if not count:
        return {"valid_windows": 0, "longest_low_energy_seconds": 0.0, "silence_runs": 0}
    blocks = samples[:count * window].reshape(count, window)
    usable = valid[:count * window].reshape(count, window).all(axis=1)
    low = (np.sqrt(np.mean(blocks.astype(np.float64) ** 2, axis=1)) < ENERGY_THRESHOLD) & usable
    longest = current = runs = 0
    for value in np.append(low, False):
        if value:
            current += 1
            longest = max(longest, current)
        else:
            runs += current * .02 >= SILENCE_SECONDS
            current = 0
    return {"threshold_rms": ENERGY_THRESHOLD, "silence_seconds": SILENCE_SECONDS,
            "window_seconds": .02, "valid_windows": int(usable.sum()), "low_energy_windows": int(low.sum()),
            "longest_low_energy_seconds": round(longest * .02, 3), "silence_runs": int(runs)}


def _longest_valid_run(valid):
    edges = np.flatnonzero(np.diff(np.r_[False, valid, False].astype(np.int8)))
    if not len(edges):
        return 0, 0
    lengths = edges[1::2] - edges[::2]
    best = int(np.argmax(lengths))
    return int(edges[2 * best]), int(edges[2 * best + 1])


def normalized_cross_correlation(reference, compared, reference_valid, compared_valid, *, rate=ANALYSIS_RATE):
    """Positive lag means compared follows reference on their shared callback-QPC timeline."""
    start, end = _longest_valid_run(reference_valid & compared_valid)
    x = np.array(reference[start:end], dtype=np.float64, copy=True)
    y = np.array(compared[start:end], dtype=np.float64, copy=True)
    result = {"best_correlation": None, "best_absolute_correlation": None, "lag_ms": None,
              "effective_seconds": 0.0, "aligned_common_seconds": round(len(x) / rate, 4),
              "max_lag_ms": MAX_LAG_MS, "near_silence_rms_threshold": NEAR_SILENCE_RMS,
              "positive_lag": "compared signal follows reference", "method": "FFT, per-lag centered normalization",
              "source_attribution": False}
    if len(x) < rate // 2:
        return {**result, "reason": "insufficient_common_contiguous_audio"}
    x -= x.mean()
    y -= y.mean()
    if min(_rms(x), _rms(y)) < NEAR_SILENCE_RMS:
        return {**result, "reason": "near_silence_or_dc_only"}
    size = 1 << (2 * len(x) - 1).bit_length()
    products = np.fft.irfft(np.fft.rfft(y, size) * np.conj(np.fft.rfft(x, size)), size)
    maximum = min(round(rate * MAX_LAG_MS / 1000), len(x) - 1)
    lags = np.arange(-maximum, maximum + 1)
    left = np.maximum(0, -lags)
    right = np.minimum(len(x), len(x) - lags)
    count = right - left

    def sums(values, begin, stop):
        prefix = np.r_[0.0, np.cumsum(values)]
        return prefix[stop] - prefix[begin]

    sx, sy = sums(x, left, right), sums(y, left + lags, right + lags)
    xx, yy = sums(x * x, left, right), sums(y * y, left + lags, right + lags)
    covariance = products[lags % size] - sx * sy / count
    denominator = np.sqrt(np.maximum(0, xx - sx * sx / count) * np.maximum(0, yy - sy * sy / count))
    correlations = np.divide(covariance, denominator, out=np.zeros_like(covariance), where=denominator > 1e-12)
    best = int(np.argmax(np.abs(correlations)))
    value = float(np.clip(correlations[best], -1, 1))
    return {**result, "best_correlation": _number(value), "best_absolute_correlation": _number(abs(value)),
            "lag_ms": round(float(lags[best]) * 1000 / rate, 4),
            "effective_seconds": round(float(count[best]) / rate, 4), "reason": None}


class ChannelAnalysis:
    def __init__(self, rate, channels, capture_start, duration, sample_format="int16"):
        self.rate, self.channels = rate, channels
        self.sample_format = sample_format
        self.bytes_per_frame = channels * (2 if sample_format == "int16" else 4)
        self.capture_start, self.duration = capture_start, duration
        self.origin = None
        self.chunks = []
        self.seconds = {}
        self.callback_count = self.status_count = self.invalid_blocks = 0
        self.received_blocks = self.received_bytes = 0
        self.first_callback = self.last_callback = None
        self.callback_jitter = []
        self.previous_offset_end = None

    def consume(self, raw, at, frame_count, offset, status):
        self.received_blocks += 1
        try:
            self.received_bytes += len(raw)
            if frame_count <= 0 or len(raw) != frame_count * self.bytes_per_frame:
                self.invalid_blocks += 1
                return
            if self.sample_format == "int16":
                samples = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
            else:
                samples = np.frombuffer(raw, dtype="<f4")
        except (TypeError, ValueError):
            self.invalid_blocks += 1
            return
        if len(samples) != frame_count * self.channels or not np.all(np.isfinite(samples)):
            self.invalid_blocks += 1
            return
        samples = samples.reshape(-1, self.channels).copy()
        self.callback_count += 1
        self.status_count += bool(status)
        self.first_callback = at if self.first_callback is None else self.first_callback
        self.last_callback = at
        if self.origin is None:
            self.origin = at - (offset + frame_count) / self.rate
        self.callback_jitter.append(at - (self.origin + (offset + frame_count) / self.rate))
        self.chunks.append((offset, samples.mean(axis=1)))
        relative = self.origin + offset / self.rate - self.capture_start
        for second in range(max(0, math.floor(relative)), min(self.duration - 1, math.floor(relative + len(samples) / self.rate)) + 1):
            first = max(0, math.ceil((second - relative) * self.rate - 1e-6))
            last = min(len(samples), math.ceil((second + 1 - relative) * self.rate - 1e-6))
            if first < last:
                self.seconds.setdefault(second, []).append(samples[first:last])

    def finish(self, resampler):
        per_second = [{"second": second, **energy_metrics(np.concatenate(chunks), self.rate)}
                      for second, chunks in sorted(self.seconds.items())]
        total = self.duration * ANALYSIS_RATE
        timeline, valid = np.zeros(total, dtype=np.float32), np.zeros(total, dtype=bool)
        if self.chunks:
            end = max(offset + len(samples) for offset, samples in self.chunks)
            dense, present = np.zeros(end, dtype=np.float32), np.zeros(end, dtype=bool)
            for offset, samples in self.chunks:
                dense[offset:offset + len(samples)] = samples
                present[offset:offset + len(samples)] = True
            converted = resampler(dense, self.rate, ANALYSIS_RATE) if self.rate != ANALYSIS_RATE else dense
            source_indices = np.minimum((np.arange(len(converted)) * self.rate / ANALYSIS_RATE).astype(np.int64), end - 1)
            coverage = present[source_indices]
            shift = round((self.origin - self.capture_start) * ANALYSIS_RATE)
            source_start, target_start = max(0, -shift), max(0, shift)
            count = min(len(converted) - source_start, total - target_start)
            if count > 0:
                timeline[target_start:target_start + count] = converted[source_start:source_start + count]
                valid[target_start:target_start + count] = coverage[source_start:source_start + count]
        report = {"per_second": per_second, "callbacks": self.callback_count,
                  "received_callback_blocks": self.received_blocks, "received_bytes": self.received_bytes,
                  "sample_format": self.sample_format, "bytes_per_frame": self.bytes_per_frame,
                  "bytes_frame_valid": self.invalid_blocks == 0,
                  "callback_status_count": self.status_count, "invalid_blocks": self.invalid_blocks,
                  "sample_rate": self.rate, "channels": self.channels,
                  "estimated_first_sample_at": self.origin, "first_callback_at": self.first_callback,
                  "last_callback_at": self.last_callback, "valid_audio_seconds": round(float(valid.sum()) / ANALYSIS_RATE, 4),
                  "callback_delivery_jitter_max_abs_ms": round(max(map(abs, self.callback_jitter), default=0) * 1000, 3),
                  "low_energy": low_energy_statistics(timeline, valid)}
        return report, timeline, valid


def run_diagnosis(args, *, audio_module=None, resampler=None, clock=None, stop_event=None, queue_size=256):
    """Caller thread owns every PortAudio lifecycle operation; callbacks only enqueue."""
    clock = time.perf_counter if clock is None else clock
    sample_format = getattr(args, "sample_format", "int16")
    stop_event = Event() if stop_event is None else stop_event
    if audio_module is None:
        import pyaudiowpatch as audio_module
    if resampler is None:
        import soxr
        resampler = soxr.resample
    report = {"ok": False, "created_at": datetime.now(timezone(timedelta(hours=8))).isoformat(),
              "duration_seconds": args.duration, "raw_audio_saved": False, "transcripts_saved": False,
              "settings_written": False, "screenshots_taken": False, "model_loading": False,
              "device_or_volume_changes": False, "owner_thread_id": get_ident(), "channels": {}, "errors": [],
              "threshold_rms": ENERGY_THRESHOLD, "silence_seconds": SILENCE_SECONDS,
              "sample_format": sample_format,
              "alignment": "shared callback QPC; first arrival minus cumulative sample duration estimates each start; "
                           "device buffering and unequal start times limit lag accuracy; no hardware-clock synchronization",
              "interpretation": "Energy, spectra and correlation are observations only; they cannot identify a sound source or device fault."}
    queue = Queue(maxsize=queue_size)
    devices = (("mic", args.mic_device), ("compare_mic", args.compare_mic), ("loopback", args.loopback_device))
    dropped = {name: 0 for name, _ in devices}
    streams, metadata, analyses = [], {}, {}
    pa = None
    stage = "initialize"
    capture_start = capture_end = None

    def callback(name):
        offset = 0

        def receive(raw, frame_count, time_info, status):
            nonlocal offset
            at = clock()
            try:
                queue.put_nowait((name, raw, at, frame_count, offset, status))
            except Full:
                dropped[name] += 1
            offset += frame_count
            return None, audio_module.paContinue
        return receive

    def consume(item):
        name, raw, at, frame_count, offset, status = item
        analyses[name].consume(raw, at, frame_count, offset, status)

    def check_active(name, stream):
        info = metadata[name]
        info["activity_checks"] = info.get("activity_checks", 0) + 1
        info["last_activity_check_at"] = clock()
        info.setdefault("all_activity_checks_active", True)
        try:
            active = bool(stream.is_active())
        except Exception:
            info["all_activity_checks_active"] = False
            raise
        info["all_activity_checks_active"] &= active
        if not active:
            raise RuntimeError("Input stream became inactive")

    try:
        pa = audio_module.PyAudio()
        for name, index in devices:
            stage = "query_" + name
            if name == "loopback" and index is None:
                info = pa.get_default_wasapi_loopback()
                index = int(info["index"])
            else:
                info = pa.get_device_info_by_index(index)
            if name == "loopback" and index in (args.mic_device, args.compare_mic):
                raise ValueError("Loopback must be a different input")
            channels, rate = min(2, int(info["maxInputChannels"])), round(float(info["defaultSampleRate"]))
            if channels < 1 or rate < 8000:
                raise ValueError("Selected device is not a supported input")
            if name == "loopback" and not info.get("isLoopbackDevice", False):
                raise ValueError("The selected loopback must be an actual loopback input")
            metadata[name] = {key: info[key] for key in ("name", "index", "hostApi", "isLoopbackDevice") if key in info}
            metadata[name].update(sample_rate=rate, channels=channels)
            stage = "open_" + name
            stream = pa.open(format=audio_module.paInt16 if sample_format == "int16" else audio_module.paFloat32,
                             input=True, input_device_index=index,
                             channels=channels, rate=rate, frames_per_buffer=1024, start=False,
                             stream_callback=callback(name))
            streams.append((name, stream))
        capture_start = clock()
        for name, _ in devices:
            info = metadata[name]
            analyses[name] = ChannelAnalysis(info["sample_rate"], info["channels"], capture_start, args.duration, sample_format)
        for name, stream in streams:
            stage = "start_" + name
            metadata[name]["stream_start_requested_at"] = clock()
            stream.start_stream()
            metadata[name]["stream_start_returned_at"] = clock()
            stage = "active_" + name
            check_active(name, stream)
        stage = "capture"
        while not stop_event.is_set() and clock() - capture_start < args.duration:
            for name, stream in streams:
                stage = "active_" + name
                check_active(name, stream)
            stage = "capture"
            try:
                consume(queue.get(timeout=.05))
            except Empty:
                pass
        for name, stream in streams:
            stage = "active_" + name
            check_active(name, stream)
    except KeyboardInterrupt:
        report["cancelled"] = True
    except Exception as exc:
        report["errors"].append({"stage": stage, "error_type": type(exc).__name__})
    finally:
        capture_end = clock()
        for name, stream in reversed(streams):
            try:
                stream.stop_stream()
            except Exception as exc:
                report["errors"].append({"stage": "stop_" + name, "error_type": type(exc).__name__})
            finally:
                try:
                    stream.close()
                except Exception as exc:
                    report["errors"].append({"stage": "close_" + name, "error_type": type(exc).__name__})
        if pa is not None:
            try:
                pa.terminate()
            except Exception as exc:
                report["errors"].append({"stage": "terminate", "error_type": type(exc).__name__})
        while True:
            try:
                consume(queue.get_nowait())
            except Empty:
                break
    report["capture_start_at"], report["capture_end_at"] = capture_start, capture_end
    report["queue_dropped_blocks"] = dropped
    report["audio_capture"] = capture_start is not None
    report["cleanup_completed"] = not any(error["stage"].startswith(("stop_", "close_", "terminate")) for error in report["errors"])
    signals = {}
    for name, analysis in analyses.items():
        result, samples, valid = analysis.finish(resampler)
        report["channels"][name] = {"device": metadata[name], **result}
        if analysis.invalid_blocks:
            report["errors"].append({"stage": "data_" + name, "error_type": "InvalidAudioBlock"})
        if name != "loopback" and analysis.callback_count == 0:
            report["errors"].append({"stage": "callbacks_" + name, "error_type": "NoCallbacks"})
        signals[name] = samples, valid
    quiet_loopback = "loopback" in analyses and analyses["loopback"].received_blocks == 0 and not dropped["loopback"] and (
        metadata["loopback"].get("all_activity_checks_active", False)
        and metadata["loopback"].get("activity_checks", 0) >= 2 and not report["errors"])
    report["loopback_observation"] = "no_callbacks_while_active" if quiet_loopback else (
        "callbacks_received" if "loopback" in analyses and analyses["loopback"].received_blocks else "no_callbacks_unverified")
    report["cross_correlation"] = {}
    for name in ("compare_mic", "loopback"):
        if "mic" in signals and name in signals:
            mic, mic_valid = signals["mic"]
            other, other_valid = signals[name]
            report["cross_correlation"]["mic_to_" + name] = normalized_cross_correlation(mic, other, mic_valid, other_valid)
    clock_info = time.get_clock_info("perf_counter")
    report["clock"] = {"function": "time.perf_counter", "implementation": clock_info.implementation,
                       "resolution_seconds": clock_info.resolution}
    report["ok"] = not report["errors"] and not report.get("cancelled") and len(analyses) == 3 and all(
        analysis.callback_count > 0 or (name == "loopback" and quiet_loopback) for name, analysis in analyses.items())
    return report


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="只读三路能量诊断；不保存原始音频、转写或用户设置。")
    parser.add_argument("--mic-device", required=True, type=int)
    parser.add_argument("--compare-mic", required=True, type=int)
    parser.add_argument("--loopback-device", type=int, help="省略时只读解析当前默认WASAPI loopback")
    parser.add_argument("--duration", type=int, choices=range(10, 31), default=15, metavar="10..30")
    parser.add_argument("--sample-format", choices=("int16", "float32"), default="int16")
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args(argv)
    indices = tuple(value for value in (args.mic_device, args.compare_mic, args.loopback_device) if value is not None)
    if min(indices) < 0 or len(set(indices)) != len(indices):
        parser.error("请显式选择三个不同的非负设备索引。")
    return args


def main(argv=None):
    args = parse_args(argv)
    if sys.platform != "win32":
        raise SystemExit("This diagnostic uses Windows PyAudioWPatch inputs.")
    report = run_diagnosis(args)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, allow_nan=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"ok": report["ok"], "report": str(args.report), "raw_audio_saved": False}))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
