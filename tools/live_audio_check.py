"""Opt-in, time-limited live GUI check. No audio or transcripts are saved by default."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from copy import deepcopy
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from difflib import SequenceMatcher
import hashlib
import json
import math
import os
from pathlib import Path
import re
import statistics
import subprocess
import sys
from threading import Event, Lock, Thread
import time
import wave

SOURCE = Path(__file__).resolve().parents[1] / "src"
if str(SOURCE) not in sys.path:
    sys.path.insert(0, str(SOURCE))

from PySide6.QtCore import QEvent, QObject, QTimer


TIMING_FIELDS = ("endpoint_ms", "queue_ms", "asr_ms", "translation_ms", "elapsed_ms")
TIMESTAMP_FIELDS = ("speech_start_at", "speech_end_at", "segment_submitted_at", "emitted_at")
PLAYBACK_STAGES = frozenset(("worker_enter", "numpy_import_start", "numpy_import_done", "pyaudio_import_start",
    "pyaudio_import_done", "soxr_import_start", "soxr_import_done", "control_reader_started", "sample_hash_start",
    "sample_hash_verified", "sample_wav_read", "pyaudio_init_start", "pyaudio_init_done", "output_device_start",
    "output_device_done", "resample_start", "resample_done", "output_pre_open_cancelled", "output_open_start",
    "output_open_done", "first_write_start", "first_write_done", "writes_done", "output_stop_start",
    "output_stop_done", "output_close_start", "output_close_done", "pyaudio_terminate_start", "pyaudio_terminate_done"))


def _finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _error_summary(message):
    # Backend exceptions may quote source text. Persist categories/codes only;
    # the GUI still receives and displays the original error through AppController.
    category = next((name for marker, name in (("识别", "recognition"), ("翻译", "translation"),
                    ("音频", "audio"), ("设备", "device"), ("模型", "model")) if marker in message), "runtime")
    return {"category": category, "numeric_codes": sorted(set(re.findall(r"-\d{3,6}\b", message))),
            "details_redacted": True}


def _summary(values):
    if not values:
        return {"count": 0}
    ordered = sorted(values)
    return {"count": len(values), "mean": round(statistics.mean(values), 3),
            "median": round(statistics.median(values), 3),
            "p95": round(ordered[max(0, math.ceil(len(ordered) * .95) - 1)], 3),
            "max": round(max(values), 3)}


def _clock_description():
    info = time.get_clock_info("perf_counter")
    return {"function": "time.perf_counter", "implementation": info.implementation,
            "resolution_seconds": info.resolution,
            "query_performance_counter": "QueryPerformanceCounter" in info.implementation}


def _prompt_normalized(text):
    return "".join(char for char in text.casefold() if char.isalnum())


class EventMetrics:
    """Thread-safe callback stamps and main-thread apply/paint proxy observations."""

    def __init__(self, *, only_fixture=False, review_text=False, clock=None, expected_phrases=()):
        self.clock = time.perf_counter if clock is None else clock
        self.started = self.clock()
        self.only_fixture = only_fixture
        self.review_text = review_text
        self.expected_phrases = tuple(_prompt_normalized(text) for text in expected_phrases)
        self.lock = Lock()
        self.counts = Counter()
        self.levels = defaultdict(lambda: {"updates": 0, "meter_peak": 0.0})
        self.states = []
        self.errors = []
        self.updates = []
        self.pending = {}
        self.queue_peak = 0

    def _stamp(self, now):
        return round((now - self.started) * 1000, 3)

    def received(self, event):
        now = self.clock()
        with self.lock:
            self.counts[event.kind] += 1
            if event.kind == "level" and _finite(event.level):
                level = self.levels[event.channel]
                level["updates"] += 1
                level["meter_peak"] = max(level["meter_peak"], max(0.0, float(event.level)))
            depth = getattr(event, "queue_depth", 0)
            if _finite(depth):
                self.queue_peak = max(self.queue_peak, int(depth))
            if event.kind == "state":
                self.states.append({"status": event.status, "callback_ms": self._stamp(now)})
            if event.kind == "error":
                self.errors.append({"channel": event.channel, "segment_id": event.segment_id,
                                    "status": event.status, "callback_ms": self._stamp(now),
                                    **_error_summary(event.message)})
            if event.kind not in {"original", "translation"}:
                return
            row = {"session_id": event.session_id, "channel": event.channel, "segment_id": event.segment_id,
                   "kind": event.kind, "callback_ms": self._stamp(now), "ui_applied_ms": None,
                   "paint_proxy_ms": None}
            for name in (*TIMING_FIELDS, *TIMESTAMP_FIELDS):
                value = getattr(event, name, None)
                if _finite(value) and (name not in TIMESTAMP_FIELDS or value > 0):
                    row[name] = value
            for name in ("end_reason", "speech_detector"):
                value = getattr(event, name, "")
                if value:
                    row[name] = value
            if event.kind == "original" and event.channel == "mic" and self.expected_phrases:
                recognized = _prompt_normalized(event.text)
                row["prompt_matches"] = [
                    {"index": index,
                     "similarity": round(1.0 if expected and expected in recognized else
                                         SequenceMatcher(None, expected, recognized, autojunk=False).ratio(), 3)}
                    for index, expected in enumerate(self.expected_phrases)
                ]
            emitted = row.get("emitted_at")
            if emitted:
                row["emit_to_callback_ms"] = round(max(0.0, now - emitted) * 1000, 3)
            if self.only_fixture:
                row["fixture_text"] = event.text
            elif self.review_text:
                row["review_text"] = event.text
            sequence = len(self.updates)
            self.updates.append(row)
            # Retain the event only in memory until applied, preventing id reuse.
            self.pending[id(event)] = (event, sequence)

    def applied(self, event, *, ui_text_matches=False, paired_data_matches=False):
        now = self.clock()
        with self.lock:
            pending = self.pending.pop(id(event), None)
            if pending is None:
                return None
            sequence = pending[1]
            row = self.updates[sequence]
            row.update(ui_text_matches=bool(ui_text_matches), paired_data_matches=bool(paired_data_matches),
                       char_count=len(event.text))
            row["ui_applied_ms"] = self._stamp(now)
            row["callback_to_apply_ms"] = round(row["ui_applied_ms"] - row["callback_ms"], 3)
            if row.get("speech_end_at"):
                row["speech_end_to_apply_ms"] = round(max(0.0, now - row["speech_end_at"]) * 1000, 3)
            if row.get("speech_start_at"):
                row["speech_start_to_apply_ms"] = round(max(0.0, now - row["speech_start_at"]) * 1000, 3)
            return sequence

    def painted(self, sequence):
        now = self.clock()
        with self.lock:
            row = self.updates[sequence]
            if row["paint_proxy_ms"] is not None:
                return
            row["paint_proxy_ms"] = self._stamp(now)
            row["apply_to_paint_proxy_ms"] = round(row["paint_proxy_ms"] - row["ui_applied_ms"], 3)
            if row.get("speech_end_at"):
                row["speech_end_to_paint_proxy_ms"] = round(max(0.0, now - row["speech_end_at"]) * 1000, 3)

    def snapshot(self):
        with self.lock:
            updates = [dict(row) for row in self.updates]
            self.pending.clear()
            pairs = defaultdict(lambda: {"original_updates": 0, "translation_updates": 0,
                                         "original_applied": 0, "translation_applied": 0,
                                         "original_verified": 0, "translation_verified": 0})
            for row in updates:
                key = (row["session_id"], row["channel"], row["segment_id"])
                pairs[key][row["kind"] + "_updates"] += 1
                pairs[key][row["kind"] + "_applied"] += row["ui_applied_ms"] is not None
                pairs[key][row["kind"] + "_verified"] += bool(row.get("ui_text_matches") and row.get("paired_data_matches"))
            paired = [{"session_id": key[0], "channel": key[1], "segment_id": key[2], **values,
                       "paired": bool(values["original_verified"] and values["translation_verified"])}
                      for key, values in pairs.items()]
            paired_keys = {(row["session_id"], row["channel"], row["segment_id"])
                           for row in paired if row["paired"]}
            matched_prompts = {match["index"] for row in updates
                               if (row["session_id"], row["channel"], row["segment_id"]) in paired_keys
                               for match in row.get("prompt_matches", []) if match["similarity"] >= .75}
            fields = (*TIMING_FIELDS, "callback_to_apply_ms", "apply_to_paint_proxy_ms",
                      "speech_start_to_apply_ms", "speech_end_to_apply_ms", "speech_end_to_paint_proxy_ms")
            performance = {kind: {field: _summary([row[field] for row in updates if row["kind"] == kind
                                                  and field in row and _finite(row[field])]) for field in fields}
                           for kind in ("original", "translation")}
            return {"clock": _clock_description(), "event_counts": dict(self.counts), "level_meter": dict(self.levels),
                    "level_meter_scale": "event.level is a scaled/clipped UI meter (RMS x10), not raw RMS",
                    "states": list(self.states),
                    "errors": list(self.errors), "queue_depth_peak": self.queue_peak, "updates": updates,
                    "segments": paired, "performance_ms": performance,
                    "expected_prompt_check": {
                        "count": len(self.expected_phrases), "matched_indices": sorted(matched_prompts),
                        "all_matched_in_paired_segments": len(matched_prompts) == len(self.expected_phrases),
                        "method": "mic only; normalized substring or SequenceMatcher ratio >= 0.75; not WER",
                        "expected_or_recognized_text_saved": False,
                    }}


class PaintProxy(QObject):
    """Stamp a queued turn after the target label's Qt paint event; never grab pixels."""

    def __init__(self, metrics, parent=None):
        super().__init__(parent)
        self.metrics = metrics
        self.pending = {}

    def watch(self, label, sequence):
        if label is None or sequence is None:
            return
        if label not in self.pending:
            label.installEventFilter(self)
        self.pending[label] = sequence
        label.update()

    def eventFilter(self, watched, event):
        if event.type() == QEvent.Type.Paint and watched in self.pending:
            if watched.isVisible() and not watched.visibleRegion().isEmpty():
                sequence = self.pending.pop(watched)
                QTimer.singleShot(0, lambda value=sequence: self.metrics.painted(value))
        return False


def play_official_sample(model_root, language, output_device, stop_event, *, audio_module=None, resampler=None, stage=None):
    """Worker-only output implementation; never call from the capture process."""
    from fanyiji_next.verification import OFFICIAL_SAMPLES

    mark = stage if stage is not None else lambda name: None
    mark("sample_hash_start")
    path = Path(model_root) / "self_test" / f"{language}.wav"
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    if digest != OFFICIAL_SAMPLES[language]:
        raise ValueError("Official playback sample SHA-256 mismatch")
    mark("sample_hash_verified")
    with wave.open(str(path), "rb") as source:
        if (source.getnchannels(), source.getsampwidth(), source.getframerate()) != (1, 2, 16000):
            raise ValueError("Official sample must be mono PCM16 at 16000 Hz")
        frames = source.readframes(source.getnframes())
    mark("sample_wav_read")
    import numpy as np
    if audio_module is None:
        import pyaudiowpatch as audio_module
    if resampler is None:
        import soxr
        resampler = soxr.resample
    mark("pyaudio_init_start")
    pa = audio_module.PyAudio()
    mark("pyaudio_init_done")
    stream = None
    try:
        mark("output_device_start")
        info = pa.get_default_output_device_info() if output_device is None else pa.get_device_info_by_index(output_device)
        mark("output_device_done")
        rate, channels = int(info["defaultSampleRate"]), min(2, int(info["maxOutputChannels"]))
        if rate <= 0 or channels <= 0:
            raise ValueError("Selected device is not an audio output")
        samples = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0
        mark("resample_start")
        if rate != 16000:
            samples = resampler(samples, 16000, rate)
        if channels > 1:
            samples = np.repeat(samples[:, None], channels, axis=1)
        data = (np.clip(samples, -1, 1) * 32767).astype("<i2").tobytes()
        mark("resample_done")
        if stop_event.is_set():
            mark("output_pre_open_cancelled")
            return {"played": False, "cancelled": True}
        mark("output_open_start")
        stream = pa.open(format=audio_module.paInt16, channels=channels, rate=rate, output=True,
                         output_device_index=int(info["index"]), frames_per_buffer=1024)
        mark("output_open_done")
        started = time.perf_counter()
        chunk_size = 1024 * channels * 2
        for offset in range(0, len(data), chunk_size):
            if stop_event.is_set():
                break
            if offset == 0:
                mark("first_write_start")
            stream.write(data[offset:offset + chunk_size])
            if offset == 0:
                mark("first_write_done")
        mark("writes_done")
        return {"played": True, "cancelled": stop_event.is_set(), "device_index": int(info["index"]),
                "sample_rate": rate, "channels": channels, "sample_sha256": digest,
                "started_at": started, "finished_at": time.perf_counter()}
    finally:
        try:
            if stream is not None:
                try:
                    mark("output_stop_start")
                    stream.stop_stream()
                    mark("output_stop_done")
                finally:
                    mark("output_close_start")
                    stream.close()
                    mark("output_close_done")
        finally:
            mark("pyaudio_terminate_start")
            pa.terminate()
            mark("pyaudio_terminate_done")


def _watch_parent_input(stream, stop_event, observer=None):
    reason, error_type = "parent_eof", None
    try:
        # OS reads avoid holding sys.stdin's buffered lock through interpreter shutdown.
        value = os.read(stream, 1) if isinstance(stream, int) else stream.read(1)
        reason = "stop_byte" if value else "parent_eof"
    except (OSError, ValueError) as exc:
        reason, error_type = "input_error", type(exc).__name__
    finally:
        # A stop byte or EOF (parent exited) both end output between chunks.
        try:
            if observer is not None:
                observer(reason, error_type)
        finally:
            stop_event.set()


def _load_playback_dependencies(stage):
    # Windows native-module imports complete before any blocking stdin reader starts.
    stage("numpy_import_start")
    import numpy
    stage("numpy_import_done")
    stage("pyaudio_import_start")
    import pyaudiowpatch
    stage("pyaudio_import_done")
    stage("soxr_import_start")
    import soxr
    stage("soxr_import_done")
    return pyaudiowpatch, soxr.resample


def _playback_worker_main(argv):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--model-root", required=True, type=Path)
    parser.add_argument("--play-sample", choices=("zh", "ja"), required=True)
    parser.add_argument("--output-device", type=int)
    args = parser.parse_args(argv)
    stop_event = Event()
    output_lock = Lock()

    def emit(payload):
        with output_lock:
            try:
                print(json.dumps({"protocol": "fanyiji-output-v1", **payload}), flush=True)
            except (OSError, ValueError):
                pass

    def stage(name):
        emit({"kind": "stage", "stage": name, "at": time.perf_counter()})

    def control(reason, error_type):
        emit({"kind": "control", "reason": reason, "error_type": error_type, "at": time.perf_counter()})

    stage("worker_enter")
    try:
        audio_module, resampler = _load_playback_dependencies(stage)
        listener = Thread(target=_watch_parent_input, args=(sys.stdin.fileno(), stop_event, control),
                          name="playback-parent-control", daemon=True)
        listener.start()
        stage("control_reader_started")
        result = play_official_sample(args.model_root, args.play_sample, args.output_device, stop_event,
                                      audio_module=audio_module, resampler=resampler, stage=stage)
        payload = {"ok": True, **result}
    except Exception as exc:
        payload = {"ok": False, "played": False,
                   "error_type": type(exc).__name__, **_error_summary(str(exc))}
    emit({**payload, "reported_worker_pid": os.getpid(), "reported_parent_pid": os.getppid()})
    return 0 if payload["ok"] else 2


def _collect_playback_process(process, *, cancel=False):
    if cancel and process.poll() is None and process.stdin is not None:
        try:
            process.stdin.write(b"stop\n")
            process.stdin.flush()
        except (OSError, ValueError):
            pass
    forced = False
    try:
        process.wait(timeout=2)
    except subprocess.TimeoutExpired:
        forced = True
        process.terminate()
        try:
            process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=2)
    return forced


def _drain_playback_pipe(stream, result, limit):
    try:
        while chunk := stream.read(4096):
            result["bytes"] += len(chunk)
            if len(result["data"]) < limit:
                result["data"].extend(chunk[:limit - len(result["data"])])
    except (OSError, ValueError) as exc:
        result["error_type"] = type(exc).__name__


def play_official_sample_process(model_root, language, output_device, stop_event, *, popen_factory=None):
    """Isolate PortAudio init/open/terminate from the process owning capture."""
    if stop_event.is_set():
        return {"isolated_output_process": True, "played": False, "cancelled": True}
    command = [sys.executable, str(Path(__file__).resolve()), "--_playback-worker",
               "--model-root", str(model_root), "--play-sample", language]
    if output_device is not None:
        command.extend(("--output-device", str(output_device)))
    options = {"stdin": subprocess.PIPE, "stdout": subprocess.PIPE, "stderr": subprocess.PIPE}
    if sys.platform == "win32":
        options["creationflags"] = subprocess.CREATE_NO_WINDOW
    process = None
    readers = []
    captures = []
    parent_times = {"parent_launch_at": time.perf_counter()}
    try:
        process = (popen_factory or subprocess.Popen)(command, **options)
        parent_times["parent_spawned_at"] = time.perf_counter()
        # Drain both pipes while waiting, without communicate() closing control stdin.
        # Only stdout's bounded protocol data is retained; stderr is counted only.
        for stream, limit in ((process.stdout, 65536), (process.stderr, 0)):
            captured = {"data": bytearray(), "bytes": 0}
            captures.append(captured)
            reader = Thread(target=_drain_playback_pipe, args=(stream, captured, limit), daemon=True)
            readers.append(reader)
            reader.start()
        while True:
            if process.poll() is not None:
                parent_times["parent_exit_observed_at"] = time.perf_counter()
                break
            if stop_event.wait(.05):
                parent_times["parent_cancel_observed_at"] = time.perf_counter()
                break
        cancelled = stop_event.is_set()
        parent_times["parent_collect_started_at"] = time.perf_counter()
        forced = _collect_playback_process(process, cancel=cancelled)
        for reader in readers:
            reader.join(timeout=2)
        output = bytes(captures[0]["data"])
        parent_times["parent_collected_at"] = time.perf_counter()
        base = {"isolated_output_process": True, "worker_pid": process.pid,
                "worker_exit_code": process.returncode, "forced_cleanup": forced, **parent_times}
        if any(reader.is_alive() for reader in readers):
            base["pipe_drain_timeout"] = True
        if captures[0]["bytes"] > len(output):
            base["stdout_truncated"] = True
        stages = []
        payload = None
        for line in output.decode("utf-8", errors="replace").splitlines():
            try:
                candidate = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(candidate, dict) or candidate.get("protocol") != "fanyiji-output-v1":
                continue
            if candidate.get("kind") == "stage":
                if candidate.get("stage") in PLAYBACK_STAGES and _finite(candidate.get("at")):
                    stages.append({"stage": candidate["stage"], "at": candidate["at"]})
            elif candidate.get("kind") == "control":
                if candidate.get("reason") in {"stop_byte", "parent_eof", "input_error"} and _finite(candidate.get("at")):
                    base["worker_control_reason"] = candidate["reason"]
                    base["worker_control_at"] = candidate["at"]
                    if candidate.get("error_type") in {"OSError", "ValueError", "UnsupportedOperation"}:
                        base["worker_control_error_type"] = candidate["error_type"]
            elif isinstance(candidate.get("ok"), bool):
                payload = candidate
        base["worker_stages"] = stages
        if payload is not None:
            for name in ("reported_worker_pid", "reported_parent_pid"):
                if isinstance(payload.get(name), int):
                    base[name] = payload[name]
        if forced:
            return {**base, "played": None, "cancelled": cancelled, "error_type": "PlaybackCleanupTimeout"}
        if base.get("pipe_drain_timeout") or any(item.get("error_type") for item in captures):
            return {**base, "played": None, "cancelled": cancelled, "error_type": "PlaybackPipeDrainError"}
        if process.returncode != 0 or payload is None or payload.get("ok") is not True:
            return {**base, "played": False, "cancelled": cancelled,
                    "error_type": (payload or {}).get("error_type", "PlaybackProcessError"),
                    "stderr_bytes": captures[1]["bytes"]}
        # Copy only protocol fields: child/library stdout must never become a transcript/log artifact.
        allowed = ("played", "cancelled", "device_index", "sample_rate", "channels", "sample_sha256", "started_at", "finished_at")
        return {**base, **{key: payload[key] for key in allowed if key in payload}}
    finally:
        if process is not None:
            if process.poll() is None:
                try:
                    _collect_playback_process(process, cancel=True)
                except Exception:
                    process.kill()
                    process.wait(timeout=2)
            if process.stdin is not None:
                process.stdin.close()
            for index, stream in enumerate((process.stdout, process.stderr)):
                if stream is not None and (index >= len(readers) or not readers[index].is_alive()):
                    stream.close()


def observe_segment_diagnostics(process_segment, active_session, engine, segment, destination):
    """Record only diagnostics produced during this segment, preserving outcomes.

    Whisper and Hy-MT replace last_diagnostics on each attempted call. Retain
    the prior objects themselves so skipped calls cannot inherit another
    segment's scores, and object identity cannot be confused by id reuse.
    """
    owners = (engine.recognizer, engine.translator)
    previous = tuple(getattr(owner, "last_diagnostics", None) for owner in owners)
    try:
        return process_segment(active_session, engine, segment)
    finally:
        current = tuple(getattr(owner, "last_diagnostics", None) for owner in owners)
        recognition, translation = (
            value if isinstance(value, dict) and value is not old else {}
            for old, value in zip(previous, current)
        )
        destination.append({
            "channel": segment.channel, "segment_id": segment.segment_id,
            "recognition": deepcopy(recognition),
            "translation": {key: deepcopy(translation[key]) for key in (
                "calls", "history_available", "history_used", "fallback", "issues", "draft_issues"
            ) if key in translation},
        })


class LiveSession(QObject):
    """Own test timers; all model/device cleanup completes before the GUI exits."""

    def __init__(self, app, window, controller, runtimes, metrics, args):
        super().__init__(window)
        self.app, self.window, self.controller = app, window, controller
        self.runtimes, self.metrics, self.args = runtimes, metrics, args
        self.started = time.perf_counter()
        self.running_at = None
        self.loading_at = None
        self.stop_requested_at = None
        self.playback_stop = Event()
        self.player = None
        self.playback = {"requested": bool(args.play_sample), "played": False}
        self.completed = False
        self.timeout = None
        self.tool_error = None
        self.ready_file = args.report.with_name(args.report.stem + ".ready.json")
        self.monitor = QTimer(self)
        self.monitor.setInterval(50)
        self.monitor.timeout.connect(self.poll)
        window.closing.connect(self.stop)

    def begin(self):
        self.monitor.start()
        if getattr(self.args, "manual_start", False):
            hint = ("电脑声音联调：点击「开始翻译」后，采集当前播放的声音。" if self.args.channel == "system" else
                    "麦克风联调：准备好后点击「开始翻译」，再按提示读测试句。")
            self.window.set_session_state("stopped", hint)
        else:
            self.controller.start_session(self.window.settings)

    def _play(self):
        if self.stop_requested_at is not None or self.player is not None:
            return

        def play():
            try:
                self.playback.update(play_official_sample_process(self.args.model_root, self.args.play_sample,
                                                                 self.args.output_device, self.playback_stop))
            except Exception as exc:
                self.playback.update(error_type=type(exc).__name__, **_error_summary(str(exc)))

        self.player = Thread(target=play, name="official-sample-output", daemon=True)
        self.player.start()

    def stop(self):
        if self.stop_requested_at is None:
            self.stop_requested_at = time.perf_counter()
            self.playback_stop.set()
            self.controller.stop_session()

    def poll(self):
        now = time.perf_counter()
        if self.controller.state == "loading" and self.loading_at is None:
            self.loading_at = now
        if self.controller.state == "running" and self.running_at is None:
            self.running_at = now
            try:
                self.ready_file.parent.mkdir(parents=True, exist_ok=True)
                self.ready_file.write_text(json.dumps({"state": "running",
                    "ready_at": datetime.now(timezone(timedelta(hours=8))).isoformat(),
                    "perf_counter": now}, indent=2) + "\n", encoding="utf-8")
            except OSError as exc:
                self.tool_error = type(exc).__name__
                self.stop()
            phrases = getattr(self.args, "expected_phrase", ())
            hint = ("请读两次，间隔3秒：" + " / ".join(phrases)) if phrases else "正在测试麦克风，请按聊天提示说话。"
            if "mic" in getattr(self.args, "expect_silence", ()):
                hint = "本轮请保持安静，正在检查底噪是否误触发字幕。"
            self.window.set_session_state("running", hint if self.args.channel in {"mic", "both"}
                                          else "正在测试电脑声音，仅在内存处理采集音频。")
            if self.args.play_sample:
                QTimer.singleShot(500, self._play)
        if self.running_at is not None and now - self.running_at >= self.args.duration:
            self.stop()
        elif self.running_at is None and (
            (not getattr(self.args, "manual_start", False) and now - self.started >= 30)
            or (getattr(self.args, "manual_start", False) and self.loading_at is not None and now - self.loading_at >= 30)
        ):
            self.timeout = "startup_30_seconds"
            self.stop()
        if self.controller.state in {"error", "stopped"} and self.runtimes:
            self.stop()
        if self.stop_requested_at is None:
            return
        busy = any(bool(getattr(runtime, "is_busy", runtime.is_running)) for runtime in self.runtimes)
        player_busy = self.player is not None and self.player.is_alive()
        controller_done = self.controller.state == "stopped" or (
            getattr(self.controller, "_closed", False) and not getattr(self.controller, "_shutdown_waiting", False))
        if not busy and not player_busy and controller_done:
            self.completed = True
            self.monitor.stop()
            self.controller.shutdown()
            self.window.close()
            QTimer.singleShot(0, self.app.quit)
        elif now - self.stop_requested_at >= 15:
            self.timeout = "stop_15_seconds"
            self.monitor.stop()
            self.controller.shutdown()
            self.window.close()
            QTimer.singleShot(0, self.app.quit)

    def report(self):
        from fanyiji_next.verification import BACKEND_OPTIONS
        from fanyiji_next.models import AppSettings
        report = self.metrics.snapshot()
        # Manual-start allows changing settings before starting. Report the
        # profile actually passed to the runtime rather than stale CLI choices.
        actual = next((runtime._audit_settings for runtime in reversed(self.runtimes)
                       if isinstance(getattr(runtime, "_audit_settings", None), AppSettings)), None)
        report["selected_backends"] = {name: getattr(actual, name) if actual is not None else getattr(self.args, name, default)
                                       for name, _choices, default in BACKEND_OPTIONS}
        report["selected_backends_source"] = "runtime_start" if actual is not None else "arguments"
        enabled = {"mic", "system"} if self.args.channel == "both" else {self.args.channel}
        silent_channels = set(getattr(self.args, "expect_silence", ()))
        paired_channels = {row["channel"] for row in report["segments"] if row["paired"]}
        unexpected_updates = [row for row in report["updates"] if row["channel"] in silent_channels]
        fatal_errors = [error for error in report["errors"] if error["status"] != "recoverable"]
        playback_ok = not self.args.play_sample or (self.playback.get("played") is True
                      and not self.playback.get("cancelled") and not self.playback.get("error_type"))
        report.update({"ok": self.completed and not fatal_errors and not self.timeout and not self.tool_error
                       and playback_ok and enabled - silent_channels <= paired_channels and not unexpected_updates
                       and report["expected_prompt_check"]["all_matched_in_paired_segments"],
                       "channel": self.args.channel, "requested_capture_seconds": self.args.duration,
                       "runtime_reached_running": self.running_at is not None,
                       "audio_capture": self.running_at is not None, "raw_audio_saved": False,
                       "settings_written": False, "screenshots_taken": False,
                       "transcripts_saved": self.args.only_fixture or getattr(self.args, "review_text", False),
                       "operator_declared_fixture_only": self.args.only_fixture,
                       "all_enabled_channels_paired": enabled <= paired_channels,
                       "silence_expectation": {"channels": sorted(silent_channels),
                                               "unexpected_text_updates": len(unexpected_updates),
                                               "passed": not unexpected_updates},
                       "cleanup_completed": self.completed, "playback": dict(self.playback),
                       "timeout": self.timeout, "tool_error_type": self.tool_error,
                       "ready_file": str(self.ready_file),
                       "total_seconds": round(time.perf_counter() - self.started, 3),
                       "measurement": "callback -> UI apply -> queued turn after visible label Qt paint event; not display photons"})
        if self.running_at is not None:
            report["capture_until_stop_request_seconds"] = round((self.stop_requested_at or time.perf_counter()) - self.running_at, 3)
        if self.stop_requested_at is not None:
            report["stop_cleanup_seconds"] = round(time.perf_counter() - self.stop_requested_at, 3)
        return report


def parse_args(argv=None):
    from fanyiji_next.verification import add_backend_arguments
    parser = argparse.ArgumentParser(description="手动授权后的限时真实声音/字幕联调；默认报告不含转写正文。")
    parser.add_argument("--model-root", required=True, type=Path)
    parser.add_argument("--channel", choices=("mic", "system", "both"), required=True)
    parser.add_argument("--duration", type=int, choices=range(20, 61), default=30, metavar="20..60")
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--play-sample", choices=("zh", "ja"))
    parser.add_argument("--manual-start", action="store_true", help="先显示联调窗口，等待用户手动点击开始后才采音")
    parser.add_argument("--expect-silence", action="append", choices=("mic", "system"), default=[],
                        help="该通道本轮预期无语音，出现字幕则判失败；其它启用通道仍需字幕配对")
    parser.add_argument("--expected-phrase", action="append", default=[],
                        help="麦克风已知测试句；只保存匹配相似度，不保存测试句或识别正文")
    parser.add_argument("--only-fixture", action="store_true", help="操作者声明只存在官方样例声音，允许在报告保存转写正文")
    parser.add_argument("--review-text", action="store_true", help="本轮人工核查：在本地报告保存真实字幕正文；默认关闭")
    parser.add_argument("--engine-diagnostics", action="store_true", help="在本地报告记录本轮识别分数和翻译保护信息；不含正文")
    add_backend_arguments(parser)
    for option in ("mic-device", "system-device", "output-device"):
        parser.add_argument("--" + option, type=int)
    args = parser.parse_args(argv)
    enabled = {"mic", "system"} if args.channel == "both" else {args.channel}
    if not set(args.expect_silence) <= enabled or ("mic" in args.expect_silence and args.expected_phrase):
        parser.error("静音预期须属于启用通道，且不能同时要求麦克风识别测试句。")
    if args.expected_phrase and (args.channel == "system" or any(not _prompt_normalized(text) for text in args.expected_phrase)):
        parser.error("--expected-phrase 需要麦克风通道与非空测试句。")
    if args.only_fixture and not args.play_sample:
        parser.error("--only-fixture 必须同时显式选择 --play-sample。")
    if any(value is not None and value < 0 for value in (args.mic_device, args.system_device, args.output_device)):
        parser.error("设备索引必须非负。")
    if args.report.resolve().is_relative_to(args.model_root.resolve()):
        parser.error("报告请保存到模型目录之外。")
    return args


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "--_playback-worker":
        return _playback_worker_main(argv[1:])
    args = parse_args(argv)
    from PySide6.QtWidgets import QApplication
    from fanyiji_next.audio_runtime import VoiceRuntime
    from fanyiji_next.controller import AppController
    from fanyiji_next.models import AppSettings, ChannelSettings
    from fanyiji_next.settings import with_local_models
    from fanyiji_next.studio import StudioWindow

    settings = with_local_models(AppSettings(font_size=30), args.model_root)
    from fanyiji_next.verification import BACKEND_OPTIONS, BACKEND_PATH_OPTIONS
    settings = replace(settings,
                       **{name: getattr(args, name) for name, _choices, _default in BACKEND_OPTIONS},
                       **{name: str(getattr(args, name)) for name in BACKEND_PATH_OPTIONS
                          if getattr(args, name, None) is not None},
                       microphone=ChannelSettings(enabled=args.channel in {"mic", "both"}, device_index=args.mic_device),
                       system=ChannelSettings(enabled=args.channel in {"system", "both"}, direction="ja_to_zh", device_index=args.system_device))
    app = QApplication.instance() or QApplication(sys.argv[:1])
    from fanyiji_next.prototype import _configure_fonts
    _configure_fonts(app)
    previous_quit = app.quitOnLastWindowClosed()
    app.setQuitOnLastWindowClosed(False)
    window = StudioWindow(settings, prototype=False)
    from fanyiji_next import __version__
    window.setWindowTitle(f"翻译姬 NEXT {__version__} · 引擎联调（新版源码）")
    metrics = EventMetrics(only_fixture=args.only_fixture, review_text=args.review_text,
                           expected_phrases=args.expected_phrase)
    painter = PaintProxy(metrics, window)
    original_apply = window.apply_event

    def apply_event(event):
        original_apply(event)
        if event.kind not in {"original", "translation"}:
            return
        panel = window.panels.get(event.channel)
        label = None
        if panel is not None:
            # Voice mode paints the session stream, including its latest row.
            row = panel.rows.get(event.segment_id)
            label = getattr(row, "original" if event.kind == "original" else "translated", None)
        caption = next((item for item in window.store.items(event.channel) if item.id == event.segment_id), None)
        value = getattr(caption, "original" if event.kind == "original" else "translated", None)
        matches = value == event.text and bool(event.text)
        paired_data_matches = matches and (event.kind == "original" or bool(caption.original))
        ui_text_matches = label is not None and label.text() == event.text
        sequence = metrics.applied(event, ui_text_matches=ui_text_matches, paired_data_matches=paired_data_matches)
        if sequence is not None and ui_text_matches and paired_data_matches:
            painter.watch(label, sequence)

    window.apply_event = apply_event
    runtimes = []
    engine_diagnostics = []

    class ObservedRuntime(VoiceRuntime):
        def start(self, settings, session_id):
            self._audit_settings = settings
            return super().start(settings, session_id)

        def _process_segment(self, active_session, engine, segment):
            process = super()._process_segment
            if not args.engine_diagnostics:
                return process(active_session, engine, segment)
            return observe_segment_diagnostics(process, active_session, engine, segment, engine_diagnostics)

    def factory(callback):
        def observed(event):
            metrics.received(event)
            callback(event)
        runtime = ObservedRuntime(observed)
        runtimes.append(runtime)
        return runtime

    controller = AppController(window, runtime_factory=factory, persist_settings=False, device_lister=lambda: [])
    window._controller = controller
    session = LiveSession(app, window, controller, runtimes, metrics, args)
    window.show()
    if args.manual_start:
        window.showNormal()
        window.raise_()
        window.activateWindow()
        visible_report = args.report.with_name(args.report.stem + ".window.json")
        visible_report.parent.mkdir(parents=True, exist_ok=True)
        from PySide6.QtGui import QGuiApplication
        visible_report.write_text(json.dumps({"pid": os.getpid(), "window_title": window.windowTitle(),
            "qt_visible": window.isVisible(), "qt_platform": QGuiApplication.platformName(),
            "native_window_id": int(window.winId()), "capture_started": False}, ensure_ascii=False, indent=2), encoding="utf-8")
    QTimer.singleShot(0, session.begin)
    try:
        app.exec()
    except Exception as exc:
        session.tool_error = type(exc).__name__
    finally:
        session.stop()
        session.monitor.stop()
        for runtime in runtimes:
            runtime.stop()
        if session.player is not None:
            # The subprocess gets 2s grace, then terminate/kill with bounded waits.
            # Keep its owner alive through that cleanup even after a Qt exception.
            session.player.join(timeout=12)
            if session.player.is_alive():
                session.tool_error = "PlaybackCleanupTimeout"
        controller.shutdown()
        window.close()
        app.setQuitOnLastWindowClosed(previous_quit)
        report = session.report()
        if args.engine_diagnostics:
            report["engine_diagnostics"] = engine_diagnostics
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if sys.stdout is not None:
        print(json.dumps({"report": str(args.report), "ok": report["ok"], "transcripts_saved": report["transcripts_saved"]}))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
