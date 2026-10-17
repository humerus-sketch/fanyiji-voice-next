"""Offline tests: no microphone access, model weights, or network requests."""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import queue
import sys
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from fanyiji_next.audio_runtime import (
    VoiceRuntime, _LocalEngine, _LoadingCancelled, _Segment, _Segmenter, _Session,
    _clean_transcript, _translation_config, list_audio_devices, validate_models,
    _PORTAUDIO_LIFETIME_LOCK, _CaptureSource, _microphone_vad_model,
)
from fanyiji_next.models import AppSettings, CaptionStore, ChannelSettings
from fanyiji_next.speech_gate import SpeechSlice


class ModelValidationTests(unittest.TestCase):
    def test_vad_discovery_is_limited_to_the_configured_asr_model_root(self):
        with tempfile.TemporaryDirectory() as folder:
            base = Path(folder)
            settings = AppSettings(asr_model=str(base / "asr" / "model.onnx"))
            self.assertIsNone(_microphone_vad_model(AppSettings()))
            self.assertIsNone(_microphone_vad_model(settings))
            model = base / "vad" / "silero_vad.onnx"
            model.parent.mkdir()
            model.write_bytes(b"fake model")
            self.assertEqual(model, _microphone_vad_model(settings))
            self.assertIsNone(_microphone_vad_model(replace(settings, asr_model=str(base / "other" / "asr" / "model.onnx"))))

    def test_empty_settings_explain_every_required_selection(self):
        errors = validate_models(AppSettings())
        self.assertEqual(4, len(errors))
        self.assertTrue(any("tokens.txt" in item for item in errors))
        self.assertTrue(any("Hy-MT2 Q8_0" in item for item in errors))
        self.assertTrue(any("llama-server" in item for item in errors))

    def test_valid_local_files_and_both_vocabulary_formats(self):
        with tempfile.TemporaryDirectory() as folder:
            base = Path(folder)
            for name in ("asr.onnx", "tokens.txt", "spm.model", "model.bin", "config.json"):
                (base / name).write_bytes(b"mock")
            settings = AppSettings(
                translation_backend="m2m100",
                asr_model=str(base / "asr.onnx"), asr_tokens=str(base / "tokens.txt"),
                translation_spm=str(base / "spm.model"), translation_model_dir=folder,
            )
            self.assertIn("vocabulary", validate_models(settings)[0])
            for name in ("source_vocabulary.json", "target_vocabulary.json"):
                (base / name).write_text("[]", encoding="utf-8")
            self.assertEqual([], validate_models(settings))
            missing = replace(settings, asr_model=str(base / "absent.onnx"))
            self.assertIn("absent.onnx", validate_models(missing)[0])


class SegmentationTests(unittest.TestCase):
    def test_silence_never_becomes_an_utterance(self):
        segmenter = _Segmenter(sample_rate=100)
        for _ in range(100):
            self.assertIsNone(segmenter.feed([0.0] * 10, 0.0))
        self.assertIsNone(segmenter.finish())

    def test_short_noise_spike_is_discarded(self):
        segmenter = _Segmenter(sample_rate=100, min_speech_seconds=0.2)
        segmenter.feed([0.5] * 5, 0.5)
        self.assertIsNone(segmenter.feed([0.0] * 60, 0.0))

    def test_utterance_preserves_pre_roll_and_ends_after_pause(self):
        segmenter = _Segmenter(sample_rate=100, silence_seconds=0.3)
        segmenter.feed([0.001] * 10, 0.001)
        self.assertIsNone(segmenter.feed([0.3] * 25, 0.3))
        result = segmenter.feed([0.0] * 30, 0.0)
        self.assertEqual([0.001] * 10 + [0.3] * 25 + [0.0] * 30, result)
        self.assertIsNone(segmenter.finish())

    def test_continuous_speech_is_bounded(self):
        segmenter = _Segmenter(sample_rate=100, max_seconds=1)
        self.assertIsNone(segmenter.feed([0.2] * 50, 0.2))
        self.assertEqual(100, len(segmenter.feed([0.2] * 50, 0.2)))

    def test_default_continuous_speech_produces_a_result_at_six_seconds(self):
        segmenter = _Segmenter()
        for _ in range(149):
            self.assertIsNone(segmenter.feed([0.2] * 640, 0.2))
        completed = segmenter.feed([0.2] * 640, 0.2)
        self.assertEqual(6 * 16000, len(completed))
        self.assertIsNone(segmenter.feed([0.2] * 640, 0.2))


class DeviceDiscoveryTests(unittest.TestCase):
    def test_classifies_inputs_and_terminates_without_opening_stream(self):
        pa = Mock()
        pa.get_device_count.return_value = 3
        pa.get_device_info_by_index.side_effect = [
            {"name": "Microphone", "maxInputChannels": 1},
            {"name": "Speakers", "maxInputChannels": 0},
            {"name": "Speakers loopback", "maxInputChannels": 2, "isLoopbackDevice": True},
        ]
        with patch.dict(sys.modules, {"pyaudiowpatch": SimpleNamespace(PyAudio=lambda: pa)}):
            devices = list_audio_devices()
        self.assertEqual([(0, "mic"), (2, "system")], [(d.index, d.channel) for d in devices])
        pa.open.assert_not_called()
        pa.terminate.assert_called_once()

    def test_device_error_still_releases_portaudio(self):
        pa = Mock()
        pa.get_device_count.side_effect = OSError("device unavailable")
        with patch.dict(sys.modules, {"pyaudiowpatch": SimpleNamespace(PyAudio=lambda: pa)}):
            with self.assertRaises(OSError):
                list_audio_devices()
        pa.terminate.assert_called_once()


class EngineTests(unittest.TestCase):
    def test_runtime_reports_model_cleanup_failure_after_stop(self):
        events = []
        runtime = VoiceRuntime(events.append)
        session = _Session(AppSettings(), 42)
        runtime._session = session
        engine = Mock()
        engine.close.side_effect = RuntimeError("fixture unload failed")

        def load(*args, **kwargs):
            session.stop.set()
            return engine

        with patch("fanyiji_next.audio_runtime.validate_models", return_value=[]), \
                patch("fanyiji_next.audio_runtime._LocalEngine", side_effect=load):
            runtime._run(session)
        failures = [event for event in events if event.kind == "error"]
        self.assertEqual(1, len(failures))
        self.assertEqual("cleanup_failed", failures[0].status)
        self.assertIn("fixture unload failed", failures[0].message)
        self.assertEqual("stopped", events[-1].status)

    def test_whisper_receives_language_from_each_channel_direction(self):
        events = []
        runtime = VoiceRuntime(events.append)
        session = _Session(AppSettings(recognition_backend="whisper"), 43)
        runtime._session = session
        engine = Mock()
        engine.recognize.return_value = "recognized"
        engine.translate.return_value = "translated"
        runtime._process_segment(session, engine, _Segment("mic", "m1", [0.1], "zh_to_ja", time.perf_counter()))
        runtime._process_segment(session, engine, _Segment("system", "s1", [0.2], "ja_to_zh", time.perf_counter()))
        self.assertEqual("zh", engine.recognize.call_args_list[0].kwargs["language"])
        self.assertEqual("ja", engine.recognize.call_args_list[1].kwargs["language"])

    def test_selected_adapters_receive_settings_and_source_language(self):
        settings = AppSettings(recognition_backend="whisper", translation_backend="nllb")
        recognition, translation = Mock(), Mock()
        recognition.recognize.return_value = "recognized"
        translation.translate.return_value = "translated"
        with patch("fanyiji_next.audio_runtime.create_recognition_engine", return_value=recognition) as rf, \
                patch("fanyiji_next.audio_runtime.create_translation_engine", return_value=translation) as tf:
            engine = _LocalEngine(settings)
        rf.assert_called_once_with(settings, stop_event=None)
        tf.assert_called_once_with(settings, stop_event=None)
        self.assertEqual("recognized", engine.recognize([0.1], language="ja"))
        recognition.recognize.assert_called_once_with([0.1], language="ja")
        self.assertEqual("translated", engine.translate("recognized", "ja_to_zh"))
        translation.translate.assert_called_once_with("recognized", "ja_to_zh")
        engine.close()
        engine.close()
        recognition.close.assert_called_once()
        translation.close.assert_called_once()

    def test_translation_constructor_failure_releases_recognition(self):
        recognition = Mock()
        with patch("fanyiji_next.audio_runtime.create_recognition_engine", return_value=recognition), \
                patch("fanyiji_next.audio_runtime.create_translation_engine", side_effect=RuntimeError("load failed")):
            with self.assertRaisesRegex(RuntimeError, "load failed"):
                _LocalEngine(AppSettings())
        recognition.close.assert_called_once()

    def test_stop_after_asr_loading_skips_translation_and_closes_asr(self):
        stop, recognition = threading.Event(), Mock()
        def load(settings, **kwargs):
            stop.set()
            return recognition
        with patch("fanyiji_next.audio_runtime.create_recognition_engine", side_effect=load), \
                patch("fanyiji_next.audio_runtime.create_translation_engine") as translation:
            with self.assertRaises(_LoadingCancelled):
                _LocalEngine(AppSettings(), stop_event=stop)
        translation.assert_not_called()
        recognition.close.assert_called_once()

    def test_stop_after_translation_loading_releases_both_models(self):
        stop, recognition, translation = threading.Event(), Mock(), Mock()
        def load(settings, **kwargs):
            stop.set()
            return translation
        with patch("fanyiji_next.audio_runtime.create_recognition_engine", return_value=recognition), \
                patch("fanyiji_next.audio_runtime.create_translation_engine", side_effect=load):
            with self.assertRaises(_LoadingCancelled):
                _LocalEngine(AppSettings(), stop_event=stop)
        translation.close.assert_called_once()
        recognition.close.assert_called_once()

    def test_adapter_cancellation_maps_to_runtime_cancellation(self):
        from fanyiji_next.recognition_engines import RecognitionCancelled
        from fanyiji_next.translation_engines import TranslationCancelled
        recognition, translation = Mock(), Mock()
        with patch("fanyiji_next.audio_runtime.create_recognition_engine", return_value=recognition), \
                patch("fanyiji_next.audio_runtime.create_translation_engine", return_value=translation):
            engine = _LocalEngine(AppSettings())
        recognition.recognize.side_effect = RecognitionCancelled("cancel")
        translation.translate.side_effect = TranslationCancelled("cancel")
        with self.assertRaises(_LoadingCancelled):
            engine.recognize([0.1])
        with self.assertRaises(_LoadingCancelled):
            engine.translate("text", "zh_to_ja")
        engine.close()

    def test_native_release_failure_still_releases_other_engine(self):
        recognition, translation = Mock(), Mock()
        translation.close.side_effect = RuntimeError("unload failed")
        with patch("fanyiji_next.audio_runtime.create_recognition_engine", return_value=recognition), \
                patch("fanyiji_next.audio_runtime.create_translation_engine", return_value=translation):
            engine = _LocalEngine(AppSettings())
        with self.assertRaisesRegex(RuntimeError, "unload failed"):
            engine.close()
        recognition.close.assert_called_once()
        self.assertIs(engine.translator, translation)
        translation.close.side_effect = None
        engine.close()
        recognition.close.assert_called_once()
        self.assertEqual(2, translation.close.call_count)
        self.assertIsNone(engine.translator)

    def test_sensevoice_metadata_is_removed(self):
        self.assertEqual("hello", _clean_transcript("<|zh|><|NEUTRAL|><|Speech|>hello<|withitn|>"))

    def test_legacy_conversion_policy_validation_remains_explicit(self):
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / "config.json"
            config.write_text(json.dumps({"add_source_eos": True}), encoding="utf-8")
            self.assertTrue(_translation_config(directory)["add_source_eos"])
            config.write_text(json.dumps({"add_source_eos": "true"}), encoding="utf-8")
            with self.assertRaises(RuntimeError):
                _translation_config(directory)


class RuntimeTests(unittest.TestCase):
    def setUp(self):
        # Every lifecycle test uses a fake manager, even when _capture itself is
        # replaced. No refactor of the production owner can open real hardware.
        self.pa = Mock()
        self.pa.get_default_input_device_info.return_value = {
            "index": 1, "maxInputChannels": 2, "defaultSampleRate": 48000,
        }
        self.pa.get_default_wasapi_loopback.return_value = {
            "index": 2, "maxInputChannels": 2, "defaultSampleRate": 48000,
            "isLoopbackDevice": True,
        }
        self.pa.open.side_effect = lambda **_kwargs: Mock(is_active=Mock(return_value=True))
        self.audio = SimpleNamespace(PyAudio=Mock(return_value=self.pa), paInt16=8, paComplete=1, paContinue=0)
        module_patch = patch.dict(sys.modules, {"pyaudiowpatch": self.audio})
        module_patch.start()
        self.addCleanup(module_patch.stop)

    def wait_for(self, predicate, timeout=2):
        deadline = time.perf_counter() + timeout
        while time.perf_counter() < deadline:
            if predicate():
                return
            time.sleep(0.005)
        self.fail("后台 mock 任务未在时限内结束")

    def test_constructor_does_not_load_engines_or_record(self):
        with patch("fanyiji_next.audio_runtime._LocalEngine") as engine:
            runtime = VoiceRuntime(Mock())
            self.assertFalse(runtime.is_running)
            engine.assert_not_called()

    def test_present_vad_load_failure_is_fatal_before_running_and_releases_streams(self):
        events = []
        runtime = VoiceRuntime(events.append)
        settings = AppSettings(system=ChannelSettings(enabled=False))
        with patch.dict(sys.modules, {"numpy": SimpleNamespace(), "soxr": SimpleNamespace()}), \
                patch("fanyiji_next.audio_runtime.validate_models", return_value=[]), \
                patch("fanyiji_next.audio_runtime._LocalEngine") as engine, \
                patch("fanyiji_next.audio_runtime._microphone_vad_model", return_value=Path("present.onnx")), \
                patch("fanyiji_next.audio_runtime.SileroSpeechGate", side_effect=RuntimeError("invalid VAD")):
            runtime.start(settings, 31)
            self.wait_for(lambda: any(e.status == "stopped" for e in events))
        error, = [event for event in events if event.kind == "error"]
        self.assertEqual("mic", error.channel)
        self.assertIn("Silero VAD 模型加载失败", error.message)
        self.assertIn("invalid VAD", error.message)
        self.assertFalse(any(event.status == "running" for event in events))
        self.assertFalse(runtime.is_busy)
        self.pa.terminate.assert_called_once()
        engine.return_value.close.assert_called_once()

    def test_missing_models_reports_error_then_stopped_without_loading(self):
        events = []
        runtime = VoiceRuntime(events.append)
        with patch("fanyiji_next.audio_runtime._LocalEngine") as engine:
            runtime.start(AppSettings(), 1)
            self.wait_for(lambda: any(e.status == "stopped" for e in events))
            engine.assert_not_called()
        self.assertTrue(any(e.kind == "error" and "SenseVoice" in e.message for e in events))
        self.assertFalse(runtime.is_running)

    def test_disabled_channels_and_invalid_direction_do_not_load_models(self):
        cases = [
            AppSettings(microphone=ChannelSettings(enabled=False), system=ChannelSettings(enabled=False)),
            AppSettings(microphone=ChannelSettings(direction="invalid"), system=ChannelSettings(enabled=False)),
        ]
        for settings in cases:
            with self.subTest(settings=settings):
                events = []
                runtime = VoiceRuntime(events.append)
                with patch("fanyiji_next.audio_runtime.validate_models", return_value=[]), \
                        patch("fanyiji_next.audio_runtime._LocalEngine") as engine:
                    runtime.start(settings, 1)
                    self.wait_for(lambda: any(e.status == "stopped" for e in events))
                    engine.assert_not_called()
                self.assertTrue(any(e.kind == "error" for e in events))

    def test_permission_failure_releases_device_and_reports_channel(self):
        events = []
        runtime = VoiceRuntime(events.append)
        session = _Session(AppSettings(), 1)
        runtime._session = session
        pa = Mock()
        pa.get_default_input_device_info.return_value = {
            "index": 2, "maxInputChannels": 1, "defaultSampleRate": 48000,
        }
        pa.open.side_effect = OSError("access denied")
        modules = {
            "pyaudiowpatch": SimpleNamespace(PyAudio=lambda: pa, paInt16=8),
            "numpy": SimpleNamespace(),
            "soxr": SimpleNamespace(ResampleStream=Mock()),
        }
        with patch.dict(sys.modules, modules), \
                patch("fanyiji_next.audio_runtime.validate_models", return_value=[]), \
                patch("fanyiji_next.audio_runtime._LocalEngine"):
            runtime._run(session)
        self.assertTrue(session.stop.is_set())
        error = next(event for event in events if event.kind == "error")
        self.assertEqual("mic", error.channel)
        self.assertIn("access denied", error.message)
        pa.terminate.assert_called_once()

    def test_explicit_wrong_channel_device_is_rejected_before_recording(self):
        runtime = VoiceRuntime(Mock())
        session = _Session(AppSettings(
            microphone=ChannelSettings(enabled=False), system=ChannelSettings(device_index=1),
        ), 1)
        runtime._session = session
        pa = Mock()
        pa.get_device_info_by_index.return_value = {"isLoopbackDevice": False}
        modules = {
            "pyaudiowpatch": SimpleNamespace(PyAudio=lambda: pa),
            "numpy": SimpleNamespace(), "soxr": SimpleNamespace(),
        }
        with patch.dict(sys.modules, modules), \
                patch("fanyiji_next.audio_runtime.validate_models", return_value=[]), \
                patch("fanyiji_next.audio_runtime._LocalEngine"):
            runtime._run(session)
        pa.open.assert_not_called()
        pa.terminate.assert_called_once()

    def test_queued_original_and_translation_share_channel_and_segment(self):
        events = []
        released = threading.Event()
        engine = Mock()
        engine.recognize.return_value = "你好"
        engine.translate.return_value = "こんにちは"
        runtime = VoiceRuntime(events.append)

        def capture(session, channel, settings, ready, source):
            session.segments.put(_Segment(channel, "s1", [0.1], settings.direction, time.perf_counter()))
            ready.set()
            session.stop.wait(2)
            released.set()

        settings = AppSettings(system=ChannelSettings(enabled=False))
        with patch("fanyiji_next.audio_runtime.validate_models", return_value=[]), \
                patch("fanyiji_next.audio_runtime._LocalEngine", return_value=engine), \
                patch.object(runtime, "_capture", side_effect=capture):
            runtime.start(settings, 11)
            self.wait_for(lambda: any(e.kind == "translation" for e in events))
            runtime.stop()
            self.wait_for(lambda: any(e.status == "stopped" for e in events))
        rows = [event for event in events if event.kind in {"original", "translation"}]
        self.assertEqual([("mic", "s1"), ("mic", "s1")], [(e.channel, e.segment_id) for e in rows])
        self.assertTrue(released.is_set())
        engine.close.assert_called_once()

    def test_stop_is_nonblocking_and_late_recognition_is_suppressed(self):
        events = []
        entered = threading.Event()
        release = threading.Event()
        engine = Mock()

        def recognize(_):
            entered.set()
            release.wait(2)
            return "late result"

        engine.recognize.side_effect = recognize
        runtime = VoiceRuntime(events.append)

        def capture(session, channel, settings, ready, source):
            session.segments.put(_Segment(channel, "s1", [0.1], settings.direction, time.perf_counter()))
            ready.set()
            session.stop.wait(2)

        with patch("fanyiji_next.audio_runtime.validate_models", return_value=[]), \
                patch("fanyiji_next.audio_runtime._LocalEngine", return_value=engine), \
                patch.object(runtime, "_capture", side_effect=capture):
            runtime.start(AppSettings(system=ChannelSettings(enabled=False)), 1)
            self.assertTrue(entered.wait(2))
            started = time.perf_counter()
            runtime.stop()
            self.assertLess(time.perf_counter() - started, 0.1)
            self.assertFalse(runtime.is_running)
            self.assertTrue(runtime.is_busy)
            release.set()
            self.wait_for(lambda: any(e.status == "stopped" for e in events))
        self.assertFalse(any(e.kind in {"original", "translation"} for e in events))
        engine.translate.assert_not_called()

    def test_loading_session_cannot_be_overwritten_until_released(self):
        events = []
        entered = threading.Event()
        release = threading.Event()
        engine = Mock()
        runtime = VoiceRuntime(events.append)

        def load(*_args, **_kwargs):
            entered.set()
            release.wait(2)
            return engine

        with patch("fanyiji_next.audio_runtime.validate_models", return_value=[]), \
                patch("fanyiji_next.audio_runtime._LocalEngine", side_effect=load), \
                patch.object(runtime, "_capture") as capture:
            runtime.start(AppSettings(), 10)
            try:
                self.assertTrue(entered.wait(2))
                runtime.stop()
                self.assertFalse(runtime.is_running)
                self.assertTrue(runtime.is_busy)
                with self.assertRaisesRegex(RuntimeError, "尚未释放"):
                    runtime.start(AppSettings(), 11)
            finally:
                release.set()
            self.wait_for(lambda: any(e.status == "stopped" for e in events))
            capture.assert_not_called()
        self.assertFalse(runtime.is_busy)
        engine.close.assert_called_once()
        self.assertFalse(any(e.kind == "error" for e in events))

    def test_stopped_callback_observes_fully_released_runtime(self):
        busy_when_stopped = []
        events = []

        def receive(event):
            events.append(event)
            if event.status == "stopped":
                busy_when_stopped.append(runtime.is_busy)

        runtime = VoiceRuntime(receive)
        runtime.start(AppSettings(), 2)
        self.wait_for(lambda: bool(busy_when_stopped))
        self.assertEqual([False], busy_when_stopped)

    def test_fatal_capture_error_precedes_stopped_and_releases_both_resources(self):
        events = []
        runtime = VoiceRuntime(events.append)
        engine = Mock()

        def fail_capture(session, channel, settings, ready, source):
            runtime._emit(session, "error", channel=channel, message="device permission denied")
            session.stop.set()
            ready.set()

        with patch("fanyiji_next.audio_runtime.validate_models", return_value=[]), \
                patch("fanyiji_next.audio_runtime._LocalEngine", return_value=engine), \
                patch.object(runtime, "_capture", side_effect=fail_capture):
            runtime.start(AppSettings(), 6)
            self.wait_for(lambda: any(e.status == "stopped" for e in events))
        kinds = [event.kind for event in events]
        self.assertIn("error", kinds)
        self.assertLess(kinds.index("error"), len(kinds) - 1)
        self.assertEqual("stopped", events[-1].status)
        self.assertFalse(runtime.is_busy)
        engine.close.assert_called_once()

    def test_same_runtime_can_restart_after_stopped_without_old_events(self):
        events = []
        runtime = VoiceRuntime(events.append)
        created = []

        def create_engine(*_args, **_kwargs):
            engine = Mock()
            created.append(engine)
            return engine

        def capture(session, channel, settings, ready, source):
            ready.set()
            session.stop.wait(2)

        with patch("fanyiji_next.audio_runtime.validate_models", return_value=[]), \
                patch("fanyiji_next.audio_runtime._LocalEngine", side_effect=create_engine), \
                patch.object(runtime, "_capture", side_effect=capture):
            for session_id in (1, 2):
                runtime.start(AppSettings(system=ChannelSettings(enabled=False)), session_id)
                self.wait_for(lambda: any(e.session_id == session_id and e.status == "running" for e in events))
                runtime.stop()
                self.wait_for(lambda: any(e.session_id == session_id and e.status == "stopped" for e in events))
                self.assertFalse(runtime.is_busy)
        self.assertEqual(2, len(created))
        for engine in created:
            engine.close.assert_called_once()
        first_second_session = next(i for i, event in enumerate(events) if event.session_id == 2)
        self.assertTrue(all(event.session_id == 2 for event in events[first_second_session:]))
        running = [event for event in events if event.status == "running"]
        self.assertEqual(2, len(running))
        self.assertTrue(all("基础能量断句" in event.message for event in running))

    def test_capture_thread_start_failure_still_unloads_engine_and_stops(self):
        events = []
        runtime = VoiceRuntime(events.append)
        session = _Session(AppSettings(), 9)
        runtime._session = session
        capture = Mock()
        capture.start.side_effect = RuntimeError("cannot create capture thread")
        engine = Mock()
        with patch("fanyiji_next.audio_runtime.validate_models", return_value=[]), \
                patch("fanyiji_next.audio_runtime._LocalEngine", return_value=engine), \
                patch("fanyiji_next.audio_runtime.threading.Thread", return_value=capture):
            runtime._run(session)
        capture.join.assert_not_called()
        engine.close.assert_called_once()
        self.assertEqual("stopped", events[-1].status)
        self.assertTrue(any(e.kind == "error" and "capture thread" in e.message for e in events))
        self.assertFalse(runtime.is_busy)

    def test_segment_queue_is_bounded_and_reports_dropped_newest(self):
        events = []
        runtime = VoiceRuntime(events.append)
        session = _Session(AppSettings(), 4)
        runtime._session = session
        for index in range(7):
            runtime._enqueue_segment(session, _Segment("mic", str(index), [0.1], "zh_to_ja", time.perf_counter()))
        self.assertEqual(6, session.segments.qsize())
        self.assertEqual([str(i) for i in range(6)], [session.segments.get_nowait().segment_id for _ in range(6)])
        depths = [e.queue_depth for e in events if e.kind == "queue"]
        self.assertEqual([1, 2, 3, 4, 5, 6, 6], depths)
        self.assertTrue(any(e.kind == "error" and e.status == "recoverable" for e in events))

    def test_stop_before_processing_skips_both_engines(self):
        runtime = VoiceRuntime(Mock())
        session = _Session(AppSettings(), 1)
        session.stop.set()
        engine = Mock()
        runtime._process_segment(session, engine, _Segment("mic", "s", [0.1], "zh_to_ja", time.perf_counter()))
        engine.recognize.assert_not_called()
        engine.translate.assert_not_called()

    def test_reused_session_id_still_discards_previous_session(self):
        events = []
        runtime = VoiceRuntime(events.append)
        old = _Session(AppSettings(), 7)
        new = _Session(AppSettings(), 7)
        runtime._session = new
        runtime._emit(old, "translation", channel="mic", segment_id="s", text="stale")
        runtime._emit(old, "state", status="stopped")
        self.assertEqual([], events)

    def test_translation_failure_preserves_original_and_identifies_row(self):
        events = []
        runtime = VoiceRuntime(events.append)
        session = _Session(AppSettings(), 3)
        runtime._session = session
        engine = Mock()
        engine.recognize.return_value = "你好"
        engine.translate.side_effect = RuntimeError("invalid model")
        runtime._process_segment(session, engine, _Segment("mic", "s", [0.2], "zh_to_ja", time.perf_counter()))
        self.assertEqual(["original", "error"], [e.kind for e in events])
        self.assertEqual("s", events[-1].segment_id)
        self.assertIn("invalid model", events[-1].message)

    def test_recognition_error_does_not_attempt_translation(self):
        events = []
        runtime = VoiceRuntime(events.append)
        session = _Session(AppSettings(), 3)
        runtime._session = session
        engine = Mock()
        engine.recognize.side_effect = ValueError("bad samples")
        runtime._process_segment(session, engine, _Segment("system", "s", [0.2], "ja_to_zh", time.perf_counter()))
        engine.translate.assert_not_called()
        self.assertEqual("error", events[0].kind)
        self.assertEqual("system", events[0].channel)


class OwnerLifecycleTests(unittest.TestCase):
    """Assert actual production lifecycle calls with a thread-recording PA fake."""

    def wait_for(self, predicate, timeout=2):
        RuntimeTests.wait_for(self, predicate, timeout)

    def fake_audio(self, calls, *, fail_second_open=False):
        def record(name, channel=""):
            calls.append((name, channel, threading.get_ident()))

        class Stream:
            def __init__(self, channel):
                self.channel = channel

            def start_stream(self):
                record("start", self.channel)

            def is_active(self):
                record("active", self.channel)
                return True

            def stop_stream(self):
                record("stop", self.channel)

            def close(self):
                record("close", self.channel)

        class Manager:
            def __init__(self):
                record("initialize")

            def get_default_input_device_info(self):
                record("device", "mic")
                return {"index": 0, "name": "fake mic", "maxInputChannels": 2, "defaultSampleRate": 48000}

            def get_default_wasapi_loopback(self):
                record("device", "system")
                return {
                    "index": 1, "name": "fake loopback", "maxInputChannels": 2,
                    "defaultSampleRate": 48000, "isLoopbackDevice": True,
                }

            def get_device_count(self):
                record("device_count")
                return 2

            def get_device_info_by_index(self, index):
                return self.get_default_input_device_info() if index == 0 else self.get_default_wasapi_loopback()

            def open(self, **kwargs):
                channel = "mic" if kwargs["input_device_index"] == 0 else "system"
                record("open", channel)
                if fail_second_open and channel == "system":
                    raise OSError("fake loopback unavailable")
                return Stream(channel)

            def terminate(self):
                record("terminate")

        return SimpleNamespace(PyAudio=Manager, paInt16=8, paComplete=1, paContinue=0)

    @staticmethod
    def consumer(session, channel, settings, ready, source):
        ready.set()
        session.stop.wait(2)

    def test_both_channels_use_one_manager_and_one_native_owner_thread(self):
        calls, events, consumer_threads = [], [], []
        runtime = VoiceRuntime(events.append)
        audio = self.fake_audio(calls)

        def consume(session, channel, settings, ready, source):
            consumer_threads.append(threading.get_ident())
            self.assertFalse(hasattr(source, "stream"))
            self.consumer(session, channel, settings, ready, source)

        with patch.dict(sys.modules, {"pyaudiowpatch": audio}), \
                patch("fanyiji_next.audio_runtime.validate_models", return_value=[]), \
                patch("fanyiji_next.audio_runtime._LocalEngine"), \
                patch.object(runtime, "_capture", side_effect=consume):
            runtime.start(AppSettings(), 1)
            self.wait_for(lambda: any(e.status == "running" for e in events))
            runtime.stop()
            self.wait_for(lambda: any(e.status == "stopped" for e in events))
        self.assertEqual(1, sum(name == "initialize" for name, _channel, _thread in calls))
        self.assertEqual(1, sum(name == "terminate" for name, _channel, _thread in calls))
        native_threads = {thread for _name, _channel, thread in calls}
        self.assertEqual(1, len(native_threads))
        self.assertNotIn(threading.get_ident(), native_threads)
        self.assertTrue(all(thread not in native_threads for thread in consumer_threads))
        lifecycle = [(name, channel) for name, channel, _thread in calls if name in {"open", "start", "stop", "close", "terminate"}]
        self.assertEqual([
            ("open", "mic"), ("open", "system"), ("start", "mic"), ("start", "system"),
            ("stop", "system"), ("close", "system"), ("stop", "mic"), ("close", "mic"), ("terminate", ""),
        ], lifecycle)

    def test_second_open_failure_closes_first_stream_before_single_termination(self):
        calls, events = [], []
        runtime = VoiceRuntime(events.append)
        audio = self.fake_audio(calls, fail_second_open=True)
        engine = Mock()
        with patch.dict(sys.modules, {"pyaudiowpatch": audio}), \
                patch("fanyiji_next.audio_runtime.validate_models", return_value=[]), \
                patch("fanyiji_next.audio_runtime._LocalEngine", return_value=engine):
            runtime.start(AppSettings(), 3)
            self.wait_for(lambda: any(e.status == "stopped" for e in events))
        error = next(event for event in events if event.kind == "error")
        self.assertEqual("system", error.channel)
        self.assertIn("fake loopback unavailable", error.message)
        self.assertFalse(any(name == "start" for name, _channel, _thread in calls))
        self.assertEqual([("stop", "mic"), ("close", "mic"), ("terminate", "")], [
            (name, channel) for name, channel, _thread in calls if name in {"stop", "close", "terminate"}
        ])
        self.assertEqual(1, len({thread for _name, _channel, thread in calls}))
        self.assertFalse(runtime.is_busy)
        engine.close.assert_called_once()

    def test_repeated_both_channel_start_stop_has_balanced_owned_lifetimes(self):
        calls, events = [], []
        runtime = VoiceRuntime(events.append)
        audio = self.fake_audio(calls)
        with patch.dict(sys.modules, {"pyaudiowpatch": audio}), \
                patch("fanyiji_next.audio_runtime.validate_models", return_value=[]), \
                patch("fanyiji_next.audio_runtime._LocalEngine"), \
                patch.object(runtime, "_capture", side_effect=self.consumer):
            for session_id in range(3):
                runtime.start(AppSettings(), session_id)
                self.wait_for(lambda: any(e.session_id == session_id and e.status == "running" for e in events))
                runtime.stop()
                self.wait_for(lambda: any(e.session_id == session_id and e.status == "stopped" for e in events))
                self.assertFalse(runtime.is_busy)
        self.assertEqual(3, sum(name == "initialize" for name, _channel, _thread in calls))
        self.assertEqual(3, sum(name == "terminate" for name, _channel, _thread in calls))
        self.assertEqual(6, sum(name == "close" for name, _channel, _thread in calls))
        owner = None
        for name, _channel, thread in calls:
            if name == "initialize":
                self.assertIsNone(owner)
                owner = thread
            self.assertEqual(owner, thread)
            if name == "terminate":
                owner = None
        self.assertIsNone(owner)

    def test_device_discovery_cannot_initialize_while_capture_owner_is_active(self):
        calls, events = [], []
        runtime = VoiceRuntime(events.append)
        audio = self.fake_audio(calls)
        with patch.dict(sys.modules, {"pyaudiowpatch": audio}), \
                patch("fanyiji_next.audio_runtime.validate_models", return_value=[]), \
                patch("fanyiji_next.audio_runtime._LocalEngine"), \
                patch.object(runtime, "_capture", side_effect=self.consumer):
            runtime.start(AppSettings(), 5)
            self.wait_for(lambda: any(e.status == "running" for e in events))
            try:
                with self.assertRaisesRegex(RuntimeError, "先停止"):
                    list_audio_devices()
                self.assertEqual(1, sum(name == "initialize" for name, _channel, _thread in calls))
            finally:
                runtime.stop()
                self.wait_for(lambda: any(e.status == "stopped" for e in events))
            self.assertEqual(2, len(list_audio_devices()))
        self.assertEqual(2, sum(name == "initialize" for name, _channel, _thread in calls))
        self.assertEqual(2, sum(name == "terminate" for name, _channel, _thread in calls))

    def test_waiting_for_discovery_lifetime_can_be_cancelled_without_initializing(self):
        events = []
        runtime = VoiceRuntime(events.append)
        audio = SimpleNamespace(PyAudio=Mock())
        engine = Mock()
        with _PORTAUDIO_LIFETIME_LOCK, \
                patch.dict(sys.modules, {"pyaudiowpatch": audio}), \
                patch("fanyiji_next.audio_runtime.validate_models", return_value=[]), \
                patch("fanyiji_next.audio_runtime._LocalEngine", return_value=engine) as factory:
            runtime.start(AppSettings(), 7)
            self.wait_for(lambda: factory.called)
            runtime.stop()
            self.wait_for(lambda: any(e.status == "stopped" for e in events))
            audio.PyAudio.assert_not_called()
        self.assertFalse(runtime.is_busy)
        engine.close.assert_called_once()


class TimingTests(unittest.TestCase):
    def process(self, *, legacy=False, fail_at=None):
        now = [100.75]
        events = []
        runtime = VoiceRuntime(events.append)
        session = _Session(AppSettings(), 22)
        runtime._session = session
        segment = _Segment("system", "timed", [0.2], "ja_to_zh", 100.55)
        if not legacy:
            segment.speech_start_at = 99.0
            segment.speech_end_at = 100.0
            segment.segment_submitted_at = 100.55
            segment.dequeued_at = 100.75

        def recognize(_samples):
            now[0] += 0.2
            if fail_at == "asr":
                raise RuntimeError("fake ASR failure")
            return "こんにちは"

        def translate(_text, _direction, *, context_id="default"):
            now[0] += 0.3
            if fail_at == "translation":
                raise RuntimeError("fake translation failure")
            return "你好"

        engine = SimpleNamespace(recognize=recognize, translate=translate)
        with patch("fanyiji_next.audio_runtime.time.perf_counter", side_effect=lambda: now[0]):
            runtime._process_segment(session, engine, segment)
        return events

    def test_stage_durations_include_endpoint_and_queue_for_both_caption_events(self):
        original, translation = self.process()
        self.assertEqual(("original", "translation"), (original.kind, translation.kind))
        for event in (original, translation):
            self.assertEqual(99.0, event.speech_start_at)
            self.assertEqual(100.0, event.speech_end_at)
            self.assertEqual(100.55, event.segment_submitted_at)
            self.assertEqual(550, event.endpoint_ms)
            self.assertEqual(200, event.queue_ms)
            self.assertEqual(200, event.asr_ms)
        self.assertEqual(0, original.translation_ms)
        self.assertEqual(950, original.elapsed_ms)
        self.assertAlmostEqual(100.95, original.emitted_at)
        self.assertEqual(300, translation.translation_ms)
        self.assertEqual(1250, translation.elapsed_ms)
        self.assertAlmostEqual(101.25, translation.emitted_at)
        self.assertEqual(
            translation.elapsed_ms,
            translation.endpoint_ms + translation.queue_ms + translation.asr_ms + translation.translation_ms,
        )

    def test_five_positional_segment_keeps_captured_time_fallback(self):
        original, translation = self.process(legacy=True)
        self.assertEqual(0.0, original.speech_start_at)
        self.assertEqual(100.55, original.speech_end_at)
        self.assertEqual(0, original.endpoint_ms)
        self.assertEqual(200, original.queue_ms)
        self.assertEqual(400, original.elapsed_ms)
        self.assertEqual(700, translation.elapsed_ms)

    def test_asr_failure_reports_elapsed_stage_time_without_translation_time(self):
        event, = self.process(fail_at="asr")
        self.assertEqual("error", event.kind)
        self.assertEqual("timed", event.segment_id)
        self.assertEqual("recoverable", event.status)
        self.assertEqual(200, event.asr_ms)
        self.assertEqual(0, event.translation_ms)
        self.assertEqual(950, event.elapsed_ms)

    def test_translation_failure_preserves_original_and_reports_both_stage_times(self):
        original, error = self.process(fail_at="translation")
        self.assertEqual("original", original.kind)
        self.assertEqual("error", error.kind)
        self.assertEqual(original.segment_id, error.segment_id)
        self.assertEqual(200, error.asr_ms)
        self.assertEqual(300, error.translation_ms)
        self.assertEqual(1250, error.elapsed_ms)

    def test_submission_time_is_recorded_at_the_queue_boundary(self):
        runtime = VoiceRuntime(Mock())
        session = _Session(AppSettings(), 22)
        runtime._session = session
        segment = _Segment("mic", "timed", [0.2], "zh_to_ja", 100.55, speech_end_at=100.0)
        with patch("fanyiji_next.audio_runtime.time.perf_counter", return_value=100.60):
            runtime._enqueue_segment(session, segment)
        self.assertEqual(100.60, segment.segment_submitted_at)
        self.assertIs(segment, session.segments.get_nowait())


    def test_segment_detector_and_end_reason_are_preserved_on_both_caption_events(self):
        events = []
        runtime = VoiceRuntime(events.append)
        session = _Session(AppSettings(), 22)
        runtime._session = session
        segment = _Segment("mic", "vad", [0.2], "zh_to_ja", time.perf_counter(),
                           end_reason="vad_silence", speech_detector="silero")
        engine = SimpleNamespace(recognize=lambda _: "fixture", translate=lambda *_: "fixture translation")
        runtime._process_segment(session, engine, segment)
        self.assertEqual([("silero", "vad_silence")] * 2, [(e.speech_detector, e.end_reason) for e in events])


class CapturePipelineTests(unittest.TestCase):
    """Real NumPy/SoXR processing with a fake PortAudio manager and byte source."""

    def wait_for(self, predicate, timeout=2):
        RuntimeTests.wait_for(self, predicate, timeout)

    def processing_modules(self):
        try:
            import numpy as np
            import soxr
        except ImportError:
            self.skipTest("合成音频检查需要项目 voice 可选依赖中的 numpy/soxr")
        return np, soxr

    def run_vad_script(self, gate, script, *, session_id=41, channel="mic", sample_rate=16000, channels=1,
                       real_resampler=False):
        """Run the real consumer with deterministic bytes/QPC and no audio API."""
        self.processing_modules()
        events = []
        runtime = VoiceRuntime(events.append)
        session = _Session(AppSettings(), session_id)
        runtime._session = session
        ready = threading.Event()
        source = _CaptureSource(sample_rate, channels)
        now = [100.0]
        actions = iter(script)

        class ScriptedQueue:
            def get(self, timeout):
                action = next(actions, None)
                if action is None:
                    session.stop.set()
                    raise queue.Empty
                if isinstance(action, float):
                    now[0] = action
                    raise queue.Empty
                if len(action) == 3:
                    source.discontinuity.set()
                raw, captured_at = action[:2]
                now[0] = captured_at
                return raw, captured_at

            def empty(self):
                # Remaining scripted actions are future callbacks, not backlog.
                return True

        source.chunks = ScriptedQueue()
        if real_resampler:
            _, native_soxr = self.processing_modules()
            resampler = native_soxr.ResampleStream(sample_rate, 16000, 1, dtype="float32")
        else:
            resampler = SimpleNamespace(resample_chunk=lambda samples, last=False: samples, clear=Mock())
        soxr = SimpleNamespace(ResampleStream=Mock(return_value=resampler))
        with patch.dict(sys.modules, {"soxr": soxr}), \
                patch("fanyiji_next.audio_runtime._microphone_vad_model", return_value=Path("fixture.onnx")), \
                patch("fanyiji_next.audio_runtime.SileroSpeechGate", return_value=gate), \
                patch("fanyiji_next.audio_runtime.time.perf_counter", side_effect=lambda: now[0]):
            runtime._capture(session, channel, ChannelSettings(direction="ja_to_zh" if channel == "system" else "zh_to_ja"), ready, source)
        self.assertTrue(ready.is_set())
        self.assertFalse(any(event.kind == "error" and event.status != "recoverable" for event in events))
        return list(session.segments.queue), events, gate, resampler

    def test_vad_sample_end_maps_to_qpc_without_native_hangover(self):
        np, _ = self.processing_modules()
        raw = np.full(640, 3200, dtype=np.int16).tobytes()
        gate = Mock()
        gate.feed.side_effect = [[]] * 13 + [[SpeechSlice([0.2] * 3200, 0, 3200, "vad_silence")]]
        script = [(raw, 100.0 + (index + 1) * 0.04) for index in range(14)]
        segments, events, _, _ = self.run_vad_script(gate, script)
        segment, = segments
        self.assertAlmostEqual(100.0, segment.speech_start_at)
        self.assertAlmostEqual(100.2, segment.speech_end_at)
        self.assertAlmostEqual(100.56, segment.captured_at)
        self.assertEqual(("silero", "vad_silence"), (segment.speech_detector, segment.end_reason))
        self.assertEqual(360, round((segment.segment_submitted_at - segment.speech_end_at) * 1000))
        gate.finish.assert_not_called()
        gate.reset.assert_called_once()
        self.assertTrue(any(event.kind == "level" and event.level > 0 for event in events))

    def test_real_capture_ids_preserve_prior_caption_history_after_a_new_session(self):
        np, _ = self.processing_modules()
        raw = np.full(640, 3200, dtype=np.int16).tobytes()
        store = CaptionStore()
        for session_id in (41, 42):
            gate = Mock()
            gate.feed.return_value = [SpeechSlice([0.2] * 512, 0, 512, "vad_silence")]
            segments, _, _, _ = self.run_vad_script(gate, [(raw, 100.04)], session_id=session_id)
            segment, = segments
            self.assertEqual(f"mic-{session_id}-1", segment.segment_id)
            store.add_original("mic", segment.segment_id, f"original {session_id}")
            store.set_translation("mic", segment.segment_id, f"translation {session_id}")
        self.assertEqual(["mic-41-1", "mic-42-1"], [caption.id for caption in store.items("mic")])
        self.assertEqual(["translation 41", "translation 42"], [caption.translated for caption in store.items("mic")])

    def test_vad_callback_gap_finishes_tail_once_and_resets_qpc_origin(self):
        np, _ = self.processing_modules()
        raw = np.full(512, 9600, dtype=np.int16).tobytes()
        partial = np.full(200, 9600, dtype=np.int16).tobytes()
        gate = Mock()
        gate.feed.return_value = []
        gate.finish.side_effect = [[SpeechSlice([0.3] * 4296, 0, 4296, "vad_silence")]] * 2
        script = []
        for origin in (100.0, 200.0):
            script.extend((raw, origin + (index + 1) * 0.032) for index in range(8))
            script.extend([(partial, origin + 4296 / 16000), origin + 0.7, origin + 1.2])
        segments, events, _, resampler = self.run_vad_script(gate, script)
        self.assertEqual(2, len(segments))
        for segment, origin in zip(segments, (100.0, 200.0)):
            self.assertAlmostEqual(origin, segment.speech_start_at)
            self.assertAlmostEqual(origin + 4296 / 16000, segment.speech_end_at)
            self.assertEqual("input_gap", segment.end_reason)
            self.assertEqual(4296, len(segment.samples))
        self.assertEqual(2, gate.finish.call_count)
        self.assertEqual(2, resampler.clear.call_count)
        self.assertEqual(2, len([e for e in events if e.kind == "level" and e.level == 0]))

    def test_vad_stop_discards_pending_sentence_without_flushing(self):
        np, _ = self.processing_modules()
        gate = Mock()
        gate.feed.return_value = []
        raw = np.full(640, 9600, dtype=np.int16).tobytes()
        segments, _, _, _ = self.run_vad_script(gate, [(raw, 100.04)] * 10)
        self.assertEqual([], segments)
        gate.finish.assert_not_called()
        gate.reset.assert_called_once()

    def test_vad_overflow_discards_old_speech_and_restarts_sample_clock(self):
        np, _ = self.processing_modules()
        gate = Mock()
        gate.feed.side_effect = [[], [SpeechSlice([0.3] * 512, 0, 512, "max_duration")]]
        raw = np.full(512, 9600, dtype=np.int16).tobytes()
        segments, events, _, resampler = self.run_vad_script(gate, [
            (raw, 100.032), (raw, 100.064, "overflow"), (raw, 200.032),
        ])
        segment, = segments
        self.assertAlmostEqual(200.0, segment.speech_start_at)
        self.assertAlmostEqual(200.032, segment.speech_end_at)
        self.assertEqual(2, gate.reset.call_count)
        self.assertEqual(1, resampler.clear.call_count)
        self.assertTrue(any(e.status == "recoverable" and "缓冲溢出" in e.message for e in events))

    def test_system_uses_vad_without_energy_hard_gate_and_routes_real_channel_and_speech_clock(self):
        np, _ = self.processing_modules()
        # This input is below the legacy .012 energy gate. VAD has sole control
        # over speech acceptance; weak source must not be removed a second time.
        raw = np.full(640, 100, dtype=np.int16).tobytes()
        gate = Mock()
        padded = SpeechSlice([.003] * 4800, 0, 4800, "vad_silence", 512, 3200)
        gate.feed.side_effect = [[]] * 13 + [[padded]]
        script = [(raw, 100.0 + (index + 1) * .04) for index in range(14)]
        segments, events, _, _ = self.run_vad_script(gate, script, channel="system")
        segment, = segments
        self.assertEqual(("system", "system-41-1", "ja_to_zh"), (segment.channel, segment.segment_id, segment.direction))
        self.assertEqual(padded.samples, segment.samples)
        self.assertAlmostEqual(100.032, segment.speech_start_at)
        self.assertAlmostEqual(100.2, segment.speech_end_at)
        self.assertEqual(("silero", "vad_silence"), (segment.speech_detector, segment.end_reason))
        self.assertTrue(any(event.kind == "level" and event.level > 0 for event in events))
        self.assertTrue(all(event.channel == "system" for event in events if event.kind in {"level", "queue", "error"}))
        gate.finish.assert_not_called()
        gate.reset.assert_called_once()

    def test_system_vad_gap_stop_and_discontinuity_keep_source_ownership(self):
        np, _ = self.processing_modules()
        raw = np.full(512, 9600, dtype=np.int16).tobytes()
        gate = Mock()
        gate.feed.return_value = []
        gate.finish.return_value = [SpeechSlice([.3] * 512, 0, 512, "vad_silence", 0, 400)]
        segments, events, _, resampler = self.run_vad_script(gate, [(raw, 100.032), 100.5, 101.0], channel="system")
        segment, = segments
        self.assertEqual(("system", "system-41-1", "input_gap"), (segment.channel, segment.segment_id, segment.end_reason))
        self.assertAlmostEqual(100.025, segment.speech_end_at)
        self.assertEqual(1, gate.finish.call_count)
        self.assertEqual(1, resampler.clear.call_count)
        self.assertTrue(all(event.channel == "system" for event in events if event.kind in {"level", "queue"}))
        gate = Mock()
        gate.feed.return_value = []
        segments, _, _, _ = self.run_vad_script(gate, [(raw, 100.032)], channel="system")
        self.assertEqual([], segments)
        gate.finish.assert_not_called()
        gate.reset.assert_called_once()
        gate = Mock()
        gate.feed.side_effect = [[], [SpeechSlice([.3] * 512, 0, 512, "max_duration")]]
        segments, events, _, resampler = self.run_vad_script(gate, [
            (raw, 100.032), (raw, 100.064, "overflow"), (raw, 200.032),
        ], channel="system")
        segment, = segments
        self.assertEqual("system-41-1", segment.segment_id)
        self.assertAlmostEqual(200.0, segment.speech_start_at)
        self.assertAlmostEqual(200.032, segment.speech_end_at)
        self.assertEqual(2, gate.reset.call_count)
        self.assertEqual(1, resampler.clear.call_count)
        errors = [event for event in events if event.kind == "error"]
        self.assertEqual(["system"], [event.channel for event in errors])

    def test_system_vad_downmixes_and_resamples_before_detection(self):
        np, _ = self.processing_modules()
        gate = Mock()
        gate.feed.return_value = []
        script = []
        for index in range(14):
            times = (np.arange(1920) + index * 1920) / 48000
            mono = .2 * np.sin(2 * np.pi * 220 * times)
            stereo = mono[:, None] * np.array([[.5, 1.5]])
            raw = (stereo * 32767).astype(np.int16).tobytes()
            script.append((raw, 100.0 + (index + 1) * .04))
        segments, events, _, _ = self.run_vad_script(gate, script, channel="system", sample_rate=48000,
                                                   channels=2, real_resampler=True)
        self.assertEqual([], segments)  # Stop discards partial speech.
        processed = [sample for call in gate.feed.call_args_list for sample in call.args[0]]
        self.assertGreater(len(processed), 8000)
        self.assertLessEqual(len(processed), 8960)
        self.assertGreater(max(processed), .19)
        self.assertLess(max(processed), .21)
        self.assertTrue(all(isinstance(sample, float) for sample in processed))
        self.assertTrue(all(event.channel == "system" for event in events if event.kind == "level"))
        gate.finish.assert_not_called()

    def test_system_vad_keeps_bounded_queue_and_reports_full_queue_on_correct_source(self):
        np, _ = self.processing_modules()
        gate = Mock()
        gate.feed.return_value = [SpeechSlice([.2] * 512, index * 512, (index + 1) * 512, "vad_silence")
                                  for index in range(8)]
        raw = np.full(4096, 6400, dtype=np.int16).tobytes()
        segments, events, _, _ = self.run_vad_script(gate, [(raw, 100.256)], channel="system")
        self.assertEqual(6, len(segments))
        self.assertEqual([f"system-41-{index}" for index in range(1, 7)], [item.segment_id for item in segments])
        errors = [event for event in events if event.kind == "error"]
        self.assertEqual(2, len(errors))
        self.assertTrue(all(event.channel == "system" and event.status == "recoverable" for event in errors))
        self.assertTrue(all("识别队列已满" in event.message for event in errors))

    @staticmethod
    def consume_with_fake_owner(runtime, session, channel, settings, ready):
        """Feed the DSP-only consumer without an inference worker draining it.

        All manager/stream methods in this fixture are mocks; production owner
        thread ownership and cleanup are checked separately below.
        """
        import pyaudiowpatch as audio
        pa = audio.PyAudio()
        item = None
        consumer = None
        try:
            item = runtime._open_input(session, pa, audio, channel, settings)
            consumer = threading.Thread(
                target=runtime._capture, args=(session, channel, settings, ready, item.source), daemon=True,
            )
            consumer.start()
            if ready.wait(1) and not session.stop.is_set():
                item.stream.start_stream()
            session.stop.wait(4)
        except Exception as exc:
            runtime._emit(session, "error", message=f"fake owner: {exc}")
        finally:
            session.stop.set()
            if item is not None:
                item.stream.stop_stream()
                item.stream.close()
            if consumer is not None:
                consumer.join(1)
            pa.terminate()

    def test_four_channel_48k_callback_audio_becomes_16k_mono_segments(self):
        np, soxr = self.processing_modules()
        for channel in ("mic", "system"):
            with self.subTest(channel=channel):
                events = []
                runtime = VoiceRuntime(events.append)
                session = _Session(AppSettings(), 15)
                runtime._session = session
                ready = threading.Event()
                closed = threading.Event()
                info = {
                    "index": 3, "maxInputChannels": 4, "defaultSampleRate": 48000,
                    "isLoopbackDevice": channel == "system",
                }
                pa = Mock()
                pa.get_default_input_device_info.return_value = info
                pa.get_default_wasapi_loopback.return_value = info
                chunks = []
                frame_count = 1920
                for index in range(30):
                    t = (np.arange(frame_count) + index * frame_count) / 48000
                    signal = 0.2 * np.sin(2 * np.pi * 220 * t) if index < 10 else np.zeros(frame_count)
                    multichannel = signal[:, None] * np.array([[0.5, 1.5, 0.5, 1.5]])
                    chunks.append((multichannel * 32767).astype(np.int16).tobytes())
                producer_threads = []
                stream = Mock()
                stream.is_active.return_value = True
                stream.stop_stream.side_effect = closed.set

                def open_stream(**kwargs):
                    def produce():
                        for raw in chunks:
                            if closed.is_set():
                                return
                            kwargs["stream_callback"](raw, frame_count, {}, 0)
                            closed.wait(0.003)

                    def start_stream():
                        producer = threading.Thread(target=produce, daemon=True)
                        producer_threads.append(producer)
                        producer.start()

                    stream.start_stream.side_effect = start_stream
                    return stream

                pa.open.side_effect = open_stream
                modules = {"pyaudiowpatch": SimpleNamespace(PyAudio=lambda: pa, paInt16=8, paComplete=1, paContinue=0)}
                worker = threading.Thread(target=self.consume_with_fake_owner, args=(runtime, session, channel, ChannelSettings(), ready), daemon=True)
                with patch.dict(sys.modules, modules):
                    worker.start()
                    try:
                        self.assertTrue(ready.wait(1))
                        self.wait_for(lambda: session.segments.qsize() > 0)
                    finally:
                        session.stop.set()
                        worker.join(1)
                        closed.set()
                        for producer in producer_threads:
                            producer.join(1)
                self.assertFalse(worker.is_alive())
                segment = session.segments.get_nowait()
                self.assertEqual(channel, segment.channel)
                self.assertEqual(f"{channel}-15-1", segment.segment_id)
                self.assertGreater(len(segment.samples), 12000)
                self.assertLess(len(segment.samples), 18000)
                self.assertGreater(max(segment.samples), 0.15)
                self.assertLess(max(segment.samples), 0.35)
                self.assertGreater(segment.speech_start_at, 0.0)
                self.assertGreaterEqual(segment.speech_end_at, segment.speech_start_at)
                self.assertGreater(segment.captured_at, segment.speech_end_at)
                self.assertGreaterEqual(segment.segment_submitted_at, segment.captured_at)
                self.assertTrue(any(e.kind == "level" and 0 < e.level <= 1 for e in events))
                self.assertFalse(any(e.kind == "error" for e in events))
                self.assertEqual(4, pa.open.call_args.kwargs["channels"])
                self.assertFalse(pa.open.call_args.kwargs["start"])
                stream.read.assert_not_called()
                stream.close.assert_called_once()
                pa.terminate.assert_called_once()

    def test_silent_callback_source_stops_without_waiting_for_audio_data(self):
        np, soxr = self.processing_modules()
        events = []
        runtime = VoiceRuntime(events.append)
        session = _Session(AppSettings(), 1)
        runtime._session = session
        ready = threading.Event()
        stream = Mock()
        stream.is_active.return_value = True
        pa = Mock()
        pa.get_default_wasapi_loopback.return_value = {
            "index": 2, "maxInputChannels": 2, "defaultSampleRate": 48000,
            "isLoopbackDevice": True,
        }
        pa.open.return_value = stream
        modules = {"pyaudiowpatch": SimpleNamespace(PyAudio=lambda: pa, paInt16=8, paComplete=1, paContinue=0)}
        worker = threading.Thread(target=self.consume_with_fake_owner, args=(runtime, session, "system", ChannelSettings(), ready), daemon=True)
        with patch.dict(sys.modules, modules):
            worker.start()
            self.assertTrue(ready.wait(1))
            started = time.perf_counter()
            runtime.stop()
            worker.join(0.5)
        self.assertFalse(worker.is_alive())
        self.assertLess(time.perf_counter() - started, 0.5)
        self.assertFalse(any(e.kind == "error" for e in events))
        callback = pa.open.call_args.kwargs["stream_callback"]
        self.assertEqual((None, 1), callback(b"late", 1, {}, 0))
        stream.read.assert_not_called()
        stream.close.assert_called_once()
        pa.terminate.assert_called_once()

    def test_loopback_tail_finishes_without_silent_callbacks_and_resumes_as_new_segment(self):
        np, soxr = self.processing_modules()
        for stop_early in (False, True):
            with self.subTest(stop_early=stop_early):
                events = []
                runtime = VoiceRuntime(events.append)
                session = _Session(AppSettings(), 17)
                runtime._session = session
                ready = threading.Event()
                closed = threading.Event()
                first_burst_done = threading.Event()
                resume = threading.Event()
                callback_times = []
                producer_threads = []
                samples = 0.2 * np.sin(2 * np.pi * 220 * np.arange(1920) / 48000)
                raw = (np.repeat(samples[:, None], 2, axis=1) * 32767).astype(np.int16).tobytes()
                pa = Mock()
                pa.get_default_wasapi_loopback.return_value = {
                    "index": 2, "maxInputChannels": 2, "defaultSampleRate": 48000,
                    "isLoopbackDevice": True,
                }
                stream = Mock()
                stream.is_active.return_value = True
                stream.stop_stream.side_effect = closed.set

                def open_stream(**kwargs):
                    def burst():
                        for _ in range(10):
                            if closed.is_set():
                                return
                            before = time.perf_counter()
                            kwargs["stream_callback"](raw, 1920, {}, 0)
                            callback_times.append((before, time.perf_counter()))
                            closed.wait(0.003)

                    def produce():
                        burst()
                        first_burst_done.set()
                        while not resume.is_set() and not closed.is_set():
                            closed.wait(0.01)
                        if not closed.is_set():
                            burst()

                    def start_stream():
                        producer = threading.Thread(target=produce, daemon=True)
                        producer_threads.append(producer)
                        producer.start()

                    stream.start_stream.side_effect = start_stream
                    return stream

                pa.open.side_effect = open_stream
                modules = {"pyaudiowpatch": SimpleNamespace(PyAudio=lambda: pa, paInt16=8, paComplete=1, paContinue=0)}
                worker = threading.Thread(target=self.consume_with_fake_owner, args=(runtime, session, "system", ChannelSettings(), ready), daemon=True)
                with patch.dict(sys.modules, modules):
                    worker.start()
                    try:
                        self.assertTrue(ready.wait(1))
                        self.assertTrue(first_burst_done.wait(1))
                        if not stop_early:
                            self.wait_for(lambda: session.segments.qsize() == 1)
                            first = session.segments.get_nowait()
                            self.assertEqual("system-17-1", first.segment_id)
                            # Flush SoXR's pending output: 0.4 s at 16 kHz.
                            self.assertEqual(6400, len(first.samples))
                            self.assertGreaterEqual(first.captured_at, callback_times[9][0])
                            self.assertLessEqual(first.captured_at, callback_times[9][1])
                            self.assertEqual(first.captured_at, first.speech_end_at)
                            self.assertGreaterEqual(first.speech_start_at, callback_times[0][0])
                            self.assertLessEqual(first.speech_start_at, callback_times[0][1])
                            self.assertGreaterEqual(first.segment_submitted_at - first.speech_end_at, 0.55)
                            resume.set()
                            self.wait_for(lambda: session.segments.qsize() == 1)
                            second = session.segments.get_nowait()
                            self.assertEqual("system-17-2", second.segment_id)
                            self.assertEqual(6400, len(second.samples))
                            self.assertGreater(second.captured_at, first.captured_at)
                            self.assertGreater(second.speech_start_at, first.speech_end_at)
                    finally:
                        runtime.stop()
                        worker.join(1)
                        closed.set()
                        for producer in producer_threads:
                            producer.join(1)
                self.assertFalse(worker.is_alive())
                self.assertTrue(session.segments.empty())
                self.assertFalse(any(e.kind == "error" for e in events))
                stream.close.assert_called_once()
                pa.terminate.assert_called_once()

    def test_audio_buffer_overflow_discards_discontinuous_speech(self):
        np, soxr = self.processing_modules()
        events = []
        runtime = VoiceRuntime(events.append)
        session = _Session(AppSettings(), 3)
        runtime._session = session
        ready = threading.Event()
        stream = Mock()
        stream.is_active.return_value = True
        pa = Mock()
        pa.get_default_input_device_info.return_value = {
            "index": 2, "maxInputChannels": 1, "defaultSampleRate": 16000,
        }
        raw = np.full(640, 3200, dtype=np.int16).tobytes()

        def open_stream(**kwargs):
            def flood():
                for _ in range(80):
                    kwargs["stream_callback"](raw, 640, {}, 0)
            stream.start_stream.side_effect = flood
            return stream

        pa.open.side_effect = open_stream
        modules = {"pyaudiowpatch": SimpleNamespace(PyAudio=lambda: pa, paInt16=8, paComplete=1, paContinue=0)}
        worker = threading.Thread(target=self.consume_with_fake_owner, args=(runtime, session, "mic", ChannelSettings(), ready), daemon=True)
        with patch.dict(sys.modules, modules):
            worker.start()
            try:
                self.assertTrue(ready.wait(1))
                self.wait_for(lambda: any(e.kind == "error" for e in events))
            finally:
                runtime.stop()
                worker.join(1)
        self.assertFalse(worker.is_alive())
        self.assertTrue(session.segments.empty())
        self.assertTrue(any(e.status == "recoverable" and "缓冲溢出" in e.message for e in events))
        self.assertTrue(any(e.kind == "queue" and e.queue_depth == 0 for e in events))


if __name__ == "__main__":
    unittest.main()
