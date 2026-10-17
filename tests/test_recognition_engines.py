"""Recognizer contract tests use fake native objects and tiny local fixtures only."""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import sys
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from fanyiji_next.models import AppSettings
from fanyiji_next.recognition_engines import (
    RecognitionCancelled, SenseVoiceRecognitionEngine, WhisperRecognitionEngine,
    create_recognition_engine, validate_whisper_model_directory,
)


class RecognitionFixture(unittest.TestCase):
    def setUp(self):
        cuda_patch = patch("fanyiji_next.native_runtime.configure_cuda_runtime", return_value=())
        self.cuda_setup = cuda_patch.start()
        self.addCleanup(cuda_patch.stop)
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.base = Path(self.folder.name)
        self.whisper = self.base / "whisper"
        self.whisper.mkdir()
        (self.whisper / "model.bin").write_bytes(b"fake, never loaded")
        for name, value in {
            "config.json": {}, "tokenizer.json": {}, "vocabulary.json": [],
            "preprocessor_config.json": {"feature_size": 128, "sampling_rate": 16000},
        }.items():
            (self.whisper / name).write_text(json.dumps(value), encoding="utf-8")
        (self.base / "sense.onnx").write_bytes(b"fake, never loaded")
        (self.base / "tokens.txt").write_text("fake", encoding="utf-8")
        self.settings = AppSettings(
            asr_model=str(self.base / "sense.onnx"),
            asr_tokens=str(self.base / "tokens.txt"),
            whisper_model_dir=str(self.whisper),
        )

    def fake_sensevoice(self, *, text="<|zh|><|NEUTRAL|>你好。<|Speech|>", on_decode=None):
        stream = Mock()
        stream.result.text = text
        recognizer = Mock()
        recognizer.create_stream.return_value = stream
        recognizer.decode_stream.side_effect = on_decode
        constructor = Mock(return_value=recognizer)
        module = SimpleNamespace(OfflineRecognizer=SimpleNamespace(from_sense_voice=constructor))
        return module, constructor, recognizer, stream

    def fake_whisper(self, *, texts=(" 你好", " 世界。"), on_construct=None):
        model = Mock()
        model.transcribe.side_effect = lambda *_args, **_kwargs: (
            (SimpleNamespace(text=text) for text in texts), SimpleNamespace(language="zh"),
        )
        if on_construct is None:
            constructor = Mock(return_value=model)
        else:
            def construct(*args, **kwargs):
                on_construct()
                return model
            constructor = Mock(side_effect=construct)
        module = SimpleNamespace(WhisperModel=constructor)
        return module, constructor, model


class WhisperDirectoryValidationTests(RecognitionFixture):
    def test_complete_local_fixture_is_valid_without_native_loading(self):
        with patch.dict(sys.modules, {"faster_whisper": None}):
            self.assertEqual([], validate_whisper_model_directory(self.whisper))

    def test_empty_and_absent_directories_explain_selection(self):
        self.assertIn("尚未选择", validate_whisper_model_directory("")[0])
        self.assertIn("不存在", validate_whisper_model_directory(self.base / "absent")[0])

    def test_missing_tokenizer_is_rejected_before_constructor_can_download(self):
        (self.whisper / "tokenizer.json").unlink()
        module, constructor, _model = self.fake_whisper()
        with patch.dict(sys.modules, {"faster_whisper": module}):
            with self.assertRaisesRegex(RuntimeError, "tokenizer.json"):
                WhisperRecognitionEngine(self.settings)
        constructor.assert_not_called()

    def test_missing_vocabulary_is_reported(self):
        (self.whisper / "vocabulary.json").unlink()
        self.assertTrue(any("vocabulary.json" in item for item in validate_whisper_model_directory(self.whisper)))

    def test_preprocessor_requires_large_v3_mel_shape_and_16khz(self):
        path = self.whisper / "preprocessor_config.json"
        path.write_text('{"feature_size": 80, "sampling_rate": 48000}', encoding="utf-8")
        issues = validate_whisper_model_directory(self.whisper)
        self.assertTrue(any("128" in item for item in issues))
        self.assertTrue(any("16000" in item for item in issues))

    def test_invalid_json_and_nonobject_configs_are_reported(self):
        (self.whisper / "config.json").write_text("[", encoding="utf-8")
        (self.whisper / "preprocessor_config.json").write_text("[]", encoding="utf-8")
        issues = validate_whisper_model_directory(self.whisper)
        self.assertTrue(any("无法读取" in item and "config.json" in item for item in issues))
        self.assertTrue(any("JSON 对象" in item for item in issues))


class SenseVoiceAdapterTests(RecognitionFixture):
    def test_factory_loads_cpu_native_model_during_construction(self):
        module, constructor, recognizer, stream = self.fake_sensevoice()
        with patch.dict(sys.modules, {"sherpa_onnx": module}):
            engine = create_recognition_engine(self.settings)
        self.addCleanup(engine.close)
        constructor.assert_called_once()
        self.assertEqual({
            "model": str((self.base / "sense.onnx").resolve()),
            "tokens": str((self.base / "tokens.txt").resolve()),
            "num_threads": min(4, max(1, __import__("os").cpu_count() or 1)),
            "sample_rate": 16000, "provider": "cpu", "language": "auto", "use_itn": True,
        }, constructor.call_args.kwargs)
        samples = [0.1, -0.2]
        self.assertEqual("你好。", engine.recognize(samples, language="zh"))
        stream.accept_waveform.assert_called_once_with(16000, samples)
        recognizer.decode_stream.assert_called_once_with(stream)
        self.assertEqual("cpu", engine.backend_info["device"])
        self.assertEqual("model", engine.backend_info["compute_type"])
        self.cuda_setup.assert_not_called()

    def test_japanese_is_accepted_and_empty_input_skips_decode(self):
        module, _, recognizer, _ = self.fake_sensevoice(text="<|ja|>こんにちは。")
        with patch.dict(sys.modules, {"sherpa_onnx": module}):
            engine = SenseVoiceRecognitionEngine(self.settings)
        self.addCleanup(engine.close)
        self.assertEqual("", engine.recognize([], language="ja"))
        recognizer.create_stream.assert_not_called()
        self.assertEqual("こんにちは。", engine.recognize([0.0], language="ja"))

    def test_missing_model_and_tokens_fail_before_native_construction(self):
        module, constructor, _, _ = self.fake_sensevoice()
        with patch.dict(sys.modules, {"sherpa_onnx": module}):
            for field in ("asr_model", "asr_tokens"):
                with self.subTest(field=field), self.assertRaisesRegex(RuntimeError, "不存在"):
                    SenseVoiceRecognitionEngine(replace(self.settings, **{field: ""}))
        constructor.assert_not_called()

    def test_cancelled_before_loading_never_constructs_native_model(self):
        stop = threading.Event()
        stop.set()
        module, constructor, _, _ = self.fake_sensevoice()
        with patch.dict(sys.modules, {"sherpa_onnx": module}):
            with self.assertRaises(RecognitionCancelled):
                SenseVoiceRecognitionEngine(self.settings, stop_event=stop)
        constructor.assert_not_called()

    def test_stop_during_native_decode_suppresses_returned_text(self):
        stop = threading.Event()
        module, _, _, _ = self.fake_sensevoice(on_decode=lambda _stream: stop.set())
        with patch.dict(sys.modules, {"sherpa_onnx": module}):
            engine = SenseVoiceRecognitionEngine(self.settings, stop_event=stop)
        self.addCleanup(engine.close)
        with self.assertRaises(RecognitionCancelled):
            engine.recognize([0.1])

    def test_decode_failure_propagates_and_close_is_idempotent(self):
        module, _, _, _ = self.fake_sensevoice(on_decode=RuntimeError("native decode failed"))
        with patch.dict(sys.modules, {"sherpa_onnx": module}):
            engine = SenseVoiceRecognitionEngine(self.settings)
        with self.assertRaisesRegex(RuntimeError, "native decode failed"):
            engine.recognize([0.1])
        engine.close()
        engine.close()
        with self.assertRaisesRegex(RuntimeError, "已关闭"):
            engine.recognize([0.1])


class WhisperAdapterTests(RecognitionFixture):
    def build_engine(self, *, settings=None, stop_event=None, texts=(" 你好", " 世界。")):
        module, constructor, model = self.fake_whisper(texts=texts)
        with patch.dict(sys.modules, {"faster_whisper": module}):
            engine = WhisperRecognitionEngine(settings or self.settings, stop_event=stop_event)
        self.addCleanup(engine.close)
        return engine, constructor, model

    def test_local_directory_is_loaded_during_construction(self):
        engine, constructor, model = self.build_engine()
        self.assertEqual((str(self.whisper.resolve()),), constructor.call_args.args)
        self.assertEqual({
            "device": "cpu", "compute_type": "int8", "local_files_only": True,
            "cpu_threads": min(4, max(1, __import__("os").cpu_count() or 1)), "num_workers": 1,
        }, constructor.call_args.kwargs)
        model.transcribe.assert_not_called()
        self.assertEqual("whisper", engine.backend_info["backend"])
        self.cuda_setup.assert_not_called()

    def test_cuda_and_requested_precision_are_forwarded_without_fallback(self):
        for precision in ("int8", "int8_float16", "float16"):
            with self.subTest(precision=precision):
                engine, constructor, _ = self.build_engine(settings=replace(
                    self.settings, recognition_device="cuda", recognition_compute_type=precision,
                ))
                self.assertEqual("cuda", constructor.call_args.kwargs["device"])
                self.assertEqual(precision, constructor.call_args.kwargs["compute_type"])
                self.assertEqual(precision, engine.backend_info["compute_type"])
                self.cuda_setup.assert_called_with("cuda")

    def test_metadata_distinguishes_native_device_and_precision_from_requested_profile(self):
        module, constructor, model = self.fake_whisper()
        model.model.device = "cpu"
        model.model.compute_type = "float32"
        settings = replace(self.settings, recognition_device="cuda", recognition_compute_type="int8_float16")
        with patch.dict(sys.modules, {"faster_whisper": module}):
            engine = WhisperRecognitionEngine(settings)
        self.addCleanup(engine.close)
        self.assertEqual("cuda", constructor.call_args.kwargs["device"])
        self.assertEqual("int8_float16", constructor.call_args.kwargs["compute_type"])
        self.assertEqual("cpu", engine.backend_info["device"])
        self.assertEqual("float32", engine.backend_info["compute_type"])
        self.assertEqual("cuda", engine.backend_info["requested_device"])
        self.assertEqual("int8_float16", engine.backend_info["requested_compute_type"])
        self.assertEqual("large-v3", engine.backend_info["model"])
        self.assertEqual("requested_profile; assets verified by preparation manifest", engine.backend_info["model_identity"])

    def test_nonstring_native_metadata_uses_requested_values_for_fakes(self):
        for device, precision in ((Mock(), Mock()), (None, 7)):
            with self.subTest(device_type=type(device).__name__):
                module, _, model = self.fake_whisper()
                model.model.device = device
                model.model.compute_type = precision
                with patch.dict(sys.modules, {"faster_whisper": module}):
                    engine = WhisperRecognitionEngine(self.settings)
                self.addCleanup(engine.close)
                self.assertEqual("cpu", engine.backend_info["device"])
                self.assertEqual("int8", engine.backend_info["compute_type"])
                self.assertEqual("cpu", engine.backend_info["requested_device"])
                self.assertEqual("int8", engine.backend_info["requested_compute_type"])

    def test_invalid_device_or_precision_never_loads_native_model(self):
        module, constructor, _ = self.fake_whisper()
        with patch.dict(sys.modules, {"faster_whisper": module}):
            for changes in (
                {"recognition_device": "auto"}, {"recognition_compute_type": "float32"},
                {"recognition_compute_type": "float16"}, {"recognition_compute_type": "int8_float16"},
            ):
                with self.subTest(changes=changes), self.assertRaises(ValueError):
                    WhisperRecognitionEngine(replace(self.settings, **changes))
        constructor.assert_not_called()

    def test_transcribe_uses_explicit_language_float32_waveform_and_fixed_options(self):
        engine, _, model = self.build_engine(texts=(" こんにちは", "。", " "))
        self.assertEqual("こんにちは。", engine.recognize([0.1, -0.2], language="ja"))
        args, kwargs = model.transcribe.call_args
        self.assertEqual(np.float32, args[0].dtype)
        np.testing.assert_allclose(args[0], [0.1, -0.2])
        self.assertEqual({
            "language": "ja", "task": "transcribe", "beam_size": 5,
            "temperature": (0.0, 0.2), "best_of": 1,
            "compression_ratio_threshold": 2.4, "log_prob_threshold": -1.0, "no_speech_threshold": 0.6,
            "condition_on_previous_text": False,
            "vad_filter": False, "without_timestamps": True, "word_timestamps": False,
        }, kwargs)

    def test_lazy_segments_are_consumed_and_word_separators_preserved(self):
        engine, _, model = self.build_engine()
        yielded = []
        finalized = threading.Event()
        def segments():
            try:
                for text in (" hello", " ", " world "):
                    yielded.append(text)
                    yield SimpleNamespace(text=text)
            finally:
                finalized.set()
        model.transcribe.return_value = (segments(), None)
        model.transcribe.side_effect = None
        self.assertEqual("hello world", engine.recognize([0.1], language="zh"))
        self.assertEqual([" hello", " ", " world "], yielded)
        self.assertTrue(finalized.is_set())

    def test_whisper_segment_diagnostics_keep_only_numeric_scores_without_changing_text(self):
        engine, _, model = self.build_engine()
        segments = [
            SimpleNamespace(text=" 公开测试", avg_logprob=-0.4, compression_ratio=1.2,
                            no_speech_prob=0.05, temperature=0.0),
            SimpleNamespace(text=" 低分结果", avg_logprob=-1.01, compression_ratio=1.3,
                            no_speech_prob=0.2, temperature=0.2),
            SimpleNamespace(text=" 重复 重复 重复。", avg_logprob=-0.5, compression_ratio=2.41,
                            no_speech_prob=0.1, temperature=0.2),
        ]
        model.transcribe.side_effect = lambda *_args, **_kwargs: (iter(segments), None)
        self.assertEqual("公开测试 低分结果 重复 重复 重复。", engine.recognize([0.1]))
        self.assertEqual({"segments": [
            {"avg_logprob": -0.4, "compression_ratio": 1.2, "no_speech_prob": 0.05,
             "temperature": 0.0, "char_count": 4, "suspect": False},
            {"avg_logprob": -1.01, "compression_ratio": 1.3, "no_speech_prob": 0.2,
             "temperature": 0.2, "char_count": 4, "suspect": True},
            {"avg_logprob": -0.5, "compression_ratio": 2.41, "no_speech_prob": 0.1,
             "temperature": 0.2, "char_count": 9, "suspect": True},
        ], "suspect": True}, engine.last_diagnostics)
        serialized = json.dumps(engine.last_diagnostics, allow_nan=False)
        self.assertNotIn("公开测试", serialized)
        self.assertNotIn("重复", serialized)
        self.assertNotIn("text", serialized)
        self.assertEqual((0.0, 0.2), engine.backend_info["transcribe_options"]["temperature"])
        self.assertEqual(1, engine.backend_info["transcribe_options"]["best_of"])

    def test_diagnostic_thresholds_use_strict_library_boundaries_and_never_prove_accuracy(self):
        engine, _, model = self.build_engine()
        for logprob, compression, suspect in ((-1.0, 2.4, False), (-1.00001, 2.4, True),
                                             (-1.0, 2.40001, True)):
            with self.subTest(logprob=logprob, compression=compression):
                model.transcribe.side_effect = lambda *_args, **_kwargs: (iter([
                    SimpleNamespace(text="结果", avg_logprob=logprob, compression_ratio=compression,
                                    no_speech_prob=0.6, temperature=0.2)
                ]), None)
                self.assertEqual("结果", engine.recognize([0.1]))
                self.assertIs(suspect, engine.last_diagnostics["segments"][0]["suspect"])
                self.assertIs(suspect, engine.last_diagnostics["suspect"])

    def test_missing_mock_nonfinite_and_nonnumeric_metrics_are_unknown_not_fabricated_scores(self):
        engine, _, model = self.build_engine()
        invalid_values = (None, Mock(), Mock(spec=float), float("nan"), float("inf"), -float("inf"), True, "-1.0")
        for value in invalid_values:
            with self.subTest(value_type=type(value).__name__):
                model.transcribe.side_effect = lambda *_args, **_kwargs: (iter([
                    SimpleNamespace(text="保留正文", avg_logprob=value, compression_ratio=value,
                                    no_speech_prob=value, temperature=value)
                ]), None)
                self.assertEqual("保留正文", engine.recognize([0.1]))
                self.assertEqual({"avg_logprob": None, "compression_ratio": None,
                                  "no_speech_prob": None, "temperature": None,
                                  "char_count": 4, "suspect": None}, engine.last_diagnostics["segments"][0])
                self.assertIsNone(engine.last_diagnostics["suspect"])
                json.dumps(engine.last_diagnostics, allow_nan=False)
        model.transcribe.side_effect = lambda *_args, **_kwargs: (iter([SimpleNamespace(text="缺字段")]), None)
        self.assertEqual("缺字段", engine.recognize([0.1]))
        self.assertIsNone(engine.last_diagnostics["segments"][0]["avg_logprob"])
        self.assertIsNone(engine.last_diagnostics["suspect"])

    def test_known_bad_score_is_still_suspicious_when_other_scores_are_missing(self):
        engine, _, model = self.build_engine()
        model.transcribe.side_effect = lambda *_args, **_kwargs: (iter([
            SimpleNamespace(text="低分", avg_logprob=np.float32(-1.5)),
            SimpleNamespace(text="重复", compression_ratio=np.float64(2.5)),
            SimpleNamespace(text="未知"),
        ]), None)
        self.assertEqual("低分重复未知", engine.recognize([0.1]))
        self.assertEqual([True, True, None], [item["suspect"] for item in engine.last_diagnostics["segments"]])
        self.assertIs(engine.last_diagnostics["suspect"], True)

    def test_diagnostics_reset_for_empty_invalid_cancelled_failed_and_closed_calls(self):
        stop = threading.Event()
        engine, _, model = self.build_engine(stop_event=stop)
        expected = {"segments": [], "suspect": None}
        self.assertEqual(expected, engine.last_diagnostics)
        engine.recognize([0.1])
        self.assertTrue(engine.last_diagnostics["segments"])
        self.assertEqual("", engine.recognize([]))
        self.assertEqual(expected, engine.last_diagnostics)
        engine.recognize([0.1])
        with self.assertRaises(ValueError):
            engine.recognize([float("nan")])
        self.assertEqual(expected, engine.last_diagnostics)
        engine.recognize([0.1])
        with self.assertRaises(ValueError):
            engine.recognize([0.1], language="en")
        self.assertEqual(expected, engine.last_diagnostics)
        engine.recognize([0.1])
        stop.set()
        with self.assertRaises(RecognitionCancelled):
            engine.recognize([0.1])
        self.assertEqual(expected, engine.last_diagnostics)

        stop.clear()
        engine.recognize([0.1])
        def failing_segments():
            yield SimpleNamespace(text="不能提交一半诊断", avg_logprob=-2.0, compression_ratio=3.0)
            raise RuntimeError("generator failed")
        model.transcribe.side_effect = lambda *_args, **_kwargs: (failing_segments(), None)
        with self.assertRaisesRegex(RuntimeError, "generator failed"):
            engine.recognize([0.1])
        self.assertEqual(expected, engine.last_diagnostics)
        engine.close()
        with self.assertRaisesRegex(RuntimeError, "已关闭"):
            engine.recognize([0.1])
        self.assertEqual(expected, engine.last_diagnostics)

    def test_no_yielded_segments_stay_unknown_without_fabricating_silence_scores(self):
        engine, _, _model = self.build_engine(texts=())
        self.assertEqual("", engine.recognize([0.1]))
        self.assertEqual({"segments": [], "suspect": None}, engine.last_diagnostics)

    def test_empty_input_does_not_call_native_transcribe(self):
        engine, _, model = self.build_engine()
        self.assertEqual("", engine.recognize([]))
        model.transcribe.assert_not_called()

    def test_nonfinite_or_multichannel_input_fails_before_transcribe(self):
        engine, _, model = self.build_engine()
        for samples in ([float("nan")], [float("inf")], [[0.1], [0.2]]):
            with self.subTest(samples=samples), self.assertRaisesRegex(ValueError, "单声道波形"):
                engine.recognize(samples)
        model.transcribe.assert_not_called()

    def test_unsupported_language_fails_before_native_transcribe(self):
        engine, _, model = self.build_engine()
        with self.assertRaisesRegex(ValueError, "不支持的识别语言"):
            engine.recognize([0.1], language="en")
        model.transcribe.assert_not_called()

    def test_cancelled_before_constructor_never_loads_native_model(self):
        stop = threading.Event()
        stop.set()
        module, constructor, _ = self.fake_whisper()
        with patch.dict(sys.modules, {"faster_whisper": module}):
            with self.assertRaises(RecognitionCancelled):
                WhisperRecognitionEngine(self.settings, stop_event=stop)
        constructor.assert_not_called()

    def test_stop_after_constructor_unloads_native_model(self):
        stop = threading.Event()
        module, _, model = self.fake_whisper(on_construct=stop.set)
        with patch.dict(sys.modules, {"faster_whisper": module}):
            with self.assertRaises(RecognitionCancelled):
                WhisperRecognitionEngine(self.settings, stop_event=stop)
        model.model.unload_model.assert_called_once_with()

    def test_cancellation_between_yields_closes_generator_without_partial_output(self):
        stop = threading.Event()
        engine, _, model = self.build_engine(stop_event=stop)
        finished = threading.Event()
        second_segment = Mock()
        def segments():
            try:
                stop.set()
                yield SimpleNamespace(text="should be discarded")
                second_segment()
                yield SimpleNamespace(text="not reached")
            finally:
                finished.set()
        model.transcribe.side_effect = None
        model.transcribe.return_value = (segments(), None)
        with self.assertRaises(RecognitionCancelled):
            engine.recognize([0.1])
        second_segment.assert_not_called()
        self.assertTrue(finished.is_set())

    def test_lazy_native_error_propagates_and_generator_is_finalized(self):
        engine, _, model = self.build_engine()
        finished = threading.Event()
        def segments():
            try:
                yield SimpleNamespace(text="not returned partially")
                raise RuntimeError("native generator failed")
            finally:
                finished.set()
        model.transcribe.side_effect = None
        model.transcribe.return_value = (segments(), None)
        with self.assertRaisesRegex(RuntimeError, "native generator failed"):
            engine.recognize([0.1])
        self.assertTrue(finished.is_set())

    def test_invalid_segment_text_does_not_masquerade_as_success(self):
        engine, _, model = self.build_engine(texts=(None,))
        with self.assertRaisesRegex(RuntimeError, "无效的分段文本"):
            engine.recognize([0.1])

    def test_close_unloads_once_and_prevents_later_inference(self):
        engine, _, model = self.build_engine()
        engine.close()
        engine.close()
        model.model.unload_model.assert_called_once_with()
        with self.assertRaisesRegex(RuntimeError, "已关闭"):
            engine.recognize([0.1])
        model.transcribe.assert_not_called()

    def test_close_waits_for_lazy_native_inference_to_finish(self):
        engine, _, model = self.build_engine()
        entered = threading.Event()
        release = threading.Event()
        close_started = threading.Event()
        close_finished = threading.Event()
        finalized = threading.Event()
        results, errors = [], []
        def segments():
            try:
                entered.set()
                if not release.wait(2):
                    raise RuntimeError("fake inference release timed out")
                yield SimpleNamespace(text="完成")
            finally:
                finalized.set()
        def recognize():
            try:
                results.append(engine.recognize([0.1]))
            except BaseException as exc:
                errors.append(exc)
        def close():
            close_started.set()
            try:
                engine.close()
            except BaseException as exc:
                errors.append(exc)
            finally:
                close_finished.set()
        model.transcribe.side_effect = None
        model.transcribe.return_value = (segments(), None)
        inference = threading.Thread(target=recognize, daemon=True)
        closer = threading.Thread(target=close, daemon=True)
        inference.start()
        try:
            self.assertTrue(entered.wait(1))
            closer.start()
            self.assertTrue(close_started.wait(1))
            self.assertFalse(close_finished.wait(0.05))
            model.model.unload_model.assert_not_called()
        finally:
            release.set()
            inference.join(2)
            if closer.ident is not None:
                closer.join(2)
        self.assertFalse(inference.is_alive())
        self.assertFalse(closer.is_alive())
        self.assertEqual([], errors)
        self.assertEqual(["完成"], results)
        self.assertTrue(finalized.is_set())
        model.model.unload_model.assert_called_once_with()


class RecognitionFactoryTests(RecognitionFixture):
    def test_whisper_backend_is_selected(self):
        module, constructor, _ = self.fake_whisper()
        with patch.dict(sys.modules, {"faster_whisper": module}):
            engine = create_recognition_engine(replace(self.settings, recognition_backend="whisper"))
        self.addCleanup(engine.close)
        self.assertIsInstance(engine, WhisperRecognitionEngine)
        constructor.assert_called_once()

    def test_legacy_settings_without_backend_keep_sensevoice(self):
        legacy = SimpleNamespace(asr_model=self.settings.asr_model, asr_tokens=self.settings.asr_tokens)
        module, constructor, _, _ = self.fake_sensevoice()
        with patch.dict(sys.modules, {"sherpa_onnx": module}):
            engine = create_recognition_engine(legacy)
        self.addCleanup(engine.close)
        self.assertIsInstance(engine, SenseVoiceRecognitionEngine)
        constructor.assert_called_once()

    def test_unknown_backend_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "不支持的识别后端"):
            create_recognition_engine(replace(self.settings, recognition_backend="unknown"))


if __name__ == "__main__":
    unittest.main()
