"""Voice channels must supply distinct context identities to the Q8 adapter."""
import time
from types import SimpleNamespace
from unittest.mock import Mock, patch

from fanyiji_next.audio_runtime import VoiceRuntime, _LocalEngine, _Segment, _Session, validate_models
from fanyiji_next.models import AppSettings


def test_hymt_channel_identity_reaches_only_its_adapter():
    translator = Mock()
    translator.translate.return_value = "translation"
    with patch("fanyiji_next.audio_runtime.create_recognition_engine", return_value=Mock()), \
         patch("fanyiji_next.audio_runtime.create_translation_engine", return_value=translator):
        engine = _LocalEngine(AppSettings(translation_backend="hymt"))
    assert engine.translate("original", "zh_to_ja", context_id="mic") == "translation"
    translator.translate.assert_called_once_with("original", "zh_to_ja", context_id="mic")
    engine.close()


def test_runtime_preserves_channel_context_and_reports_guard_error_as_recoverable():
    events = []
    runtime = VoiceRuntime(events.append)
    session = _Session(AppSettings(translation_backend="hymt"), 1)
    runtime._session = session
    engine = SimpleNamespace(recognize=Mock(return_value="original"), translate=Mock(return_value="translated"))
    runtime._process_segment(session, engine, _Segment("mic", "one", [0.1], "zh_to_ja", time.perf_counter()))
    engine.translate.assert_called_once_with("original", "zh_to_ja", context_id="mic")
    engine.translate.side_effect = RuntimeError("数值未通过检查")
    runtime._process_segment(session, engine, _Segment("system", "two", [0.1], "zh_to_ja", time.perf_counter()))
    assert events[-2].kind == "original" and events[-2].text == "original"
    assert events[-1].kind == "error" and events[-1].status == "recoverable"
    assert events[-1].segment_id == "two"


def test_hymt_static_validation_uses_its_own_local_files(tmp_path):
    model, exe = tmp_path / "model.gguf", tmp_path / "llama-server.exe"
    model.write_bytes(b"GGUFfake")
    exe.write_bytes(b"fake")
    settings = AppSettings(recognition_backend="whisper", hymt_model_file=str(model), hymt_server_exe=str(exe))
    with patch("fanyiji_next.recognition_engines.validate_whisper_model_directory", return_value=[]):
        assert validate_models(settings) == []
