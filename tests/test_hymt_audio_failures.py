"""Hy/audio ownership boundaries with fake models, streams and PortAudio only."""

from dataclasses import replace
import sys
import threading
import time
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from fanyiji_next import audio_runtime as module
from fanyiji_next.hymt_runtime import HyRuntimeError
from fanyiji_next.models import AppSettings, ChannelSettings
from fanyiji_next.translation_engines import TranslationCancelled


def settings():
    return replace(AppSettings(), recognition_backend="sensevoice", translation_backend="hymt",
                   microphone=ChannelSettings(enabled=True, direction="zh_to_ja"),
                   system=ChannelSettings(enabled=True, direction="zh_to_ja"))


@pytest.fixture
def models(monkeypatch):
    recognizer, translator = Mock(), Mock()
    recognizer.recognize.return_value = "recognized-source"
    translator.translate.return_value = "translated-current"
    stop_events = []

    def make_translation(_settings, *, stop_event=None):
        stop_events.append(stop_event)
        return translator

    monkeypatch.setattr(module, "create_recognition_engine", Mock(return_value=recognizer))
    monkeypatch.setattr(module, "create_translation_engine", make_translation)
    return SimpleNamespace(recognizer=recognizer, translator=translator, stop_events=stop_events)


def run_fake_owner(monkeypatch, models):
    events, sessions = [], []
    stopped = threading.Event()

    def receive(event):
        events.append(event)
        if event.kind == "state" and event.status == "stopped":
            stopped.set()

    runtime = module.VoiceRuntime(receive)
    pa = Mock()
    streams = {channel: Mock() for channel in ("mic", "system")}
    for stream in streams.values():
        stream.is_active.return_value = True
    monkeypatch.setitem(sys.modules, "pyaudiowpatch", SimpleNamespace(PyAudio=Mock(return_value=pa)))
    monkeypatch.setattr(module, "validate_models", lambda _: [])

    def open_input(session, owner, audio, channel, selected):
        assert owner is pa
        return SimpleNamespace(channel=channel, settings=selected, source=object(), stream=streams[channel])

    def consume(session, channel, selected, ready, source):
        if channel == "mic":
            sessions.append(session)
            for number in range(3):
                session.segments.put(module._Segment(channel, f"55-mic-{number}", [0.1],
                                                      selected.direction, time.perf_counter()))
        ready.set()
        session.stop.wait(2)

    monkeypatch.setattr(runtime, "_open_input", open_input)
    monkeypatch.setattr(runtime, "_capture", consume)
    runtime.start(settings(), 55)
    try:
        assert stopped.wait(1), "Hy terminal failure/cancellation did not retire its audio session"
    finally:
        runtime.stop()
        assert stopped.wait(1)
        if sessions:
            sessions[0].worker.join(timeout=1)
    return SimpleNamespace(runtime=runtime, events=events, session=sessions[0], pa=pa, streams=streams)


def assert_owner_released(result, models):
    assert not result.runtime.is_busy
    assert result.session.stop.is_set() and result.session.segments.empty()
    assert result.session.worker is not None and not result.session.worker.is_alive()
    for stream in result.streams.values():
        stream.start_stream.assert_called_once()
        stream.stop_stream.assert_called_once()
        stream.close.assert_called_once()
    result.pa.terminate.assert_called_once()
    models.recognizer.close.assert_called_once()
    models.translator.close.assert_called_once()
    assert result.events[-1].kind == "state" and result.events[-1].status == "stopped"


def test_hy_runtime_failure_stops_owner_and_discards_following_queued_sentences(monkeypatch, models):
    models.translator.translate.side_effect = HyRuntimeError("fixture retired child")
    result = run_fake_owner(monkeypatch, models)
    assert_owner_released(result, models)
    models.recognizer.recognize.assert_called_once()
    models.translator.translate.assert_called_once_with("recognized-source", "zh_to_ja", context_id="mic")
    originals = [event for event in result.events if event.kind == "original"]
    assert [(event.channel, event.segment_id) for event in originals] == [("mic", "55-mic-0")]
    assert not any(event.kind == "translation" for event in result.events)
    errors = [event for event in result.events if event.kind == "error"]
    assert len(errors) == 1 and errors[0].status != "recoverable"
    assert "retired child" in errors[0].message


@pytest.mark.parametrize("cancel_exception", [False, True])
def test_stop_during_hy_translation_suppresses_late_output_and_retires_owner(monkeypatch, models, cancel_exception):
    def translate(*args, **kwargs):
        models.stop_events[0].set()
        if cancel_exception:
            raise TranslationCancelled("fixture stop during generation")
        return "late-result-must-not-reach-ui"

    models.translator.translate.side_effect = translate
    result = run_fake_owner(monkeypatch, models)
    assert_owner_released(result, models)
    assert len([event for event in result.events if event.kind == "original"]) == 1
    assert not any(event.kind in {"translation", "error"} for event in result.events)
    models.translator.translate.assert_called_once()


def test_audio_and_composer_forward_distinct_channel_context_ids(models):
    events = []
    runtime = module.VoiceRuntime(events.append)
    session = module._Session(settings(), 66)
    runtime._session = session
    engine = module._LocalEngine(session.settings, stop_event=session.stop)
    models.recognizer.recognize.side_effect = ["mic-first", "system-first", "mic-second", "system-second"]
    histories = {}

    def translate(text, direction, *, context_id):
        previous = histories.setdefault((context_id, direction), [])
        result = f"{context_id}:{'|'.join(previous)}>{text}"
        previous.append(text)
        return result

    models.translator.translate.side_effect = translate
    try:
        for index, channel in enumerate(("mic", "system", "mic", "system")):
            runtime._process_segment(session, engine, module._Segment(
                channel, f"66-{channel}-{index}", [.1], "zh_to_ja", time.perf_counter()))
    finally:
        engine.close()
    assert [call.kwargs["context_id"] for call in models.translator.translate.call_args_list] == ["mic", "system", "mic", "system"]
    translated = [event for event in events if event.kind == "translation"]
    assert [event.text for event in translated] == ["mic:>mic-first", "system:>system-first",
        "mic:mic-first>mic-second", "system:system-first>system-second"]
    assert [(event.channel, event.segment_id) for event in translated] == [
        ("mic", "66-mic-0"), ("system", "66-system-1"), ("mic", "66-mic-2"), ("system", "66-system-3")]


def test_composer_retains_failed_hy_cleanup_reference_for_retry(models):
    engine = module._LocalEngine(settings())
    models.translator.close.side_effect = [HyRuntimeError("fixture kill failed"), None]
    with pytest.raises(HyRuntimeError, match="kill failed"):
        engine.close()
    assert engine.translator is models.translator
    assert engine.recognizer is None
    models.recognizer.close.assert_called_once()
    engine.close()
    assert engine.translator is None
    assert models.translator.close.call_count == 2
    models.recognizer.close.assert_called_once()
