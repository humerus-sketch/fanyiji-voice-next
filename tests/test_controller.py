"""Session and main-thread UI contracts using fake audio and temporary paths."""

import os
from dataclasses import replace
from threading import Event, Thread, current_thread
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtCore import QObject, QThread, Signal
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from fanyiji_next import controller as controller_module
from fanyiji_next.controller import AppController
from fanyiji_next.models import AppSettings, RuntimeEvent


def settle(app):
    for _ in range(6):
        app.processEvents()


class FakeWindow(QObject):
    start_requested = Signal(object)
    stop_requested = Signal()
    demo_requested = Signal()
    settings_changed = Signal(object)
    closing = Signal()
    clear_requested = Signal()
    refresh_devices_requested = Signal()

    def __init__(self):
        super().__init__()
        self.settings = AppSettings()
        self.events = []
        self.event_threads = []
        self.states = []
        self.errors = []
        self.clear_count = 0
        self.devices = []
        self.device_errors = []
        self.device_threads = []

    def update_settings(self, settings):
        self.settings = settings

    def set_devices(self, devices):
        self.devices.append(devices)
        self.device_threads.append(QThread.currentThread())

    def set_device_error(self, message):
        self.device_errors.append(message)

    def apply_event(self, event):
        self.events.append(event)
        self.event_threads.append(QThread.currentThread())

    def set_session_state(self, state, message=""):
        self.states.append((state, message))

    def show_error(self, message):
        self.errors.append(message)

    def clear_captions(self):
        self.clear_count += 1
        self.events.clear()


class FakeRuntime:
    def __init__(self, callback):
        self.callback = callback
        self.is_running = False
        self.starts = []
        self.stop_count = 0

    def start(self, settings, session_id):
        self.starts.append((settings, session_id))
        self.is_running = True

    def stop(self):
        self.stop_count += 1

    def finish(self):
        self.is_running = False
        self.callback(RuntimeEvent(kind="state", session_id=self.starts[-1][1], status="stopped"))


@pytest.fixture(scope="module")
def qt_app():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def harness(qt_app, monkeypatch, tmp_path):
    saved = []
    instances = []
    monkeypatch.setattr(controller_module.settings_store, "save_settings",
                        lambda settings, path=None: saved.append((settings, path)))

    def create(*, persist=False, runtime_factory=None, device_lister=lambda: []):
        window = FakeWindow()
        runtimes = []

        def factory(callback):
            runtime = FakeRuntime(callback)
            runtimes.append(runtime)
            return runtime

        settings_file = tmp_path / f"settings-{len(instances)}.json"
        controller = AppController(window, settings_file=settings_file,
                                   runtime_factory=runtime_factory or factory, persist_settings=persist,
                                   device_lister=device_lister)
        result = SimpleNamespace(window=window, controller=controller, runtimes=runtimes,
                                 saved=saved, settings_file=settings_file)
        instances.append(result)
        return result

    yield create
    for instance in instances:
        instance.controller.shutdown()
        instance.window.deleteLater()
    settle(qt_app)


def start(instance):
    instance.window.start_requested.emit(instance.window.settings)
    return instance.runtimes[-1]


def test_construction_does_not_start_audio_or_write_settings(harness):
    instance = harness()
    assert instance.controller.parent() is instance.window
    assert instance.runtimes == []
    assert instance.window.clear_count == 0
    assert instance.controller.state == "stopped"
    assert instance.saved == []
    assert not instance.settings_file.exists()


def test_only_manual_start_creates_runtime_without_discarding_history(harness):
    instance = harness()
    runtime = start(instance)
    assert runtime.starts == [(instance.window.settings, instance.controller.session_id)]
    assert instance.controller.session_id > 0
    assert instance.controller.state == "loading"
    assert instance.window.clear_count == 0
    assert instance.saved == []


def test_worker_callbacks_reach_widgets_only_via_the_main_thread(harness, qt_app):
    instance = harness()
    runtime = start(instance)
    event = RuntimeEvent(kind="original", session_id=instance.controller.session_id,
                         channel="mic", segment_id="a", text="测试文本")
    thread = Thread(target=runtime.callback, args=(event,))
    thread.start()
    thread.join(timeout=2)
    assert not thread.is_alive()
    assert instance.window.events == []
    settle(qt_app)
    assert instance.window.events == [event]
    assert instance.window.event_threads == [qt_app.thread()]


def test_retired_session_results_do_not_reach_the_next_session(harness, qt_app):
    instance = harness()
    old = start(instance)
    old_session = instance.controller.session_id
    instance.window.stop_requested.emit()
    old.finish()
    settle(qt_app)
    new = start(instance)
    new_session = instance.controller.session_id
    assert new_session > old_session
    old.callback(RuntimeEvent(kind="original", session_id=old_session, channel="mic", text="过期文本"))
    old.callback(RuntimeEvent(kind="error", session_id=old_session, message="过期错误"))
    expected = RuntimeEvent(kind="original", session_id=new_session, channel="mic", text="当前文本")
    new.callback(expected)
    settle(qt_app)
    assert instance.window.events == [expected]
    assert "过期错误" not in instance.window.errors


def test_pause_and_restart_retain_history_but_explicit_clear_discards_it(harness, qt_app):
    instance = harness()
    old = start(instance)
    old_id = instance.controller.session_id
    first = RuntimeEvent("original", old_id, "mic", f"{old_id}:mic-1", text="前一轮原文")
    old.callback(first)
    settle(qt_app)
    instance.controller.stop_session()
    old.finish()
    settle(qt_app)
    new = start(instance)
    new_id = instance.controller.session_id
    second = RuntimeEvent("original", new_id, "mic", f"{new_id}:mic-1", text="继续发言")
    new.callback(second)
    settle(qt_app)
    assert instance.window.events == [first, second]
    assert instance.window.clear_count == 0
    instance.controller.stop_session()
    new.finish()
    settle(qt_app)
    instance.controller.clear_captions()
    assert instance.window.events == []
    assert instance.window.clear_count == 1


def test_stop_is_nonblocking_and_disallows_restart_until_stopped(harness, qt_app):
    instance = harness()
    runtime = start(instance)
    instance.window.stop_requested.emit()
    instance.window.stop_requested.emit()
    assert runtime.is_running
    assert runtime.stop_count == 1
    assert instance.controller.state == "stopping"
    instance.window.start_requested.emit(instance.window.settings)
    assert len(instance.runtimes) == 1
    assert "等待" in instance.window.errors[-1]
    runtime.callback(RuntimeEvent(kind="state", session_id=instance.controller.session_id, status="running"))
    runtime.callback(RuntimeEvent(kind="translation", session_id=instance.controller.session_id, text="停止后译文"))
    settle(qt_app)
    assert instance.controller.state == "stopping"
    assert instance.window.events == []
    runtime.finish()
    settle(qt_app)
    assert instance.controller.state == "stopped"
    start(instance)
    assert len(instance.runtimes) == 2


def test_stopped_notification_does_not_allow_restart_while_worker_is_alive(harness, qt_app):
    instance = harness()
    runtime = start(instance)
    instance.window.stop_requested.emit()
    runtime.callback(RuntimeEvent(kind="state", session_id=instance.controller.session_id, status="stopped"))
    settle(qt_app)
    assert runtime.is_running
    instance.window.start_requested.emit(instance.window.settings)
    assert len(instance.runtimes) == 1
    runtime.is_running = False
    start(instance)
    assert len(instance.runtimes) == 2


def test_runtime_error_is_visible_and_requires_the_final_stopped_event(harness, qt_app):
    instance = harness()
    runtime = start(instance)
    runtime.callback(RuntimeEvent(kind="error", session_id=instance.controller.session_id, message="模拟模型错误"))
    settle(qt_app)
    assert instance.controller.state == "error"
    assert instance.window.errors[-1] == "模拟模型错误"
    runtime.is_running = False
    instance.window.start_requested.emit(instance.window.settings)
    assert len(instance.runtimes) == 1
    runtime.finish()
    settle(qt_app)
    start(instance)
    assert len(instance.runtimes) == 2


@pytest.mark.parametrize("segment_id", ["failed-segment", ""])
def test_recoverable_segment_or_queue_errors_do_not_freeze_later_subtitles(harness, qt_app, segment_id):
    instance = harness()
    runtime = start(instance)
    session_id = instance.controller.session_id
    runtime.callback(RuntimeEvent(kind="state", session_id=session_id, status="running"))
    error = RuntimeEvent(kind="error", session_id=session_id, channel="mic", segment_id=segment_id,
                         status="recoverable", message="当前片段处理失败")
    runtime.callback(error)
    later = RuntimeEvent(kind="original", session_id=session_id, channel="mic", segment_id="later", text="后续文本")
    runtime.callback(later)
    settle(qt_app)
    assert instance.controller.state == "running"
    assert instance.window.errors[-1] == error.message
    assert instance.window.events == [error, later]


def test_demo_is_explicitly_simulated_and_needs_no_runtime_or_model_paths(harness, qt_app):
    instance = harness(persist=True)
    assert instance.window.settings.asr_model == ""
    instance.window.demo_requested.emit()
    assert instance.controller.state == "demo"
    assert "模拟" in instance.window.states[-1][1]
    assert instance.controller._demo_timer.isActive()
    for _ in range(12):
        instance.controller._demo_timer.timeout.emit()
        settle(qt_app)
    events = instance.window.events
    assert len(events) == 12
    assert {event.channel for event in events} == {"mic", "system"}
    assert sum(event.kind == "original" for event in events) == 6
    assert sum(event.kind == "translation" for event in events) == 6
    assert all(event.status == "demo" and "模拟" in event.message for event in events)
    assert not instance.controller._demo_timer.isActive()
    assert instance.controller.state == "demo"
    assert "未进行真实识别" in instance.window.states[-1][1]
    assert instance.runtimes == []
    assert instance.saved == []


def test_stopping_demo_discards_queued_steps_and_pending_events(harness, qt_app):
    instance = harness()
    instance.window.demo_requested.emit()
    instance.controller._demo_timer.timeout.emit()
    instance.window.stop_requested.emit()
    settle(qt_app)
    assert instance.window.events == []
    assert not instance.controller._demo_timer.isActive()
    assert not instance.controller._demo_events
    assert instance.controller.state == "stopped"
    assert instance.runtimes == []


def test_manual_start_can_replace_demo_without_leaking_demo_results(harness, qt_app):
    instance = harness()
    instance.window.demo_requested.emit()
    demo_session = instance.controller.session_id
    instance.controller._demo_timer.timeout.emit()
    start(instance)
    settle(qt_app)
    assert instance.controller.session_id > demo_session
    assert not instance.controller._demo_timer.isActive()
    assert instance.window.events == []
    assert instance.controller.state == "loading"
    assert instance.window.clear_count == 2


def test_persist_false_never_saves_settings(harness):
    instance = harness()
    instance.window.settings_changed.emit(replace(instance.window.settings, font_size=28))
    assert instance.saved == []
    assert not instance.settings_file.exists()


def test_idle_settings_save_uses_the_explicit_path(harness):
    instance = harness(persist=True)
    settings = replace(instance.window.settings, font_size=28)
    instance.window.settings_changed.emit(settings)
    assert instance.saved == [(settings, instance.settings_file)]
    assert instance.runtimes == []


@pytest.mark.parametrize("activity", ["audio", "demo"])
def test_active_appearance_settings_save_without_restarting_runtime(harness, activity):
    instance = harness(persist=True)
    if activity == "audio":
        start(instance)
    else:
        instance.window.demo_requested.emit()
    updated = replace(instance.window.settings, font_size=28, window_opacity=80)
    instance.window.settings_changed.emit(updated)
    assert instance.saved == [(updated, instance.settings_file)]
    assert len(instance.runtimes) == (1 if activity == "audio" else 0)
    if instance.runtimes:
        assert len(instance.runtimes[0].starts) == 1
        assert instance.runtimes[0].stop_count == 0


@pytest.mark.parametrize("activity", ["audio", "demo"])
@pytest.mark.parametrize("change", ["model", "source", "backend", "device"])
def test_active_model_and_source_settings_are_rejected_without_restart(harness, activity, change):
    instance = harness(persist=True)
    if activity == "audio":
        start(instance)
    else:
        instance.window.demo_requested.emit()
    settings = instance.window.settings
    updates = {
        "model": {"asr_model": "changed.onnx"},
        "source": {"microphone": replace(settings.microphone, enabled=False)},
        "backend": {"recognition_backend": "whisper"},
        "device": {"translation_device": "cuda"},
    }
    updated = replace(settings, **updates[change])
    instance.window.settings_changed.emit(updated)
    assert instance.saved == []
    assert "先停止" in instance.window.errors[-1]
    assert len(instance.runtimes) == (1 if activity == "audio" else 0)


def test_save_failure_is_reported_without_changing_session_state(harness, monkeypatch):
    instance = harness(persist=True)

    def fail_save(*args, **kwargs):
        raise ValueError("fixture storage failure")

    monkeypatch.setattr(controller_module.settings_store, "save_settings", fail_save)
    instance.window.settings_changed.emit(instance.window.settings)
    assert "fixture storage failure" in instance.window.errors[-1]
    assert instance.controller.state == "stopped"
    assert instance.runtimes == []


def test_shutdown_ignores_queued_and_late_events_and_future_start_requests(harness, qt_app):
    instance = harness()
    runtime = start(instance)
    event = RuntimeEvent(kind="original", session_id=instance.controller.session_id, text="迟到文本")
    runtime.callback(event)
    instance.window.closing.emit()
    runtime.callback(event)
    instance.window.start_requested.emit(instance.window.settings)
    instance.window.demo_requested.emit()
    instance.controller.shutdown()
    settle(qt_app)
    assert instance.window.events == []
    assert runtime.stop_count == 1
    assert len(instance.runtimes) == 1
    assert not instance.controller._demo_timer.isActive()


def test_synchronous_start_failure_is_visible_and_does_not_lock_future_starts(harness):
    attempts = []

    class FailingRuntime(FakeRuntime):
        def start(self, settings, session_id):
            attempts.append(session_id)
            raise RuntimeError("fixture startup failure")

    instance = harness(runtime_factory=FailingRuntime)
    instance.window.start_requested.emit(instance.window.settings)
    assert instance.controller.state == "stopped"
    assert instance.window.errors[-1] == "fixture startup failure"
    instance.window.stop_requested.emit()
    assert instance.controller.state == "stopped"
    instance.window.start_requested.emit(instance.window.settings)
    assert len(attempts) == 2


def test_clear_is_blocked_until_worker_released_and_rejects_late_demo_events(harness, qt_app):
    instance = harness()
    runtime = start(instance)
    before = instance.window.clear_count
    instance.window.clear_requested.emit()
    assert instance.window.clear_count == before
    assert "先停止" in instance.window.errors[-1]
    runtime.finish()
    settle(qt_app)
    instance.window.demo_requested.emit()
    instance.controller._demo_timer.timeout.emit()
    instance.window.clear_requested.emit()
    settle(qt_app)
    assert instance.controller.state == "stopped"
    assert not instance.window.events
    assert not instance.controller._demo_timer.isActive()


def test_busy_native_cleanup_blocks_restart_even_after_running_flag_is_false(harness, qt_app):
    class BusyRuntime(FakeRuntime):
        is_busy = True

    instance = harness(runtime_factory=BusyRuntime)
    instance.window.start_requested.emit(instance.window.settings)
    runtime = instance.controller._runtime
    runtime.is_running = False
    runtime.callback(RuntimeEvent("state", instance.controller.session_id, status="stopped"))
    settle(qt_app)
    instance.window.start_requested.emit(instance.window.settings)
    assert instance.controller._runtime is runtime
    assert "等待" in instance.window.errors[-1]
    runtime.is_busy = False
    instance.controller._release_stopped_runtime()
    assert instance.controller._runtime is None
    assert instance.controller.state == "stopped"


def test_device_refresh_is_explicit_background_work_and_ui_update_is_queued(harness, qt_app):
    entered, release = Event(), Event()
    workers = []

    def discover():
        workers.append(current_thread())
        entered.set()
        assert release.wait(2)
        return ["fixture microphone"]

    instance = harness(device_lister=discover)
    assert not workers
    instance.window.refresh_devices_requested.emit()
    assert entered.wait(2)
    instance.window.refresh_devices_requested.emit()
    assert len(workers) == 1
    assert workers[0] is not current_thread()
    assert not instance.window.devices
    release.set()
    instance.controller._device_worker.join(2)
    settle(qt_app)
    assert instance.window.devices == [["fixture microphone"]]
    assert instance.window.device_threads == [qt_app.thread()]
    assert not instance.runtimes
    assert not instance.saved


def test_device_failure_and_late_discovery_result_are_safe(harness, qt_app):
    def fail():
        raise OSError("fixture device failure")

    instance = harness(device_lister=fail)
    instance.window.refresh_devices_requested.emit()
    instance.controller._device_worker.join(2)
    settle(qt_app)
    assert "fixture device failure" in instance.window.device_errors[-1]
    instance.controller._handle_devices((instance.controller._device_generation - 1, ["old"], ""))
    assert not instance.window.devices
    instance.controller.shutdown()
    instance.controller._handle_devices((instance.controller._device_generation, ["late"], ""))
    assert not instance.window.devices


def test_device_discovery_blocks_start_without_clearing_subtitles(harness, qt_app):
    entered, release = Event(), Event()

    def discover():
        entered.set()
        assert release.wait(2)
        return []

    instance = harness(device_lister=discover)
    try:
        instance.window.refresh_devices_requested.emit()
        assert entered.wait(2)
        instance.window.start_requested.emit(instance.window.settings)
        assert not instance.runtimes
        assert instance.window.clear_count == 0
        assert "等待" in instance.window.errors[-1]
        release.set()
        instance.controller._device_worker.join(2)
        settle(qt_app)
        start(instance)
        assert len(instance.runtimes) == 1
    finally:
        release.set()
        instance.controller._device_worker.join(2)


def test_device_refresh_waits_for_the_runtime_to_release_all_resources(harness, qt_app):
    calls = []
    instance = harness(device_lister=lambda: calls.append(True) or [])
    runtime = start(instance)
    instance.window.refresh_devices_requested.emit()
    instance.window.stop_requested.emit()
    instance.window.refresh_devices_requested.emit()
    assert not calls
    assert instance.controller._device_worker is None
    assert "资源释放" in instance.window.device_errors[-1]
    runtime.finish()
    settle(qt_app)
    instance.window.refresh_devices_requested.emit()
    instance.controller._device_worker.join(2)
    settle(qt_app)
    assert calls == [True]


def test_idle_window_close_waits_for_native_device_discovery(qt_app):
    from fanyiji_next.studio import StudioWindow

    entered, release = Event(), Event()

    def discover():
        entered.set()
        assert release.wait(2)
        return []

    window = StudioWindow(prototype=False)
    controller = AppController(window, persist_settings=False, device_lister=discover)
    try:
        window.show()
        controller.refresh_devices()
        assert entered.wait(2)
        assert not window.close()
        settle(qt_app)
        assert window.isVisible() and not window.closing_ready
        assert controller._shutdown_waiting
        assert controller._retire_timer.isActive()
        release.set()
        controller._device_worker.join(2)
        QTest.qWait(60)
        settle(qt_app)
        assert window.closing_ready and not window.isVisible()
        assert not controller._retire_timer.isActive()
        assert not controller._shutdown_waiting
    finally:
        release.set()
        if controller._device_worker is not None:
            controller._device_worker.join(2)
        window.finish_shutdown()
        window.deleteLater()
        settle(qt_app)


def test_rejected_runtime_or_failed_save_restores_last_accepted_ui_settings(harness, monkeypatch):
    instance = harness(persist=True)
    accepted = instance.window.settings
    start(instance)
    instance.window.settings = replace(accepted, asr_model="changed.onnx")
    instance.window.settings_changed.emit(instance.window.settings)
    assert instance.window.settings == accepted
    instance.controller._active_settings = None

    def fail(*_args, **_kwargs):
        raise OSError("fixture disk error")

    monkeypatch.setattr(controller_module.settings_store, "save_settings", fail)
    instance.window.settings = replace(accepted, font_size=32)
    instance.window.settings_changed.emit(instance.window.settings)
    assert instance.window.settings == accepted
    assert "fixture disk error" in instance.window.errors[-1]


def test_close_waits_for_stopped_cleanup_while_discarding_captions(harness, qt_app):
    instance = harness()
    finished = []
    instance.window.finish_shutdown = lambda: finished.append(True)
    runtime = start(instance)
    session_id = instance.controller.session_id
    instance.controller.shutdown()
    assert instance.controller._shutdown_waiting
    runtime.callback(RuntimeEvent("original", session_id, "mic", "late", text="关闭后的字幕"))
    runtime.finish()
    settle(qt_app)
    assert finished == [True]
    assert not instance.window.events
    assert instance.controller._runtime is None
    assert not instance.controller._shutdown_waiting
    instance.controller.shutdown()
    assert finished == [True]


def test_real_window_close_deferred_when_cleanup_finishes_inside_close_event(qt_app):
    from fanyiji_next.studio import StudioWindow

    class BusyRuntime(FakeRuntime):
        is_busy = True

    window = StudioWindow(prototype=False)
    controller = AppController(window, runtime_factory=BusyRuntime, persist_settings=False,
                               device_lister=lambda: [])
    try:
        window.show()
        window.start_button.click()
        runtime = controller._runtime
        runtime.is_running = False
        runtime.callback(RuntimeEvent("state", controller.session_id, status="stopped"))
        settle(qt_app)
        assert controller._runtime_stopped and runtime.is_busy
        runtime.is_busy = False
        window.close()
        settle(qt_app)
        assert controller._runtime is None
        assert window.closing_ready
        assert not window.isVisible()
    finally:
        window.finish_shutdown()
        window.deleteLater()
        settle(qt_app)
