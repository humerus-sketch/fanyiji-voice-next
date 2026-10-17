"""Main-thread session control for manually started audio and explicit demos."""

from __future__ import annotations

from collections import deque
from pathlib import Path
from threading import Thread

from PySide6.QtCore import QObject, QTimer, Qt, Signal, Slot

from .audio_runtime import VoiceRuntime, list_audio_devices
from .models import AppSettings, RuntimeEvent
from . import settings as settings_store


class _EventBridge(QObject):
    received = Signal(object)
    devices_received = Signal(object)


class AppController(QObject):
    """Keep worker callbacks out of widgets and discard retired sessions."""

    def __init__(
        self,
        window,
        *,
        settings_file: Path | None = None,
        runtime_factory=VoiceRuntime,
        persist_settings: bool = True,
        device_lister=list_audio_devices,
    ) -> None:
        super().__init__(window)
        self.window = window
        self._settings_file = settings_file
        self._runtime_factory = runtime_factory
        self._persist_settings = persist_settings
        self._accepted_settings = window.settings
        self._device_lister = device_lister
        self._device_worker = None
        self._device_generation = 0
        self._runtime = None
        self._runtime_stopped = False
        self._active_settings: AppSettings | None = None
        self._session_id = 0
        self._state = "stopped"
        self._closed = False
        self._shutdown_waiting = False
        self._bridge = _EventBridge(self)
        self._bridge.received.connect(self._handle_event, Qt.ConnectionType.QueuedConnection)
        self._bridge.devices_received.connect(self._handle_devices, Qt.ConnectionType.QueuedConnection)
        self._demo_events: deque[RuntimeEvent] = deque()
        self._demo_timer = QTimer(self)
        self._demo_timer.setInterval(450)
        self._demo_timer.timeout.connect(self._advance_demo)
        self._retire_timer = QTimer(self)
        self._retire_timer.setInterval(20)
        self._retire_timer.timeout.connect(self._release_stopped_runtime)
        window.start_requested.connect(self.start_session)
        window.stop_requested.connect(self.stop_session)
        window.demo_requested.connect(self.start_demo)
        window.settings_changed.connect(self._settings_changed)
        window.closing.connect(self.shutdown)
        if hasattr(window, "clear_requested"):
            window.clear_requested.connect(self.clear_captions)
        if hasattr(window, "refresh_devices_requested"):
            window.refresh_devices_requested.connect(self.refresh_devices)
        self._set_state("stopped", "尚未开始；可手动开始识别，或查看模拟演示。")

    @property
    def session_id(self) -> int:
        return self._session_id

    @property
    def state(self) -> str:
        return self._state

    def _set_state(self, state: str, message: str = "") -> None:
        self._state = state
        self.window.set_session_state(state, message)

    def _post_event(self, event: RuntimeEvent) -> None:
        if self._closed and not (self._shutdown_waiting and event.kind == "state" and event.status == "stopped"):
            return
        try:
            self._bridge.received.emit(event)
        except RuntimeError:
            # A worker can finish after Qt has destroyed its parent window.
            return

    @Slot(object)
    def _handle_event(self, event: RuntimeEvent) -> None:
        if not isinstance(event, RuntimeEvent) or event.session_id != self._session_id:
            return
        if self._closed:
            if self._shutdown_waiting and event.kind == "state" and event.status == "stopped":
                self._runtime_stopped = True
                self._release_stopped_runtime()
                if self._runtime is not None:
                    self._retire_timer.start()
            return
        if event.kind == "state":
            if self._runtime is None:
                return
            if event.status == "stopped":
                self._runtime_stopped = True
                self._release_stopped_runtime()
                if self._runtime is not None:
                    self._retire_timer.start()
            elif event.status == "error":
                self._report_runtime_error(event.message)
            elif self._state not in {"stopping", "stopped", "error"} and event.status in {"loading", "running"}:
                self._set_state(event.status, event.message)
            return
        if event.kind == "error":
            if self._runtime is not None:
                if event.status == "recoverable":
                    if self._state in {"loading", "running"}:
                        self.window.show_error(event.message or event.text or "当前片段处理失败。")
                        self.window.apply_event(event)
                else:
                    self._report_runtime_error(event.message or event.text)
            return
        if self._state in {"loading", "running", "demo"}:
            self.window.apply_event(event)

    def _report_runtime_error(self, message: str) -> None:
        message = message or "声音处理失败，请检查模型和声音来源设置。"
        self._set_state("error", message)
        self.window.show_error(message)
        if self._runtime is not None:
            self._runtime.stop()

    def _runtime_busy(self) -> bool:
        if self._runtime is None:
            return False
        return bool(getattr(self._runtime, "is_busy", self._runtime.is_running))

    def _discovering_devices(self) -> bool:
        return self._device_worker is not None and self._device_worker.is_alive()

    @Slot()
    def _release_stopped_runtime(self) -> None:
        if self._closed and not self._shutdown_waiting:
            return
        if self._runtime is not None and self._runtime_stopped and not self._runtime_busy():
            self._runtime = None
            self._active_settings = None
            if not self._shutdown_waiting:
                self._retire_timer.stop()
                self._set_state("stopped", "声音处理已停止；可以调整设置或重新开始。")
        if self._shutdown_waiting and self._runtime is None and not self._discovering_devices():
            self._finish_shutdown()

    @Slot(object)
    def start_session(self, settings: AppSettings) -> None:
        if self._closed:
            return
        self._release_stopped_runtime()
        if self._runtime is not None:
            self.window.show_error("上一轮声音处理尚未结束，请等待停止完成后再开始。")
            return
        if self._discovering_devices():
            self.window.show_error("声音设备列表仍在读取，请等待完成后再开始。")
            return
        if not isinstance(settings, AppSettings):
            self.window.show_error("无法开始：设置格式无效。")
            return
        try:
            settings_store.validate_settings(settings)
        except settings_store.SettingsError as exc:
            self.window.show_error(f"无法开始：{exc}")
            return
        if not settings.microphone.enabled and not settings.system.enabled:
            self.window.show_error("请先开启至少一个声音来源。")
            return
        leaving_demo = self._state == "demo"
        self._cancel_demo()
        self._session_id += 1
        self._active_settings = settings
        if leaving_demo:
            self.window.clear_captions()
        self._set_state("loading", "正在准备模型和声音来源…")
        try:
            self._runtime = self._runtime_factory(self._post_event)
            self._runtime_stopped = False
            # The runtime starts its worker and validates/loads models there.
            self._runtime.start(settings, self._session_id)
        except Exception as exc:
            self._report_runtime_error(str(exc))
            if self._runtime is not None:
                if not self._runtime_busy():
                    self._runtime = None
            if self._runtime is None:
                self._active_settings = None
                self._set_state("stopped", "启动未完成，请检查错误提示并调整设置。")

    @Slot()
    def clear_captions(self) -> None:
        if self._closed:
            return
        self._release_stopped_runtime()
        if self._runtime is not None:
            self.window.show_error("请先停止声音处理，再清空字幕。")
            return
        self._cancel_demo()
        self._session_id += 1
        self._active_settings = None
        self.window.clear_captions()
        self._set_state("stopped", "字幕已清空，可以开始新的对话。")

    @Slot()
    def refresh_devices(self) -> None:
        if self._closed or self._discovering_devices():
            return
        self._release_stopped_runtime()
        if self._runtime is not None:
            message = "请先停止声音处理并等待资源释放，再刷新设备列表。"
            if hasattr(self.window, "set_device_error"):
                self.window.set_device_error(message)
            else:
                self.window.show_error(message)
            return
        self._device_generation += 1
        generation = self._device_generation

        def discover():
            try:
                payload = (generation, self._device_lister(), "")
            except Exception as exc:
                payload = (generation, [], f"声音设备读取失败：{exc}")
            if not self._closed:
                try:
                    self._bridge.devices_received.emit(payload)
                except RuntimeError:
                    pass

        self._device_worker = Thread(target=discover, name="fanyiji-device-discovery", daemon=True)
        self._device_worker.start()

    @Slot(object)
    def _handle_devices(self, payload) -> None:
        generation, devices, error = payload
        if self._closed or generation != self._device_generation:
            return
        if error:
            if hasattr(self.window, "set_device_error"):
                self.window.set_device_error(error)
            else:
                self.window.show_error(error)
        else:
            self.window.set_devices(devices)

    @Slot()
    def stop_session(self) -> None:
        if self._closed:
            return
        if self._state == "demo":
            self._cancel_demo()
            self._session_id += 1
            self._active_settings = None
            self._set_state("stopped", "模拟演示已结束，未采集声音。")
        elif self._runtime is not None:
            if self._state != "stopping":
                self._set_state("stopping", "正在停止声音处理，请等待当前任务结束…")
                self._runtime.stop()
        elif self._state == "error":
            self._set_state("stopped", "当前处理已停止，可以调整设置后重新开始。")

    def _cancel_demo(self) -> None:
        self._demo_timer.stop()
        self._demo_events.clear()

    @Slot()
    def start_demo(self) -> None:
        if self._closed:
            return
        self._release_stopped_runtime()
        if self._runtime is not None:
            self.window.show_error("请先停止声音处理，再查看模拟演示。")
            return
        self._cancel_demo()
        self._session_id += 1
        self._active_settings = self.window.settings
        self.window.clear_captions()
        self._set_state("demo", "模拟演示：以下为预置中日文本，不采集声音，也不代表真实识别结果。")
        samples = (
            ("今天的天气很好。", "今日は天気がいいです。", "こんばんは。配信を始めます。", "晚上好，直播现在开始。"),
            ("请稍等，我正在调整声音。", "少々お待ちください。音声を調整しています。", "音はちゃんと聞こえていますか。", "声音听得清楚吗？"),
            ("谢谢大家，我们下次再见。", "皆さん、ありがとうございます。またお会いしましょう。", "今日も見てくれてありがとう。", "谢谢大家今天的观看。"),
        )
        for index, (mic_original, mic_translation, system_original, system_translation) in enumerate(samples):
            for kind, channel, text in (
                ("original", "mic", mic_original), ("original", "system", system_original),
                ("translation", "mic", mic_translation), ("translation", "system", system_translation),
            ):
                self._demo_events.append(RuntimeEvent(
                    kind=kind, session_id=self._session_id, channel=channel,
                    segment_id=f"demo-{index}-{channel}", text=text, status="demo",
                    message="模拟演示：预置文本，不是真实识别结果。",
                ))
        self._demo_timer.start()

    @Slot()
    def _advance_demo(self) -> None:
        if self._closed or self._state != "demo":
            return
        if self._demo_events:
            self._post_event(self._demo_events.popleft())
        if not self._demo_events:
            self._demo_timer.stop()
            self._set_state("demo", "模拟演示已播放完毕；以上为预置文本，未进行真实识别。")

    @Slot(object)
    def _settings_changed(self, settings: AppSettings) -> None:
        if self._closed:
            return
        if not isinstance(settings, AppSettings):
            self.window.show_error("设置格式无效，未保存。")
            return
        try:
            settings_store.validate_settings(settings)
        except settings_store.SettingsError as exc:
            self._restore_settings()
            self.window.show_error(f"设置格式无效，未保存：{exc}")
            return
        if self._active_settings is not None:
            runtime_fields = (
                "microphone", "system", "asr_model", "asr_tokens", "translation_model_dir", "translation_spm",
                "recognition_backend", "whisper_model_dir", "recognition_device", "recognition_compute_type",
                "translation_backend", "nllb_model_dir", "nllb_tokenizer_dir", "translation_device",
                "translation_compute_type", "hymt_model_file", "hymt_server_exe", "vad_model",
            )
            if any(getattr(settings, name) != getattr(self._active_settings, name) for name in runtime_fields):
                self.window.show_error("请先停止当前处理，再修改模型或声音来源。")
                self._restore_settings()
                return
        if self._persist_settings:
            try:
                settings_store.save_settings(settings, path=self._settings_file)
            except (OSError, ValueError) as exc:
                self._restore_settings()
                self.window.show_error(f"设置保存失败：{exc}")
                return
        self._accepted_settings = settings

    def _restore_settings(self) -> None:
        if hasattr(self.window, "update_settings"):
            self.window.update_settings(self._accepted_settings)

    @Slot()
    def shutdown(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._device_generation += 1
        self._cancel_demo()
        self._retire_timer.stop()
        if self._runtime is not None or self._discovering_devices():
            self._shutdown_waiting = hasattr(self.window, "finish_shutdown")
        if self._runtime is not None:
            self._runtime.stop()
        if self._shutdown_waiting:
            self._set_state("stopping", "正在结束声音任务和设备读取，完成后自动关闭…")
            self._retire_timer.start()
            self._release_stopped_runtime()
            return
        self._session_id += 1
        self._bridge.received.disconnect(self._handle_event)

    def _finish_shutdown(self) -> None:
        self._shutdown_waiting = False
        self._retire_timer.stop()
        self._session_id += 1
        self._bridge.received.disconnect(self._handle_event)
        # A close request can itself enter this path; defer to avoid nested closeEvent.
        QTimer.singleShot(0, self.window.finish_shutdown)
