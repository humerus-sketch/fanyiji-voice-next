"""Standalone interface prototype using preset captions and in-memory settings."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from PySide6.QtCore import QObject, QTimer, Slot, qVersion

from .models import AppSettings, RuntimeEvent


# Every pair has the same meaning in both languages; direction selects its order.
SAMPLES = (
    ("今天的天气很好。", "今日は天気がいいです。"),
    ("请稍等，我正在调整字幕。", "少々お待ちください。字幕を調整しています。"),
    ("欢迎来到今天的直播。", "今日の配信へようこそ。"),
    ("这是一段较长的模拟字幕，用来查看换行、字号和双通道阅读效果。",
     "これは長めの模擬字幕です。折り返しや文字サイズ、二つのチャンネルの読みやすさを確認できます。"),
    ("可以暂停播放，回看前面的字幕。", "再生を一時停止して、前の字幕を読み返すことができます。"),
    ("谢谢观看，我们下次再见。", "ご覧いただきありがとうございます。またお会いしましょう。"),
)

BASE_READING_MS = 2800
MAX_READING_MS = 6000
FIRST_SAMPLE_DELAY_MS = 80


def _reading_duration(texts) -> int:
    """Estimate preset-caption reading time, never audio or translation latency."""
    readable = max((sum(character.isalnum() for character in text) for text in texts), default=0)
    return min(MAX_READING_MS, BASE_READING_MS + max(0, readable - 12) * 85)


class PrototypeController(QObject):
    """Play complete caption pairs without accessing audio, models, or disk."""

    def __init__(self, window) -> None:
        super().__init__(window)
        self.window = window
        self._settings = window.settings
        self._index = 0
        self._session_id = 0
        self._closed = False
        self._state = "idle"
        self._playback_speed = 1.0
        self._reading_ms = BASE_READING_MS
        self._waiting_first_sample = False
        self._timer = QTimer(self)
        self._timer.setInterval(BASE_READING_MS)
        self._timer.timeout.connect(self._advance)
        window.demo_requested.connect(self.toggle_play)
        window.replay_requested.connect(self.replay)
        window.stop_requested.connect(self.clear)
        window.settings_changed.connect(self._settings_changed)
        window.closing.connect(self.shutdown)
        speed_signal = getattr(window, "playback_speed_changed", None)
        if speed_signal is not None:
            speed_signal.connect(self._set_playback_speed)
        self._report_progress()
        self._set_state("idle", "界面原型 · 点击播放预置模拟字幕；不采集声音。")

    @property
    def state(self) -> str:
        return self._state

    def _set_state(self, state: str, message: str) -> None:
        self._state = state
        self.window.set_session_state(state, message)

    def _has_enabled_channel(self) -> bool:
        return self._settings.microphone.enabled or self._settings.system.enabled

    def _report_progress(self) -> None:
        callback = getattr(self.window, "set_playback_progress", None)
        if callable(callback):
            callback(self._index, len(SAMPLES))

    def _next_interval(self) -> int:
        return FIRST_SAMPLE_DELAY_MS if self._waiting_first_sample else round(self._reading_ms / self._playback_speed)

    @Slot(float)
    def _set_playback_speed(self, speed: float) -> None:
        if self._closed or speed not in (0.75, 1.0, 1.25):
            return
        self._playback_speed = float(speed)
        # setInterval updates a running timer but never starts a paused one.
        self._timer.setInterval(self._next_interval())

    @Slot()
    def toggle_play(self) -> None:
        if self._closed:
            return
        if self._state == "demo":
            self._timer.stop()
            self._set_state("paused", "模拟字幕已暂停；可以调整方向、字号或回看历史。")
        elif self._state == "paused":
            if not self._has_enabled_channel():
                self.window.show_error("请开启至少一个字幕通道，再播放模拟字幕。")
                return
            self._set_state("demo", "正在播放预置模拟字幕 · 正常速度每句约3–6秒，仅为演示阅读节奏。")
            self._timer.start(self._next_interval())
        else:
            self.replay()

    @Slot()
    def replay(self) -> None:
        if self._closed:
            return
        self._timer.stop()
        self._session_id += 1
        self._index = 0
        self._reading_ms = BASE_READING_MS
        self._waiting_first_sample = True
        self.window.clear_captions()
        self._report_progress()
        if not self._has_enabled_channel():
            self._set_state("paused", "请开启至少一个字幕通道，再播放模拟字幕。")
            return
        self._set_state("demo", "正在播放预置模拟字幕 · 正常速度每句约3–6秒，仅为演示阅读节奏。")
        self._timer.start(FIRST_SAMPLE_DELAY_MS)

    @Slot()
    def _advance(self) -> None:
        if self._closed or self._state != "demo":
            return
        if not self._has_enabled_channel():
            self._timer.stop()
            self._set_state("paused", "所有字幕通道已关闭；开启通道后可继续播放。")
            return
        settings = self._settings
        chinese, japanese = SAMPLES[self._index]
        visible_texts = []
        for channel, config in (("mic", settings.microphone), ("system", settings.system)):
            if not config.enabled:
                continue
            original, translated = (chinese, japanese) if config.direction == "zh_to_ja" else (japanese, chinese)
            visible_texts.append(translated)
            if settings.show_original:
                visible_texts.append(original)
            segment_id = f"prototype-{self._session_id}-{self._index}-{channel}"
            # Capture both texts before applying either event, so setting changes
            # cannot make an original and its translation use different directions.
            for kind, text in (("original", original), ("translation", translated)):
                self.window.apply_event(RuntimeEvent(
                    kind=kind, session_id=self._session_id, channel=channel,
                    segment_id=segment_id, text=text, status="demo",
                    message="界面原型：预置模拟字幕，未采集声音。",
                ))
        self._reading_ms = _reading_duration(visible_texts)
        self._waiting_first_sample = False
        self._timer.setInterval(self._next_interval())
        self._index += 1
        self._report_progress()
        if self._index == len(SAMPLES):
            self._timer.stop()
            self._set_state("finished", "模拟字幕已播放完毕 · 字幕已保留，再次播放可从头查看。")

    @Slot(object)
    def _settings_changed(self, settings: AppSettings) -> None:
        if self._closed or not isinstance(settings, AppSettings):
            return
        self._settings = settings
        if self._state == "demo" and not self._has_enabled_channel():
            self._timer.stop()
            self._set_state("paused", "所有字幕通道已关闭；开启通道后可继续播放。")

    @Slot()
    def clear(self) -> None:
        if self._closed:
            return
        self._timer.stop()
        self._index = 0
        self._session_id += 1
        self._reading_ms = BASE_READING_MS
        self._waiting_first_sample = False
        self._timer.setInterval(self._next_interval())
        self.window.clear_captions()
        self._report_progress()
        self._set_state("idle", "字幕已清空 · 点击播放预置模拟字幕；不采集声音。")

    @Slot()
    def shutdown(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._timer.stop()


def _forbidden_modules() -> list[str]:
    backends = {"fanyiji_next.audio_runtime", "fanyiji_next.controller", "fanyiji_next.app"}
    heavy = {"numpy", "sherpa_onnx", "ctranslate2", "pyaudio", "pyaudiowpatch", "sentencepiece", "soxr", "torch", "transformers"}
    return sorted({
        name if name in backends else name.split(".")[0]
        for name in sys.modules
        if name in backends or name.split(".")[0].lower() in heavy
    })


def _report_base() -> dict:
    forbidden = _forbidden_modules()
    return {
        "prototype_only": True,
        "voice_backend_loaded": bool(forbidden),
        "audio_capture": False,
        "model_loading": False,
        "settings_written": False,
        "forbidden_modules": forbidden,
        "qt_version": qVersion(),
    }


def _write_report(report: dict, path: Path | None) -> None:
    content = json.dumps(report, ensure_ascii=False, indent=2)
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content + "\n", encoding="utf-8")
    else:
        print(content)


def _configure_fonts(app) -> None:
    from PySide6.QtGui import QFont, QFontDatabase

    for name in ("msyh.ttc", "msyhbd.ttc"):
        font = Path("C:/Windows/Fonts") / name
        if font.is_file():
            QFontDatabase.addApplicationFont(str(font))
    app.setFont(QFont("Microsoft YaHei UI", 10))


def _settle(app) -> None:
    for _ in range(8):
        app.processEvents()


def _smoke_test(app, window, controller, screenshot: Path | None) -> dict:
    checks = {"initially_idle": controller.state == "idle" and not controller._timer.isActive()}
    window.demo_requested.emit()
    controller._timer.timeout.emit()
    _settle(app)
    checks["plays_paired_captions"] = bool(window.store.items()) and all(
        item.original and item.translated for item in window.store.items())
    window.demo_requested.emit()
    count = len(window.store.items())
    controller._timer.timeout.emit()
    checks["pause_preserves_captions"] = controller.state == "paused" and len(window.store.items()) == count
    window.demo_requested.emit()
    controller._timer.timeout.emit()
    checks["resume_continues"] = len(window.store.items()) > count
    window.replay_requested.emit()
    checks["replay_clears_previous_run"] = not window.store.items()
    controller._timer.timeout.emit()
    window.stop_requested.emit()
    checks["clear_stops_playback"] = controller.state == "idle" and not window.store.items() and not controller._timer.isActive()
    window.replay_requested.emit()
    for _ in range(3):
        controller._timer.timeout.emit()
    window.demo_requested.emit()
    _settle(app)
    if screenshot is not None:
        screenshot.parent.mkdir(parents=True, exist_ok=True)
        checks["screenshot_saved"] = window.grab().save(str(screenshot))
    report = _report_base()
    report.update({"checks": checks, "caption_count": len(window.store.items()),
                   "ok": not report["forbidden_modules"] and all(checks.values())})
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="翻译姬全新界面原型：仅预置模拟字幕，不接入声音。")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--check-runtime", action="store_true")
    parser.add_argument("--screenshot", type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args(argv)
    from PySide6.QtWidgets import QApplication

    if args.check_runtime:
        report = _report_base()
        report.update({"qt_available": True, "ok": not report["forbidden_modules"]})
        _write_report(report, args.report)
        return 0 if report["ok"] else 1

    from .studio import StudioWindow as VoiceWindow

    app = QApplication.instance() or QApplication(sys.argv[:1])
    _configure_fonts(app)
    window = VoiceWindow(prototype=True)
    controller = PrototypeController(window)
    window._prototype_controller = controller
    app.aboutToQuit.connect(controller.shutdown)
    window.show()
    _settle(app)
    if args.smoke_test:
        try:
            report = _smoke_test(app, window, controller, args.screenshot)
        except Exception as exc:
            report = _report_base()
            report.update({"ok": False, "error": str(exc)})
        finally:
            controller.shutdown()
            window.close()
            window.deleteLater()
            _settle(app)
        _write_report(report, args.report)
        return 0 if report["ok"] else 1
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
