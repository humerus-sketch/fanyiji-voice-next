"""Prototype interaction tests with real Qt, preset text, and no voice backend."""

import json
import os
from dataclasses import replace
from pathlib import Path
import subprocess
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtCore import Signal
from PySide6.QtWidgets import QApplication

from fanyiji_next.models import AppSettings
from fanyiji_next.prototype import (
    BASE_READING_MS, FIRST_SAMPLE_DELAY_MS, MAX_READING_MS,
    PrototypeController, SAMPLES, _configure_fonts, _reading_duration,
)
from fanyiji_next.ui import VoiceWindow


ROOT = Path(__file__).resolve().parents[1]


class ProgressWindow(VoiceWindow):
    playback_speed_changed = Signal(float)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.progress_updates = []

    def set_playback_progress(self, current, total):
        self.progress_updates.append((current, total))


def settle(app):
    for _ in range(6):
        app.processEvents()


@pytest.fixture
def prototype():
    app = QApplication.instance() or QApplication([])
    _configure_fonts(app)
    windows = []

    def create(settings=None, *, window_class=VoiceWindow):
        window = window_class(settings, prototype=True)
        controller = PrototypeController(window)
        windows.append((window, controller))
        window.show()
        settle(app)
        return window, controller, app

    yield create
    for window, controller in windows:
        controller.shutdown()
        window.close()
        window.deleteLater()
    settle(app)


def advance(controller, app, count=1):
    for _ in range(count):
        controller._timer.timeout.emit()
        settle(app)


def test_initial_window_is_idle_without_model_paths_or_automatic_playback(prototype):
    window, controller, app = prototype()
    assert controller.parent() is window
    assert controller.state == "idle"
    assert not controller._timer.isActive()
    assert not window.store.items()
    assert window.settings.asr_model == ""
    assert window.model_button.isHidden()
    advance(controller, app)
    assert not window.store.items()


def test_main_button_plays_pauses_and_continues_complete_caption_pairs(prototype):
    window, controller, app = prototype()
    window.start_button.click()
    advance(controller, app)
    first = [(item.id, item.original, item.translated) for item in window.store.items()]
    assert len(first) == 2
    assert all(original and translated for _, original, translated in first)
    assert controller.state == "demo"
    window.start_button.click()
    assert controller.state == "paused"
    assert not controller._timer.isActive()
    advance(controller, app)
    assert [(item.id, item.original, item.translated) for item in window.store.items()] == first
    window.start_button.click()
    advance(controller, app)
    assert len(window.store.items()) == 4
    assert controller.state == "demo"


def test_first_caption_appears_after_a_short_delay_then_long_captions_get_more_reading_time(prototype):
    window, controller, app = prototype()
    assert controller._timer.interval() == BASE_READING_MS
    window.start_button.click()
    assert controller._timer.interval() == FIRST_SAMPLE_DELAY_MS
    assert not window.store.items()
    advance(controller, app)
    assert len(window.store.items()) == 2
    assert controller._timer.interval() == BASE_READING_MS
    advance(controller, app, 3)
    assert BASE_READING_MS < controller._timer.interval() <= MAX_READING_MS


def test_preset_reading_estimate_ignores_spacing_and_caps_long_text():
    assert _reading_duration(("短句", "  …… !? \n" * 100)) == BASE_READING_MS
    assert _reading_duration(("長い字幕" * 100,)) == MAX_READING_MS


def test_optional_progress_callback_tracks_playback_replay_and_clear(prototype):
    window, controller, app = prototype(window_class=ProgressWindow)
    assert window.progress_updates == [(0, len(SAMPLES))]
    window.start_button.click()
    assert window.progress_updates[-1] == (0, len(SAMPLES))
    advance(controller, app)
    assert window.progress_updates[-1] == (1, len(SAMPLES))
    advance(controller, app, len(SAMPLES) - 1)
    assert window.progress_updates[-1] == (len(SAMPLES), len(SAMPLES))
    assert controller.state == "finished"
    window.replay_requested.emit()
    assert window.progress_updates[-1] == (0, len(SAMPLES))
    advance(controller, app)
    window.stop_requested.emit()
    assert window.progress_updates[-1] == (0, len(SAMPLES))


@pytest.mark.parametrize("speed", [0.75, 1.0, 1.25])
def test_optional_speed_control_changes_reading_rate_without_delaying_the_first_sample(prototype, speed):
    window, controller, app = prototype(window_class=ProgressWindow)
    window.playback_speed_changed.emit(speed)
    assert not controller._timer.isActive()
    window.start_button.click()
    assert controller._timer.interval() == FIRST_SAMPLE_DELAY_MS
    advance(controller, app)
    assert controller._timer.interval() == pytest.approx(BASE_READING_MS / speed, abs=1)
    captions = len(window.store.items())
    window.playback_speed_changed.emit(1.0)
    assert controller._timer.interval() == BASE_READING_MS
    assert len(window.store.items()) == captions


def test_speed_changes_never_resume_paused_playback_or_accept_invalid_rates(prototype):
    window, controller, app = prototype(window_class=ProgressWindow)
    window.start_button.click()
    advance(controller, app)
    window.start_button.click()
    window.playback_speed_changed.emit(0.75)
    assert controller.state == "paused"
    assert not controller._timer.isActive()
    interval = controller._timer.interval()
    window.playback_speed_changed.emit(0.0)
    window.playback_speed_changed.emit(float("nan"))
    assert controller._timer.interval() == interval
    window.start_button.click()
    assert controller._timer.isActive()
    assert controller._timer.interval() == interval


def test_channel_and_direction_changes_affect_new_pairs_without_rewriting_history(prototype):
    window, controller, app = prototype()
    window.start_button.click()
    advance(controller, app)
    previous = [(item.id, item.original, item.translated) for item in window.store.items()]
    microphone_enabled = window.channel_controls["mic"][0]
    system_direction = window.channel_controls["system"][1]
    assert microphone_enabled.isEnabled() and system_direction.isEnabled()
    microphone_enabled.setChecked(False)
    system_direction.setCurrentIndex(system_direction.findData("zh_to_ja"))
    advance(controller, app)
    captions = window.store.items()
    assert [(item.id, item.original, item.translated) for item in captions[:2]] == previous
    assert len(captions) == 3
    assert captions[-1].channel == "system"
    assert (captions[-1].original, captions[-1].translated) == SAMPLES[1]


@pytest.mark.parametrize("direction", ["zh_to_ja", "ja_to_zh"])
def test_enabled_channel_uses_the_selected_translation_direction(prototype, direction):
    settings = AppSettings()
    settings = replace(settings, microphone=replace(settings.microphone, enabled=False),
                       system=replace(settings.system, direction=direction))
    window, controller, app = prototype(settings)
    window.start_button.click()
    advance(controller, app)
    captions = window.store.items()
    assert len(captions) == 1 and captions[0].channel == "system"
    expected = SAMPLES[0] if direction == "zh_to_ja" else SAMPLES[0][::-1]
    assert (captions[0].original, captions[0].translated) == expected


def test_direction_change_between_events_cannot_mismatch_a_caption_pair(prototype, monkeypatch):
    window, controller, app = prototype()
    apply = window.apply_event
    changed = False

    def apply_with_setting_change(event):
        nonlocal changed
        apply(event)
        if not changed and event.kind == "original" and event.channel == "mic":
            changed = True
            settings = replace(window.settings, microphone=replace(window.settings.microphone, direction="ja_to_zh"))
            window.update_settings(settings)
            window.settings_changed.emit(settings)

    monkeypatch.setattr(window, "apply_event", apply_with_setting_change)
    window.start_button.click()
    advance(controller, app)
    microphone = window.store.items("mic")
    assert (microphone[0].original, microphone[0].translated) == SAMPLES[0]
    advance(controller, app)
    microphone = window.store.items("mic")
    assert (microphone[1].original, microphone[1].translated) == SAMPLES[1][::-1]


def test_finished_playback_keeps_captions_and_next_play_restarts(prototype):
    window, controller, app = prototype()
    window.start_button.click()
    advance(controller, app, len(SAMPLES))
    assert controller.state == "finished"
    assert not controller._timer.isActive()
    captions = window.store.items()
    assert len(captions) == len(SAMPLES) * 2
    old_ids = {item.id for item in captions}
    advance(controller, app)
    assert len(window.store.items()) == len(captions)
    window.start_button.click()
    assert not window.store.items()
    advance(controller, app)
    assert len(window.store.items()) == 2
    assert not old_ids.intersection(item.id for item in window.store.items())


def test_sidebar_replay_and_clear_controls_reset_only_the_demo(prototype):
    window, controller, app = prototype()
    window.start_button.click()
    advance(controller, app, 2)
    window.demo_button.click()
    assert controller.state == "demo"
    assert not window.store.items()
    advance(controller, app)
    assert len(window.store.items()) == 2
    window.stop_button.click()
    assert controller.state == "idle"
    assert not controller._timer.isActive()
    assert not window.store.items()
    advance(controller, app)
    assert not window.store.items()


def test_disabling_all_channels_pauses_without_consuming_more_samples(prototype):
    window, controller, app = prototype()
    window.start_button.click()
    advance(controller, app)
    window.channel_controls["mic"][0].setChecked(False)
    window.channel_controls["system"][0].setChecked(False)
    assert controller.state == "paused"
    advance(controller, app)
    assert len(window.store.items()) == 2
    window.channel_controls["mic"][0].setChecked(True)
    window.start_button.click()
    advance(controller, app)
    microphone = window.store.items("mic")
    assert len(microphone) == 2
    assert (microphone[-1].original, microphone[-1].translated) == SAMPLES[1]


def test_settings_remain_in_memory_through_playback_and_appearance_changes(prototype, monkeypatch):
    def forbidden_save(*args, **kwargs):
        pytest.fail("The interface prototype must never read or save user settings")

    monkeypatch.setattr("fanyiji_next.settings.load_settings", forbidden_save)
    monkeypatch.setattr("fanyiji_next.settings.save_settings", forbidden_save)
    window, controller, app = prototype()
    window.start_button.click()
    advance(controller, app)
    window.font_size.setValue(30)
    assert window.settings.font_size == 30
    assert controller.state == "demo"
    assert len(window.store.items()) == 2
    window.stop_button.click()
    assert window.settings.font_size == 30


def test_shutdown_stops_timer_and_ignores_future_interactions(prototype):
    window, controller, app = prototype()
    window.start_button.click()
    advance(controller, app)
    count = len(window.store.items())
    window.closing.emit()
    window.demo_requested.emit()
    window.replay_requested.emit()
    window.stop_requested.emit()
    advance(controller, app)
    assert not controller._timer.isActive()
    assert len(window.store.items()) == count


@pytest.mark.parametrize("mode", ["--check-runtime", "--smoke-test"])
@pytest.mark.parametrize("entrypoint, extra", [("run_prototype.py", ()), ("run.py", ("--prototype",))])
def test_cli_prototype_has_no_voice_modules_and_writes_no_settings(tmp_path, mode, entrypoint, extra):
    report = tmp_path / "report.json"
    screenshot = tmp_path / "prototype.png"
    profile = tmp_path / "isolated-profile"
    environment = os.environ.copy()
    environment.update(QT_QPA_PLATFORM="offscreen", LOCALAPPDATA=str(profile))
    command = [sys.executable, str(ROOT / entrypoint), *extra, mode, "--report", str(report)]
    if mode == "--smoke-test":
        command.extend(("--screenshot", str(screenshot)))
    result = subprocess.run(command, cwd=ROOT, env=environment, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    data = json.loads(report.read_text(encoding="utf-8"))
    assert data["ok"] and data["prototype_only"]
    assert not data["voice_backend_loaded"]
    assert not data["audio_capture"] and not data["model_loading"] and not data["settings_written"]
    assert data["forbidden_modules"] == []
    assert not (profile / "FanyiJiVoiceNext" / "settings.json").exists()
    if mode == "--smoke-test":
        assert all(data["checks"].values())
        assert screenshot.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
