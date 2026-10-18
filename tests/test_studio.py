"""Warm-white studio interactions, using preset text and real Qt only."""

import os
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtCore import QAbstractAnimation, QPoint, QPointF, QRect, Qt
from PySide6.QtGui import QFontDatabase, QWheelEvent
from PySide6.QtWidgets import QApplication, QDialogButtonBox, QFileDialog, QLabel, QPushButton
from PySide6.QtTest import QTest

from fanyiji_next.models import AppSettings, RuntimeEvent
from fanyiji_next.studio import StudioWindow


def settle(app):
    for _ in range(8):
        app.processEvents()


@pytest.fixture
def studio():
    app = QApplication.instance() or QApplication([])
    if not QFontDatabase.families():
        for name in ("msyh.ttc", "msyhbd.ttc"):
            path = Path("C:/Windows/Fonts") / name
            if path.is_file():
                QFontDatabase.addApplicationFont(str(path))
    windows = []

    def create(settings=None, *, prototype=True):
        widget = StudioWindow(settings, prototype=prototype)
        windows.append(widget)
        widget.show()
        settle(app)
        return widget, app

    yield create
    for widget in windows:
        if widget.prototype:
            widget.close()
        else:
            widget.finish_shutdown()
        widget.deleteLater()
    settle(app)


def add_caption(widget, index=1, channel="system", original=None, translated=None):
    original = original or f"今日の字幕 {index}：よろしくお願いします。"
    translated = translated or f"今天的字幕 {index}：请多关照。"
    segment_id = str(index)
    widget.apply_event(RuntimeEvent("original", 1, channel, segment_id, text=original))
    widget.apply_event(RuntimeEvent("translation", 1, channel, segment_id, text=translated))
    return original, translated


def history_labels(panel, text):
    return [label for label in panel.history_container.findChildren(QLabel) if label.text() == text]


def font_size(label):
    font = label.font()
    return font.pointSizeF() if font.pointSizeF() > 0 else font.pixelSize()


def assert_inside(widget, child):
    assert child.isVisible(), f"{child.objectName() or type(child).__name__} should be visible"
    rectangle = QRect(child.mapTo(widget, QPoint()), child.size())
    assert rectangle.width() > 0 and rectangle.height() > 0
    assert widget.rect().contains(rectangle), f"{rectangle} extends outside {widget.rect()}"


def wheel(viewport, delta):
    position = QPointF(viewport.rect().center())
    event = QWheelEvent(position, QPointF(viewport.mapToGlobal(position.toPoint())),
                        QPoint(), QPoint(0, delta), Qt.MouseButton.NoButton,
                        Qt.KeyboardModifier.NoModifier, Qt.ScrollPhase.NoScrollPhase, False)
    QApplication.sendEvent(viewport, event)


def test_default_studio_has_no_model_or_device_entry_and_drawer_is_closed(studio):
    widget, _app = studio()
    assert widget.drawer.isHidden()
    assert not widget.store.items()
    assert widget.settings.asr_model == "" and widget.settings.translation_model_dir == ""
    assert widget.start_button.isEnabled()
    assert not any(
        button.isVisible() and any(word in button.text() for word in ("模型", "设备"))
        for button in widget.findChildren(QPushButton)
    )


def test_settings_button_toggles_drawer_without_changing_caption_data(studio):
    widget, app = studio()
    add_caption(widget)
    widget.settings_button.click()
    settle(app)
    assert widget.drawer.isVisible()
    assert widget.font_size.isVisible() and widget.original_checkbox.isVisible()
    assert all(controls[0].isEnabled() for controls in widget.channel_controls.values())
    widget.settings_button.click()
    settle(app)
    assert widget.drawer.isHidden()
    assert len(widget.store.items()) == 1


def test_latest_caption_updates_plain_selectable_original_and_translation(studio):
    widget, _app = studio()
    original, translated = add_caption(widget, original="<b>原文です</b>", translated="<b>这是正文</b>")
    panel = widget.panels["system"]
    assert panel.latest_original.text() == original
    assert panel.latest_translation.text() == translated
    for label in (panel.latest_original, panel.latest_translation):
        assert label.textFormat() == Qt.TextFormat.PlainText
        assert label.textInteractionFlags() & Qt.TextInteractionFlag.TextSelectableByMouse
    widget.apply_event(RuntimeEvent("translation", 1, "system", "1", text="修订后的译文"))
    assert panel.latest_translation.text() == "修订后的译文"
    assert len(widget.store.items()) == 1


def test_original_switch_changes_current_and_existing_history(studio):
    widget, app = studio()
    widget.history_button.setChecked(True)
    original, _translated = add_caption(widget)
    settle(app)
    panel = widget.panels["system"]
    originals = history_labels(panel, original)
    assert originals
    widget.original_checkbox.setChecked(False)
    settle(app)
    assert not panel.latest_original.isVisible()
    assert all(not label.isVisible() for label in originals)
    widget.original_checkbox.setChecked(True)
    settle(app)
    assert panel.latest_original.isVisible()
    assert all(label.isVisible() for label in originals)


def test_font_size_changes_current_and_existing_history(studio):
    widget, app = studio()
    widget.history_button.setChecked(True)
    _original, translated = add_caption(widget)
    settle(app)
    panel = widget.panels["system"]
    historical = history_labels(panel, translated)
    assert historical
    previous_current = font_size(panel.latest_translation)
    previous_history = [font_size(label) for label in historical]
    widget.font_size.setValue(32)
    settle(app)
    assert widget.settings.font_size == 32
    assert font_size(panel.latest_translation) > previous_current
    assert all(font_size(label) > before for label, before in zip(historical, previous_history))


def test_reviewing_history_does_not_jump_when_new_captions_arrive(studio):
    widget, app = studio()
    widget.history_button.setChecked(True)
    for index in range(22):
        add_caption(widget, index, translated=f"历史第 {index} 条。" + "长字幕用于回看，不应被新字幕打断。" * 5)
    settle(app)
    panel = widget.panels["system"]
    bar = panel.history_scroll.verticalScrollBar()
    assert bar.maximum() > 0
    assert bar.value() == bar.maximum()
    bar.setValue(0)
    add_caption(widget, 23, translated="刚到达的一条新字幕。" * 8)
    settle(app)
    assert bar.value() == 0
    assert not panel.follow_button.isChecked()
    panel.follow_button.click()
    settle(app)
    assert panel.follow_button.isChecked()
    assert bar.value() == bar.maximum()


def test_history_visibility_toggle_keeps_all_captions(studio):
    widget, app = studio()
    original, translated = add_caption(widget)
    before = [(item.channel, item.id, item.original, item.translated) for item in widget.store.items()]
    widget.history_button.setChecked(True)
    settle(app)
    assert widget.history_area.isVisible()
    widget.history_button.setChecked(False)
    settle(app)
    assert not widget.history_area.isVisible()
    widget.history_button.setChecked(True)
    settle(app)
    assert [(item.channel, item.id, item.original, item.translated) for item in widget.store.items()] == before
    assert history_labels(widget.panels["system"], original)
    assert history_labels(widget.panels["system"], translated)


def test_copy_latest_writes_only_requested_channel_to_clipboard(studio):
    widget, app = studio()
    add_caption(widget, channel="mic", translated="麦克风这边的译文。")
    add_caption(widget, channel="system", translated="系统这边的译文。")
    settle(app)
    before = widget.stage.geometry()
    widget.copy_latest("system")
    settle(app)
    assert widget.stage.geometry() == before
    copied = QApplication.clipboard().text()
    assert "系统这边的译文。" in copied
    assert "麦克风这边的译文。" not in copied


def test_clear_captions_hides_existing_history_before_deferred_deletion(studio):
    widget, app = studio()
    widget.history_button.setChecked(True)
    original, translated = add_caption(widget)
    settle(app)
    panel = widget.panels["system"]
    old_labels = history_labels(panel, original) + history_labels(panel, translated)
    assert old_labels and all(label.isVisible() for label in old_labels)
    widget.clear_captions()
    assert not widget.store.items()
    assert all(not label.isVisible() for label in old_labels)
    assert panel.latest_original.text() != original
    assert panel.latest_translation.text() != translated
    widget.apply_event(RuntimeEvent("translation", 1, "system", "1", text="过期的返回结果"))
    assert not widget.store.items()
    assert panel.latest_translation.text() != "过期的返回结果"


@pytest.mark.parametrize("size", [(1000, 600), (1264, 632), (1100, 760), (1440, 900)])
def test_current_cards_playback_dock_and_open_drawer_stay_inside_window(studio, size):
    widget, app = studio()
    widget.resize(*size)
    for channel in ("mic", "system"):
        add_caption(widget, channel=channel, translated="较长的模拟字幕，用来验证主区域换行和边界。" * 8)
    settle(app)
    assert widget.size().toTuple() == size
    for panel in widget.panels.values():
        assert_inside(widget, panel.current_card)
        assert panel.current_card.rect().contains(panel.current_scroll.geometry())
        assert panel.latest_translation.isVisible()
    for control in (widget.start_button, widget.replay_button, widget.clear_button,
                    widget.settings_button, widget.history_button, widget.playback_dock):
        assert_inside(widget, control)
    widget.settings_button.click()
    settle(app)
    assert_inside(widget, widget.drawer)
    assert_inside(widget, widget.start_button)
    assert_inside(widget, widget.playback_dock)
    for panel in widget.panels.values():
        assert_inside(widget, panel.current_card)
        assert panel.current_card.rect().contains(panel.current_scroll.geometry())
        assert panel.latest_translation.isVisible()


def test_playback_state_changes_main_button_and_progress(studio):
    widget, app = studio()
    widget.set_session_state("idle", "等待播放模拟字幕。")
    idle_label = widget.start_button.text()
    widget.set_session_state("demo", "播放中。")
    playing_label = widget.start_button.text()
    assert playing_label != idle_label
    widget.set_playback_progress(2, 6)
    settle(app)
    assert widget.playback_progress.maximum() == 6
    assert widget.playback_progress.value() == 2
    widget.set_session_state("paused", "已暂停。")
    assert widget.start_button.text() != playing_label
    assert widget.start_button.isEnabled()
    assert widget.playback_progress.value() == 2
    widget.set_session_state("finished", "播放完毕。")
    widget.set_playback_progress(6, 6)
    assert widget.start_button.isEnabled()
    assert widget.playback_progress.value() == widget.playback_progress.maximum()


def test_disabled_sources_can_be_reenabled_from_drawer(studio):
    settings = AppSettings()
    settings = replace(settings,
                       microphone=replace(settings.microphone, enabled=False),
                       system=replace(settings.system, enabled=False))
    widget, app = studio(settings)
    assert widget.settings_button.isEnabled()
    widget.settings_button.click()
    settle(app)
    microphone = widget.channel_controls["mic"][0]
    assert microphone.isVisible() and microphone.isEnabled()
    microphone.setChecked(True)
    settle(app)
    assert widget.settings.microphone.enabled
    assert widget.start_button.isEnabled()
    assert widget.panels["mic"].latest_translation.isVisible()


def test_focus_mode_preserves_captions_and_keeps_playback_available(studio):
    widget, app = studio()
    add_caption(widget)
    before = [(item.id, item.translated) for item in widget.store.items()]
    widget.focus_button.click()
    settle(app)
    assert widget.focus_button.isChecked()
    assert widget.start_button.isVisible()
    assert widget.panels["system"].latest_translation.isVisible()
    widget.focus_button.click()
    settle(app)
    assert not widget.focus_button.isChecked()
    assert [(item.id, item.translated) for item in widget.store.items()] == before


def test_play_replay_and_clear_buttons_emit_the_public_signals(studio):
    widget, _app = studio()
    actions = []
    widget.demo_requested.connect(lambda: actions.append("play"))
    widget.replay_requested.connect(lambda: actions.append("replay"))
    widget.stop_requested.connect(lambda: actions.append("clear"))
    widget.start_button.click()
    widget.replay_button.click()
    add_caption(widget)
    widget.clear_button.click()
    assert actions == ["play", "replay", "clear"]


def test_vertical_reading_uses_focus_space_and_escape_restores_side_by_side(studio):
    widget, app = studio()
    add_caption(widget)
    widget.layout_combo.setCurrentIndex(1)
    settle(app)
    assert widget.focus_button.isChecked()
    assert not widget.history_section.isVisible()
    for panel in widget.panels.values():
        assert_inside(widget, panel.current_card)
        assert panel.current_card.rect().contains(panel.current_scroll.geometry())
    widget._escape()
    settle(app)
    assert widget.layout_combo.currentIndex() == 0
    assert not widget.focus_button.isChecked()


def test_caption_fade_finishes_and_reduced_motion_keeps_new_text_visible(studio):
    widget, app = studio()
    widget.animations_enabled = True
    add_caption(widget)
    QTest.qWait(290)
    settle(app)
    panel = widget.panels["system"]
    assert panel.latest_translation.graphicsEffect().opacity() == pytest.approx(1.0)
    add_caption(widget, 2, translated="在淡入期间关闭动效，应立即显示完整文字。")
    QTest.qWait(20)
    assert panel._animation.state() == QAbstractAnimation.State.Running
    widget.motion_checkbox.setChecked(True)
    assert panel._animation.state() == QAbstractAnimation.State.Stopped
    assert panel.latest_translation.graphicsEffect().opacity() == pytest.approx(1.0)
    add_caption(widget, 3, translated="减少动态效果后，这句仍然完整可见。")
    settle(app)
    assert not widget.animations_enabled
    assert panel.latest_translation.graphicsEffect().opacity() == pytest.approx(1.0)


def test_space_preserves_checkbox_and_focused_button_native_operation(studio):
    widget, app = studio()
    actions = []
    widget.demo_requested.connect(lambda: actions.append("play"))
    widget.replay_requested.connect(lambda: actions.append("replay"))
    widget.activateWindow()
    widget.settings_button.click()
    widget.original_checkbox.setFocus()
    settle(app)
    assert QApplication.focusWidget() is widget.original_checkbox
    before = widget.original_checkbox.isChecked()
    QTest.keyClick(widget.original_checkbox, Qt.Key.Key_Space)
    settle(app)
    assert widget.original_checkbox.isChecked() is not before
    assert not actions
    widget.replay_button.setFocus()
    settle(app)
    QTest.keyClick(widget.replay_button, Qt.Key.Key_Space)
    assert actions == ["replay"]
    widget.setFocus()
    settle(app)
    QTest.keyClick(widget, Qt.Key.Key_Space)
    assert actions == ["replay", "play"]


def test_small_work_area_fits_window_and_keeps_reading_controls_accessible(studio):
    widget, app = studio()
    work_area = QRect(120, 40, 1280, 680)
    widget.fit_to_screen(work_area)
    widget.settings_button.click()
    widget.font_size.setValue(40)
    settle(app)
    assert widget.size().toTuple() == (1264, 632)
    assert not widget.welcome.isVisible()
    assert widget.history_section.height() == 120
    assert work_area.contains(widget.frameGeometry())
    for control in (widget.drawer, widget.playback_dock, widget.start_button):
        assert_inside(widget, control)
    for panel in widget.panels.values():
        assert_inside(widget, panel.current_card)
        assert panel.current_scroll.height() > 100
    widget.focus_button.click()
    widget.focus_button.click()
    settle(app)
    assert not widget.welcome.isVisible()
    widget.resize(1440, 900)
    settle(app)
    assert widget.welcome.isVisible()
    assert widget.history_section.height() == 160


def test_voice_mode_waits_for_manual_start_and_has_no_preset_playback_controls(studio):
    widget, _app = studio(prototype=False)
    starts, refreshes = [], []
    widget.start_requested.connect(starts.append)
    widget.refresh_devices_requested.connect(lambda: refreshes.append(True))
    assert not widget.prototype
    assert not widget.store.items()
    assert not starts and not refreshes
    assert widget.model_button.isVisible() and widget.runtime_metrics.isVisible()
    assert not widget.replay_button.isVisible() and not widget.speed_combo.isVisible()
    assert not widget.playback_progress.isVisible()
    assert widget.start_button.text() == "开始翻译"
    assert not widget.stop_button.isEnabled()
    widget.start_button.click()
    assert starts == [widget.settings]
    assert not widget.store.items()


def test_voice_start_stop_and_clear_are_separate_actions(studio):
    widget, _app = studio(prototype=False)
    actions = []
    widget.start_requested.connect(lambda settings: actions.append(("start", settings)))
    widget.stop_requested.connect(lambda: actions.append(("stop", None)))
    widget.clear_requested.connect(lambda: actions.append(("clear", None)))
    widget.start_button.click()
    widget.set_session_state("loading")
    add_caption(widget)
    assert not widget.clear_button.isEnabled()
    widget.clear_button.click()
    widget.stop_button.click()
    assert [action[0] for action in actions] == ["start", "stop"]
    widget.set_session_state("stopping")
    assert not widget.start_button.isEnabled() and not widget.stop_button.isEnabled()
    widget.set_session_state("stopped")
    assert widget.clear_button.isEnabled()
    widget.clear_button.click()
    assert [action[0] for action in actions] == ["start", "stop", "clear"]
    assert widget.store.items()  # The controller owns retiring results and clearing the session.


def test_voice_runtime_locks_sources_but_keeps_reading_settings_available(studio):
    widget, _app = studio(prototype=False)
    accepted = []
    widget.settings_changed.connect(accepted.append)
    before_source = widget.settings.microphone
    widget.set_session_state("running")
    assert widget.model_button.isEnabled()
    assert all(not control.isEnabled() for pair in widget.channel_controls.values() for control in pair)
    assert widget.font_size.isEnabled() and widget.original_checkbox.isEnabled()
    widget.font_size.setValue(32)
    widget.original_checkbox.setChecked(False)
    assert widget.settings.font_size == 32 and not widget.settings.show_original
    assert accepted and accepted[-1].microphone == before_source
    widget.set_session_state("stopped")
    assert widget.model_button.isEnabled()
    assert all(control.isEnabled() for pair in widget.channel_controls.values() for control in pair)


def test_voice_transparency_toggle_uses_default_level_and_restores_saved_level(studio):
    widget, app = studio(prototype=False)
    saved = []
    widget.settings_changed.connect(saved.append)
    assert not widget.transparency_button.isChecked()
    assert widget.transparency_slider.value() == 0
    assert widget.transparency_value_label.text() == "0%"
    assert widget.windowOpacity() == pytest.approx(1.0)
    widget.transparency_button.click()
    settle(app)
    assert widget.settings.window_transparency_enabled
    assert widget.settings.window_opacity == 80
    assert widget.transparency_slider.value() == 20
    assert widget.transparency_value_label.text() == "20%"
    assert widget.windowOpacity() == pytest.approx(.8, abs=1 / 255)
    widget.transparency_button.click()
    assert not widget.settings.window_transparency_enabled
    assert widget.settings.window_opacity == 80
    assert widget.windowOpacity() == pytest.approx(1.0)
    assert widget.transparency_value_label.text() == "20% · 已关闭"
    widget.transparency_button.click()
    assert widget.settings.window_opacity == 80
    assert widget.windowOpacity() == pytest.approx(.8, abs=1 / 255)
    assert len(saved) == 3 and saved[-1] == widget.settings


def test_voice_legacy_transparency_and_disabled_settings_restore_without_writes(studio):
    widget, app = studio(AppSettings(window_opacity=85), prototype=False)
    saved = []
    widget.settings_changed.connect(saved.append)
    assert widget.transparency_button.isChecked()
    assert widget.transparency_slider.value() == 15
    assert widget.windowOpacity() == pytest.approx(.85, abs=1 / 255)
    disabled = replace(widget.settings, window_transparency_enabled=False)
    widget.update_settings(disabled)
    settle(app)
    assert not saved
    assert not widget.transparency_button.isChecked()
    assert widget.transparency_slider.value() == 15
    assert widget.windowOpacity() == pytest.approx(1.0)
    reopened, _app = studio(disabled, prototype=False)
    assert reopened.windowOpacity() == pytest.approx(1.0)
    assert reopened.transparency_slider.value() == 15
    reopened.transparency_button.click()
    assert reopened.settings.window_opacity == 85
    assert reopened.windowOpacity() == pytest.approx(.85, abs=1 / 255)
    external = replace(widget.settings, window_opacity=60, window_transparency_enabled=True)
    widget.update_settings(external)
    settle(app)
    assert not saved
    assert widget.transparency_button.isChecked()
    assert widget.transparency_slider.value() == 40
    assert widget.transparency_value_label.text() == "40%"
    assert widget.windowOpacity() == pytest.approx(.6, abs=1 / 255)


def test_voice_transparency_slider_enables_effect_and_zero_is_opaque(studio):
    widget, _app = studio(AppSettings(window_opacity=85, window_transparency_enabled=False), prototype=False)
    saved = []
    widget.settings_changed.connect(saved.append)
    assert widget.transparency_slider.minimum() == 0 and widget.transparency_slider.maximum() == 70
    widget.transparency_slider.setValue(70)
    assert widget.settings.window_transparency_enabled and widget.settings.window_opacity == 30
    assert widget.transparency_value_label.text() == "70%"
    assert widget.transparency_button.isChecked()
    assert widget.windowOpacity() == pytest.approx(.3, abs=1 / 255)
    widget.transparency_slider.setValue(0)
    assert widget.settings.window_transparency_enabled and widget.settings.window_opacity == 100
    assert not widget.transparency_button.isChecked()
    assert widget.windowOpacity() == pytest.approx(1.0)
    widget.transparency_button.click()
    assert widget.settings.window_opacity == 80 and widget.transparency_slider.value() == 20
    assert len(saved) == 3


def test_voice_transparency_changes_only_main_window_without_restarting_runtime_or_rebuilding_history(studio, monkeypatch):
    from fanyiji_next.controller import AppController

    runtimes = []
    class QuietRuntime:
        def __init__(self, callback):
            self.callback = callback
            self.is_running = False
            self.stop_count = 0
            self.start_count = 0
            runtimes.append(self)

        def start(self, _settings, session_id):
            self.start_count += 1
            self.is_running = True
            self.session_id = session_id
            self.callback(RuntimeEvent("state", session_id, status="running"))

        def stop(self):
            self.stop_count += 1
            self.is_running = False
            self.callback(RuntimeEvent("state", self.session_id, status="stopped"))

    widget, app = studio(prototype=False)
    controller = AppController(widget, runtime_factory=QuietRuntime, persist_settings=False, device_lister=lambda: [])
    widget.start_button.click()
    settle(app)
    for index in range(40):
        add_caption(widget, index, "system", translated="保留这段字幕与回看位置。" * 5)
    settle(app)
    panel = widget.panels["system"]
    bar = panel.history_scroll.verticalScrollBar()
    bar.setValue(bar.maximum() // 2)
    settle(app)
    anchor_id, top = panel._find_review_anchor()
    rows = tuple(panel.rows.values())
    captions = [(id(item), item.original, item.translated) for item in widget.store.items()]
    original_settings = widget.settings
    dialog = widget.open_runtime_settings()
    settle(app)

    def no_caption_rebuild(*_args):
        raise AssertionError("Transparency must not rebuild the caption stream")

    with monkeypatch.context() as scoped:
        for channel_panel in widget.panels.values():
            scoped.setattr(channel_panel, "update_settings", no_caption_rebuild)
        widget.transparency_button.click()
        widget.transparency_slider.setValue(35)
        widget.transparency_button.click()
        widget.transparency_slider.setValue(45)
    settle(app)
    assert controller.state == "running" and len(runtimes) == 1
    assert runtimes[0].start_count == 1 and runtimes[0].stop_count == 0
    assert runtimes[0].is_running and widget._voice_resources_active
    assert controller._accepted_settings == widget.settings
    assert widget.settings.microphone == original_settings.microphone
    assert widget.settings.system == original_settings.system
    assert widget.settings.recognition_backend == original_settings.recognition_backend
    assert widget.settings.translation_backend == original_settings.translation_backend
    assert widget.settings.system_audio_processing == original_settings.system_audio_processing
    assert captions == [(id(item), item.original, item.translated) for item in widget.store.items()]
    assert rows == tuple(panel.rows.values())
    assert panel.rows[anchor_id].mapTo(panel.history_scroll.viewport(), QPoint()).y() == top
    assert not panel.follow_button.isChecked()
    assert dialog.windowOpacity() == pytest.approx(1.0)
    assert dialog.isVisible() and not dialog.apply_button.isEnabled()
    assert widget.windowOpacity() == pytest.approx(.55, abs=1 / 255)
    dialog.reject()
    controller.stop_session()
    settle(app)
    controller.shutdown()


@pytest.mark.parametrize("size", [(1000, 600), (1100, 760), (1264, 632), (1440, 900)])
def test_voice_transparency_controls_fit_narrow_windows_and_remain_in_reading_settings(studio, size):
    settings = AppSettings(recognition_backend="whisper", translation_backend="hymt", system_audio_processing="mdx_gtcrn")
    widget, app = studio(settings, prototype=False)
    widget.resize(*size)
    settle(app)
    assert widget.size().toTuple() == size
    assert widget.transparency_button.isVisible() is (size[0] >= 1180)
    for control in (widget.model_button, widget.history_button, widget.focus_button, widget.settings_button):
        assert_inside(widget, control)
    if widget.transparency_button.isVisible():
        assert_inside(widget, widget.transparency_button)
    widget.settings_button.click()
    settle(app)
    assert widget.drawer.isVisible() and widget.transparency_slider.isVisible()
    assert widget.transparency_slider.isEnabled()
    assert widget.transparency_slider.accessibleName() == "窗口透明度"
    assert widget.transparency_slider.width() > 100


@pytest.mark.parametrize("state", ["stopped", "running"])
def test_space_on_transparency_slider_never_starts_or_stops_audio(studio, state):
    widget, app = studio(prototype=False)
    widget.set_session_state(state)
    actions = []
    widget.start_requested.connect(lambda _settings: actions.append("start"))
    widget.stop_requested.connect(lambda: actions.append("stop"))
    widget.settings_button.click()
    widget.transparency_slider.setFocus()
    settle(app)
    assert QApplication.focusWidget() is widget.transparency_slider
    assert all(not shortcut.isEnabled() for _key, shortcut in widget._reading_shortcuts)
    QTest.keyClick(widget.transparency_slider, Qt.Key.Key_Space)
    QTest.keyClick(widget.transparency_slider, Qt.Key.Key_Right)
    settle(app)
    assert actions == []
    assert widget.transparency_slider.value() == 1
    assert widget.settings.window_opacity == 99


def test_voice_error_keeps_resources_locked_until_stopped(studio):
    widget, _app = studio(prototype=False)
    widget.set_session_state("loading")
    widget.set_session_state("error", "模型载入失败。")
    assert widget.state_chip.text() == "处理出错"
    assert not widget.start_button.isEnabled() and widget.model_button.isEnabled()
    dialog = widget.open_runtime_settings()
    assert dialog is not None and not dialog.apply_button.isEnabled()
    widget.set_session_state("stopped")
    assert widget.start_button.isEnabled() and widget.model_button.isEnabled()
    assert dialog.apply_button.isEnabled()


def test_voice_stopped_cancels_pending_rows_without_fabricating_translations(studio):
    widget, app = studio(prototype=False)
    widget.history_button.setChecked(True)
    widget.set_session_state("running")
    add_caption(widget, index="completed", channel="mic", translated="已经完成的真实译文。")
    for channel in ("mic", "system"):
        widget.apply_event(RuntimeEvent("original", 1, channel, "pending", text=f"{channel} 已识别但还未翻译。"))
    widget.set_session_state("stopping")
    assert all(item.status == "translating" for item in widget.store.items() if item.id == "pending")
    widget.set_session_state("stopped")
    settle(app)
    for channel in ("mic", "system"):
        pending = next(item for item in widget.store.items(channel) if item.id == "pending")
        assert pending.status == "cancelled" and pending.translated == ""
        assert pending.original == f"{channel} 已识别但还未翻译。"
        panel = widget.panels[channel]
        assert panel.latest_original.text() == pending.original
        assert panel.latest_translation.text() == "已停止翻译，原文已保留"
        assert "已停止翻译" in panel.current_title.text()
        assert history_labels(panel, "已停止翻译，原文已保留")
        assert not panel.copy_button.isEnabled()
    completed = next(item for item in widget.store.items("mic") if item.id == "completed")
    assert completed.status == "complete" and completed.translated == "已经完成的真实译文。"
    exported = widget.transcript()
    assert exported.count("[已停止翻译，原文已保留]") == 2
    assert "已经完成的真实译文。" in exported
    assert all(item.original in exported for item in widget.store.items())
    assert all(item.translated == "" for item in widget.store.items() if item.status == "cancelled")
    widget.set_session_state("stopped")
    assert len(widget.store.items()) == 3


def test_prototype_stopped_state_does_not_cancel_preset_caption_rows(studio):
    widget, _app = studio()
    widget.apply_event(RuntimeEvent("original", 1, "mic", "pending", text="尚未展示的样例原文。"))
    widget.set_session_state("stopped")
    assert widget.store.items("mic")[0].status == "translating"
    assert widget.panels["mic"].latest_translation.text() == "正在呈现这句样例…"


def test_voice_config_cancel_keeps_paths_channels_and_devices_unchanged(studio):
    widget, app = studio(prototype=False)
    accepted = []
    widget.settings_changed.connect(accepted.append)
    before = widget.settings
    dialog = widget.open_runtime_settings()
    settle(app)
    dialog.path_edits["asr_model"].setText("C:/models/new.onnx")
    dialog.channel_controls["mic"][0].setChecked(False)
    dialog.reject()
    assert dialog.result_settings is None
    assert widget.settings == before and accepted == []


def test_voice_config_apply_emits_staged_settings_without_starting(studio):
    widget, app = studio(prototype=False)
    accepted, starts = [], []
    widget.settings_changed.connect(accepted.append)
    widget.start_requested.connect(starts.append)
    dialog = widget.open_runtime_settings()
    settle(app)
    dialog.path_edits["asr_model"].setText(" C:/models/voice.onnx ")
    dialog.path_edits["translation_model_dir"].setText("C:/models/m2m100")
    dialog.channel_controls["system"][1].setCurrentIndex(0)
    assert widget.settings.asr_model == ""
    dialog.apply_button.click()
    assert len(accepted) == 1
    assert widget.settings.asr_model == "C:/models/voice.onnx"
    assert widget.settings.translation_model_dir == "C:/models/m2m100"
    assert widget.settings.system.direction == "zh_to_ja"
    assert not starts and not widget.store.items()


def test_voice_device_refresh_is_explicit_and_preserves_per_channel_selection(studio):
    widget, app = studio(prototype=False)
    refreshes = []
    widget.refresh_devices_requested.connect(lambda: refreshes.append(True))
    dialog = widget.open_runtime_settings()
    settle(app)
    assert refreshes == []
    dialog.refresh_button.click()
    assert refreshes == [True]
    devices = [
        SimpleNamespace(index=7, name="USB Microphone", channel="mic"),
        SimpleNamespace(index=19, name="Speaker loopback", channel="system"),
    ]
    widget.set_devices(devices)
    assert dialog.device_combos["mic"].findData(7) >= 0
    assert dialog.device_combos["mic"].findData(19) == -1
    dialog.device_combos["system"].setCurrentIndex(dialog.device_combos["system"].findData(19))
    dialog.apply_button.click()
    assert widget.settings.system.device_index == 19
    assert widget.settings.microphone.device_index is None


def test_voice_device_failure_stays_visible_in_the_settings_dialog(studio):
    widget, app = studio(prototype=False)
    dialog = widget.open_runtime_settings()
    settle(app)
    dialog.refresh_button.click()
    widget.set_device_error("设备枚举失败：WASAPI 不可用。")
    assert "WASAPI" in dialog.device_hint.text()
    assert dialog.isVisible() and dialog.apply_button.isEnabled()
    assert not widget.store.items()


def test_voice_volume_and_backlog_events_update_metrics_without_creating_captions(studio):
    widget, _app = studio(prototype=False)
    widget.set_session_state("running")
    widget.apply_event(RuntimeEvent("level", 1, "mic", level=.46))
    widget.apply_event(RuntimeEvent("level", 1, "system", level=2.0))
    widget.apply_event(RuntimeEvent("queue", 1, queue_depth=4, message="等待本地识别"))
    assert widget.level_meters["mic"].value() == 460
    assert widget.level_meters["system"].value() == 1000
    assert widget.queue_label.text() == "待处理 4 段"
    assert "等待本地识别" in widget.queue_label.toolTip()
    assert not widget.store.items()
    widget.set_session_state("stopped")
    assert all(meter.value() == 0 for meter in widget.level_meters.values())
    assert widget.queue_label.text() == "待处理 0 段"


@pytest.mark.parametrize("show_original", [True, False])
def test_voice_segment_failure_displays_only_original_and_keeps_failure_details(studio, show_original):
    widget, app = studio(AppSettings(show_original=show_original), prototype=False)
    widget.set_session_state("running")
    widget.apply_event(RuntimeEvent("original", 1, "mic", "failed", text="请保留这句原文。"))
    widget.apply_event(RuntimeEvent("error", 1, "mic", "failed", message="翻译模型执行失败。", status="recoverable"))
    settle(app)
    caption = widget.store.items("mic")[0]
    assert caption.original == "请保留这句原文。"
    assert caption.status == "failed"
    assert caption.translated == ""
    assert caption.error_message == "翻译模型执行失败。"
    assert widget.panels["mic"].latest_original.text() == caption.original
    assert widget.panels["mic"].latest_original.isHidden()
    assert widget.panels["mic"].latest_translation.text() == caption.original
    assert widget.panels["mic"].latest_translation.toolTip() == caption.error_message
    row = widget.panels["mic"].rows["failed"]
    assert row.original.isHidden()
    assert row.translated.isVisible()
    assert row.translated.text() == caption.original
    assert row.translated.toolTip() == caption.error_message
    assert caption.error_message in row.toolTip()
    assert font_size(row.translated) >= 18
    assert "翻译失败" in widget.panels["mic"].current_title.text()
    assert widget.state_chip.text() == "正在翻译"
    widget.update_settings(replace(widget.settings, show_original=not show_original, font_size=30))
    settle(app)
    assert row.original.isHidden()
    assert row.translated.isVisible() and row.translated.text() == caption.original
    assert widget.panels["mic"].latest_original.isHidden()
    widget.copy_latest("mic")
    assert QApplication.clipboard().text() == caption.original
    assert widget._toast_timer.interval() == 1600
    assert widget.transcript().endswith("\n" + caption.original)
    assert caption.error_message not in widget.transcript()
    add_caption(widget, 2, "mic", original="下一句原文。", translated="次の翻訳。")
    assert widget.panels["mic"].latest_translation.text() == "次の翻訳。"
    assert widget.panels["mic"].latest_original.isHidden() is show_original


@pytest.mark.parametrize("status", ["error", "failed"])
def test_failed_translation_events_keep_only_original_on_reading_surface(studio, status):
    widget, _app = studio(AppSettings(show_original=False), prototype=False)
    widget.apply_event(RuntimeEvent("original", 1, "system", "failed", text="日本語の原文です。"))
    widget.apply_event(RuntimeEvent("translation", 1, "system", "failed", text="兼容来源的失败原因。", status=status))
    caption = widget.store.items()[0]
    assert caption.status == "failed" and caption.translated == ""
    assert caption.error_message == "兼容来源的失败原因。"
    row = widget.panels["system"].rows["failed"]
    assert row.original.isHidden()
    assert row.translated.text() == caption.original
    assert row.translated.toolTip() == caption.error_message


def test_recognition_failure_without_original_does_not_create_an_orphan_caption(studio):
    widget, _app = studio(prototype=False)
    widget.apply_event(RuntimeEvent("error", 1, "mic", "failed-asr", status="recoverable", message="语音识别失败。"))
    assert not widget.store.items()
    assert not widget.panels["mic"].rows


def test_prototype_failure_still_displays_original_with_separate_error_details(studio):
    widget, _app = studio(AppSettings(show_original=False))
    widget.apply_event(RuntimeEvent("original", 1, "mic", "failed", text="预览原文。"))
    widget.apply_event(RuntimeEvent("error", 1, "mic", "failed", message="预览失败原因。", status="recoverable"))
    panel = widget.panels["mic"]
    assert panel.latest_original.isHidden()
    assert panel.latest_translation.text() == "预览原文。"
    assert panel.latest_translation.toolTip() == "预览失败原因。"
    assert panel.rows["failed"].translated.text() == "预览原文。"


def test_recoverable_notice_expires_after_two_seconds_without_extension_for_duplicates(studio):
    widget, app = studio(prototype=False)
    widget.show_recoverable_error("本句翻译未完成。")
    settle(app)
    assert widget.notice.isVisible()
    assert widget._toast_timer.interval() == 2000
    QTest.qWait(1200)
    remaining = widget._toast_timer.remainingTime()
    assert 0 < remaining < 1200
    widget.show_recoverable_error("本句翻译未完成。")
    assert widget._toast_timer.remainingTime() <= remaining + 100
    QTest.qWait(1000)
    assert not widget.notice.isVisible()
    assert not widget._toast_timer.isActive()


def test_fatal_error_replaces_transient_notice_and_remains_visible(studio):
    widget, app = studio(prototype=False)
    widget.show_recoverable_error("本句翻译未完成。")
    widget.show_error("模型载入失败，请检查路径。")
    settle(app)
    assert not widget._toast_timer.isActive()
    QTest.qWait(2100)
    assert widget.notice.isVisible()
    assert widget.notice.text() == "模型载入失败，请检查路径。"


@pytest.mark.parametrize("state", ["loading", "stopped"])
def test_new_or_stopped_session_clears_only_recoverable_notice(studio, state):
    widget, _app = studio(prototype=False)
    widget.show_recoverable_error("本句翻译未完成。")
    widget.set_session_state(state)
    assert not widget.notice.isVisible()
    assert not widget._toast_timer.isActive()
    widget.show_error("设备配置错误。")
    widget.set_session_state(state)
    assert widget.notice.isVisible()
    assert widget.notice.text() == "设备配置错误。"


def test_voice_config_scrolls_with_apply_and_cancel_outside_the_scroll_view(studio):
    widget, app = studio(prototype=False)
    dialog = widget.open_runtime_settings()
    dialog.resize(560, 420)
    settle(app)
    assert dialog.scroll.verticalScrollBar().maximum() > 0
    assert dialog.rect().contains(dialog.buttons.geometry())
    assert dialog.apply_button.isVisible()
    dialog.reject()


def test_voice_config_initial_fit_preserves_buttons_and_scrolls_on_small_work_area(studio):
    widget, app = studio(AppSettings(translation_backend="m2m100"), prototype=False)
    dialog = widget.open_runtime_settings()
    work_area = QRect(120, 40, 1280, 680)
    dialog.fit_to_screen(work_area)
    settle(app)
    assert dialog.height() <= 632
    assert dialog.minimumSize().toTuple() == (560, 420)
    assert work_area.contains(dialog.frameGeometry())
    assert dialog.rect().contains(dialog.buttons.geometry())
    for button in dialog.buttons.buttons():
        assert button.isVisible()
        assert dialog.rect().contains(QRect(button.mapTo(dialog, QPoint()), button.size()))
    bar = dialog.scroll.verticalScrollBar()
    assert bar.maximum() > 0
    bar.setValue(bar.maximum())
    settle(app)
    last_path = dialog.path_edits["translation_spm"]
    assert dialog.scroll.viewport().rect().contains(QRect(last_path.mapTo(dialog.scroll.viewport(), QPoint()), last_path.size()))
    dialog.reject()


@pytest.mark.parametrize("size", [(1000, 600), (1440, 900)])
def test_voice_controls_and_current_cards_fit_compact_and_default_windows(studio, size):
    widget, app = studio(prototype=False)
    widget.resize(*size)
    widget.set_session_state("running")
    add_caption(widget, translated="真实模式只使用预置测试文本，不开启任何声音采集。" * 8)
    settle(app)
    assert widget.size().toTuple() == size
    for child in (widget.playback_dock, widget.start_button, widget.stop_button,
                  widget.model_button, widget.queue_label):
        assert_inside(widget, child)
    for panel in widget.panels.values():
        assert_inside(widget, panel.current_card)
        assert panel.current_card.rect().contains(panel.history_scroll.geometry())
        assert panel.current_scroll.isHidden()


def test_voice_close_waits_for_resource_release_without_duplicate_shutdown(studio):
    widget, app = studio(prototype=False)
    shutdowns = []
    widget.closing.connect(lambda: shutdowns.append(True))
    widget.set_session_state("loading")
    widget.set_session_state("error", "模型正在清理。")
    assert not widget.close()
    settle(app)
    assert widget.isVisible() and not widget.closing_ready
    assert "正在关闭" in widget.status_hint.text()
    assert not widget.start_button.isEnabled() and not widget.model_button.isEnabled()
    assert shutdowns == [True]
    widget.close()
    settle(app)
    assert widget.isVisible() and shutdowns == [True]
    # Even a stopped state does not bypass the controller's final busy check.
    widget.set_session_state("stopped")
    widget.close()
    assert widget.isVisible() and shutdowns == [True]
    widget.finish_shutdown()
    settle(app)
    assert widget.closing_ready and not widget.isVisible()
    assert shutdowns == [True]


def test_idle_close_waits_when_controller_upgrades_shutdown_to_stopping(studio):
    widget, app = studio(prototype=False)
    shutdowns = []

    def wait_for_device_enumeration():
        shutdowns.append(True)
        widget.set_session_state("stopping", "正在等待设备列表读取完成…")

    widget.closing.connect(wait_for_device_enumeration)
    assert not widget.close()
    settle(app)
    assert widget.isVisible() and shutdowns == [True]
    assert "等待设备" in widget.status_hint.text()
    assert not widget.centralWidget().isEnabled()
    assert not widget.close()
    assert shutdowns == [True]
    widget.finish_shutdown()
    settle(app)
    assert not widget.isVisible() and widget.closing_ready


@pytest.mark.parametrize("prototype", [False, True])
def test_idle_voice_and_prototype_windows_close_immediately(studio, prototype):
    widget, app = studio(prototype=prototype)
    shutdowns = []
    widget.closing.connect(lambda: shutdowns.append(True))
    assert widget.close()
    settle(app)
    assert not widget.isVisible() and shutdowns == [True]


def test_voice_continuous_history_is_visible_and_includes_current_sentence(studio):
    widget, app = studio(prototype=False)
    original, translated = add_caption(widget, 1)
    add_caption(widget, 2)
    settle(app)
    panel = widget.panels["system"]
    assert not widget.history_button.isCheckable() and widget.history_section.isHidden()
    assert "本次字幕 · 2" == widget.history_button.text()
    assert "全部保留" in widget.caption_count.text()
    assert panel.history_scroll.parentWidget() is panel.current_card
    assert panel.current_scroll.isHidden()
    assert not panel.latest_translation.isVisible()
    assert panel.rows["1"].isVisible() and panel.rows["2"].isVisible()

    assert panel.rows["1"].original.text() == original and panel.rows["1"].translated.text() == translated
    assert font_size(panel.rows["1"].translated) >= 14
    widget.apply_event(RuntimeEvent("translation", 1, "system", "1", text="第一句的迟到修订。"))
    settle(app)
    assert panel.latest_id == "2" and panel.latest_translation.text() == widget.store.items("system")[-1].translated
    assert panel.rows["1"].isVisible() and panel.rows["2"].isVisible()


@pytest.mark.parametrize("size", [(1000, 600), (1264, 632), (1440, 900)])
def test_voice_main_reading_shows_at_least_three_short_original_translation_pairs(studio, size):
    widget, app = studio(prototype=False)
    widget.resize(*size)
    for channel in ("mic", "system"):
        for index in range(6):
            add_caption(widget, index, channel=channel, original=f"公開例{index}です。", translated=f"这是公开例{index}。")
    settle(app)
    assert widget.size().toTuple() == size
    assert widget.history_section.isHidden()
    assert widget.stage.height() >= 350
    for panel in widget.panels.values():
        viewport = panel.history_scroll.viewport()
        for index in range(3, 6):
            row = panel.rows[str(index)]
            rectangle = QRect(row.mapTo(viewport, QPoint()), row.size())
            assert viewport.rect().contains(rectangle), (index, rectangle, viewport.rect())
            assert row.original.isVisible() and row.translated.isVisible()
            assert row.original.text() == f"公開例{index}です。"
            assert row.translated.text() == f"这是公开例{index}。"
        assert panel.latest_id == "5"
        assert panel.follow_button.isChecked()


def test_voice_focus_keeps_continuous_stream_and_session_action_returns_to_latest(studio):
    widget, app = studio(prototype=False)
    for index in range(24):
        add_caption(widget, index, translated="公开字幕，保持焦点模式中的连续阅读。" * 4)
    widget.layout_combo.setCurrentIndex(1)
    settle(app)
    assert widget.focus_button.isChecked() and widget.history_section.isHidden()
    panel = widget.panels["system"]
    assert_inside(widget, panel.history_scroll)
    bar = panel.history_scroll.verticalScrollBar()
    bar.setValue(0)
    settle(app)
    assert not panel.follow_button.isChecked()
    widget.history_button.click()
    settle(app)
    assert widget.focus_button.isChecked() and widget.layout_combo.currentIndex() == 1
    assert panel.follow_button.isChecked() and bar.value() == bar.maximum()
    assert panel.rows["1"].isVisible()


@pytest.mark.parametrize("size", [(1000, 600), (1264, 632), (1440, 900)])
def test_voice_small_window_keeps_single_stream_viewports_inside(studio, size):
    widget, app = studio(prototype=False)
    widget.resize(*size)
    for channel in ("mic", "system"):
        for index in range(3):
            add_caption(widget, index, channel=channel)
    settle(app)
    assert widget.size().toTuple() == size
    assert widget.history_section.isHidden()
    assert_inside(widget, widget.stage)
    assert_inside(widget, widget.history_button)
    for panel in widget.panels.values():
        assert_inside(widget, panel.history_scroll)
        assert panel.history_scroll.viewport().height() >= 300
        assert panel.current_scroll.isHidden()
        assert panel.current_card.rect().contains(panel.history_scroll.geometry())


def test_voice_review_anchor_survives_new_and_late_text_and_font_change(studio):
    widget, app = studio(prototype=False)
    for index in range(24):
        add_caption(widget, index, translated=f"公开历史样例 {index}。" * 4)
    settle(app)
    panel = widget.panels["system"]
    bar = panel.history_scroll.verticalScrollBar()
    bar.setValue(bar.maximum() // 2)
    settle(app)
    assert not panel.follow_button.isChecked()
    visible = next(row for row in panel.rows.values() if row.isVisible()
                   and row.mapTo(panel.history_scroll.viewport(), QPoint()).y() + row.height() > 0)
    before = visible.mapTo(panel.history_scroll.viewport(), QPoint()).y()
    add_caption(widget, 25, translated="新来的公开字幕。" * 8)
    settle(app)
    assert visible.mapTo(panel.history_scroll.viewport(), QPoint()).y() == before
    widget.apply_event(RuntimeEvent("translation", 1, "system", "0", text="最早一句的迟到长译文。" * 60))
    settle(app)
    assert visible.mapTo(panel.history_scroll.viewport(), QPoint()).y() == before
    widget.font_size.setValue(40)
    settle(app)
    assert visible.mapTo(panel.history_scroll.viewport(), QPoint()).y() == before
    assert not panel.follow_button.isChecked()
    panel.follow_button.click()
    settle(app)
    assert panel.follow_button.isChecked() and bar.value() == bar.maximum()


def test_voice_stopping_and_loading_keep_continuous_records_and_manual_transcript(studio, tmp_path):
    widget, app = studio(prototype=False)
    records = [add_caption(widget, index) for index in range(4)]
    widget.set_session_state("running")
    widget.set_session_state("stopped")
    widget.set_session_state("loading")
    settle(app)
    assert len(widget.store.items("system")) == 4
    panel = widget.panels["system"]
    assert all(row.isVisible() for row in panel.rows.values())
    transcript = widget.transcript()
    for original, translated in records:
        assert original in transcript and translated in transcript
    assert list(tmp_path.iterdir()) == []
    widget.clear_captions()
    settle(app)
    assert not widget.store.items() and widget.transcript() == "" and not panel.rows


def test_voice_mouse_wheel_reviews_without_new_or_late_subtitles_stealing_position(studio):
    widget, app = studio(prototype=False)
    for index in range(30):
        add_caption(widget, index, translated=f"公开示例 {index}：滚轮回看时继续接收字幕。" * 3)
    settle(app)
    panel = widget.panels["system"]
    bar = panel.history_scroll.verticalScrollBar()
    assert bar.maximum() > 0 and bar.value() == bar.maximum()
    wheel(panel.history_scroll.viewport(), 720)
    settle(app)
    assert bar.value() < bar.maximum() and not panel.follow_button.isChecked()
    anchor_id, top = panel._find_review_anchor()
    anchor_row = panel.rows[anchor_id]
    add_caption(widget, 30, translated="最新到达的公开字幕。" * 9)
    widget.apply_event(RuntimeEvent("translation", 1, "system", "0", text="较早一句迟到的长译文。" * 50))
    settle(app)
    assert anchor_row.mapTo(panel.history_scroll.viewport(), QPoint()).y() == top
    assert panel.follow_button.text() == "返回最新"
    panel.follow_button.click()
    settle(app)
    assert bar.value() == bar.maximum() and panel.follow_button.isChecked()
    add_caption(widget, 31, translated="返回后继续跟随最新字幕。" * 8)
    settle(app)
    assert bar.value() == bar.maximum()


def test_voice_scrolling_down_to_end_resumes_following(studio):
    widget, app = studio(prototype=False)
    for index in range(24):
        add_caption(widget, index, translated="滚轮公开测试字幕。" * 8)
    settle(app)
    panel = widget.panels["system"]
    bar = panel.history_scroll.verticalScrollBar()
    wheel(panel.history_scroll.viewport(), 360)
    settle(app)
    assert not panel.follow_button.isChecked()
    wheel(panel.history_scroll.viewport(), -100000)
    settle(app)
    assert panel.follow_button.isChecked() and bar.value() == bar.maximum()


def test_voice_session_stream_keeps_over_one_hundred_rows_and_accepts_first_late_translation(studio):
    widget, app = studio(prototype=False)
    widget.apply_event(RuntimeEvent("original", 1, "system", "first", text="最早一句公开原文。"))
    for index in range(125):
        add_caption(widget, index)
    settle(app)
    panel = widget.panels["system"]
    assert len(widget.store.items("system")) == len(panel.rows) == 126
    assert list(panel.rows)[0] == "first"
    widget.apply_event(RuntimeEvent("translation", 1, "system", "first", text="最早一句的迟到译文。"))
    settle(app)
    assert panel.rows["first"].translated.text() == "最早一句的迟到译文。"
    assert widget.store.items("system")[0].translated == "最早一句的迟到译文。"
    assert panel.latest_id == "124"
    assert "本次字幕 · 126" == widget.history_button.text()
    assert "最早一句公开原文。" in widget.transcript()
    assert "最早一句的迟到译文。" in widget.transcript()
    widget.set_session_state("running")
    widget.set_session_state("stopped")
    widget.set_session_state("loading")
    assert len(panel.rows) == 126


@pytest.mark.parametrize("size", [(1000, 600), (1440, 900)])
def test_voice_only_enabled_system_stream_uses_whole_width_and_reading_preferences(studio, size):
    settings = AppSettings()
    settings = replace(settings, microphone=replace(settings.microphone, enabled=False))
    widget, app = studio(settings, prototype=False)
    widget.resize(*size)
    for index in range(28):
        add_caption(widget, index, translated=f"第 {index} 句公开字幕。" * 6)
    settle(app)
    panel = widget.panels["system"]
    assert widget.panels["mic"].current_card.isHidden()
    assert panel.current_card.width() == widget.stage.width()
    assert_inside(widget, panel.history_scroll)
    assert panel.history_scroll.viewport().height() >= 300
    bar = panel.history_scroll.verticalScrollBar()
    bar.setValue(bar.maximum() // 2)
    settle(app)
    anchor_id, top = panel._find_review_anchor()
    row = panel.rows[anchor_id]
    old_font = font_size(row.translated)
    widget.font_size.setValue(40)
    widget.original_checkbox.setChecked(False)
    settle(app)
    assert not panel.follow_button.isChecked()
    assert row.mapTo(panel.history_scroll.viewport(), QPoint()).y() == top
    assert font_size(row.translated) > old_font
    assert all(item.original.isHidden() for item in panel.rows.values())
    widget.focus_button.click()
    settle(app)
    assert panel.history_scroll.isVisible()
    assert row.mapTo(panel.history_scroll.viewport(), QPoint()).y() == top


def runtime_edit_controls(dialog):
    controls = [dialog.apply_button, dialog.refresh_button, *dialog.device_combos.values(),
                *dialog.engine_combos.values(), *dialog.path_edits.values(), *dialog.browse_buttons.values()]
    controls.extend(control for pair in dialog.channel_controls.values() for control in pair)
    return controls


@pytest.mark.parametrize("state,hint", [("loading", "正在加载模型"), ("running", "正在翻译"),
                                         ("stopping", "正在停止"), ("error", "资源释放")])
def test_voice_runtime_model_entry_opens_readonly_settings_in_every_active_state(studio, monkeypatch, state, hint):
    widget, app = studio(prototype=False)
    widget.set_session_state("loading")
    widget.set_session_state(state)
    before = widget.settings
    saved, refreshes, stops = [], [], []
    widget.settings_changed.connect(saved.append)
    widget.refresh_devices_requested.connect(lambda: refreshes.append(True))
    widget.stop_requested.connect(lambda: stops.append(True))
    widget.model_button.click()
    settle(app)
    dialog = widget._runtime_dialog
    assert dialog is not None and dialog.isVisible()
    assert widget.model_button.isEnabled()
    assert dialog.runtime_hint.isVisible() and hint in dialog.runtime_hint.text()
    assert "停止完成后才能修改" in dialog.runtime_hint.text()
    assert all(not control.isEnabled() for control in runtime_edit_controls(dialog))
    assert dialog.scroll.isEnabled()
    assert dialog.buttons.button(QDialogButtonBox.StandardButton.Cancel).isEnabled()
    assert dialog.stop_to_edit_button.isVisible()
    assert dialog.stop_to_edit_button.isEnabled() is (state != "stopping")
    assert dialog.engine_combos["recognition_backend"].currentData() == before.recognition_backend
    assert dialog.engine_combos["translation_backend"].currentData() == before.translation_backend
    assert dialog.path_edits["hymt_model_file"].text() == before.hymt_model_file

    def forbidden(*_args, **_kwargs):
        raise AssertionError("A locked settings dialog must not open a file chooser")

    monkeypatch.setattr(QFileDialog, "getExistingDirectory", forbidden)
    monkeypatch.setattr(QFileDialog, "getOpenFileName", forbidden)
    dialog._browse("whisper_model_dir")
    dialog._browse("asr_model")
    dialog._request_refresh()
    dialog.accept()  # Enter or a direct call cannot bypass the runtime lock.
    assert dialog.isVisible() and dialog.result_settings is None
    assert widget.settings == before and saved == [] and refreshes == []
    dialog.result_settings = replace(before, asr_model="C:/must-not-apply.onnx")
    widget._apply_runtime_settings(dialog)
    assert widget.settings == before and saved == []
    dialog.buttons.button(QDialogButtonBox.StandardButton.Cancel).click()
    settle(app)
    assert not dialog.isVisible() and widget._voice_resources_active
    assert stops == []


@pytest.mark.parametrize("state", ["idle", "stopped"])
def test_voice_inactive_runtime_settings_remain_editable_without_stop_action(studio, state):
    widget, app = studio(prototype=False)
    widget.set_session_state(state)
    dialog = widget.open_runtime_settings()
    settle(app)
    assert all(control.isEnabled() for control in runtime_edit_controls(dialog))
    assert dialog.runtime_hint.isHidden() and dialog.stop_to_edit_button.isHidden()
    dialog.path_edits["asr_model"].setText("C:/models/after-stop.onnx")
    dialog.accept()
    assert widget.settings.asr_model == "C:/models/after-stop.onnx"


def test_voice_open_runtime_settings_lock_on_start_and_unlock_only_after_stopped(studio):
    widget, app = studio(prototype=False)
    add_caption(widget)
    dialog = widget.open_runtime_settings()
    settle(app)
    stops = []
    widget.stop_requested.connect(lambda: stops.append(True))
    widget.set_session_state("running")
    assert all(not control.isEnabled() for control in runtime_edit_controls(dialog))
    dialog.stop_to_edit_button.click()
    dialog.stop_to_edit_button.click()
    assert stops == [True]
    assert dialog._stop_pending and not dialog.stop_to_edit_button.isEnabled()
    assert "等待声音资源释放" in dialog.runtime_hint.text()
    assert widget._voice_resources_active and not dialog.apply_button.isEnabled()
    widget.set_session_state("stopping")
    widget.set_session_state("error", "仍在等待模型释放。")
    assert dialog._stop_pending and not dialog.apply_button.isEnabled()
    widget.set_session_state("stopped")
    settle(app)
    assert not dialog._stop_pending and dialog.stop_to_edit_button.isHidden()
    assert all(control.isEnabled() for control in runtime_edit_controls(dialog))
    assert len(widget.store.items()) == 1 and len(widget.panels["system"].rows) == 1
    dialog.path_edits["asr_model"].setText("C:/models/unlocked.onnx")
    dialog.accept()
    assert widget.settings.asr_model == "C:/models/unlocked.onnx"


def test_voice_readonly_model_settings_wait_for_controller_native_resource_release(studio):
    from fanyiji_next.controller import AppController

    class DeferredRuntime:
        def __init__(self, callback):
            self.callback = callback
            self.is_running = False
            self.is_busy = True
            self.stop_count = 0

        def start(self, _settings, session_id):
            self.is_running = True
            self.session_id = session_id

        def stop(self):
            self.stop_count += 1
            self.is_running = False
            self.callback(RuntimeEvent("state", self.session_id, status="stopped"))

    widget, app = studio(prototype=False)
    controller = AppController(widget, runtime_factory=DeferredRuntime, persist_settings=False,
                               device_lister=lambda: [])
    add_caption(widget)
    widget.start_button.click()
    runtime = controller._runtime
    dialog = widget.open_runtime_settings()
    settle(app)
    dialog.stop_to_edit_button.click()
    settle(app)
    assert runtime.stop_count == 1 and controller._runtime_stopped
    assert not runtime.is_running and runtime.is_busy
    assert controller.state == "stopping" and widget._voice_resources_active
    assert dialog.isVisible() and dialog._stop_pending
    assert all(not control.isEnabled() for control in runtime_edit_controls(dialog))
    dialog.accept()
    assert dialog.isVisible() and dialog.result_settings is None
    assert len(widget.store.items()) == 1
    runtime.is_busy = False
    controller._release_stopped_runtime()
    settle(app)
    assert controller._runtime is None and controller.state == "stopped"
    assert not widget._voice_resources_active
    assert all(control.isEnabled() for control in runtime_edit_controls(dialog))
    assert dialog.runtime_hint.isHidden() and dialog.stop_to_edit_button.isHidden()
    assert len(widget.store.items()) == 1
    controller.shutdown()


@pytest.mark.parametrize("mode,mdx_visible,gtcrn_visible", [("raw", False, False), ("gtcrn", False, True), ("mdx", True, False),
                                                         ("gtcrn_preserve", False, True),
                                                         ("mdx_gtcrn", True, True)])
def test_voice_system_audio_processing_shows_only_required_paths_and_applies_without_changing_mic(
        studio, mode, mdx_visible, gtcrn_visible):
    settings = AppSettings(mdx_model_file="D:/models/saved-mdx.onnx", gtcrn_model_file="D:/models/saved-gtcrn.onnx",
                           system_audio_processing="gtcrn_preserve" if mode == "gtcrn_preserve" else "raw")
    widget, app = studio(settings, prototype=False)
    accepted, starts = [], []
    widget.settings_changed.connect(accepted.append)
    widget.start_requested.connect(starts.append)
    dialog = widget.open_runtime_settings()
    combo = dialog.engine_combos["system_audio_processing"]
    expected = [
        ("raw", "原声直通"), ("gtcrn", "GTCRN 流式降噪"),
        ("mdx", "MDX人声提取"), ("mdx_gtcrn", "MDX + GTCRN")]
    if mode == "gtcrn_preserve":
        expected.append(("gtcrn_preserve", "GTCRN 并行诊断（原声识别，试验）"))
    assert [(combo.itemData(index), combo.itemText(index)) for index in range(combo.count())] == expected
    combo.setCurrentIndex(combo.findData(mode))
    settle(app)
    assert dialog.path_groups["mdx_model_file"].isHidden() is not mdx_visible
    assert dialog.path_groups["gtcrn_model_file"].isHidden() is not gtcrn_visible
    assert "麦克风保持原设置" in dialog.processing_hint.text()
    if mode in {"mdx", "mdx_gtcrn"}:
        assert "首批需约5.8秒原声，另加推理时间" in dialog.processing_hint.text()
        assert "CPU可能继续积压" in dialog.processing_hint.text()
    elif mode == "gtcrn":
        assert "流式降噪" in dialog.processing_hint.text()
        assert "无需等待MDX整窗" in dialog.processing_hint.text()
        assert "不能保证分离游戏声或背景音乐" in dialog.processing_hint.text()
        assert "5.8秒" not in dialog.processing_hint.text()
    elif mode == "gtcrn_preserve":
        assert "原声" in dialog.processing_hint.text() and "试验模式" in dialog.processing_hint.text()
        assert "背景音乐和游戏声音也会保留" in dialog.processing_hint.text()
        assert "GTCRN仅作并行诊断" in dialog.processing_hint.text()
        assert "5.8秒" not in dialog.processing_hint.text()
    dialog.path_edits["mdx_model_file"].setText(" D:/模型/Kim_Vocal_2.onnx ")
    dialog.path_edits["gtcrn_model_file"].setText(" D:/模型/gtcrn_simple.onnx ")
    dialog.accept()
    assert len(accepted) == 1 and starts == []
    assert widget.settings.system_audio_processing == mode
    assert widget.settings.mdx_model_file == "D:/模型/Kim_Vocal_2.onnx"
    assert widget.settings.gtcrn_model_file == "D:/模型/gtcrn_simple.onnx"
    assert widget.settings.microphone == settings.microphone
    assert widget.settings.system == settings.system
    assert not widget.store.items()


def test_voice_switching_enhancement_back_to_raw_keeps_paths_and_cancel_discards_edits(studio):
    settings = AppSettings(system_audio_processing="mdx_gtcrn", mdx_model_file="D:/mdx.onnx",
                           gtcrn_model_file="D:/gtcrn.onnx")
    widget, app = studio(settings, prototype=False)
    changed = []
    widget.settings_changed.connect(changed.append)
    dialog = widget.open_runtime_settings()
    combo = dialog.engine_combos["system_audio_processing"]
    assert combo.currentData() == "mdx_gtcrn"
    combo.setCurrentIndex(combo.findData("raw"))
    settle(app)
    assert dialog.path_groups["mdx_model_file"].isHidden() and dialog.path_groups["gtcrn_model_file"].isHidden()
    assert dialog.path_edits["mdx_model_file"].text() == settings.mdx_model_file
    assert dialog.path_edits["gtcrn_model_file"].text() == settings.gtcrn_model_file
    dialog.path_edits["mdx_model_file"].setText("D:/must-not-save.onnx")
    dialog.reject()
    assert widget.settings == settings and changed == []
    dialog = widget.open_runtime_settings()
    combo = dialog.engine_combos["system_audio_processing"]
    combo.setCurrentIndex(combo.findData("raw"))
    dialog.accept()
    assert widget.settings.system_audio_processing == "raw"
    assert widget.settings.mdx_model_file == settings.mdx_model_file
    assert widget.settings.gtcrn_model_file == settings.gtcrn_model_file


@pytest.mark.parametrize("state", ["loading", "running", "stopping", "error"])
@pytest.mark.parametrize("mode", ["gtcrn", "gtcrn_preserve", "mdx_gtcrn"])
def test_voice_active_system_enhancement_controls_stay_readonly_until_stopped(studio, monkeypatch, state, mode):
    settings = AppSettings(system_audio_processing=mode, mdx_model_file="D:/mdx.onnx", gtcrn_model_file="D:/gtcrn.onnx")
    widget, app = studio(settings, prototype=False)
    widget.set_session_state("loading")
    widget.set_session_state(state)
    dialog = widget.open_runtime_settings()
    settle(app)
    combo = dialog.engine_combos["system_audio_processing"]
    assert combo.currentData() == mode and not combo.isEnabled()
    for field in ("mdx_model_file", "gtcrn_model_file"):
        assert dialog.path_groups[field].isVisible() is (field == "gtcrn_model_file" or mode == "mdx_gtcrn")
        assert not dialog.path_edits[field].isEnabled()
        assert not dialog.browse_buttons[field].isEnabled()

    def forbidden(*_args, **_kwargs):
        raise AssertionError("Running audio enhancement must not open a file chooser")

    monkeypatch.setattr(QFileDialog, "getOpenFileName", forbidden)
    dialog._browse("mdx_model_file")
    dialog._browse("gtcrn_model_file")
    dialog.accept()
    assert dialog.result_settings is None and widget.settings == settings
    widget.set_session_state("stopped")
    assert combo.isEnabled() and dialog.apply_button.isEnabled()
    assert all(dialog.path_edits[field].isEnabled() and dialog.browse_buttons[field].isEnabled()
               for field in ("mdx_model_file", "gtcrn_model_file"))


@pytest.mark.parametrize("mode,label", [("raw", "原声直通"), ("gtcrn", "GTCRN 流式降噪"), ("mdx", "MDX 人声提取"),
                                        ("gtcrn_preserve", "GTCRN 诊断 · 原声识别"),
                                        ("mdx_gtcrn", "MDX + GTCRN")])
def test_voice_header_shows_selected_processing_and_updates_when_system_is_disabled(studio, mode, label):
    settings = AppSettings(system_audio_processing=mode)
    widget, app = studio(settings, prototype=False)
    engines = "SenseVoice · Hy-MT2 Q8\n"
    assert widget.mode_label.text() == engines + label
    widget.update_settings(replace(settings, system=replace(settings.system, enabled=False)))
    settle(app)
    assert widget.mode_label.text() == engines + "原声直通"
    widget.update_settings(settings)
    assert widget.mode_label.text() == engines + label
    dialog = widget.open_runtime_settings()
    selector = dialog.engine_combos["system_audio_processing"]
    selector.setCurrentIndex(selector.findData("mdx_gtcrn" if mode != "mdx_gtcrn" else "raw"))
    dialog.accept()
    assert widget.mode_label.text() == engines + ("MDX + GTCRN" if mode != "mdx_gtcrn" else "原声直通")


@pytest.mark.parametrize("recognition,translation,label", [
    ("whisper", "hymt", "Whisper large-v3 · Hy-MT2 Q8"),
    ("sensevoice", "m2m100", "SenseVoice · M2M100"),
    ("whisper", "nllb", "Whisper large-v3 · NLLB 1.3B"),
])
def test_voice_header_identifies_selected_models_through_running_and_reading_changes(studio, recognition, translation, label):
    settings = AppSettings(recognition_backend=recognition, translation_backend=translation)
    widget, app = studio(settings, prototype=False)
    widget.resize(1000, 600)
    for state in ("loading", "running", "stopping", "stopped"):
        widget.set_session_state(state)
        widget.focus_button.setChecked(True)
        settle(app)
        assert widget.mode_label.text() == label + "\n原声直通"
        assert_inside(widget, widget.mode_label)
        assert_inside(widget, widget.settings_button)
    widget.update_settings(replace(settings, recognition_backend="sensevoice", translation_backend="m2m100"))
    assert widget.mode_label.text() == "SenseVoice · M2M100\n原声直通"


def test_voice_settings_source_is_absolute_selectable_and_available_while_runtime_is_locked(studio, tmp_path):
    widget, app = studio(prototype=False)
    dialog = widget.open_runtime_settings()
    assert dialog.settings_source_label.isHidden()
    path = tmp_path / "独立配置" / "settings.json"
    expected = str(path.resolve())
    widget.set_settings_source(str(path))
    assert expected in widget.mode_label.toolTip()
    assert expected in widget.model_button.toolTip()
    assert dialog.settings_source_label.text() == f"配置文件：{expected}"
    assert dialog.settings_source_label.textInteractionFlags() & Qt.TextInteractionFlag.TextSelectableByMouse
    widget.set_session_state("running")
    settle(app)
    assert dialog.settings_source_label.isVisible()
    assert not dialog.apply_button.isEnabled()
    dialog.reject()
    reopened = widget.open_runtime_settings()
    settle(app)
    assert reopened.settings_source_label.text() == f"配置文件：{expected}"
    assert reopened.settings_source_label.isVisible()
    widget.set_settings_source("")
    assert reopened.settings_source_label.isHidden()
    assert not widget.mode_label.toolTip() and not widget.model_button.toolTip()


def test_prototype_header_is_unchanged_by_system_processing_choices(studio):
    widget, _app = studio(AppSettings(system_audio_processing="mdx_gtcrn"))
    assert widget.mode_label.text() == "界面体验版 · 预置字幕"
    widget.update_settings(replace(widget.settings, system_audio_processing="mdx"))
    assert widget.mode_label.text() == "界面体验版 · 预置字幕"
