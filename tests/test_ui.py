"""Real Qt layout and interaction checks without microphones or model files."""
import os
from dataclasses import replace
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtCore import QPoint, QRect
from PySide6.QtGui import QFontDatabase
from PySide6.QtWidgets import QApplication

from fanyiji_next.models import AppSettings, RuntimeEvent
from fanyiji_next.ui import ModelSettingsDialog, VoiceWindow


def settle(app):
    for _ in range(8):
        app.processEvents()


def test_error_notice_has_three_second_timer_and_duplicates_do_not_extend_it(window):
    widget, app = window
    widget.show_error("设备错误")
    assert widget.notice.isVisible() and widget._error_timer.interval() == 3000
    remaining = widget._error_timer.remainingTime()
    widget.show_error("设备错误")
    assert widget._error_timer.remainingTime() <= remaining
    widget._error_timer.timeout.emit()
    assert not widget.notice.isVisible()


@pytest.fixture
def window():
    app = QApplication.instance() or QApplication([])
    for name in ("msyh.ttc", "msyhbd.ttc"):
        path = Path("C:/Windows/Fonts") / name
        if path.exists():
            QFontDatabase.addApplicationFont(str(path))
    widget = VoiceWindow()
    widget.show()
    settle(app)
    yield widget, app
    widget.close()
    widget.deleteLater()
    settle(app)


def add_caption(widget, index, text="今天也请多关照。", channel="system"):
    widget.apply_event(RuntimeEvent("original", 1, channel, str(index), text="今日もよろしくお願いします。" * 5))
    widget.apply_event(RuntimeEvent("translation", 1, channel, str(index), text=text))


def test_initial_window_waits_for_manual_start_and_blocks_empty_sources(window):
    widget, _app = window
    starts = []
    widget.start_requested.connect(starts.append)
    assert widget.state_label.text() == "待机"
    assert widget.stop_button.isEnabled() is False
    assert not widget.store.items()
    for channel in ("mic", "system"):
        widget.channel_controls[channel][0].setChecked(False)
    widget.start_button.click()
    assert starts == []
    assert "开启" in widget.notice.text()


def test_translation_updates_existing_row_and_late_results_cannot_create_one(window):
    widget, _app = window
    add_caption(widget, 1)
    widget.apply_event(RuntimeEvent("translation", 1, "system", "1", text="修订后的译文"))
    assert len(widget.panels["system"].rows) == 1
    assert widget.panels["system"].rows["1"].translation_label.text() == "修订后的译文"
    widget.clear_captions()
    widget.apply_event(RuntimeEvent("translation", 1, "system", "1", text="已过期"))
    assert not widget.store.items()
    assert not widget.panels["system"].rows


def test_font_original_visibility_copy_and_plain_text(window):
    widget, app = window
    add_caption(widget, 1, text="<b>这是正文</b>")
    widget.original_checkbox.setChecked(False)
    widget.font_size.setValue(30)
    settle(app)
    row = widget.panels["system"].rows["1"]
    assert not row.original_label.isVisible()
    assert row.translation_label.font().pointSize() == 30
    assert row.translation_label.text() == "<b>这是正文</b>"
    row.copy_caption()
    assert QApplication.clipboard().text().endswith("<b>这是正文</b>")


@pytest.mark.parametrize("show_original", [True, False])
@pytest.mark.parametrize("kind,status", [("error", "recoverable"), ("translation", "failed"), ("translation", "error")])
def test_failed_caption_displays_original_and_keeps_diagnostics_out_of_copy_and_export(window, show_original, kind, status):
    widget, app = window
    original = "この文は原文のまま残します。"
    reason = "翻译模型执行失败。"
    widget.update_settings(replace(widget.settings, show_original=show_original))
    widget.apply_event(RuntimeEvent("original", 1, "system", "failed", text=original))
    widget.apply_event(RuntimeEvent(kind, 1, "system", "failed", text=reason, message=reason, status=status))
    settle(app)

    row = widget.panels["system"].rows["failed"]
    assert row.translation_label.isVisible()
    assert row.translation_label.text() == original
    assert not row.original_label.isVisible()
    assert row.translation_label.toolTip() == reason
    assert row.stage_label.text() == "原文已保留"
    row.copy_caption()
    assert QApplication.clipboard().text() == original
    transcript = widget.transcript()
    assert transcript.count(original) == 1
    assert reason not in transcript
    assert "正在翻译" not in transcript


def test_failed_caption_recovery_restores_translation_and_clears_diagnostic_tooltip(window):
    widget, app = window
    widget.apply_event(RuntimeEvent("original", 1, "system", "recover", text="原文です。"))
    widget.apply_event(RuntimeEvent("error", 1, "system", "recover", message="模型执行失败。"))
    widget.apply_event(RuntimeEvent("translation", 1, "system", "recover", text="这是原文。"))
    settle(app)
    row = widget.panels["system"].rows["recover"]
    assert row.translation_label.text() == "这是原文。"
    assert row.translation_label.toolTip() == ""
    assert row.original_label.isVisible()


def test_history_review_is_preserved_until_follow_button_clicked(window):
    widget, app = window
    widget.update_settings(replace(widget.settings, microphone=replace(widget.settings.microphone, enabled=False)))
    for index in range(12):
        add_caption(widget, index, text="长字幕可以回看，新内容不会打断阅读。" * 6)
    settle(app)
    panel = widget.panels["system"]
    bar = panel.scroll.verticalScrollBar()
    assert bar.maximum() > 0 and bar.value() == bar.maximum()
    bar.setValue(0)
    add_caption(widget, 13, text="刚到达的新字幕" * 30)
    settle(app)
    assert bar.value() == 0 and not panel.follow_button.isChecked()
    panel.follow_button.click()
    settle(app)
    assert bar.value() == bar.maximum() and panel.follow_button.isChecked()


@pytest.mark.parametrize("size", [(1060, 700), (1380, 840)])
def test_dual_panels_and_controls_stay_inside_the_window(window, size):
    widget, app = window
    widget.resize(*size)
    for channel in ("mic", "system"):
        add_caption(widget, 1, text="这是一条较长的真实布局测试字幕。" * 12, channel=channel)
    widget.show_error("具体错误可以选择并复制，完整内容也可以悬停查看。" * 5)
    settle(app)
    assert widget.size().toTuple() == size
    for panel in widget.panels.values():
        bounds = QRect(panel.mapTo(widget, QPoint()), panel.size())
        assert widget.rect().contains(bounds)
        assert panel.rect().contains(panel.scroll.geometry())
        assert panel.scroll.height() >= 150
    for control in (widget.start_button, widget.stop_button, widget.layout_button, widget.export_button):
        assert widget.rect().contains(QRect(control.mapTo(widget, QPoint()), control.size()))


def test_stopping_disables_source_changes_but_keeps_appearance_available(window):
    widget, _app = window
    widget.set_session_state("stopping")
    assert not widget.start_button.isEnabled()
    assert not widget.stop_button.isEnabled()
    assert not widget.model_button.isEnabled()
    assert all(not enabled.isEnabled() for enabled, _, _ in widget.channel_controls.values())
    assert widget.font_size.isEnabled()
    widget.set_session_state("stopped")
    assert widget.start_button.isEnabled()


def test_model_dialog_cancel_has_no_result_and_accept_keeps_other_settings(window):
    widget, _app = window
    dialog = ModelSettingsDialog(widget.settings)
    dialog.reject()
    assert dialog.result_settings is None
    dialog.path_edits["asr_model"].setText("C:/models/sensevoice.onnx")
    dialog.accept()
    assert dialog.result_settings.asr_model == "C:/models/sensevoice.onnx"
    assert dialog.result_settings.font_size == widget.settings.font_size
    dialog.deleteLater()


def test_demo_remains_visible_with_both_real_sources_disabled_without_changing_settings(window):
    widget, app = window
    for channel in ("mic", "system"):
        widget.channel_controls[channel][0].setChecked(False)
    assert not widget.splitter.isVisible()
    widget.set_session_state("demo", "模拟字幕，不采集声音。")
    widget.font_size.setValue(24)
    settle(app)
    assert widget.splitter.isVisible() and all(panel.isVisible() for panel in widget.panels.values())
    assert not widget.idle_label.isVisible()
    assert not widget.settings.microphone.enabled and not widget.settings.system.enabled
    widget.set_session_state("stopped")
    settle(app)
    assert not widget.splitter.isVisible() and widget.idle_label.isVisible()


def test_clear_hides_old_rows_before_deferred_widget_destruction(window):
    widget, app = window
    add_caption(widget, 1)
    settle(app)
    row = widget.panels["system"].rows["1"]
    assert row.isVisible()
    widget.clear_captions()
    assert not row.isVisible()
