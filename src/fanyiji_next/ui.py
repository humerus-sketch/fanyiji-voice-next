"""Independent voice workspace. No recording or model loading occurs in this module."""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QCloseEvent, QFont
from PySide6.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QDialog, QDialogButtonBox, QFileDialog,
    QFormLayout, QFrame, QHBoxLayout, QLabel, QLineEdit, QMainWindow, QProgressBar,
    QPushButton, QScrollArea, QSpinBox, QSplitter, QVBoxLayout, QWidget,
)

from .models import AppSettings, Caption, CaptionStore, RuntimeEvent
from .style import STYLE

APP_TITLE = "翻译姬 · 语音版 NEXT"
CHANNEL_NAMES = {"mic": "麦克风", "system": "电脑声音"}
DIRECTIONS = (("zh_to_ja", "中文 → 日文"), ("ja_to_zh", "日文 → 中文"))
LOCAL_TIMEZONE = timezone(timedelta(hours=8))


def label(text="", *, name="", wrap=False):
    widget = QLabel(text)
    widget.setTextFormat(Qt.TextFormat.PlainText)
    widget.setWordWrap(wrap)
    if name:
        widget.setObjectName(name)
    return widget


def button(text, *, name="", callback=None):
    widget = QPushButton(text)
    if name:
        widget.setObjectName(name)
    if callback is not None:
        widget.clicked.connect(callback)
    return widget


class ModelSettingsDialog(QDialog):
    """Edit explicit model and audio paths without reading legacy settings."""

    def __init__(self, settings: AppSettings, parent=None):
        super().__init__(parent)
        self.settings = settings
        self.result_settings = None
        self.setWindowTitle("模型与音频设备")
        self.resize(720, 480)
        self.setMinimumWidth(620)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 22, 24, 22)
        layout.setSpacing(16)
        layout.addWidget(label("为真实翻译准备本地模型", name="sectionTitle"))
        layout.addWidget(label("识别使用 SenseVoice ONNX；翻译使用 M2M100 CTranslate2。模型单独选择，不随程序分发。",
                               name="muted", wrap=True))
        form = QFormLayout()
        form.setSpacing(12)
        self.path_edits = {}
        for field, title, directory in (
            ("asr_model", "识别模型 .onnx", False),
            ("asr_tokens", "识别词表 tokens.txt", False),
            ("translation_model_dir", "翻译模型目录", True),
            ("translation_spm", "SentencePiece 模型", False),
        ):
            row = QWidget()
            row_layout = QHBoxLayout(row)
            row_layout.setContentsMargins(0, 0, 0, 0)
            edit = QLineEdit(getattr(settings, field))
            edit.setPlaceholderText("选择本机文件或目录")
            self.path_edits[field] = edit
            row_layout.addWidget(edit, 1)
            row_layout.addWidget(button("浏览", callback=lambda _checked=False, e=edit, d=directory: self._browse(e, d)))
            form.addRow(title, row)
        self.device_combos = {}
        for channel, source in (("mic", settings.microphone), ("system", settings.system)):
            combo = QComboBox()
            combo.addItem("系统默认设备", None)
            if source.device_index is not None:
                combo.addItem(f"已保存设备 #{source.device_index}（刷新可确认名称）", source.device_index)
                combo.setCurrentIndex(1)
            self.device_combos[channel] = combo
            form.addRow(CHANNEL_NAMES[channel], combo)
        layout.addLayout(form)
        self.device_status = label("刷新只读取设备列表，不开启录音。", name="muted", wrap=True)
        layout.addWidget(self.device_status)
        layout.addWidget(button("刷新音频设备", callback=self._refresh_devices))
        layout.addStretch()
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel)
        buttons.button(QDialogButtonBox.StandardButton.Save).setText("保存设置")
        buttons.button(QDialogButtonBox.StandardButton.Cancel).setText("取消")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _browse(self, edit, directory):
        if directory:
            chosen = QFileDialog.getExistingDirectory(self, "选择翻译模型目录", edit.text())
        else:
            chosen, _ = QFileDialog.getOpenFileName(self, "选择本地模型文件", edit.text(), "所有文件 (*)")
        if chosen:
            edit.setText(chosen)

    def _refresh_devices(self):
        from .audio_runtime import list_audio_devices
        try:
            devices = list_audio_devices()
            for channel, combo in self.device_combos.items():
                previous = combo.currentData()
                combo.clear()
                combo.addItem("系统默认设备", None)
                for device in devices:
                    if device.channel == channel:
                        combo.addItem(device.name, device.index)
                found = combo.findData(previous)
                if previous is not None and found < 0:
                    combo.addItem(f"已保存设备 #{previous}（当前未找到）", previous)
                    found = combo.count() - 1
                combo.setCurrentIndex(max(0, found))
            self.device_status.setText(f"找到 {len(devices)} 个可用输入或电脑声音设备。")
        except Exception as exc:
            self.device_status.setText(f"读取设备列表失败：{exc}")

    def accept(self):
        values = {name: edit.text().strip() for name, edit in self.path_edits.items()}
        self.result_settings = replace(
            self.settings, **values,
            microphone=replace(self.settings.microphone, device_index=self.device_combos["mic"].currentData()),
            system=replace(self.settings.system, device_index=self.device_combos["system"].currentData()),
        )
        super().accept()


class CaptionRow(QFrame):
    def __init__(self, caption: Caption, settings: AppSettings, parent=None):
        super().__init__(parent)
        self.setObjectName("captionRow")
        self.caption = caption
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 12, 16, 14)
        layout.setSpacing(9)
        header = QHBoxLayout()
        timestamp = datetime.fromtimestamp(caption.created_at, LOCAL_TIMEZONE).strftime("%H:%M:%S")
        header.addWidget(label(timestamp, name="muted"))
        self.stage_label = label("", name="muted")
        header.addWidget(self.stage_label)
        header.addStretch()
        header.addWidget(button("复制", name="quiet", callback=self.copy_caption))
        layout.addLayout(header)
        self.original_label = label(name="original", wrap=True)
        self.translation_label = label(wrap=True)
        for text_label in (self.original_label, self.translation_label):
            text_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            text_label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
            layout.addWidget(text_label)
        self.update_caption(caption, settings)

    def update_caption(self, caption, settings):
        self.caption = caption
        self.original_label.setText(caption.original)
        self.original_label.setVisible(settings.show_original)
        self.original_label.setFont(QFont("Microsoft YaHei UI", max(10, round(settings.font_size * .58))))
        self.original_label.setStyleSheet(f"font-size: {max(10, round(settings.font_size * .58))}pt; color: #a6b8ca;")
        self.translation_label.setFont(QFont("Microsoft YaHei UI", settings.font_size))
        self.translation_label.setText(caption.translated or "正在翻译…")
        self.stage_label.setText({"translating": "翻译中", "complete": "译文", "failed": "翻译未完成"}.get(caption.status, caption.status))
        color = "#f1c58d" if caption.status == "failed" else "#f0f6fc"
        self.translation_label.setStyleSheet(f"font-size: {settings.font_size}pt; color: {color};")

    def copy_caption(self):
        QApplication.clipboard().setText("\n".join(text for text in (self.caption.original, self.caption.translated) if text))


class CaptionPanel(QFrame):
    def __init__(self, channel, parent=None, *, prototype=False):
        super().__init__(parent)
        self.channel = channel
        self.rows: OrderedDict[str, CaptionRow] = OrderedDict()
        self._following_scroll = False
        self.setObjectName("captionPanel")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 12, 14)
        layout.setSpacing(14)
        header = QHBoxLayout()
        header.addWidget(label(CHANNEL_NAMES[channel], name="sectionTitle"))
        self.direction_label = label(name="muted")
        header.addWidget(self.direction_label)
        header.addStretch()
        self.follow_button = button("跟随最新", name="quiet")
        self.follow_button.setCheckable(True)
        self.follow_button.setChecked(True)
        self.follow_button.toggled.connect(self._follow_changed)
        self.follow_button.setToolTip("向上滚动后暂停自动跟随，点击返回最新字幕。")
        header.addWidget(self.follow_button)
        layout.addLayout(header)
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.content = QWidget()
        self.rows_layout = QVBoxLayout(self.content)
        self.rows_layout.setContentsMargins(0, 0, 7, 0)
        self.rows_layout.setSpacing(12)
        self.placeholder = QFrame()
        self.placeholder.setObjectName("emptyState")
        placeholder_layout = QVBoxLayout(self.placeholder)
        placeholder_layout.setContentsMargins(18, 48, 18, 48)
        placeholder_layout.addWidget(label("字幕将显示在这里", name="sectionTitle"))
        placeholder_layout.addWidget(label("点击“播放字幕演示”查看双语字幕。\n可选择文字，也可逐句复制。" if prototype else
                                           "开始翻译后显示原文与译文。\n可选择文字，也可逐句复制。", name="muted", wrap=True))
        self.rows_layout.addWidget(self.placeholder)
        self.rows_layout.addStretch()
        self.scroll.setWidget(self.content)
        bar = self.scroll.verticalScrollBar()
        bar.rangeChanged.connect(lambda _minimum, _maximum: self._scroll_latest())
        bar.valueChanged.connect(self._scroll_changed)
        layout.addWidget(self.scroll, 1)

    def _scroll_latest(self):
        if self.follow_button.isChecked():
            self._following_scroll = True
            bar = self.scroll.verticalScrollBar()
            try:
                bar.setValue(bar.maximum())
            finally:
                self._following_scroll = False

    def _scroll_changed(self, value):
        if not self._following_scroll and value < self.scroll.verticalScrollBar().maximum() - 2:
            self.follow_button.setChecked(False)

    def _follow_changed(self, checked):
        self.follow_button.setText("跟随最新" if checked else "返回最新")
        self._scroll_latest()

    def render_caption(self, caption, settings):
        row = self.rows.get(caption.id)
        if row is None:
            row = CaptionRow(caption, settings)
            self.rows[caption.id] = row
            self.rows_layout.insertWidget(self.rows_layout.count() - 1, row)
        else:
            row.update_caption(caption, settings)
        self.placeholder.hide()
        self._scroll_latest()

    def prune(self, retained_ids):
        for key in list(self.rows):
            if key not in retained_ids:
                row = self.rows.pop(key)
                self.rows_layout.removeWidget(row)
                row.hide()
                row.deleteLater()

    def clear(self):
        self.prune(set())
        self.placeholder.show()
        self.follow_button.setChecked(True)


class VoiceWindow(QMainWindow):
    start_requested = Signal(object)
    stop_requested = Signal()
    demo_requested = Signal()
    replay_requested = Signal()
    settings_changed = Signal(object)
    closing = Signal()

    def __init__(self, settings: AppSettings | None = None, *, prototype=False):
        super().__init__()
        self.prototype = prototype
        self.settings = settings or AppSettings()
        self.store = CaptionStore()
        self._applying_settings = False
        self._active = False
        self._session_state = "idle"
        self.setWindowTitle(APP_TITLE + (" · 界面原型" if prototype else ""))
        self.setMinimumSize(1060, 700)
        self.resize(1380, 840)
        self.setStyleSheet(STYLE)
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        sidebar = QFrame()
        sidebar.setObjectName("sidebar")
        sidebar.setFixedWidth(250)
        side = QVBoxLayout(sidebar)
        side.setContentsMargins(20, 28, 20, 22)
        side.setSpacing(18)
        side.addWidget(label("翻译姬", name="brand"))
        side.addWidget(label("VOICE NEXT  /  界面原型" if prototype else "VOICE NEXT  /  独立语音版", name="brandTag"))
        side.addSpacing(10)
        side.addWidget(label("字幕通道" if prototype else "声音来源", name="sectionTitle"))
        self.channel_controls = {}
        for channel in ("mic", "system"):
            card = QFrame()
            card.setObjectName("sourceCard")
            card_layout = QVBoxLayout(card)
            card_layout.setContentsMargins(13, 14, 13, 14)
            card_layout.setSpacing(12)
            enabled = QCheckBox(CHANNEL_NAMES[channel])
            direction = QComboBox()
            for value, title in DIRECTIONS:
                direction.addItem(title, value)
            level = QProgressBar()
            level.setRange(0, 100)
            level.setValue(0)
            level.setTextVisible(False)
            level.setFixedHeight(5)
            level.setVisible(not prototype)
            card_layout.addWidget(enabled)
            card_layout.addWidget(direction)
            card_layout.addWidget(level)
            enabled.toggled.connect(self._source_changed)
            direction.currentIndexChanged.connect(self._source_changed)
            self.channel_controls[channel] = (enabled, direction, level)
            side.addWidget(card)
        self.model_button = button("模型与音频设备", callback=self._edit_models)
        side.addWidget(self.model_button)
        self.model_button.setVisible(not prototype)
        self.model_status = label("", name="muted", wrap=True)
        side.addWidget(self.model_status)
        side.addStretch()
        self.demo_button = button("重新播放样例" if prototype else "体验模拟字幕",
                                  callback=self.replay_requested.emit if prototype else self.demo_requested.emit)
        side.addWidget(self.demo_button)
        side.addWidget(label("模拟字幕不采集声音，\n不代表真实识别或翻译结果。", name="muted", wrap=True))
        root.addWidget(sidebar)

        body = QWidget()
        layout = QVBoxLayout(body)
        layout.setContentsMargins(26, 26, 26, 20)
        layout.setSpacing(18)
        header = QHBoxLayout()
        heading = QVBoxLayout()
        heading.setSpacing(6)
        heading.addWidget(label("实时双语字幕", name="pageTitle"))
        heading.addWidget(label("全新界面与字幕交互 · 使用预置样例" if prototype else "本地语音识别 · 中日互译 · 独立工作区", name="muted"))
        header.addLayout(heading)
        header.addStretch()
        self.state_label = label("待机", name="statusPill")
        header.addWidget(self.state_label)
        self.start_button = button("播放字幕演示" if prototype else "开始翻译", name="primary", callback=self._start)
        header.addWidget(self.start_button)
        self.stop_button = button("清空字幕" if prototype else "停止", name="stop", callback=self.stop_requested.emit)
        header.addWidget(self.stop_button)
        layout.addLayout(header)
        toolbar = QHBoxLayout()
        self.original_checkbox = QCheckBox("显示原文")
        self.original_checkbox.toggled.connect(self._appearance_changed)
        toolbar.addWidget(self.original_checkbox)
        toolbar.addSpacing(12)
        toolbar.addWidget(label("译文字号", name="muted"))
        self.font_size = QSpinBox()
        self.font_size.setRange(16, 40)
        self.font_size.valueChanged.connect(self._appearance_changed)
        toolbar.addWidget(self.font_size)
        self.topmost_checkbox = QCheckBox("窗口置顶")
        self.topmost_checkbox.toggled.connect(self._appearance_changed)
        toolbar.addWidget(self.topmost_checkbox)
        toolbar.addStretch()
        self.layout_button = button("上下排列", name="quiet", callback=self._toggle_layout)
        toolbar.addWidget(self.layout_button)
        self.export_button = button("导出文本", name="quiet", callback=self._export)
        toolbar.addWidget(self.export_button)
        layout.addLayout(toolbar)
        self.notice = label(name="notice", wrap=True)
        self.notice.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.notice.setMaximumHeight(78)
        self.notice.hide()
        layout.addWidget(self.notice)
        self.splitter = QSplitter(Qt.Orientation.Horizontal)
        self.splitter.setHandleWidth(14)
        self.panels = {channel: CaptionPanel(channel, prototype=prototype) for channel in ("mic", "system")}
        for panel in self.panels.values():
            panel.setMinimumWidth(320)
            self.splitter.addWidget(panel)
        self.splitter.setChildrenCollapsible(False)
        self.idle_label = label("显示至少一路字幕通道，即可预览样例。" if prototype else "开启至少一路声音来源，即可开始翻译。", name="muted", wrap=True)
        self.idle_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.idle_label)
        layout.addWidget(self.splitter, 1)
        footer = QHBoxLayout()
        self.summary = label("本次会话 · 0 条字幕", name="muted")
        footer.addWidget(self.summary)
        footer.addStretch()
        self.last_latency = label("", name="muted")
        footer.addWidget(self.last_latency)
        layout.addLayout(footer)
        root.addWidget(body, 1)
        self.update_settings(self.settings)
        self.set_session_state("idle")

    def update_settings(self, settings):
        self.settings = settings
        self._applying_settings = True
        try:
            for channel, source in (("mic", settings.microphone), ("system", settings.system)):
                enabled, direction, _level = self.channel_controls[channel]
                enabled.setChecked(source.enabled)
                direction.setCurrentIndex(direction.findData(source.direction))
                self.panels[channel].direction_label.setText(dict(DIRECTIONS)[source.direction])
                self.panels[channel].setVisible(source.enabled or (self._session_state == "demo" and not self.prototype))
            any_enabled = settings.microphone.enabled or settings.system.enabled or (self._session_state == "demo" and not self.prototype)
            self.idle_label.setVisible(not any_enabled)
            self.splitter.setVisible(any_enabled)
            self.original_checkbox.setChecked(settings.show_original)
            self.font_size.setValue(settings.font_size)
            self.topmost_checkbox.setChecked(settings.always_on_top)
            visible = self.isVisible()
            self.setWindowFlag(Qt.WindowType.WindowStaysOnTopHint, settings.always_on_top)
            if visible:
                self.show()
            self.setWindowOpacity(settings.window_opacity / 100)
            ready = all((settings.asr_model, settings.asr_tokens, settings.translation_model_dir, settings.translation_spm))
            self.model_status.setText("界面与交互验收版\n语音翻译待验收后接入。" if self.prototype else
                                     "模型路径已配置\n开始时校验并加载模型。" if ready else "真实翻译需要先选择模型。\n可先体验模拟字幕。")
            for caption in self.store.items():
                row = self.panels[caption.channel].rows.get(caption.id)
                if row:
                    row.update_caption(caption, settings)
        finally:
            self._applying_settings = False

    def _source_changed(self, *_args):
        if self._applying_settings or (self._active and not self.prototype):
            return
        values = {}
        for channel, field in (("mic", "microphone"), ("system", "system")):
            enabled, direction, _level = self.channel_controls[channel]
            values[field] = replace(getattr(self.settings, field), enabled=enabled.isChecked(), direction=direction.currentData())
        self.update_settings(replace(self.settings, **values))
        self.settings_changed.emit(self.settings)

    def _appearance_changed(self, *_args):
        if self._applying_settings:
            return
        self.update_settings(replace(self.settings, font_size=self.font_size.value(), show_original=self.original_checkbox.isChecked(),
                                     always_on_top=self.topmost_checkbox.isChecked()))
        self.settings_changed.emit(self.settings)

    def _edit_models(self):
        if self._active or self.prototype:
            return
        dialog = ModelSettingsDialog(self.settings, self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            self.update_settings(dialog.result_settings)
            self.settings_changed.emit(self.settings)

    def _start(self):
        if self.prototype:
            self.demo_requested.emit()
            return
        if not self.settings.microphone.enabled and not self.settings.system.enabled:
            self.show_error("请先开启麦克风或电脑声音。")
            return
        self.notice.hide()
        self.start_requested.emit(self.settings)

    def set_session_state(self, state, message=""):
        self._session_state = state
        self._active = state == "demo" if self.prototype else state not in ("idle", "stopped")
        for channel, source in (("mic", self.settings.microphone), ("system", self.settings.system)):
            self.panels[channel].setVisible(source.enabled or (state == "demo" and not self.prototype))
        any_enabled = self.settings.microphone.enabled or self.settings.system.enabled or (state == "demo" and not self.prototype)
        self.splitter.setVisible(any_enabled)
        self.idle_label.setVisible(not any_enabled)
        descriptions = {"idle": "待机", "loading": "准备模型", "running": "正在翻译", "stopping": "正在停止",
                        "stopped": "已停止", "error": "遇到问题", "demo": "模拟演示"}
        if self.prototype:
            descriptions.update(idle="待演示", demo="样例播放中", paused="演示已暂停", finished="演示已结束")
        self.state_label.setText(descriptions.get(state, state))
        self.state_label.setToolTip(message or descriptions.get(state, state))
        self.state_label.setProperty("active", state in ("running", "demo"))
        self.state_label.style().unpolish(self.state_label)
        self.state_label.style().polish(self.state_label)
        self.start_button.setEnabled(not self._active)
        self.stop_button.setEnabled(self._active and state != "stopping")
        self.demo_button.setEnabled(not self._active)
        self.model_button.setEnabled(not self._active)
        if self.prototype:
            self.start_button.setText("暂停演示" if state == "demo" else "继续演示" if state == "paused" else "播放字幕演示")
            self.start_button.setEnabled(True)
            self.stop_button.setEnabled(bool(self.store.items()) or state in ("demo", "paused"))
            self.demo_button.setEnabled(True)
        if message and (self.prototype or state == "demo"):
            self.show_error(message)
        for enabled, direction, level in self.channel_controls.values():
            enabled.setEnabled(self.prototype or not self._active)
            direction.setEnabled(self.prototype or not self._active)
            if not self._active:
                level.setValue(0)

    def show_error(self, message):
        self.notice.setText(message)
        self.notice.setToolTip(message)
        self.notice.show()

    def apply_event(self, event: RuntimeEvent):
        if event.channel not in self.panels:
            return
        if event.kind == "level":
            self.channel_controls[event.channel][2].setValue(round(max(0.0, min(1.0, event.level)) * 100))
            return
        if event.kind == "original":
            caption = self.store.add_original(event.channel, event.segment_id, event.text)
        elif event.kind == "translation":
            caption = self.store.set_translation(event.channel, event.segment_id, event.text, failed=event.status in ("error", "failed"))
        elif event.kind == "error" and event.segment_id:
            caption = self.store.set_translation(event.channel, event.segment_id, event.message or "本句处理未完成", failed=True)
        else:
            return
        if caption is not None:
            panel = self.panels[event.channel]
            panel.render_caption(caption, self.settings)
            panel.prune({item.id for item in self.store.items(event.channel)})
            self.summary.setText(f"本次会话 · {len(self.store.items())} 条字幕")
            if event.elapsed_ms:
                self.last_latency.setText(f"最近处理耗时 {event.elapsed_ms / 1000:.1f} 秒")

    def clear_captions(self):
        self.store.clear()
        for panel in self.panels.values():
            panel.clear()
        self.summary.setText("本次会话 · 0 条字幕")
        self.last_latency.clear()
        self.notice.hide()

    def _toggle_layout(self):
        vertical = self.splitter.orientation() == Qt.Orientation.Horizontal
        self.splitter.setOrientation(Qt.Orientation.Vertical if vertical else Qt.Orientation.Horizontal)
        self.layout_button.setText("并排排列" if vertical else "上下排列")

    def transcript(self):
        parts = []
        for caption in self.store.items():
            timestamp = datetime.fromtimestamp(caption.created_at, LOCAL_TIMEZONE).strftime("%H:%M:%S")
            parts.append(f"[{timestamp}] {CHANNEL_NAMES[caption.channel]}\n{caption.original}\n{caption.translated}")
        return "\n\n".join(parts)

    def _export(self):
        if not self.store.items():
            self.show_error("本次会话还没有可导出的字幕。")
            return
        filename, _ = QFileDialog.getSaveFileName(self, "导出本次字幕", "翻译姬字幕.txt", "文本文件 (*.txt)")
        if filename:
            try:
                Path(filename).write_text(self.transcript(), encoding="utf-8-sig")
            except OSError as exc:
                self.show_error(f"导出失败：{exc}")

    def closeEvent(self, event: QCloseEvent):
        self.closing.emit()
        super().closeEvent(event)
