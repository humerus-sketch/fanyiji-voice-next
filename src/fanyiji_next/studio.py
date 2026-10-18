"""Quiet reading surfaces for preset captions and manually started local voice."""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path

from PySide6.QtCore import QEvent, QEasingCurve, QObject, QPropertyAnimation, QSize, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QFont, QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QAbstractButton, QApplication, QCheckBox, QComboBox, QFileDialog, QFrame, QGraphicsDropShadowEffect,
    QDialog, QDialogButtonBox, QGraphicsOpacityEffect, QHBoxLayout, QLabel, QLineEdit, QMainWindow, QProgressBar,
    QPushButton, QScrollArea, QSpinBox, QVBoxLayout, QWidget,
)

from .models import AppSettings, CaptionStore, RuntimeEvent
from . import __version__
from .studio_theme import ACCENT, INK, SECONDARY, STUDIO_STYLE, icon

CHANNELS = {"mic": "麦克风", "system": "电脑声音"}
DIRECTIONS = (("zh_to_ja", "中文 → 日文"), ("ja_to_zh", "日文 → 中文"))
ZONE = timezone(timedelta(hours=8))


def text_label(text="", name="muted", *, wrap=False):
    label = QLabel(text)
    label.setObjectName(name)
    label.setTextFormat(Qt.TextFormat.PlainText)
    label.setWordWrap(wrap)
    return label


def action(text="", glyph=None, *, name="quietButton", checkable=False):
    result = QPushButton(text)
    result.setObjectName(name)
    result.setCheckable(checkable)
    result.setCursor(Qt.CursorShape.PointingHandCursor)
    if glyph:
        result.setIcon(icon(glyph))
        result.setIconSize(QSize(18, 18))
    return result


def font(label, size, color=INK, weight=QFont.Weight.Normal):
    label.setFont(QFont("Microsoft YaHei UI", size, weight))
    label.setStyleSheet(f"font-size: {size}pt; color: {color}; border: none; background: transparent;")


class RuntimeSettingsDialog(QDialog):
    """Stage local paths and device choices; applying never loads a model or records."""

    refresh_requested = Signal()
    stop_requested = Signal()

    def __init__(self, settings, devices=(), parent=None):
        super().__init__(parent)
        self._settings = settings
        self._runtime_active = False
        self._runtime_state = "idle"
        self._stop_pending = False
        self.result_settings = None
        self.setWindowTitle("声音与离线模型")
        self.setObjectName("runtimeSettings")
        self.setStyleSheet(STUDIO_STYLE)
        self.resize(720, 660)
        self.setMinimumSize(560, 420)
        self.setWindowModality(Qt.WindowModality.WindowModal)
        shell = QVBoxLayout(self)
        shell.setContentsMargins(20, 18, 20, 18)
        self.runtime_hint = text_label("", "notice", wrap=True)
        self.runtime_hint.hide()
        shell.addWidget(self.runtime_hint)
        self.settings_source_label = text_label("", "footerHint", wrap=True)
        self.settings_source_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.settings_source_label.hide()
        shell.addWidget(self.settings_source_label)
        scroll = QScrollArea()
        self.scroll = scroll
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(4, 4, 14, 8)
        layout.setSpacing(13)
        layout.addWidget(text_label("先选声音，再连接本地模型", "brandName"))
        layout.addWidget(text_label("只读取你选择的本地文件；此处不会下载模型或开启录音。", wrap=True))
        refresh_row = QHBoxLayout()
        refresh_row.addWidget(text_label("声音来源", "smallCaps"))
        refresh_row.addStretch()
        self.refresh_button = action("刷新设备列表", "replay")
        self.refresh_button.clicked.connect(self._request_refresh)
        refresh_row.addWidget(self.refresh_button)
        layout.addLayout(refresh_row)
        self.device_hint = text_label("设备尚未刷新；可使用系统默认设备。", "footerHint", wrap=True)
        layout.addWidget(self.device_hint)
        self.channel_controls = {}
        self.device_combos = {}
        self.engine_combos = {}
        for channel, field_name in (("mic", "microphone"), ("system", "system")):
            source = getattr(settings, field_name)
            heading = QHBoxLayout()
            enabled = QCheckBox(CHANNELS[channel])
            enabled.setChecked(source.enabled)
            direction = QComboBox()
            for value, title in DIRECTIONS:
                direction.addItem(title, value)
            direction.setCurrentIndex(direction.findData(source.direction))
            heading.addWidget(enabled)
            heading.addStretch()
            heading.addWidget(direction)
            layout.addLayout(heading)
            device = QComboBox()
            device.setMinimumContentsLength(12)
            device.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
            device.addItem("系统默认设备", None)
            if source.device_index is not None:
                device.addItem(f"设备 {source.device_index}（尚未刷新）", source.device_index)
                device.setCurrentIndex(1)
            self.channel_controls[channel] = (enabled, direction)
            self.device_combos[channel] = device
            layout.addWidget(device)
            if channel == "system":
                processing = QHBoxLayout()
                processing.addWidget(text_label("电脑声音增强"))
                combo = QComboBox()
                for value, title in (("raw", "原声直通"), ("gtcrn", "GTCRN 流式降噪"),
                                     ("mdx", "MDX人声提取"), ("mdx_gtcrn", "MDX + GTCRN")):
                    combo.addItem(title, value)
                # Unaccepted DSP experiments are available only to explicitly
                # selected diagnostic profiles, never normal mode selection.
                if settings.system_audio_processing == "gtcrn_preserve":
                    combo.addItem("GTCRN 并行诊断（原声识别，试验）", "gtcrn_preserve")
                combo.setCurrentIndex(combo.findData(settings.system_audio_processing))
                self.engine_combos["system_audio_processing"] = combo
                processing.addWidget(combo, 1)
                layout.addLayout(processing)
                self.processing_hint = text_label("", "footerHint", wrap=True)
                layout.addWidget(self.processing_hint)
        layout.addSpacing(10)
        layout.addWidget(text_label("识别与翻译引擎", "smallCaps"))
        self.engine_rows = {}
        choices = (
            ("recognition_backend", "语音识别", (("sensevoice", "SenseVoice · CPU"), ("whisper", "Whisper large-v3"))),
            ("recognition_device", "Whisper 运算设备", (("cpu", "CPU"), ("cuda", "NVIDIA GPU"))),
            ("recognition_compute_type", "Whisper 运算精度", (("int8", "INT8"), ("int8_float16", "INT8 + FP16"), ("float16", "FP16"))),
            ("translation_backend", "本地翻译", (("hymt", "Hy-MT2-1.8B Q8_0"), ("m2m100", "M2M100 · 现有引擎"), ("nllb", "NLLB-200 distilled 1.3B"))),
            ("translation_device", "翻译运算设备", (("cpu", "CPU"), ("cuda", "NVIDIA GPU"))),
            ("translation_compute_type", "翻译运算精度", (("int8", "INT8"), ("int8_float16", "INT8 + FP16"), ("float16", "FP16"))),
        )
        for field_name, title, options in choices:
            container = QWidget()
            row = QHBoxLayout(container)
            row.setContentsMargins(0, 0, 0, 0)
            row.addWidget(text_label(title))
            combo = QComboBox()
            for value, label in options:
                combo.addItem(label, value)
            combo.setCurrentIndex(combo.findData(getattr(settings, field_name)))
            row.addWidget(combo, 1)
            self.engine_combos[field_name] = combo
            self.engine_rows[field_name] = container
            layout.addWidget(container)
        layout.addWidget(text_label("离线模型文件", "smallCaps"))
        self.path_edits = {}
        self.path_groups = {}
        self.browse_buttons = {}
        fields = (
            ("asr_model", "SenseVoice ONNX 模型", "选择 .onnx 文件"),
            ("asr_tokens", "SenseVoice tokens.txt", "选择 tokens.txt 文件"),
            ("whisper_model_dir", "Whisper large-v3 CTranslate2 目录", "包含 model.bin、tokenizer.json 与配套配置"),
            ("translation_model_dir", "M2M100 CTranslate2 模型目录", "包含 model.bin 与词表的目录"),
            ("translation_spm", "M2M100 SentencePiece 模型", "选择与翻译模型匹配的 .model 文件"),
            ("nllb_model_dir", "NLLB CTranslate2 模型目录", "NLLB-200-distilled-1.3B 转换后的目录"),
            ("nllb_tokenizer_dir", "NLLB tokenizer 目录", "包含配套 tokenizer.json 的目录"),
            ("hymt_model_file", "Hy-MT2-1.8B Q8_0 模型文件", "选择 Hy-MT2-1.8B-Q8_0.gguf"),
            ("hymt_server_exe", "Hy-MT2 本地运行程序", "选择 CUDA 版 llama-server.exe"),
            ("vad_model", "语音 Silero VAD", "选择 silero_vad.onnx；留空时按模型根目录查找"),
            ("mdx_model_file", "MDX 人声提取模型（电脑声音）", "选择 Kim_Vocal_2.onnx"),
            ("gtcrn_model_file", "GTCRN 降噪模型（电脑声音）", "选择 gtcrn_simple.onnx"),
        )
        for field_name, label, placeholder in fields:
            group = QWidget()
            group_layout = QVBoxLayout(group)
            group_layout.setContentsMargins(0, 0, 0, 0)
            group_layout.addWidget(text_label(label))
            row = QHBoxLayout()
            edit = QLineEdit(getattr(settings, field_name))
            edit.setPlaceholderText(placeholder)
            edit.setClearButtonEnabled(True)
            self.path_edits[field_name] = edit
            row.addWidget(edit, 1)
            browse = action("选择…")
            self.browse_buttons[field_name] = browse
            browse.clicked.connect(lambda _checked=False, name=field_name: self._browse(name))
            row.addWidget(browse)
            group_layout.addLayout(row)
            self.path_groups[field_name] = group
            layout.addWidget(group)
        self.engine_hint = text_label("", "footerHint", wrap=True)
        layout.addWidget(self.engine_hint)
        for combo in self.engine_combos.values():
            combo.currentIndexChanged.connect(self._update_engine_fields)
        self._update_engine_fields()
        layout.addWidget(text_label("应用后保存到本地配置；点击主窗口“开始翻译”后才会验证并加载模型。", "footerHint", wrap=True))
        layout.addStretch()
        scroll.setWidget(content)
        shell.addWidget(scroll, 1)
        self.stop_to_edit_button = action("停止以修改", name="quietButton")
        self.stop_to_edit_button.clicked.connect(self._request_stop)
        self.stop_to_edit_button.hide()
        shell.addWidget(self.stop_to_edit_button)
        self.buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Apply | QDialogButtonBox.StandardButton.Cancel)
        self.apply_button = self.buttons.button(QDialogButtonBox.StandardButton.Apply)
        self.apply_button.setText("应用设置")
        self.buttons.button(QDialogButtonBox.StandardButton.Cancel).setText("取消")
        self.apply_button.clicked.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        shell.addWidget(self.buttons)
        if devices:
            self.set_devices(devices)
        if QApplication.platformName() != "offscreen":
            self.fit_to_screen()

    def fit_to_screen(self, available=None):
        area = self.screen().availableGeometry() if available is None else available
        target = QSize(min(720, area.width() - 16), min(660, area.height() - 48))
        self.resize(target.expandedTo(self.minimumSize()))
        self.move(area.x() + max(0, (area.width() - self.width()) // 2),
                  area.y() + max(0, (area.height() - self.height() - 32) // 2))

    def set_settings_source(self, path):
        source = str(Path(path).resolve()) if path else ""
        self.settings_source_label.setText(f"配置文件：{source}" if source else "")
        self.settings_source_label.setToolTip(source)
        self.settings_source_label.setVisible(bool(source))

    def _request_refresh(self):
        if self._editing_locked():
            return
        self.device_hint.setText("正在读取设备列表…")
        self.refresh_requested.emit()

    def _editing_locked(self):
        parent = self.parentWidget()
        return self._runtime_active or bool(getattr(parent, "_voice_resources_active", False))

    def set_runtime_state(self, state, active):
        """The controller's stopped state alone confirms resources are released."""
        self._runtime_state = state
        self._runtime_active = bool(active)
        if not active:
            self._stop_pending = False
        elif state == "stopping":
            self._stop_pending = True
        controls = [self.refresh_button, self.apply_button, *self.engine_combos.values(),
                    *self.device_combos.values(), *self.path_edits.values(), *self.browse_buttons.values()]
        controls.extend(control for pair in self.channel_controls.values() for control in pair)
        for control in controls:
            control.setEnabled(not active)
        status = {"loading": "正在加载模型", "running": "正在翻译", "stopping": "正在停止",
                  "error": "处理出错，正在等待声音资源释放"}.get(state, "声音处理尚未结束")
        if self._stop_pending:
            status = "正在停止，等待声音资源释放"
        self.runtime_hint.setText(status + "。可查看当前设置，停止完成后才能修改。" if active else "")
        self.runtime_hint.setVisible(active)
        self.stop_to_edit_button.setVisible(active)
        self.stop_to_edit_button.setText("正在停止…" if self._stop_pending else "停止以修改")
        self.stop_to_edit_button.setEnabled(active and not self._stop_pending)

    def _request_stop(self):
        if not self._editing_locked() or self._stop_pending:
            return
        self._stop_pending = True
        self.set_runtime_state(self._runtime_state, True)
        self.stop_requested.emit()

    def set_devices(self, devices):
        count = 0
        for channel, combo in self.device_combos.items():
            selected = combo.currentData()
            combo.clear()
            combo.addItem("系统默认设备", None)
            for device in devices:
                if device.channel == channel:
                    combo.addItem(device.name, device.index)
                    count += 1
            if selected is not None and combo.findData(selected) < 0:
                combo.addItem(f"设备 {selected}（当前列表中不可用）", selected)
            combo.setCurrentIndex(max(0, combo.findData(selected)))
        self.device_hint.setText(f"已发现 {count} 个输入设备；电脑声音使用 WASAPI loopback。")

    def _browse(self, field_name):
        if self._editing_locked():
            return
        current = self.path_edits[field_name].text()
        if field_name in {"translation_model_dir", "whisper_model_dir", "nllb_model_dir", "nllb_tokenizer_dir"}:
            selected = QFileDialog.getExistingDirectory(self, "选择本地模型或 tokenizer 目录", current)
        else:
            filters = {
                "asr_model": "ONNX 模型 (*.onnx);;所有文件 (*)",
                "asr_tokens": "词表 (*.txt);;所有文件 (*)",
                "translation_spm": "SentencePiece 模型 (*.model);;所有文件 (*)",
                "vad_model": "ONNX 模型 (*.onnx);;所有文件 (*)",
                "mdx_model_file": "ONNX 模型 (*.onnx);;所有文件 (*)",
                "gtcrn_model_file": "ONNX 模型 (*.onnx);;所有文件 (*)",
                "hymt_model_file": "GGUF 模型 (*.gguf);;所有文件 (*)",
                "hymt_server_exe": "本地运行程序 (*.exe);;所有文件 (*)",
            }
            selected, _ = QFileDialog.getOpenFileName(self, "选择本地模型文件", current, filters[field_name])
        if selected:
            self.path_edits[field_name].setText(selected)

    def _update_engine_fields(self):
        whisper = self.engine_combos["recognition_backend"].currentData() == "whisper"
        nllb = self.engine_combos["translation_backend"].currentData() == "nllb"
        hymt = self.engine_combos["translation_backend"].currentData() == "hymt"
        processing = self.engine_combos["system_audio_processing"].currentData()
        self.path_groups["mdx_model_file"].setVisible(processing in {"mdx", "mdx_gtcrn"})
        self.path_groups["gtcrn_model_file"].setVisible(processing in {"gtcrn", "gtcrn_preserve", "mdx_gtcrn"})
        self.processing_hint.setText({
            "raw": "电脑原声直接进入人声检测与识别。麦克风保持原设置。",
            "gtcrn": "电脑声音经流式降噪并保留50%原声后进入人声检测与识别，无需等待MDX整窗。原声保护用于保留弱人声，也不能保证分离游戏声或背景音乐；游戏混音请先对照原声或MDX。麦克风保持原设置。",
            "gtcrn_preserve": "试验模式：使用原声断句和识别，GTCRN仅作并行诊断，不增加或合并识别片段。背景音乐和游戏声音也会保留；这不提供识别降噪收益，也不能保证识别准确。麦克风保持原设置。",
            "mdx": "先提取电脑人声，再检测与识别。首批需约5.8秒原声，另加推理时间；CPU可能继续积压。麦克风保持原设置。",
            "mdx_gtcrn": "先提取电脑人声，再降噪、检测与识别。首批需约5.8秒原声，另加推理时间；CPU可能继续积压。麦克风保持原设置。",
        }[processing])
        for name in ("asr_model", "asr_tokens"):
            self.path_groups[name].setVisible(not whisper)
        self.path_groups["whisper_model_dir"].setVisible(whisper)
        for name in ("translation_model_dir", "translation_spm"):
            self.path_groups[name].setVisible(not nllb and not hymt)
        for name in ("nllb_model_dir", "nllb_tokenizer_dir"):
            self.path_groups[name].setVisible(nllb)
        for name in ("hymt_model_file", "hymt_server_exe"):
            self.path_groups[name].setVisible(hymt)
        for name in ("translation_device", "translation_compute_type"):
            self.engine_rows[name].setVisible(not hymt)
        for name in ("recognition_device", "recognition_compute_type"):
            self.engine_rows[name].setVisible(whisper)
        for role in ("recognition", "translation"):
            if role == "translation" and hymt:
                continue
            gpu = self.engine_combos[role + "_device"].currentData() == "cuda"
            combo = self.engine_combos[role + "_compute_type"]
            if not gpu and combo.currentData() != "int8":
                combo.setCurrentIndex(combo.findData("int8"))
            for index in range(combo.count()):
                combo.model().item(index).setEnabled(gpu or combo.itemData(index) == "int8")
        message = "模型切换在下次手动开始时生效；两个声道共享模型。CUDA 需要配套 NVIDIA 运行库。"
        if nllb:
            message += " NLLB 权重使用 CC-BY-NC-4.0，定位研究及非商业用途。"
        elif hymt:
            message += " Hy-MT2-1.8B Q8_0 使用本地 GGUF，量化由文件固定；CUDA 自动运行，不自动回退到 CPU。"
        self.engine_hint.setText(message)

    def accept(self):
        if self._editing_locked():
            return
        values = {name: edit.text().strip() for name, edit in self.path_edits.items()}
        values.update({name: combo.currentData() for name, combo in self.engine_combos.items()})
        for channel, field_name in (("mic", "microphone"), ("system", "system")):
            enabled, direction = self.channel_controls[channel]
            values[field_name] = replace(
                getattr(self._settings, field_name), enabled=enabled.isChecked(),
                direction=direction.currentData(), device_index=self.device_combos[channel].currentData(),
            )
        self.result_settings = replace(self._settings, **values)
        super().accept()


class _HistoryRow(QFrame):
    def __init__(self, caption, settings, *, prototype=True):
        super().__init__()
        self.prototype = prototype
        self.setObjectName("historyRow")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 10, 14, 12)
        layout.setSpacing(6)
        if not prototype:
            layout.setContentsMargins(10, 4, 10, 5)
            layout.setSpacing(2)
        if prototype:
            heading = QHBoxLayout()
            heading.addWidget(text_label(datetime.fromtimestamp(caption.created_at, ZONE).strftime("%H:%M"), "smallCaps"))
            heading.addStretch()
            layout.addLayout(heading)
        self.original = text_label(name="originalText", wrap=True)
        self.translated = text_label(name="translationText", wrap=True)
        for label in (self.original, self.translated):
            label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            layout.addWidget(label)
        self.update_caption(caption, settings)

    def update_caption(self, caption, settings):
        self.original.setText(caption.original)
        failed_original = caption.status == "failed"
        timestamp = datetime.fromtimestamp(caption.created_at, ZONE).strftime("%H:%M:%S")
        self.setToolTip(f"{timestamp}\n{caption.error_message}" if failed_original else timestamp)
        pending = "样例准备中…" if self.prototype else "正在本地翻译…"
        cancelled = caption.status == "cancelled"
        self.translated.setText(caption.original if failed_original else
                                "已停止翻译，原文已保留" if cancelled else caption.translated or pending)
        self.translated.setToolTip(caption.error_message if caption.status == "failed" else "")
        self.original.setVisible(settings.show_original and not failed_original)
        font(self.original, max(9, round(settings.font_size * .34)) if self.prototype else max(11, round(settings.font_size * .42)), SECONDARY)
        color = SECONDARY if cancelled else INK
        font(self.translated, max(11, round(settings.font_size * .48)) if self.prototype else max(18, round(settings.font_size * .75)), color)


class _ChannelPanel(QObject):
    def __init__(self, channel, window):
        super().__init__(window)
        self.channel, self.window = channel, window
        self.rows = OrderedDict()
        self.latest_id = None
        self._following = False
        self._review_anchor_state = None
        self._review_restore_scheduled = False
        self.current_card = QFrame()
        self.current_card.setObjectName("currentCard")
        layout = QVBoxLayout(self.current_card)
        layout.setContentsMargins(25, 22, 25, 20)
        layout.setSpacing(18)
        if not window.prototype:
            layout.setContentsMargins(12, 7, 12, 7)
            layout.setSpacing(4)
        header = QHBoxLayout()
        header.addWidget(text_label(CHANNELS[channel], "channelTag"))
        self.direction_label = text_label()
        header.addWidget(self.direction_label)
        header.addStretch()
        self.copy_button = action(glyph="copy", name="iconButton")
        if not window.prototype:
            self.copy_button.setStyleSheet("padding: 2px 6px; min-height: 18px;")
        self.copy_button.setToolTip("复制当前一句")
        self.copy_button.setAccessibleName("复制" + CHANNELS[channel] + "当前字幕")
        self.copy_button.clicked.connect(lambda: window.copy_latest(channel))
        header.addWidget(self.copy_button)
        layout.addLayout(header)
        self.current_scroll = QScrollArea()
        self.current_scroll.setWidgetResizable(True)
        self.current_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.current_scroll.setFrameShape(QFrame.Shape.NoFrame)
        content = QWidget()
        content.setStyleSheet("background: transparent;")
        current = QVBoxLayout(content)
        current.setContentsMargins(1, 14, 8, 20)
        current.setSpacing(24)
        if not window.prototype:
            current.setContentsMargins(1, 2, 8, 2)
            current.setSpacing(4)
        self.latest_original = text_label(name="originalText", wrap=True)
        self.latest_translation = text_label(name="translationText", wrap=True)
        current.addStretch()
        for label in (self.latest_original, self.latest_translation):
            label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
            current.addWidget(label)
        current.addStretch()
        self.current_scroll.setWidget(content)
        layout.addWidget(self.current_scroll, 1)
        self.current_title = text_label("等待第一句", "smallCaps")
        if window.prototype:
            layout.addWidget(self.current_title)
        else:
            self.current_title.setParent(self.current_card)
            self.current_title.hide()
        self.history_group = QFrame()
        history = QVBoxLayout(self.history_group)
        history.setContentsMargins(0, 0, 0, 0)
        history.setSpacing(6)
        history_header = QHBoxLayout()
        history_header.addWidget(text_label(CHANNELS[channel], "smallCaps"))
        history_header.addStretch()
        self.follow_button = action("跟随最新", name="chipButton", checkable=True)
        self.follow_button.setChecked(True)
        self.follow_button.setToolTip("向上滚动回看；点击返回最新字幕")
        self.follow_button.toggled.connect(self._follow_changed)
        history_header.addWidget(self.follow_button)
        history.addLayout(history_header)
        self.history_scroll = QScrollArea()
        self.history_viewport = self.history_scroll.viewport()
        self.history_scroll.setWidgetResizable(True)
        self.history_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.history_scroll.setFrameShape(QFrame.Shape.NoFrame)
        self.history_container = QWidget()
        self.history_container.installEventFilter(self)
        self.history_container.setStyleSheet("background: transparent;")
        self.history_layout = QVBoxLayout(self.history_container)
        self.history_layout.setContentsMargins(0, 0, 6, 0)
        self.history_layout.setSpacing(5)
        self.history_placeholder = text_label("播放后，前面的句子会留在这里。" if window.prototype else
            "开始翻译后，本次字幕会依次出现在这里。\n向上滚动回看，点击“返回最新”继续跟随。", "footerHint", wrap=not window.prototype)
        self.history_layout.addWidget(self.history_placeholder)
        self.history_layout.addStretch()
        self.history_scroll.setWidget(self.history_container)
        history.addWidget(self.history_scroll, 1)
        if not window.prototype:
            # Voice has one reading surface: the complete session stream.
            # Keep latest_* for copying/status without showing a duplicate card.
            self.current_scroll.hide()
            layout.removeWidget(self.current_scroll)
            header.insertWidget(header.count() - 1, self.follow_button)
            layout.addWidget(self.history_scroll, 1)
            self.history_scroll.setAccessibleName(CHANNELS[channel] + "本次连续字幕")
            self.history_viewport.installEventFilter(self)
            self.history_group.hide()
        bar = self.history_scroll.verticalScrollBar()
        bar.rangeChanged.connect(lambda *_: self.scroll_latest())
        bar.valueChanged.connect(self._position_changed)
        self._animation = None
        self.reset_current()

    def reset_current(self):
        self.latest_id = None
        self.latest_original.setText("原文与译文，一起慢慢读。" if self.window.prototype else "尚未收到语音识别结果。")
        placeholder = "让对话，\n自然发生。" if self.channel == "mic" else "听见另一种语言，\n也听懂其中的温度。"
        self.latest_translation.setText(placeholder if self.window.prototype else "开始翻译后，\n字幕会出现在这里。")
        self.current_title.setText("等待第一句")
        self.copy_button.setEnabled(False)

    def update_settings(self, settings):
        anchor = self._review_anchor()
        source = settings.microphone if self.channel == "mic" else settings.system
        captions = {item.id: item for item in self.window.store.items(self.channel)}
        latest = captions.get(self.latest_id)
        failed_original = latest is not None and latest.status == "failed"
        self.direction_label.setText(dict(DIRECTIONS)[source.direction])
        self.current_card.setVisible(source.enabled)
        self.history_group.setVisible(source.enabled)
        self.latest_original.setVisible(settings.show_original and not failed_original)
        font(self.latest_original, max(12, round(settings.font_size * (.48 if self.window.prototype else .4))), SECONDARY)
        font(self.latest_translation, settings.font_size if self.window.prototype else max(18, round(settings.font_size * .75)))
        for key, row in self.rows.items():
            caption = captions.get(key)
            if caption:
                row.update_caption(caption, settings)
        self._restore_review_anchor(anchor)

    def render(self, caption, settings):
        anchor = self._review_anchor()
        row = self.rows.get(caption.id)
        if row is None:
            row = _HistoryRow(caption, settings, prototype=self.window.prototype)
            self.rows[caption.id] = row
            self.history_layout.insertWidget(self.history_layout.count() - 1, row)
        else:
            row.update_caption(caption, settings)
        retained = {item.id for item in self.window.store.items(self.channel)}
        for key in list(self.rows):
            if key not in retained:
                removed = self.rows.pop(key)
                self.history_layout.removeWidget(removed)
                removed.hide()
                removed.deleteLater()
        latest = self.window.store.items(self.channel)
        self.history_placeholder.hide()
        self._restore_review_anchor(anchor)
        if latest and latest[-1].id != caption.id:
            self.scroll_latest()
            return
        changed = self.latest_id != caption.id or self.latest_translation.text() != caption.translated
        self.latest_id = caption.id
        self.latest_original.setText(caption.original)
        pending = "正在呈现这句样例…" if self.window.prototype else "正在本地翻译…"
        cancelled = caption.status == "cancelled"
        failed_original = caption.status == "failed"
        self.latest_original.setVisible(settings.show_original and not failed_original)
        self.latest_translation.setText(caption.original if failed_original else
                                        "已停止翻译，原文已保留" if cancelled else caption.translated or pending)
        self.latest_translation.setToolTip(caption.error_message if caption.status == "failed" else "")
        color = SECONDARY if cancelled else INK
        font(self.latest_translation, settings.font_size if self.window.prototype else max(18, round(settings.font_size * .75)), color)
        title = "已停止翻译 · " if cancelled else "翻译失败 · " if caption.status == "failed" else "当前一句 · "
        self.current_title.setText(title + datetime.fromtimestamp(caption.created_at, ZONE).strftime("%H:%M"))
        self.copy_button.setEnabled(bool(caption.original if failed_original else caption.translated))
        if cancelled:
            self.stop_animation()
        if changed:
            self.current_scroll.verticalScrollBar().setValue(0)
        if self.window.prototype and caption.translated and changed and self.window.animations_enabled:
            effect = self.latest_translation.graphicsEffect()
            if effect is None:
                effect = QGraphicsOpacityEffect(self.latest_translation)
                self.latest_translation.setGraphicsEffect(effect)
            if self._animation:
                self._animation.stop()
                self._animation.deleteLater()
            self._animation = QPropertyAnimation(effect, b"opacity", self)
            self._animation.setDuration(240)
            self._animation.setStartValue(.35)
            self._animation.setEndValue(1.0)
            self._animation.setEasingCurve(QEasingCurve.Type.OutCubic)
            self._animation.start()
        self.scroll_latest()

    def _review_anchor(self):
        if self.window.prototype or self.follow_button.isChecked():
            return None
        return self._review_anchor_state or self._find_review_anchor()

    def _find_review_anchor(self):
        viewport = self.history_scroll.viewport()
        for key, row in self.rows.items():
            if not row.isHidden():
                top = row.mapTo(viewport, row.rect().topLeft()).y()
                if top + row.height() > 0:
                    return key, top
        return None

    def _restore_review_anchor(self, anchor):
        if anchor is None:
            return
        self._review_anchor_state = anchor
        if self._review_restore_scheduled:
            return
        self._review_restore_scheduled = True

        def restore():
            self._review_restore_scheduled = False
            if self._review_anchor_state is None or self.follow_button.isChecked():
                return
            key, top = self._review_anchor_state
            self.history_layout.activate()
            row = self.rows.get(key)
            if row is not None and not row.isHidden() and not self.follow_button.isChecked():
                offset = row.mapTo(self.history_scroll.viewport(), row.rect().topLeft()).y() - top
                bar = self.history_scroll.verticalScrollBar()
                self._following = True
                try:
                    bar.setValue(bar.value() + offset)
                finally:
                    self._following = False
        QTimer.singleShot(0, restore)

    def eventFilter(self, watched, event):
        if watched is self.history_viewport and event.type() == QEvent.Type.Wheel:
            bar = self.history_scroll.verticalScrollBar()
            if event.angleDelta().y() > 0 and bar.maximum() > 0:
                self.follow_button.setChecked(False)
            elif event.angleDelta().y() < 0:
                # Resume only after the reader deliberately scrolls down to the
                # end; layout changes must not cancel an active review anchor.
                QTimer.singleShot(0, self._follow_if_at_bottom)
        if watched is self.history_container and event.type() in (QEvent.Type.LayoutRequest, QEvent.Type.Resize):
            # Wrapped text and QScrollArea resize on separate turns; keep the
            # reader's anchor through both instead of pinning a stale bar value.
            self._restore_review_anchor(self._review_anchor())
        return False

    def _follow_if_at_bottom(self):
        bar = self.history_scroll.verticalScrollBar()
        if bar.value() >= bar.maximum() - 2:
            self.follow_button.setChecked(True)

    def scroll_latest(self):
        if self.follow_button.isChecked():
            self._following = True
            bar = self.history_scroll.verticalScrollBar()
            try:
                bar.setValue(bar.maximum())
            finally:
                self._following = False

    def _position_changed(self, value):
        if self._following:
            return
        if value < self.history_scroll.verticalScrollBar().maximum() - 2:
            self.follow_button.setChecked(False)
        if not self.window.prototype and not self.follow_button.isChecked():
            self._review_anchor_state = self._find_review_anchor()

    def _follow_changed(self, enabled):
        self._review_anchor_state = None if enabled else self._find_review_anchor()
        self.follow_button.setText("跟随最新" if enabled else "返回最新")
        self.scroll_latest()

    def stop_animation(self):
        if self._animation:
            self._animation.stop()
        effect = self.latest_translation.graphicsEffect()
        if effect:
            effect.setOpacity(1.0)

    def clear(self):
        self.stop_animation()
        self._review_anchor_state = None
        for row in self.rows.values():
            self.history_layout.removeWidget(row)
            row.hide()
            row.deleteLater()
        self.rows.clear()
        self.history_placeholder.show()
        self.follow_button.setChecked(True)
        self.reset_current()


class StudioWindow(QMainWindow):
    start_requested = Signal(object)
    demo_requested = Signal()
    replay_requested = Signal()
    stop_requested = Signal()
    clear_requested = Signal()
    refresh_devices_requested = Signal()
    settings_changed = Signal(object)
    playback_speed_changed = Signal(float)
    closing = Signal()
    update_requested = Signal()

    def __init__(self, settings=None, *, prototype=True):
        super().__init__()
        self.prototype = bool(prototype)
        self.settings = settings or AppSettings(font_size=30)
        self._devices = []
        self._runtime_dialog = None
        self._settings_source = ""
        self._voice_resources_active = False
        self.closing_ready = False
        self._close_requested = False
        self.store = CaptionStore(max_per_channel=100 if self.prototype else None)
        self._applying = False
        self._state = "idle"
        self.animations_enabled = QApplication.platformName() != "offscreen"
        self.setObjectName("studioRoot")
        self.setWindowTitle(f"翻译姬 NEXT {__version__} · 轻声字幕")
        self.resize(1440, 900)
        self.setMinimumSize(1000, 600)
        self.setFont(QFont("Microsoft YaHei UI", 10))
        self.setStyleSheet(STUDIO_STYLE)
        central = QWidget()
        central.setObjectName("studioRoot")
        self.setCentralWidget(central)
        outer = QVBoxLayout(central)
        outer.setContentsMargins(32, 16, 32, 18)
        outer.setSpacing(16)
        if not self.prototype:
            outer.setContentsMargins(24, 10, 24, 12)
            outer.setSpacing(10)

        top = QFrame()
        top.setObjectName("topbar")
        top_layout = QHBoxLayout(top)
        top_layout.setContentsMargins(0, 0, 0, 15)
        if not self.prototype:
            top_layout.setContentsMargins(0, 0, 0, 8)
        mark = text_label("译", "brandMark")
        mark.setAlignment(Qt.AlignmentFlag.AlignCenter)
        mark.setFixedSize(44, 44)
        mark.setFont(QFont("Microsoft YaHei UI", 20, QFont.Weight.Medium))
        top_layout.addWidget(mark)
        brand = QVBoxLayout()
        brand.setSpacing(2)
        brand.addWidget(text_label("翻译姬", "brandName"))
        brand.addWidget(text_label("让语言，轻一点。", "footerHint"))
        top_layout.addLayout(brand)
        top_layout.addStretch()
        self.mode_label = text_label("界面体验版 · 预置字幕" if self.prototype else "本地语音 · 手动开始", "smallCaps")
        if not self.prototype:
            self.mode_label.setWordWrap(True)
            self.mode_label.setMaximumWidth(300)
        top_layout.addWidget(self.mode_label)
        self.model_button = action("声音与模型", "settings")
        self.model_button.setVisible(not self.prototype)
        self.model_button.clicked.connect(self.open_runtime_settings)
        top_layout.addWidget(self.model_button)
        self.history_button = action("历史" if self.prototype else "本次字幕", "history", checkable=self.prototype)
        self.history_button.setChecked(self.prototype)
        if not self.prototype:
            self.history_button.setToolTip("本窗口内保留全部字幕；点击返回最新。需要留存时可手动导出。")
            self.history_button.clicked.connect(self._return_to_latest)
        self.focus_button = action("专注", "focus", checkable=True)
        self.settings_button = action("阅读设置", "settings", checkable=True)
        for control in (self.history_button, self.focus_button, self.settings_button):
            top_layout.addWidget(control)
        outer.addWidget(top)

        self.welcome = QWidget()
        welcome_layout = QHBoxLayout(self.welcome)
        welcome_layout.setContentsMargins(2, 0, 0, 0)
        titles = QVBoxLayout()
        titles.setSpacing(8)
        titles.addWidget(text_label("中日双语 · 从容阅读", "smallCaps"))
        titles.addWidget(text_label("让理解，自然流动。", "pageTitle"))
        welcome_layout.addLayout(titles)
        welcome_layout.addStretch()
        self.state_chip = text_label("准备好了", "statusChip")
        welcome_layout.addWidget(self.state_chip)
        outer.addWidget(self.welcome)

        self.notice = text_label(name="notice", wrap=True)
        self.notice.setMaximumHeight(54)
        self.notice.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.notice.setParent(central)
        self.notice.hide()
        body_row = QHBoxLayout()
        body_row.setSpacing(20)
        reading = QWidget()
        reading_layout = QVBoxLayout(reading)
        reading_layout.setContentsMargins(0, 0, 0, 0)
        reading_layout.setSpacing(15)
        if not self.prototype:
            reading_layout.setSpacing(10)
        self.panels = {channel: _ChannelPanel(channel, self) for channel in CHANNELS}
        self.stage = QWidget()
        stage_layout = QHBoxLayout(self.stage)
        stage_layout.setContentsMargins(0, 0, 0, 0)
        stage_layout.setSpacing(20)
        for panel in self.panels.values():
            stage_layout.addWidget(panel.current_card, 1)
        reading_layout.addWidget(self.stage, 1)
        self.empty_label = text_label("在阅读设置中显示一个字幕通道，即可开始。", wrap=True)
        self.empty_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        reading_layout.addWidget(self.empty_label)
        self.history_section = QWidget()
        section_layout = QVBoxLayout(self.history_section)
        section_layout.setContentsMargins(2, 0, 2, 0)
        section_layout.setSpacing(9)
        history_heading = QHBoxLayout()
        history_heading.addWidget(text_label("刚刚说过的" if self.prototype else "连续对话 · 当前句与前文", "smallCaps"))
        history_heading.addStretch()
        self.caption_count = text_label("还没有记录", "footerHint")
        history_heading.addWidget(self.caption_count)
        section_layout.addLayout(history_heading)
        self.history_area = QWidget()
        history_layout = QHBoxLayout(self.history_area)
        history_layout.setContentsMargins(0, 0, 0, 0)
        history_layout.setSpacing(20)
        for panel in self.panels.values():
            history_layout.addWidget(panel.history_group, 1)
        section_layout.addWidget(self.history_area, 1)
        if self.prototype:
            self.history_section.setFixedHeight(160)
            reading_layout.addWidget(self.history_section)
        else:
            self.history_section.setParent(reading)
            self.history_section.hide()
        body_row.addWidget(reading, 1)
        self.drawer = self._make_drawer()
        body_row.addWidget(self.drawer)
        self.drawer.hide()
        outer.addLayout(body_row, 1)

        self.playback_dock = QFrame()
        self.playback_dock.setObjectName("dock")
        shadow = QGraphicsDropShadowEffect(self.playback_dock)
        shadow.setBlurRadius(24)
        shadow.setOffset(0, 4)
        shadow.setColor(QColor(70, 78, 61, 18))
        self.playback_dock.setGraphicsEffect(shadow)
        dock = QVBoxLayout(self.playback_dock)
        dock.setContentsMargins(20, 13, 20, 10)
        dock.setSpacing(10)
        if not self.prototype:
            dock.setContentsMargins(14, 8, 14, 8)
            dock.setSpacing(6)
        controls = QHBoxLayout()
        controls.setSpacing(13)
        self.start_button = action("播放样例", "play", name="primaryButton")
        self.start_button.setIcon(icon("play", "#FFFEFB"))
        self.start_button.clicked.connect(self._main_action)
        self.stop_button = action("停止", name="quietButton")
        self.stop_button.clicked.connect(self.stop_requested.emit)
        self.stop_button.setVisible(not self.prototype)
        self.replay_button = action(glyph="replay", name="iconButton")
        self.replay_button.setToolTip("从头播放 · R")
        self.replay_button.setAccessibleName("重新播放")
        self.replay_button.clicked.connect(self.replay_requested.emit)
        controls.addWidget(self.start_button)
        controls.addWidget(self.stop_button)
        controls.addWidget(self.replay_button)
        self.progress_text = text_label("00 / 06", "smallCaps")
        self.progress_text.setMinimumWidth(65)
        controls.addWidget(self.progress_text)
        self.playback_progress = QProgressBar()
        self.playback_progress.setRange(0, 6)
        self.playback_progress.setValue(0)
        self.playback_progress.setTextVisible(False)
        self.playback_progress.setFixedHeight(5)
        controls.addWidget(self.playback_progress, 1)
        self.speed_combo = QComboBox()
        for title, value in (("慢一点 · 0.75×", .75), ("自然 · 1×", 1.0), ("轻快 · 1.25×", 1.25)):
            self.speed_combo.addItem(title, value)
        self.speed_combo.setCurrentIndex(1)
        self.speed_combo.setToolTip("只调整模拟字幕停留时间")
        self.speed_combo.currentIndexChanged.connect(lambda _index: self.playback_speed_changed.emit(self.speed_combo.currentData()))
        controls.addWidget(self.speed_combo)
        self.clear_button = action("清空", name="quietButton")
        self.clear_button.clicked.connect(self._clear_action)
        self.export_button = action("导出", "export", name="quietButton")
        self.export_button.clicked.connect(self._export)
        controls.addWidget(self.clear_button)
        controls.addWidget(self.export_button)
        dock.addLayout(controls)
        self.runtime_metrics = QWidget()
        metrics = QHBoxLayout(self.runtime_metrics)
        metrics.setContentsMargins(0, 0, 0, 0)
        metrics.setSpacing(12)
        self.level_meters = {}
        for channel in CHANNELS:
            metrics.addWidget(text_label(CHANNELS[channel], "footerHint"))
            meter = QProgressBar()
            meter.setRange(0, 1000)
            meter.setValue(0)
            meter.setTextVisible(False)
            meter.setFixedSize(72, 6)
            meter.setAccessibleName(CHANNELS[channel] + "音量")
            metrics.addWidget(meter)
            self.level_meters[channel] = meter
        metrics.addStretch()
        self.queue_label = text_label("待处理 0 段", "footerHint")
        metrics.addWidget(self.queue_label)
        self.runtime_metrics.setVisible(not self.prototype)
        dock.addWidget(self.runtime_metrics)
        for control in (self.replay_button, self.progress_text, self.playback_progress, self.speed_combo):
            control.setVisible(self.prototype)
        dock_bottom = QHBoxLayout()
        self.status_hint = text_label("按一下播放，给每句话留一点时间。", "footerHint", wrap=True)
        dock_bottom.addWidget(self.status_hint)
        dock_bottom.addStretch()
        self.update_button = action("检查更新", name="quietButton")
        self.update_button.setAccessibleName("检查软件更新")
        self.update_button.setVisible(not self.prototype)
        self.update_button.setStyleSheet("padding: 2px 10px; min-height: 22px;")
        self.update_button.clicked.connect(self.update_requested.emit)
        dock_bottom.addWidget(self.update_button)
        dock_bottom.addWidget(text_label("空格 播放 / 暂停" if self.prototype else "空格 开始 / 停止", "footerHint"))
        dock.addLayout(dock_bottom)
        outer.addWidget(self.playback_dock)

        self.history_button.toggled.connect(self._history_changed)
        self.focus_button.toggled.connect(self._focus_changed)
        self.settings_button.toggled.connect(self.drawer.setVisible)
        self._toast_timer = QTimer(self)
        self._toast_timer.setSingleShot(True)
        self._toast_timer.timeout.connect(self.notice.hide)
        self._recoverable_notice_message = None
        self._shortcuts = []
        self._reading_shortcuts = []
        for key, callback in (("Space", self._main_action), ("R", self._replay_action), ("Ctrl+E", self._export)):
            shortcut = QShortcut(QKeySequence(key), self)
            shortcut.setAutoRepeat(False)
            shortcut.activated.connect(lambda cb=callback: self._shortcut(cb))
            self._shortcuts.append(shortcut)
            self._reading_shortcuts.append((key, shortcut))
        escape = QShortcut(QKeySequence("Escape"), self)
        escape.activated.connect(self._escape)
        self._shortcuts.append(escape)
        QApplication.instance().focusChanged.connect(self._keyboard_focus_changed)
        self._keyboard_focus_changed()
        self.update_settings(self.settings)
        self.set_session_state("idle")
        if QApplication.platformName() != "offscreen":
            self.fit_to_screen()

    def _make_drawer(self):
        drawer = QFrame()
        drawer.setObjectName("drawer")
        drawer.setFixedWidth(285)
        shell = QVBoxLayout(drawer)
        shell.setContentsMargins(0, 0, 0, 0)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        content = QWidget()
        content.setStyleSheet("background: transparent;")
        layout = QVBoxLayout(content)
        layout.setContentsMargins(20, 22, 20, 22)
        layout.setSpacing(18)
        heading = QHBoxLayout()
        title = text_label("读得舒服一点", "brandName")
        heading.addWidget(title)
        heading.addStretch()
        close = action(glyph="close", name="iconButton")
        close.setToolTip("收起阅读设置")
        close.clicked.connect(lambda: self.settings_button.setChecked(False))
        heading.addWidget(close)
        layout.addLayout(heading)
        layout.addWidget(text_label("文字与窗口", "smallCaps"))
        self.original_checkbox = QCheckBox("同时显示原文")
        self.original_checkbox.toggled.connect(self._appearance_changed)
        layout.addWidget(self.original_checkbox)
        font_row = QHBoxLayout()
        font_row.addWidget(text_label("译文字号"))
        font_row.addStretch()
        self.font_size = QSpinBox()
        self.font_size.setRange(16, 40)
        self.font_size.valueChanged.connect(self._appearance_changed)
        font_row.addWidget(self.font_size)
        layout.addLayout(font_row)
        self.topmost_checkbox = QCheckBox("保持窗口置顶")
        self.topmost_checkbox.toggled.connect(self._appearance_changed)
        layout.addWidget(self.topmost_checkbox)
        self.motion_checkbox = QCheckBox("减少动态效果")
        self.motion_checkbox.toggled.connect(self._motion_changed)
        layout.addWidget(self.motion_checkbox)
        layout.addWidget(text_label("阅读布局", "smallCaps"))
        self.layout_combo = QComboBox()
        self.layout_combo.addItem("并排对照", False)
        self.layout_combo.addItem("上下阅读 · 专注", True)
        self.layout_combo.currentIndexChanged.connect(self._layout_changed)
        layout.addWidget(self.layout_combo)
        layout.addSpacing(5)
        layout.addWidget(text_label("字幕通道", "smallCaps"))
        self.channel_controls = {}
        for channel in CHANNELS:
            enabled = QCheckBox(CHANNELS[channel])
            enabled.toggled.connect(self._source_changed)
            direction = QComboBox()
            for value, title in DIRECTIONS:
                direction.addItem(title, value)
            direction.currentIndexChanged.connect(self._source_changed)
            layout.addWidget(enabled)
            layout.addWidget(direction)
            self.channel_controls[channel] = (enabled, direction)
        layout.addStretch()
        hint = ("现在只体验界面与阅读。\n语音翻译等验收后再接入。" if self.prototype else
                "声音和模型在顶部单独设置。\n运行时仍可调整字号、原文和置顶。")
        layout.addWidget(text_label(hint, "footerHint", wrap=True))
        scroll.setWidget(content)
        shell.addWidget(scroll)
        return drawer

    def _main_action(self):
        if self.prototype:
            self.demo_requested.emit()
        elif self._voice_resources_active:
            if self._state != "stopping":
                self.stop_requested.emit()
        elif not (self.settings.microphone.enabled or self.settings.system.enabled):
            self.show_error("请在阅读设置或声音设置中开启至少一个字幕通道。")
        else:
            self.start_requested.emit(self.settings)

    def _replay_action(self):
        if self.prototype:
            self.replay_requested.emit()

    def _clear_action(self):
        if self.prototype:
            self.stop_requested.emit()
        elif not self._voice_resources_active:
            self.clear_requested.emit()

    def open_runtime_settings(self):
        if self.prototype:
            return None
        if self._runtime_dialog is not None and self._runtime_dialog.isVisible():
            self._runtime_dialog.set_runtime_state(self._state, self._voice_resources_active)
            self._runtime_dialog.raise_()
            return self._runtime_dialog
        if self._runtime_dialog is not None:
            self._runtime_dialog.deleteLater()
        dialog = RuntimeSettingsDialog(self.settings, self._devices, self)
        dialog.set_settings_source(self._settings_source)
        self._runtime_dialog = dialog
        dialog.refresh_requested.connect(self.refresh_devices_requested.emit)
        dialog.stop_requested.connect(self.stop_requested.emit)
        dialog.accepted.connect(lambda: self._apply_runtime_settings(dialog))
        dialog.set_runtime_state(self._state, self._voice_resources_active)
        dialog.open()
        return dialog

    def _apply_runtime_settings(self, dialog):
        if self._voice_resources_active:
            self.show_error("当前声音处理尚未结束，未应用设备和模型改动。")
            return
        selected = dialog.result_settings
        if selected is None:
            return
        fields = ("microphone", "system", "asr_model", "asr_tokens", "translation_model_dir", "translation_spm",
                  "recognition_backend", "whisper_model_dir", "recognition_device", "recognition_compute_type",
                  "translation_backend", "nllb_model_dir", "nllb_tokenizer_dir", "translation_device",
                  "translation_compute_type", "hymt_model_file", "hymt_server_exe", "vad_model",
                  "system_audio_processing", "mdx_model_file", "gtcrn_model_file")
        self.update_settings(replace(self.settings, **{name: getattr(selected, name) for name in fields}))
        self.settings_changed.emit(self.settings)

    def set_devices(self, devices):
        self._devices = list(devices)
        if self._runtime_dialog is not None:
            self._runtime_dialog.set_devices(self._devices)

    def set_settings_source(self, path):
        if self.prototype:
            return
        self._settings_source = str(Path(path).resolve()) if path else ""
        hint = f"配置文件：{self._settings_source}" if self._settings_source else ""
        self.mode_label.setToolTip(hint)
        self.model_button.setToolTip(hint)
        if self._runtime_dialog is not None:
            self._runtime_dialog.set_settings_source(self._settings_source)

    def set_device_error(self, message):
        if self._runtime_dialog is not None:
            self._runtime_dialog.device_hint.setText(message)
        self.show_error(message)

    def _shortcut(self, callback):
        if not isinstance(QApplication.focusWidget(), (QLineEdit, QComboBox, QSpinBox, QCheckBox)):
            callback()

    def _keyboard_focus_changed(self, *_args):
        focused = QApplication.focusWidget()
        editing = isinstance(focused, (QLineEdit, QComboBox, QSpinBox, QCheckBox))
        for key, shortcut in self._reading_shortcuts:
            # Disable before Qt resolves shortcuts so controls receive their native keys.
            shortcut.setEnabled(not editing and not (key == "Space" and isinstance(focused, QAbstractButton)))

    def _motion_changed(self, reduced):
        self.animations_enabled = not reduced and QApplication.platformName() != "offscreen"
        if not self.animations_enabled:
            for panel in self.panels.values():
                panel.stop_animation()

    def _escape(self):
        self.settings_button.setChecked(False)
        self.focus_button.setChecked(False)
        self.notice.hide()

    def update_settings(self, settings):
        self.settings = settings
        self._applying = True
        try:
            if not self.prototype:
                processing = settings.system_audio_processing if settings.system.enabled else "raw"
                recognition = {"sensevoice": "SenseVoice", "whisper": "Whisper large-v3"}[settings.recognition_backend]
                translation = {"hymt": "Hy-MT2 Q8", "m2m100": "M2M100", "nllb": "NLLB 1.3B"}[settings.translation_backend]
                enhancement = {"raw": "原声直通", "gtcrn": "GTCRN 流式降噪",
                               "gtcrn_preserve": "GTCRN 诊断 · 原声识别",
                               "mdx": "MDX 人声提取", "mdx_gtcrn": "MDX + GTCRN"}[processing]
                self.mode_label.setText(f"{recognition} · {translation}\n{enhancement}")
            self.original_checkbox.setChecked(settings.show_original)
            self.font_size.setValue(settings.font_size)
            self.topmost_checkbox.setChecked(settings.always_on_top)
            for channel, source in (("mic", settings.microphone), ("system", settings.system)):
                enabled, direction = self.channel_controls[channel]
                enabled.setChecked(source.enabled)
                direction.setCurrentIndex(direction.findData(source.direction))
                self.panels[channel].update_settings(settings)
            any_enabled = settings.microphone.enabled or settings.system.enabled
            self.stage.setVisible(any_enabled)
            self.empty_label.setVisible(not any_enabled)
            visible = self.isVisible()
            if bool(self.windowFlags() & Qt.WindowType.WindowStaysOnTopHint) != settings.always_on_top:
                self.setWindowFlag(Qt.WindowType.WindowStaysOnTopHint, settings.always_on_top)
                if visible:
                    self.show()
            self.setWindowOpacity(max(30, min(100, settings.window_opacity)) / 100)
            if not self.prototype:
                self._sync_voice_controls()
        finally:
            self._applying = False

    def _appearance_changed(self, *_args):
        if self._applying:
            return
        self.update_settings(replace(self.settings, font_size=self.font_size.value(), show_original=self.original_checkbox.isChecked(), always_on_top=self.topmost_checkbox.isChecked()))
        self.settings_changed.emit(self.settings)

    def _source_changed(self, *_args):
        if self._applying:
            return
        if not self.prototype and self._voice_resources_active:
            self.update_settings(self.settings)
            return
        values = {}
        for channel, field in (("mic", "microphone"), ("system", "system")):
            enabled, direction = self.channel_controls[channel]
            values[field] = replace(getattr(self.settings, field), enabled=enabled.isChecked(), direction=direction.currentData())
        self.update_settings(replace(self.settings, **values))
        self.settings_changed.emit(self.settings)

    def _history_changed(self, checked):
        if self.prototype:
            self.history_section.setVisible(checked and not self.focus_button.isChecked())

    def _return_to_latest(self):
        for panel in self.panels.values():
            panel.follow_button.setChecked(True)
            panel.scroll_latest()

    def _focus_changed(self, checked):
        self._sync_reading_density()
        if self.prototype:
            self.history_section.setVisible(not checked and self.history_button.isChecked())
        if checked:
            self.settings_button.setChecked(False)
        elif self.layout_combo.currentData():
            self.layout_combo.setCurrentIndex(0)

    def _layout_changed(self, _index):
        vertical = self.layout_combo.currentData()
        self.stage.layout().setDirection(QVBoxLayout.Direction.TopToBottom if vertical else QVBoxLayout.Direction.LeftToRight)
        if vertical:
            self.focus_button.setChecked(True)

    def set_session_state(self, state, message=""):
        self._state = state
        if state in {"idle", "loading", "stopped"} and self._recoverable_notice_message is not None:
            self._toast_timer.stop()
            self.notice.hide()
            self._recoverable_notice_message = None
        if not self.prototype:
            if state in {"loading", "running", "stopping"}:
                self._voice_resources_active = True
            elif state in {"idle", "stopped"}:
                self._voice_resources_active = False
                if state == "stopped":
                    for caption in self.store.cancel_pending():
                        self.panels[caption.channel].render(caption, self.settings)
                for meter in self.level_meters.values():
                    meter.setValue(0)
                self.queue_label.setText("待处理 0 段")
            labels = {"idle": "尚未开始", "stopped": "已停止", "loading": "正在加载模型", "running": "正在翻译", "stopping": "正在停止", "error": "处理出错"}
            self.state_chip.setText(labels.get(state, state))
            self.start_button.setText({"loading": "正在加载…", "running": "翻译中", "stopping": "正在停止…"}.get(state, "开始翻译"))
            self.start_button.setIcon(icon("play", "#FFFEFB"))
            hints = {"idle": "手动开始后才会采集声音。请先配置本地模型与声音来源。", "stopped": "声音处理已停止，字幕保留在本次会话中。", "loading": "正在初始化本地模型，可以停止；此过程不下载文件。", "running": "正在采集已启用的声音来源，识别与翻译在本地进行。", "stopping": "正在等待当前任务结束并释放声音设备。", "error": "请检查详细错误、设备权限或本地模型设置。"}
            self.status_hint.setText(message or hints.get(state, ""))
            self.status_hint.setToolTip(message or hints.get(state, ""))
            self._sync_voice_controls()
            return
        self.start_button.setText({"idle": "播放样例", "demo": "暂停片刻", "paused": "继续阅读", "finished": "再读一遍"}.get(state, "播放样例"))
        self.start_button.setIcon(icon("pause" if state == "demo" else "play", "#FFFEFB"))
        self.state_chip.setText({"idle": "准备好了", "demo": "正在呈现", "paused": "慢慢读，不着急", "finished": "这一段，读完了"}.get(state, state))
        self.status_hint.setText({"idle": "预置字幕演示 · 不采集声音", "demo": "每句话都有停留的时间，随时可以暂停。", "paused": "已经暂停。可以回看、复制，或调整阅读设置。", "finished": "样例已保留在历史中，喜欢的句子可以复制或导出。"}.get(state, message))
        self.start_button.setEnabled(True)
        self.clear_button.setEnabled(bool(self.store.items()))
        self.export_button.setEnabled(bool(self.store.items()))

    def _sync_voice_controls(self):
        active = self._voice_resources_active
        self.start_button.setEnabled(not active and (self.settings.microphone.enabled or self.settings.system.enabled))
        self.stop_button.setEnabled(active and self._state != "stopping")
        self.model_button.setEnabled(True)
        for enabled, direction in self.channel_controls.values():
            enabled.setEnabled(not active)
            direction.setEnabled(not active)
        self.clear_button.setEnabled(not active and bool(self.store.items()))
        self.export_button.setEnabled(bool(self.store.items()))
        if self._runtime_dialog is not None:
            self._runtime_dialog.set_runtime_state(self._state, active)

    def set_playback_progress(self, current, total):
        self.playback_progress.setRange(0, max(1, total))
        self.playback_progress.setValue(current)
        self.progress_text.setText(f"{current:02d} / {total:02d}")
        self.playback_progress.setAccessibleName(f"样例进度，第{current}句，共{total}句")

    def apply_event(self, event: RuntimeEvent):
        if event.kind == "queue" and not self.prototype:
            depth = max(0, int(getattr(event, "queue_depth", 0)))
            self.queue_label.setText(f"待处理 {depth} 段")
            self.queue_label.setToolTip(event.message)
            return
        if event.channel not in self.panels:
            return
        if event.kind == "level" and not self.prototype:
            self.level_meters[event.channel].setValue(round(max(0.0, min(1.0, event.level)) * 1000))
            return
        if event.kind == "original":
            caption = self.store.add_original(event.channel, event.segment_id, event.text)
        elif event.kind == "translation":
            caption = self.store.set_translation(event.channel, event.segment_id, event.text,
                                                 failed=event.status in {"error", "failed"})
        elif event.kind == "error" and event.segment_id:
            caption = self.store.set_translation(event.channel, event.segment_id, event.message or "这句翻译失败，请查看错误信息。", failed=True)
        else:
            return
        if caption:
            self.panels[event.channel].render(caption, self.settings)
            count = len(self.store.items())
            self.caption_count.setText(f"{count} 条记录 · " + ("仅本次预览" if self.prototype else "当前窗口全部保留"))
            if not self.prototype:
                self.history_button.setText(f"本次字幕 · {count}")
            self.clear_button.setEnabled(self.prototype or not self._voice_resources_active)
            self.export_button.setEnabled(True)

    def clear_captions(self):
        self.store.clear()
        for panel in self.panels.values():
            panel.clear()
            panel.update_settings(self.settings)
        self.caption_count.setText("还没有记录")
        if not self.prototype:
            self.history_button.setText("本次字幕")
        self.clear_button.setEnabled(False)
        self.export_button.setEnabled(False)
        self.notice.hide()

    def copy_latest(self, channel):
        captions = self.store.items(channel)
        if not captions:
            return
        caption = captions[-1]
        text = caption.original if caption.status == "failed" else caption.original + "\n" + caption.translated
        QApplication.clipboard().setText(text)
        self.show_error("已复制这句字幕。")
        self._toast_timer.start(1600)

    def show_error(self, message):
        self._toast_timer.stop()
        self._recoverable_notice_message = None
        self.notice.setText(message)
        self.notice.setToolTip(message)
        self.notice.setFixedWidth(min(560, self.centralWidget().width() - 64))
        self.notice.adjustSize()
        self._position_notice()
        self.notice.show()
        self.notice.raise_()

    def show_recoverable_error(self, message):
        # Repeated failures must not turn a two-second notice into a permanent banner.
        if (self._recoverable_notice_message == message and self.notice.isVisible()
                and self._toast_timer.isActive()):
            return
        self.show_error(message)
        self._recoverable_notice_message = message
        self._toast_timer.start(2000)

    def _position_notice(self):
        self.notice.move(max(16, (self.centralWidget().width() - self.notice.width()) // 2),
                         max(16, self.playback_dock.y() - self.notice.height() - 12))

    def fit_to_screen(self, available=None):
        area = self.screen().availableGeometry() if available is None else available
        # Leave room for native title bar and borders in logical desktop pixels.
        target = QSize(min(1440, area.width() - 16), min(900, area.height() - 48))
        self.resize(target.expandedTo(self.minimumSize()))
        self.move(area.x() + max(0, (area.width() - self.width()) // 2),
                  area.y() + max(0, (area.height() - self.height() - 32) // 2))

    def _sync_reading_density(self):
        compact = self.height() < 760
        self.welcome.setVisible(not compact and not self.focus_button.isChecked())
        if self.prototype:
            self.history_section.setFixedHeight(120 if compact else 160)
        else:
            anchors = {channel: panel._review_anchor() for channel, panel in self.panels.items()}
            self.stage.setMinimumHeight(0)
            self.stage.setMaximumHeight(16777215)
            self.stage.parentWidget().layout().setStretchFactor(self.stage, 1)
            for channel, panel in self.panels.items():
                panel._restore_review_anchor(anchors[channel])

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if hasattr(self, "history_section"):
            self._sync_reading_density()
        if hasattr(self, "notice") and self.notice.isVisible():
            self._position_notice()

    def transcript(self):
        parts = []
        for item in self.store.items():
            heading = f"[{datetime.fromtimestamp(item.created_at, ZONE):%H:%M:%S}] {CHANNELS[item.channel]}"
            if item.status == "failed":
                parts.append(f"{heading}\n{item.original}")
                continue
            translated = item.translated
            if item.status == "cancelled" and not translated:
                translated = "[已停止翻译，原文已保留]"
            parts.append(f"{heading}\n{item.original}\n{translated}")
        return "\n\n".join(parts)

    def _export(self):
        if not self.store.items():
            return
        path, _ = QFileDialog.getSaveFileName(self, "把这段对话留存下来", "翻译姬·字幕记录.txt", "文本文件 (*.txt)")
        if path:
            try:
                Path(path).write_text(self.transcript(), encoding="utf-8-sig")
                self.show_error("这段字幕已经保存。")
                self._toast_timer.start(2000)
            except OSError as exc:
                self.show_error(f"保存未完成：{exc}")

    def finish_shutdown(self):
        """Called by the controller after all native audio/model work has ended."""
        self.closing_ready = True
        self.close()

    def closeEvent(self, event):
        if self._runtime_dialog is not None:
            self._runtime_dialog.close()
        if not self.prototype:
            if self.closing_ready:
                super().closeEvent(event)
                return
            if self._close_requested:
                event.ignore()
                return
            self._close_requested = True
            if self._voice_resources_active:
                event.ignore()
                self.set_session_state("stopping", "正在关闭窗口，等待当前任务结束并释放声音设备…")
                self.centralWidget().setEnabled(False)
                self.closing.emit()
                return
        self.closing.emit()
        if not self.prototype and not self.closing_ready and self._voice_resources_active:
            # shutdown() may discover a still-running device enumeration only
            # after this idle window has emitted its close request.
            event.ignore()
            self.centralWidget().setEnabled(False)
            return
        super().closeEvent(event)
