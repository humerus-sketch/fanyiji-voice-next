"""User-triggered application updates; network work never runs on Qt's thread."""
from __future__ import annotations

from pathlib import Path
import sys
import threading

from PySide6.QtCore import QObject, Qt, QTimer, Signal
from PySide6.QtWidgets import (QDialog, QHBoxLayout, QLabel, QPlainTextEdit,
                               QProgressBar, QPushButton, QVBoxLayout)

from . import __version__
from .app_update import DEFAULT_REPOSITORY, UpdateCancelled, UpdateClient, launch_helper


class _UpdateSignals(QObject):
    completed = Signal(object)
    progress = Signal(object)


class UpdateDialog(QDialog):
    def __init__(self, window, *, client=None, frozen=None, install_dir=None):
        super().__init__(window)
        self.window = window
        self.client = client or UpdateClient(DEFAULT_REPOSITORY)
        self.frozen = bool(getattr(sys, "frozen", False)) if frozen is None else frozen
        self.install_dir = Path(install_dir or Path(sys.executable).resolve().parent)
        self.info = self.prepared = None
        self._cancel = threading.Event()
        self._worker = None
        self._closed = False
        self._generation = 0
        self._signals = _UpdateSignals(self)
        self._signals.completed.connect(self._completed, Qt.ConnectionType.QueuedConnection)
        self._signals.progress.connect(self._progress, Qt.ConnectionType.QueuedConnection)
        self.setWindowTitle("翻译姬 · 软件更新")
        self.resize(540, 350)
        layout = QVBoxLayout(self)
        self.version_label = QLabel(f"当前版本 {__version__}")
        layout.addWidget(self.version_label)
        self.status_label = QLabel("点击检查更新，获取可用的新版本。")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)
        self.notes = QPlainTextEdit()
        self.notes.setReadOnly(True)
        self.notes.setPlaceholderText("版本说明会显示在这里。更新保留模型和用户设置。")
        layout.addWidget(self.notes, 1)
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 1000)
        self.progress_bar.setValue(0)
        layout.addWidget(self.progress_bar)
        controls = QHBoxLayout()
        self.check_button = QPushButton("检查更新")
        self.download_button = QPushButton("下载更新")
        self.install_button = QPushButton("安装并重启")
        self.close_button = QPushButton("关闭")
        for button in (self.check_button, self.download_button, self.install_button, self.close_button):
            controls.addWidget(button)
        layout.addLayout(controls)
        self.check_button.clicked.connect(self.check)
        self.download_button.clicked.connect(self.download)
        self.install_button.clicked.connect(self.install)
        self.close_button.clicked.connect(self.close)
        self._controls_timer = QTimer(self)
        self._controls_timer.setInterval(250)
        self._controls_timer.timeout.connect(self._refresh_controls)
        self._controls_timer.start()
        self._error_timer = QTimer(self)
        self._error_timer.setSingleShot(True)
        self._error_timer.timeout.connect(self._clear_error)
        self._last_error = ""
        self._refresh_controls()

    def _show_error(self, message):
        if message == self._last_error and self._error_timer.isActive() and self.status_label.text() == message:
            return
        self._last_error = message
        self.status_label.setText(message)
        self._error_timer.start(3000)

    def _clear_error(self):
        if self.status_label.text() == self._last_error:
            self.status_label.clear()

    def _busy(self):
        return self._worker is not None and self._worker.is_alive()

    def _refresh_controls(self):
        busy = self._busy()
        self.check_button.setEnabled(not busy)
        self.download_button.setEnabled(not busy and self.info is not None and self.prepared is None and self.frozen)
        active = bool(getattr(self.window, "_voice_resources_active", False))
        self.install_button.setEnabled(not busy and self.prepared is not None and self.frozen and not active)
        self.install_button.setToolTip("请先停止翻译并等待资源释放。" if active else "安装完成后自动重新打开软件。")
        if not self.frozen:
            self.download_button.setToolTip("源码运行仅检查版本；打包后的 EXE 支持下载和安装更新。")

    def _run(self, kind, operation):
        if self._closed or self._busy():
            return
        self._cancel = threading.Event()
        self._generation += 1
        generation = self._generation

        def work():
            result = error = None
            try:
                result = operation()
            except Exception as exc:
                error = exc
            try:
                self._signals.completed.emit((generation, kind, result, error))
            except RuntimeError:
                pass  # The application can close during a bounded HTTP read.

        self._worker = threading.Thread(target=work, name="fanyiji-app-update", daemon=True)
        self._worker.start()
        self.check_button.setEnabled(False)
        self.download_button.setEnabled(False)
        self.install_button.setEnabled(False)

    def check(self):
        if self._closed or self._busy():
            return
        self.info = self.prepared = None
        self.notes.clear()
        self.status_label.setText("正在检查更新…")
        self.progress_bar.setRange(0, 0)
        self._run("check", lambda: self.client.check(__version__))

    def download(self):
        if self._closed or self._busy() or not self.frozen or self.info is None:
            return
        info = self.info
        self.status_label.setText("正在下载程序更新；模型和设置保留。")
        self.progress_bar.setRange(0, 1000)
        self.progress_bar.setValue(0)
        generation = self._generation + 1

        def progress(done, total):
            try:
                self._signals.progress.emit((generation, done, total))
            except RuntimeError:
                self._cancel.set()

        def download_and_prepare():
            downloaded = self.client.download(info, progress=progress, cancelled=self._cancel.is_set)
            if self._cancel.is_set():
                raise UpdateCancelled("更新已取消。")
            return self.client.prepare(downloaded, self.install_dir)

        self._run("download", download_and_prepare)

    def _completed(self, payload):
        generation, kind, result, error = payload
        if self._closed or generation != self._generation:
            return
        self._worker = None
        self.progress_bar.setRange(0, 1000)
        if error is not None:
            self._show_error("更新已取消。" if isinstance(error, UpdateCancelled) else f"更新未完成：{error}")
            self.progress_bar.setValue(0)
        elif kind == "check":
            self.info = result
            if result is None:
                self.status_label.setText("当前已是最新版本。")
            else:
                size = result.size / (1024 * 1024)
                self.version_label.setText(f"当前 {__version__} → 新版本 {result.version}")
                self.status_label.setText(f"发现新版本，程序更新约 {size:.1f} MB。")
                self.notes.setPlainText(result.notes)
                if not self.frozen:
                    self.status_label.setText("发现新版本。当前以源码运行，请在打包后的软件内安装更新。")
            self.progress_bar.setValue(0)
        else:
            self.prepared = result
            self.progress_bar.setValue(1000)
            self.status_label.setText("下载和校验完成。停止翻译后，点击「安装并重启」。")
        self._refresh_controls()

    def _progress(self, payload):
        generation, done, total = payload
        if not self._closed and generation == self._generation and total > 0:
            self.progress_bar.setValue(min(1000, max(0, int(done * 1000 / total))))

    def install(self):
        if self._closed or self._busy() or self.prepared is None or not self.frozen:
            return
        if getattr(self.window, "_voice_resources_active", False):
            self._show_error("请先停止翻译，等待声音与模型资源释放后再安装。")
            self._refresh_controls()
            return
        try:
            launch_helper(self.prepared, restart_args=getattr(self.window, "_update_restart_args", ()))
        except Exception as exc:
            self._show_error(f"无法启动更新：{exc}")
            return
        self.status_label.setText("正在退出并安装更新…")
        self.install_button.setEnabled(False)
        self.close()
        self.window.close()

    def closeEvent(self, event):
        self._closed = True
        self._cancel.set()
        self._controls_timer.stop()
        self._error_timer.stop()
        super().closeEvent(event)


def open_update_dialog(window):
    dialog = getattr(window, "_update_dialog", None)
    if dialog is not None and not dialog._closed:
        dialog._refresh_controls()
        dialog.show()
        dialog.raise_()
        dialog.activateWindow()
        return dialog
    dialog = UpdateDialog(window)
    window._update_dialog = dialog
    window.closing.connect(dialog.close)
    dialog.show()
    dialog.check()
    return dialog
