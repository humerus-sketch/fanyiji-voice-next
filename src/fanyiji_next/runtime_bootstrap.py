"""Prepare missing program runtimes before importing any audio/model modules."""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import threading

from PySide6.QtCore import QObject, QTimer, Qt, Signal
from PySide6.QtWidgets import QApplication, QDialog, QHBoxLayout, QLabel, QProgressBar, QPushButton, QVBoxLayout


APPLICATION = "fanyiji-voice-next"
DESCRIPTOR_PATH = Path("_updater/runtime-bootstrap.json")
READY_PATH = Path("_updater/runtime-ready.json")
MAX_DESCRIPTOR_BYTES = 2 * 1024 * 1024


def _read_object(path: Path) -> dict:
    for target in (path, *path.parents):
        if target.is_symlink() or (hasattr(target, "is_junction") and target.is_junction()):
            raise ValueError("运行库描述不得使用链接。")
    with path.open("rb") as stream:
        data = stream.read(MAX_DESCRIPTOR_BYTES + 1)
    if len(data) > MAX_DESCRIPTOR_BYTES:
        raise ValueError("运行库描述超过大小限制。")
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("运行库描述含重复字段。")
            result[key] = value
        return result
    value = json.loads(data.decode("utf-8"), object_pairs_hook=unique)
    if not isinstance(value, dict):
        raise ValueError("运行库描述格式无效。")
    return value


def read_bootstrap_descriptor(install_dir: Path) -> dict:
    from .app_update import parse_runtime_bundle_descriptor

    descriptor = _read_object(Path(install_dir) / DESCRIPTOR_PATH)
    parse_runtime_bundle_descriptor(descriptor)
    return descriptor


def runtime_is_ready(install_dir: Path, descriptor: dict) -> bool:
    try:
        marker = _read_object(Path(install_dir) / READY_PATH)
    except (OSError, ValueError, UnicodeError):
        return False
    return (type(marker.get("schema_version")) is int
            and marker == {"schema_version": 1, "application": APPLICATION,
                           "version": descriptor["version"], "build_id": descriptor["build_id"]})


def _default_cache() -> Path:
    base = Path(os.environ.get("LOCALAPPDATA") or Path.home() / "AppData/Local")
    return base / "FanyiJiVoiceNext/updates"


def configure_bootstrap_application(app):
    from PySide6.QtGui import QFont, QFontDatabase

    app.setApplicationName("FanyiJiVoiceNext")
    app.setApplicationDisplayName("翻译姬 · 准备运行库")
    fonts = Path(os.environ.get("SystemRoot", "C:/Windows")) / "Fonts"
    for name in ("msyh.ttc", "msyhbd.ttc"):
        path = fonts / name
        if path.is_file():
            QFontDatabase.addApplicationFont(str(path))
    app.setFont(QFont("Microsoft YaHei UI", 10))


class _BootstrapSignals(QObject):
    completed = Signal(object)
    progress = Signal(object)
    phase = Signal(object)


class RuntimeBootstrapDialog(QDialog):
    """Qt and stdlib only; preparation and the installer handshake run off-thread."""

    def __init__(self, install_dir, descriptor=None, restart_args=(), *, cache_dir=None,
                 download=None, prepare=None, launcher=None, auto_start=True):
        super().__init__()
        self.install_dir = Path(install_dir)
        self.descriptor = descriptor
        self.restart_args = tuple(restart_args)
        self.cache_dir = Path(cache_dir) if cache_dir is not None else _default_cache()
        self._download_fn, self._prepare_fn, self._launch_fn = download, prepare, launcher
        self._cancel = threading.Event()
        self._worker = None
        self._generation = 0
        self._closed = False
        self._installing = False
        self.last_error = ""
        self._signals = _BootstrapSignals(self)
        self._signals.completed.connect(self._completed, Qt.ConnectionType.QueuedConnection)
        self._signals.progress.connect(self._progress, Qt.ConnectionType.QueuedConnection)
        self._signals.phase.connect(self._phase, Qt.ConnectionType.QueuedConnection)
        self.setWindowTitle("翻译姬 · 准备运行库")
        self.resize(560, 240)
        layout = QVBoxLayout(self)
        heading = QLabel("正在补齐识别与翻译运行库")
        heading.setStyleSheet("font-size: 16px; font-weight: 600;")
        layout.addWidget(heading)
        hint = QLabel("完成后自动重新打开软件。已下载的模型和个人设置会保留。")
        hint.setWordWrap(True)
        layout.addWidget(hint)
        self.status_label = QLabel("运行库尚未就绪，正在准备下载。")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 1000)
        layout.addWidget(self.progress_bar)
        self.notice = QLabel("")
        self.notice.setWordWrap(True)
        self.notice.setStyleSheet("color: #8f3f2d;")
        self.notice.hide()
        layout.addWidget(self.notice)
        self._notice_timer = QTimer(self)
        self._notice_timer.setSingleShot(True)
        self._notice_timer.timeout.connect(self.notice.hide)
        row = QHBoxLayout()
        self.retry_button = QPushButton("重试")
        self.retry_button.clicked.connect(self.start_download)
        row.addWidget(self.retry_button)
        row.addStretch()
        self.cancel_button = QPushButton("取消")
        self.cancel_button.clicked.connect(self.reject)
        row.addWidget(self.cancel_button)
        layout.addLayout(row)
        if auto_start:
            QTimer.singleShot(0, self.start_download)

    def _show_error(self, message):
        self.last_error = str(message)
        self.retry_button.setToolTip(self.last_error)
        if self.notice.text() != self.last_error or self.notice.isHidden() or not self._notice_timer.isActive():
            self.notice.setText(self.last_error)
            self.notice.show()
            self._notice_timer.start(3000)
        self.status_label.setText("运行库尚未准备完成，可点击重试；现有模型和设置未更改。")
        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            report = {"application": APPLICATION, "runtime_bootstrap_error": self.last_error,
                      "audio_capture": False, "models_loaded": False, "settings_written": False}
            (self.cache_dir / "runtime-bootstrap-error.json").write_text(
                json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        except OSError:
            pass

    def _run(self, kind, operation):
        self._generation += 1
        generation, cancelled = self._generation, self._cancel
        def worker():
            result = error = None
            try:
                result = operation(generation, cancelled)
            except Exception as exc:
                error = exc
            try:
                self._signals.completed.emit((generation, kind, result, error, cancelled))
            except RuntimeError:
                cancelled.set()
        self._worker = threading.Thread(target=worker, name="fanyiji-runtime-bootstrap", daemon=True)
        self._worker.start()

    def start_download(self):
        if self._closed or self._worker is not None or self._installing:
            return
        self._cancel = threading.Event()
        self._notice_timer.stop()
        self.notice.hide()
        self.retry_button.setEnabled(False)
        self.progress_bar.setValue(0)
        self.status_label.setText("正在下载并校验运行库…")
        def operation(generation, cancelled):
            from .app_update import download_runtime_bundle, prepare_runtime_bundle, UpdateCancelled

            descriptor = self.descriptor or read_bootstrap_descriptor(self.install_dir)
            self.descriptor = descriptor
            def progress(done, total):
                try:
                    self._signals.progress.emit((generation, done, total))
                except RuntimeError:
                    cancelled.set()
            downloaded = (self._download_fn or download_runtime_bundle)(
                descriptor, self.cache_dir, progress_cb=progress, cancel_event=cancelled)
            if cancelled.is_set():
                raise UpdateCancelled("运行库准备已取消。")
            self._signals.phase.emit((generation, "正在校验并准备安装运行库…"))
            prepared = (self._prepare_fn or prepare_runtime_bundle)(downloaded, self.install_dir, cache_dir=self.cache_dir)
            if cancelled.is_set():
                raise UpdateCancelled("运行库准备已取消。")
            return prepared
        self._run("prepare", operation)

    def _progress(self, payload):
        generation, done, total = payload
        if not self._closed and generation == self._generation and total > 0:
            self.progress_bar.setValue(min(1000, max(0, int(done * 1000 / total))))
            self.status_label.setText(f"正在下载运行库：{done / 1024**3:.2f} / {total / 1024**3:.2f} GiB")

    def _phase(self, payload):
        generation, text = payload
        if not self._closed and generation == self._generation:
            self.status_label.setText(text)

    def _completed(self, payload):
        generation, kind, result, error, cancelled = payload
        if self._closed or generation != self._generation:
            return
        self._worker = None
        if cancelled.is_set():
            self.status_label.setText("已取消，可点击重试继续；现有模型和设置未更改。")
            self.retry_button.setEnabled(True)
            return
        if error is not None:
            self._installing = False
            self.cancel_button.setEnabled(True)
            self.retry_button.setEnabled(True)
            self._show_error(error)
            return
        if kind == "prepare":
            # Commit the install only after the queued cancellation check. From
            # this point the helper owns the transaction, so closing must wait.
            self._installing = True
            self.cancel_button.setEnabled(False)
            self.status_label.setText("正在退出并安装运行库，完成后自动重新打开…")
            def install(_generation, _cancelled):
                from .app_update import launch_helper
                return (self._launch_fn or launch_helper)(result, restart_args=self.restart_args, frozen=True)
            self._run("install", install)
        else:
            self._closed = True
            self._installing = False
            self.accept()

    def reject(self):
        if self._installing:
            return
        self._closed = True
        self._cancel.set()
        super().reject()

    def closeEvent(self, event):
        if self._installing:
            event.ignore()
            return
        self._closed = True
        self._cancel.set()
        super().closeEvent(event)


def run_runtime_bootstrap_if_needed(argv=None, *, install_dir=None, frozen=None):
    """Return None to continue startup, or an exit code without importing app."""
    packaged = bool(getattr(sys, "frozen", False)) if frozen is None else frozen
    if not packaged:
        return None
    install = Path(install_dir) if install_dir is not None else Path(sys.executable).absolute().parent
    path = install / DESCRIPTOR_PATH
    if not path.exists() and not path.is_symlink():
        # Full runtime archives carry a ready marker, not a self-referential
        # descriptor containing their own archive SHA256.
        return None
    error = None
    try:
        descriptor = read_bootstrap_descriptor(install)
        if runtime_is_ready(install, descriptor):
            return None
    except Exception as exc:
        descriptor, error = None, exc
    app = QApplication.instance() or QApplication(sys.argv[:1])
    configure_bootstrap_application(app)
    dialog = RuntimeBootstrapDialog(install, descriptor, sys.argv[1:] if argv is None else argv,
                                    auto_start=error is None)
    if error is not None:
        dialog._show_error(error)
    result = dialog.exec()
    dialog.deleteLater()
    app.processEvents()
    return 0 if result == QDialog.DialogCode.Accepted else 2
