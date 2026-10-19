"""Download the two supported model bundles without changing runtime settings."""
from __future__ import annotations

import atexit
from html import escape
from pathlib import Path
import threading
import time
from typing import Callable, Mapping

from PySide6.QtCore import QThread, QTimer, Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox, QDialog, QFrame, QHBoxLayout, QLabel, QProgressBar,
    QPushButton, QScrollArea, QVBoxLayout, QWidget,
)

from .model_catalog import MODEL_CATALOG, MODEL_COMBINATIONS
from .model_download import DownloadCancelled, ModelDownloadError, ModelDownloadManager
from .studio_theme import STUDIO_STYLE


# Jobs have no QObject parent: closing/deleting a dialog must not destroy its
# running QThread. They release themselves after cooperative cancellation.
_ACTIVE_JOBS: set[_ModelJob] = set()


def _finish_jobs_at_exit() -> None:
    jobs = tuple(_ACTIVE_JOBS)
    for job in jobs:
        job.cancel.set()
    # The event loop has ended here; this cannot freeze an open dialog. Network
    # reads have backend timeouts, so threads can finish without terminate().
    for job in jobs:
        try:
            job.wait()
        except RuntimeError:
            pass


atexit.register(_finish_jobs_at_exit)


def _bytes(value: int | float) -> str:
    amount = max(0.0, float(value))
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if amount < 1024 or unit == "TiB":
            return f"{amount:.1f} {unit}" if unit != "B" else f"{int(amount)} B"
        amount /= 1024
    raise AssertionError("unreachable")


def _label(text: str = "", *, name: str = "muted", wrap: bool = True) -> QLabel:
    label = QLabel(text)
    label.setObjectName(name)
    label.setWordWrap(wrap)
    label.setTextFormat(Qt.TextFormat.PlainText)
    return label


def _button(text: str, *, primary: bool = False) -> QPushButton:
    button = QPushButton(text)
    button.setObjectName("primaryButton" if primary else "quietButton")
    button.setCursor(Qt.CursorShape.PointingHandCursor)
    return button


class _ModelJob(QThread):
    progress = Signal(object)
    completed = Signal(object)
    phase = Signal(str)

    def __init__(self, manager, model_ids, *, action: str, bundle: str = "", acknowledged=False):
        super().__init__()
        self.manager = manager
        self.model_ids = tuple(model_ids)
        self.action = action
        self.bundle = bundle
        self.acknowledged = acknowledged
        self.cancel = threading.Event()

    def run(self) -> None:
        states = {}
        try:
            for model_id in self.model_ids:
                if self.cancel.is_set():
                    raise DownloadCancelled("已取消")
                self.phase.emit(model_id)
                states[model_id] = self.manager.inspect(model_id)
                if self.action == "download" and not states[model_id].ready:
                    self.manager.download(
                        model_id, progress=self.progress.emit, cancel_event=self.cancel,
                        license_acknowledged=self.acknowledged,
                    )
                    if self.cancel.is_set():
                        raise DownloadCancelled("已取消")
                    states[model_id] = self.manager.inspect(model_id)
                    if not states[model_id].ready:
                        raise ModelDownloadError("下载后的模型未通过完整校验，请重试。")
            if self.cancel.is_set():
                raise DownloadCancelled("已取消")
            self.completed.emit({"action": self.action, "bundle": self.bundle,
                                 "states": states, "cancel_event": self.cancel})
        except DownloadCancelled:
            self.completed.emit({"action": self.action, "bundle": self.bundle,
                                 "states": states, "cancelled": True})
        except Exception as exc:
            self.completed.emit({"action": self.action, "bundle": self.bundle,
                                 "states": states, "error": str(exc)})


class ModelDownloadDialog(QDialog):
    """Model downloads only; the caller owns applying a selected A/B bundle.

    ``bundle_selected`` is emitted after a fresh complete bundle verification and
    a final capture-state check. The receiver must also enforce its runtime gate.
    All manager calls, including SHA256 inspection, run outside the GUI thread.
    """

    bundle_selected = Signal(str)

    def __init__(
        self, model_root: Path, *, manager=None,
        capture_active: Callable[[], bool] | None = None,
        catalog: Mapping | None = None, combinations: Mapping | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        # Preserve reparse-point identity for the backend's path safety checks.
        self.model_root = Path(model_root).absolute()
        self.manager = manager if manager is not None else ModelDownloadManager(self.model_root)
        self.catalog = MODEL_CATALOG if catalog is None else catalog
        self.combinations = MODEL_COMBINATIONS if combinations is None else combinations
        self._capture_callback = capture_active
        self._capture_active = False
        self._closed = False
        self._refresh_on_finish = False
        self._job: _ModelJob | None = None
        self._states: dict = {}
        self._retry_bundle = ""
        self._speed_sample: tuple[str, float, int] | None = None
        self._model_ids = tuple(dict.fromkeys(
            model_id for bundle in ("A", "B") for model_id in self.combinations[bundle]
        ))
        self._shared_model_ids = tuple(model_id for model_id in self._model_ids
                                       if all(model_id in self.combinations[bundle] for bundle in ("A", "B")))

        self.setWindowTitle("下载识别与翻译模型")
        self.setWindowModality(Qt.WindowModality.WindowModal)
        self.setStyleSheet(STUDIO_STYLE)
        self.resize(760, 720)
        self.setMinimumSize(580, 500)
        shell = QVBoxLayout(self)
        shell.setContentsMargins(20, 18, 20, 18)
        shell.setSpacing(10)
        shell.addWidget(_label("识别与翻译模型", name="brandName"))
        shell.addWidget(_label("选择方案 A 或 B。下载完成后点击“使用方案”，才会切换当前设置。"))
        self.root_label = _label(str(self.model_root), name="footerHint")
        self.root_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        shell.addWidget(_label("固定模型目录", name="smallCaps"))
        shell.addWidget(self.root_label)
        shell.addWidget(_label("模型保存在上方目录；默认目录名为“主程序文件名_models”，也可由启动配置指定共享目录。", name="footerHint"))

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        body = QWidget()
        content = QVBoxLayout(body)
        content.setContentsMargins(0, 0, 6, 0)
        content.setSpacing(10)
        self.bundle_status_labels = {}
        self.bundle_cards = {}
        self.download_buttons = {}
        self.apply_buttons = {}
        self.model_status_labels = {}
        self.shared_card = QFrame()
        self.shared_card.setObjectName("paper")
        shared_layout = QVBoxLayout(self.shared_card)
        shared_layout.setContentsMargins(16, 10, 16, 10)
        shared_layout.setSpacing(4)
        shared_layout.addWidget(_label("共用模型 · 仅下载一次", name="smallCaps"))
        for model_id in self._shared_model_ids:
            status = _label(self.catalog[model_id].display_name + "：正在检查…", name="footerHint")
            shared_layout.addWidget(status)
            self.model_status_labels[model_id] = status
        self.shared_card.setVisible(bool(self._shared_model_ids))
        content.addWidget(self.shared_card)
        for bundle in ("A", "B"):
            card = QFrame()
            card.setObjectName("paper")
            layout = QVBoxLayout(card)
            layout.setContentsMargins(16, 10, 16, 10)
            layout.setSpacing(4)
            layout.addWidget(_label(f"方案 {bundle}", name="brandName"))
            for model_id in self.combinations[bundle]:
                if model_id in self._shared_model_ids:
                    continue
                model_status = _label("识别：" + self.catalog[model_id].display_name + " · 正在检查…")
                layout.addWidget(model_status)
                self.model_status_labels[model_id] = model_status
            status = _label("正在检查本地文件…", name="footerHint")
            layout.addWidget(status)
            row = QHBoxLayout()
            download = _button(f"下载方案 {bundle}", primary=True)
            apply = _button(f"使用方案 {bundle}")
            download.clicked.connect(lambda _checked=False, value=bundle: self.download_bundle(value))
            apply.clicked.connect(lambda _checked=False, value=bundle: self.use_bundle(value))
            row.addWidget(download)
            row.addWidget(apply)
            layout.addLayout(row)
            content.addWidget(card)
            self.bundle_cards[bundle] = card
            self.bundle_status_labels[bundle] = status
            self.download_buttons[bundle] = download
            self.apply_buttons[bundle] = apply

        gated = [self.catalog[model_id] for model_id in self._model_ids
                 if self.catalog[model_id].requires_license_ack]
        self.license_label = _label("", name="footerHint")
        self.license_label.setTextFormat(Qt.TextFormat.RichText)
        self.license_label.setOpenExternalLinks(True)
        links = []
        for spec in gated:
            license_url = getattr(spec, "license_url", "") or "https://ai.google.dev/gemma/terms"
            links.append(
                f'<a href="{escape(license_url, quote=True)}">'
                f'{escape(spec.license_name)} 条款</a> · '
                f'<a href="{escape(spec.source_url, quote=True)}">第三方量化来源</a>'
            )
        self.license_label.setText("<br>".join(links))
        self.license_checkbox = QCheckBox("我已阅读并确认有权按上述条款使用 Gemma 及其第三方量化模型")
        self.license_checkbox.setChecked(False)
        self.license_checkbox.setVisible(bool(gated))
        self.license_checkbox.toggled.connect(self._refresh_controls)
        content.addStretch(1)
        scroll.setWidget(body)
        shell.addWidget(scroll, 1)
        shell.addWidget(self.license_label)
        shell.addWidget(self.license_checkbox)

        self.notice = _label("", name="notice")
        self.notice.hide()
        self._notice_timer = QTimer(self)
        self._notice_timer.setSingleShot(True)
        self._notice_timer.timeout.connect(self.notice.hide)
        shell.addWidget(self.notice)
        self.progress_label = _label("检查和校验在后台执行，不会加载模型。", name="footerHint")
        shell.addWidget(self.progress_label)
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 1000)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setFixedHeight(8)
        shell.addWidget(self.progress_bar)
        footer = QHBoxLayout()
        self.refresh_button = _button("重新检查")
        self.refresh_button.clicked.connect(self.refresh_states)
        self.cancel_button = _button("取消下载")
        self.cancel_button.clicked.connect(self.cancel_download)
        close_button = _button("关闭")
        close_button.clicked.connect(self.close)
        footer.addWidget(self.refresh_button)
        footer.addStretch(1)
        footer.addWidget(self.cancel_button)
        footer.addWidget(close_button)
        shell.addLayout(footer)
        self._control_timer = QTimer(self)
        self._control_timer.setInterval(250)
        self._control_timer.timeout.connect(self._refresh_controls)
        self._control_timer.start()
        self._refresh_controls()
        QTimer.singleShot(0, self.refresh_states)

    def set_capture_active(self, active: bool) -> None:
        self._capture_active = bool(active)
        self._refresh_controls()

    def _capturing(self) -> bool:
        if self._capture_active:
            return True
        if self._capture_callback is not None:
            try:
                return bool(self._capture_callback())
            except Exception:
                return True
        return False

    def _bundle_ready(self, bundle: str) -> bool:
        return all(self._states.get(model_id) is not None and self._states[model_id].ready
                   for model_id in self.combinations[bundle])

    def _requires_ack(self, bundle: str) -> bool:
        return any(self.catalog[model_id].requires_license_ack
                   and not getattr(self._states.get(model_id), "ready", False)
                   for model_id in self.combinations[bundle])

    def _refresh_controls(self) -> None:
        busy = self._job is not None
        for bundle in ("A", "B"):
            known = all(model_id in self._states for model_id in self.combinations[bundle])
            ready = self._bundle_ready(bundle)
            self.download_buttons[bundle].setEnabled(
                not self._closed and not busy and known and not ready
                and (not self._requires_ack(bundle) or self.license_checkbox.isChecked())
            )
            self.download_buttons[bundle].setText(
                f"重试方案 {bundle}" if self._retry_bundle == bundle else f"下载方案 {bundle}"
            )
            self.apply_buttons[bundle].setEnabled(
                not self._closed and not busy and ready and not self._capturing()
            )
            self.apply_buttons[bundle].setToolTip("请先停止采集，再切换方案" if self._capturing() else "")
        self.refresh_button.setEnabled(not self._closed and not busy)
        self.cancel_button.setEnabled(not self._closed and busy and not self._job.cancel.is_set())

    def _show_notice(self, message: str) -> None:
        if message and self.notice.text() == message and not self.notice.isHidden() and self._notice_timer.isActive():
            return
        self._notice_timer.stop()
        self.notice.setText(message)
        self.notice.setVisible(bool(message))
        if message:
            self._notice_timer.start(3000)

    def refresh_states(self) -> None:
        if self._closed or self._job is not None:
            return
        self._start_job("inspect", self._model_ids)

    def download_bundle(self, bundle: str) -> None:
        if self._closed or self._job is not None or bundle not in ("A", "B"):
            return
        if not all(model_id in self._states for model_id in self.combinations[bundle]):
            return
        if self._requires_ack(bundle) and not self.license_checkbox.isChecked():
            self._show_notice("请先阅读模型条款，并确认有权使用后再下载。")
            return
        if self._bundle_ready(bundle):
            return
        self._start_job("download", self.combinations[bundle], bundle=bundle)

    def use_bundle(self, bundle: str) -> None:
        if self._closed or self._job is not None or bundle not in ("A", "B"):
            return
        if self._capturing():
            self._show_notice("请先停止采集，再切换模型方案。")
            return
        if not self._bundle_ready(bundle):
            self._show_notice("请先下载并完成整套模型校验。")
            return
        self._start_job("apply", self.combinations[bundle], bundle=bundle)

    def _start_job(self, action: str, model_ids, *, bundle: str = "") -> None:
        self._show_notice("")
        self._speed_sample = None
        self.progress_bar.setValue(0)
        self.progress_label.setText("正在校验本地模型文件…")
        job = _ModelJob(self.manager, model_ids, action=action, bundle=bundle,
                        acknowledged=self.license_checkbox.isChecked())
        self._job = job
        _ACTIVE_JOBS.add(job)
        job.progress.connect(self._on_progress)
        job.phase.connect(self._on_phase)
        job.completed.connect(self._on_completed)
        job.finished.connect(self._on_finished)
        # These callbacks remain valid even when the dialog has been deleted.
        job.finished.connect(lambda task=job: _ACTIVE_JOBS.discard(task))
        job.finished.connect(job.deleteLater)
        self.destroyed.connect(job.cancel.set)
        job.start()
        self._refresh_controls()

    def _on_phase(self, model_id: str) -> None:
        if not self._closed:
            self.progress_label.setText("正在检查：" + self.catalog[model_id].display_name)

    def _on_progress(self, progress) -> None:
        if self._closed:
            return
        now = time.monotonic()
        done = max(0, int(progress.downloaded_bytes))
        total = max(0, int(progress.total_bytes))
        speed = 0.0
        old = self._speed_sample
        if old is not None and old[0] == progress.model_id and now > old[1] and done >= old[2]:
            speed = (done - old[2]) / (now - old[1])
        self._speed_sample = (progress.model_id, now, done)
        percent = min(100.0, 100 * done / total) if total else 0.0
        self.progress_bar.setValue(round(percent * 10))
        phase = {"downloading": "下载", "verifying": "校验", "ready": "已校验",
                 "retrying": "连接失败，正在重试", "switching_source": "正在切换下载源",
                 "quarantined": "已隔离校验失败的临时文件，准备重新下载"}.get(
            progress.status, str(progress.status))
        source = getattr(progress, "source_name", "")
        attempt = getattr(progress, "attempt", 0)
        source_index = getattr(progress, "source_index", 0)
        detail = source
        if source_index:
            detail += f" · 下载源 {source_index}"
        if attempt:
            detail += f" · 第 {attempt} 次尝试"
        self.progress_label.setText(
            f"{self.catalog[progress.model_id].display_name} · {phase}\n"
            f"{_bytes(done)} / {_bytes(total)}（{percent:.1f}%） · {_bytes(speed)}/s"
            + ("\n" + detail if detail else "")
        )

    def _on_completed(self, result: dict) -> None:
        if self._closed:
            return
        self._states.update(result["states"])
        self._render_states()
        bundle = result["bundle"]
        cancel_event = result.get("cancel_event")
        if result.get("cancelled") or (cancel_event is not None and cancel_event.is_set()):
            self._retry_bundle = bundle
            self.progress_label.setText("已取消。已完成文件和临时下载保留，可点击重试继续。")
        elif result.get("error"):
            self._retry_bundle = bundle
            self._show_notice("模型操作失败：" + result["error"])
            self.progress_label.setText("请检查错误后重试；不会自动切换模型设置。")
        elif result["action"] == "apply":
            if not self._bundle_ready(bundle):
                self._show_notice("模型文件未通过完整校验，请重新下载。")
            elif self._capturing():
                self._show_notice("采集已开始，请先停止采集再切换方案。")
            else:
                self.bundle_selected.emit(bundle)
                self.progress_label.setText(f"方案 {bundle} 已校验，已请求切换。")
        elif result["action"] == "download":
            self._retry_bundle = ""
            self.progress_bar.setValue(1000)
            self.progress_label.setText(f"方案 {bundle} 下载与校验完成。点击“使用方案 {bundle}”后才会切换。")
        else:
            self.progress_label.setText("本地检查完成。下载和切换均需手动点击。")

    def _render_states(self) -> None:
        for model_id, state in self._states.items():
            if state.ready:
                text = "已完成校验"
            else:
                text = f"尚需下载 {_bytes(state.remaining_bytes)}"
                if state.partial_bytes:
                    text += f" · 临时文件 {_bytes(state.partial_bytes)}（未完成校验）"
                if state.remaining_bytes == 0:
                    text = "文件已在本地，仍需完成校验和就绪标记"
            name = self.catalog[model_id].display_name
            self.model_status_labels[model_id].setText(
                name + "：" + text if model_id in self._shared_model_ids else "识别：" + name + " · " + text)
        for bundle in ("A", "B"):
            if self._bundle_ready(bundle):
                text = "整套模型已校验，可在停止采集后使用。"
            elif all(model_id in self._states for model_id in self.combinations[bundle]):
                missing = sum(self._states[model_id].remaining_bytes for model_id in self.combinations[bundle])
                text = f"整套尚需下载 {_bytes(missing)}"
            else:
                text = "本地检查未完成，可重新检查。"
            self.bundle_status_labels[bundle].setText(text)

    def _on_finished(self) -> None:
        job = self.sender()
        if self._job is job:
            self._job = None
        if not self._closed:
            self._refresh_controls()
            if self._refresh_on_finish:
                self._refresh_on_finish = False
                self.refresh_states()

    def showEvent(self, event) -> None:
        super().showEvent(event)
        if self._closed:
            self._closed = False
            self.license_checkbox.setChecked(False)
            self._control_timer.start()
            self._refresh_controls()
            if self._job is None:
                self.refresh_states()
            else:
                # A cancelled job may still be unwinding when the user reopens.
                self._refresh_on_finish = True

    def cancel_download(self) -> None:
        if self._job is not None:
            self._job.cancel.set()
            if not self._closed:
                self.progress_label.setText("正在取消，等待当前文件操作安全结束…")
                self._refresh_controls()

    def closeEvent(self, event) -> None:
        self._closed = True
        self.cancel_download()
        self._control_timer.stop()
        super().closeEvent(event)

    def reject(self) -> None:
        self._closed = True
        self.cancel_download()
        self._control_timer.stop()
        super().reject()
