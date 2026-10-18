"""Exercise the download UI with bounded local fakes, without model weights."""
from __future__ import annotations

import os
from pathlib import Path
import threading
import time
from types import SimpleNamespace

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QCoreApplication, QEvent
from PySide6.QtWidgets import QApplication, QLabel, QWidget
from PySide6.QtTest import QTest

from fanyiji_next import model_download_dialog as ui


@pytest.fixture(scope="module")
def qt_app():
    return QApplication.instance() or QApplication([])


def settle(app, predicate, timeout=3):
    deadline = time.monotonic() + timeout
    while not predicate():
        app.processEvents()
        if time.monotonic() >= deadline:
            raise AssertionError("Model dialog job did not finish")
        threading.Event().wait(.002)
    app.processEvents()


def state(model_id, ready=False, partial=0):
    return SimpleNamespace(model_id=model_id, ready=ready, total_bytes=1000,
                           verified_bytes=1000 if ready else 0, partial_bytes=partial,
                           remaining_bytes=0 if ready else 1000-partial)


class FakeManager:
    def __init__(self):
        self.ready = set()
        self.downloads = []
        self.calls = []
        self.fail_once = False
        self.block = False
        self.entered = threading.Event()
        self.cancel_seen = threading.Event()
        self.release = threading.Event()
        self.hold_after_cancel = False
        self.progress_seen = threading.Event()

    def inspect(self, model_id):
        self.calls.append(("inspect", model_id, threading.get_ident()))
        return state(model_id, model_id in self.ready)

    def download(self, model_id, *, progress, cancel_event, license_acknowledged):
        self.calls.append(("download", model_id, threading.get_ident()))
        self.downloads.append((model_id, license_acknowledged))
        self.entered.set()
        if self.fail_once:
            self.fail_once = False
            raise RuntimeError("本地测试连接中断")
        progress(SimpleNamespace(model_id=model_id, downloaded_bytes=250,
                                 total_bytes=1000, status="downloading", file_path="weights.bin"))
        self.progress_seen.set()
        while self.block and not self.release.is_set():
            if cancel_event.is_set():
                self.cancel_seen.set()
                if not self.hold_after_cancel:
                    raise ui.DownloadCancelled("test cancel")
            threading.Event().wait(.003)
        if cancel_event.is_set():
            self.cancel_seen.set()
            raise ui.DownloadCancelled("test cancel")
        self.ready.add(model_id)
        progress(SimpleNamespace(model_id=model_id, downloaded_bytes=1000,
                                 total_bytes=1000, status="ready", file_path="weights.bin"))
        return Path("/unused")


@pytest.fixture
def catalog():
    result = {}
    for model_id, name, gated in (("q", "Qwen ASR", False), ("f", "Fun ASR", False),
                                  ("g", "TranslateGemma 第三方量化", True)):
        result[model_id] = SimpleNamespace(model_id=model_id, display_name=name,
            directory_name=model_id, requires_license_ack=gated, license_name="Gemma" if gated else "Apache-2.0",
            license_url="https://ai.google.dev/gemma/terms" if gated else "",
            source_url="https://huggingface.co/example/third-party", files=())
    return result


@pytest.fixture
def dialog(qt_app, catalog, tmp_path):
    manager = FakeManager()
    parent = QWidget()
    value = ui.ModelDownloadDialog(tmp_path / "FanyiJiVoiceNext_models", manager=manager,
        catalog=catalog, combinations={"A": ("q", "g"), "B": ("f", "g")}, parent=parent)
    settle(qt_app, lambda: value._job is None and len(value._states) == len(value._model_ids))
    yield value, manager
    manager.release.set()
    try:
        value.close()
    except RuntimeError:
        pass
    settle(qt_app, lambda: not ui._ACTIVE_JOBS)
    parent.close()
    parent.deleteLater()
    qt_app.processEvents()


def test_open_inspects_only_off_gui_thread_and_shows_fixed_root(dialog):
    value, manager = dialog
    assert not manager.downloads
    assert {call[0] for call in manager.calls} == {"inspect"}
    assert all(call[2] != threading.get_ident() for call in manager.calls)
    assert value.root_label.text() == str(value.model_root)
    assert value.root_label.text().endswith("FanyiJiVoiceNext_models")
    assert not value.license_checkbox.isChecked()
    assert "第三方量化来源" in value.license_label.text()
    assert "https://ai.google.dev/gemma/terms" in value.license_label.text()
    assert "尚需下载" in value.bundle_status_labels["A"].text()
    assert not value.apply_buttons["A"].isEnabled()


def test_unacknowledged_terms_block_even_programmatic_download(dialog):
    value, manager = dialog
    assert not value.download_buttons["A"].isEnabled()
    value.download_bundle("A")
    assert not manager.downloads
    assert value._job is None
    assert "条款" in value.notice.text()


def test_download_bundles_share_gemma_and_never_automatically_apply(dialog, qt_app):
    value, manager = dialog
    selections = []
    value.bundle_selected.connect(selections.append)
    value.license_checkbox.setChecked(True)
    value.download_buttons["A"].click()
    settle(qt_app, lambda: value._job is None)
    assert manager.downloads == [("q", True), ("g", True)]
    assert value.apply_buttons["A"].isEnabled()
    assert not selections
    value.download_buttons["B"].click()
    settle(qt_app, lambda: value._job is None)
    assert [model_id for model_id, _ack in manager.downloads] == ["q", "g", "f"]
    assert value._bundle_ready("B")
    assert value.progress_bar.value() == 1000
    assert not selections
    assert all(call[2] != threading.get_ident() for call in manager.calls)


def test_shared_models_appear_once_and_refresh_in_one_region(qt_app, catalog, tmp_path):
    catalog["v"] = SimpleNamespace(model_id="v", display_name="Silero VAD", requires_license_ack=False)
    manager = FakeManager()
    value = ui.ModelDownloadDialog(tmp_path / "Example_models", manager=manager, catalog=catalog,
        combinations={"A": ("q", "g", "v"), "B": ("f", "g", "v")})
    try:
        settle(qt_app, lambda: value._job is None and len(value._states) == 4)
        assert value._shared_model_ids == ("g", "v")
        assert set(value.model_status_labels) == {"q", "f", "g", "v"}
        labels = [label.text() for label in value.findChildren(QLabel)]
        assert sum("共用" in text for text in labels) == 1
        for model_id in ("g", "v"):
            assert value.shared_card.isAncestorOf(value.model_status_labels[model_id])
            assert sum(catalog[model_id].display_name in text for text in labels) == 1
        for bundle, asr_id in (("A", "q"), ("B", "f")):
            card_labels = [label.text() for label in value.bundle_cards[bundle].findChildren(QLabel)]
            assert any(catalog[asr_id].display_name in text for text in card_labels)
            assert all(catalog[model_id].display_name not in text
                       for model_id in ("g", "v") for text in card_labels)
            assert value.bundle_status_labels[bundle].text() == "整套尚需下载 2.9 KiB"
        manager.ready.update(("g", "v"))
        value.refresh_states()
        settle(qt_app, lambda: value._job is None)
        assert all("已完成校验" in value.model_status_labels[model_id].text() for model_id in ("g", "v"))
        assert all(status.text() == "整套尚需下载 1000 B" for status in value.bundle_status_labels.values())
        assert not manager.downloads
        assert all(call[2] != threading.get_ident() for call in manager.calls)
    finally:
        value.close()
        settle(qt_app, lambda: not ui._ACTIVE_JOBS)
        value.deleteLater()
        qt_app.processEvents()


def test_progress_shows_bytes_percent_and_speed_and_cancels_cooperatively(dialog, qt_app):
    value, manager = dialog
    value.license_checkbox.setChecked(True)
    manager.block = True
    value.download_bundle("A")
    settle(qt_app, lambda: manager.entered.is_set() and "25.0%" in value.progress_label.text())
    assert "250 B / 1000 B" in value.progress_label.text()
    assert "/s" in value.progress_label.text()
    assert value.progress_bar.value() == 250
    assert not value.download_buttons["B"].isEnabled()
    value.cancel_button.click()
    settle(qt_app, lambda: value._job is None)
    assert manager.cancel_seen.is_set()
    assert "已取消" in value.progress_label.text()
    assert "重试" in value.download_buttons["A"].text()
    assert not value._bundle_ready("A")


@pytest.mark.parametrize("status,message", [("retrying", "正在重试"),
    ("switching_source", "切换下载源"), ("quarantined", "已隔离校验失败")])
def test_retry_progress_is_chinese_and_includes_source_and_attempt(dialog, status, message):
    value, _manager = dialog
    value._on_progress(SimpleNamespace(model_id="q", downloaded_bytes=0, total_bytes=1000,
        status=status, source_name="ModelScope（国内）", source_index=2, attempt=1))
    text = value.progress_label.text()
    assert message in text and status not in text
    assert "ModelScope（国内）" in text and "下载源 2" in text and "第 1 次尝试" in text


def test_failure_can_retry_without_automatic_loop(dialog, qt_app):
    value, manager = dialog
    value.license_checkbox.setChecked(True)
    manager.fail_once = True
    value.download_bundle("A")
    settle(qt_app, lambda: value._job is None)
    assert len(manager.downloads) == 1
    assert "本地测试连接中断" in value.notice.text()
    assert value.download_buttons["A"].text() == "重试方案 A"
    value.download_buttons["A"].click()
    settle(qt_app, lambda: value._job is None)
    assert value._bundle_ready("A")


def test_apply_reverifies_and_capture_gate_is_checked_again(dialog, qt_app):
    value, manager = dialog
    manager.ready.update(("q", "g"))
    value.refresh_states()
    settle(qt_app, lambda: value._job is None)
    selected = []
    value.bundle_selected.connect(lambda bundle: selected.append((bundle, threading.get_ident())))
    value.set_capture_active(True)
    assert not value.apply_buttons["A"].isEnabled()
    value.use_bundle("A")
    assert not selected and value._job is None
    value.set_capture_active(False)
    previous = len(manager.calls)
    value.use_bundle("A")
    # A capture starting while background SHA verification runs must still block.
    value.set_capture_active(True)
    settle(qt_app, lambda: value._job is None)
    assert not selected
    assert len(manager.calls) == previous + 2
    value.set_capture_active(False)
    value.use_bundle("A")
    settle(qt_app, lambda: value._job is None)
    assert selected == [("A", threading.get_ident())]


def test_apply_detects_file_removed_since_previous_inspection(dialog, qt_app):
    value, manager = dialog
    manager.ready.update(("q", "g"))
    value.refresh_states()
    settle(qt_app, lambda: value._job is None)
    manager.ready.remove("g")
    selected = []
    value.bundle_selected.connect(selected.append)
    value.use_bundle("A")
    settle(qt_app, lambda: value._job is None)
    assert not selected
    assert "未通过完整校验" in value.notice.text()
    assert not value.apply_buttons["A"].isEnabled()


def test_close_and_delete_do_not_block_or_destroy_running_thread(dialog, qt_app):
    value, manager = dialog
    value.license_checkbox.setChecked(True)
    manager.block = True
    manager.hold_after_cancel = True
    value.download_bundle("A")
    settle(qt_app, lambda: manager.entered.is_set())
    job = value._job
    began = time.monotonic()
    value.close()
    assert time.monotonic() - began < .2
    settle(qt_app, lambda: manager.cancel_seen.is_set())
    assert job.isRunning() and job in ui._ACTIVE_JOBS
    value.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    qt_app.processEvents()
    manager.release.set()
    settle(qt_app, lambda: not ui._ACTIVE_JOBS)


def test_escape_reject_also_cancels(dialog, qt_app):
    value, manager = dialog
    value.license_checkbox.setChecked(True)
    manager.block = True
    value.download_bundle("A")
    settle(qt_app, lambda: manager.entered.is_set())
    value.reject()
    settle(qt_app, lambda: value._job is None)
    assert manager.cancel_seen.is_set()
    assert value._closed


@pytest.mark.parametrize("close_during_download", [False, True])
def test_reopen_refreshes_and_remains_usable(dialog, qt_app, close_during_download):
    value, manager = dialog
    value.show()
    qt_app.processEvents()
    value.license_checkbox.setChecked(True)
    if close_during_download:
        manager.block = True
        manager.hold_after_cancel = True
        value.download_bundle("A")
        settle(qt_app, lambda: manager.entered.is_set())
    previous = len(manager.calls)
    value.close()
    value.show()
    qt_app.processEvents()
    assert not value._closed and value._control_timer.isActive()
    assert not value.license_checkbox.isChecked()
    manager.release.set()
    settle(qt_app, lambda: value._job is None and not value._refresh_on_finish)
    assert len(manager.calls) >= previous + len(value._model_ids)
    value.license_checkbox.setChecked(True)
    assert value.download_buttons["A"].isEnabled()
    value.download_buttons["A"].click()
    settle(qt_app, lambda: value._job is None)
    assert value._bundle_ready("A")


def test_failed_capture_callback_fails_closed(dialog):
    value, _manager = dialog
    value._capture_callback = lambda: (_ for _ in ()).throw(RuntimeError("unavailable"))
    assert value._capturing()


def test_download_error_notice_expires_without_repeated_error_extending_it(dialog, qt_app):
    value, _manager = dialog
    value.show()
    value._show_notice("模型操作失败：本地测试连接中断")
    qt_app.processEvents()
    assert value.notice.isVisible() and value._notice_timer.interval() == 3000
    QTest.qWait(1200)
    remaining = value._notice_timer.remainingTime()
    value._show_notice("模型操作失败：本地测试连接中断")
    assert value._notice_timer.remainingTime() <= remaining + 100
    QTest.qWait(2000)
    assert not value.notice.isVisible()
    assert value.notice.text() == "模型操作失败：本地测试连接中断"


def test_apply_cancelled_during_final_inspection_cannot_succeed():
    class CancelLastInspect:
        def inspect(self, model_id):
            if model_id == "g":
                job.cancel.set()
            return state(model_id, ready=True)

    job = ui._ModelJob(CancelLastInspect(), ("q", "g"), action="apply", bundle="A")
    results = []
    job.completed.connect(results.append)
    job.run()
    assert len(results) == 1 and results[0]["cancelled"]


def test_apply_cancelled_after_worker_emit_before_gui_delivery_cannot_switch(dialog, qt_app):
    value, manager = dialog
    manager.ready.update(("q", "g"))
    selections = []
    value.bundle_selected.connect(selections.append)
    # Simulate the exact queued-signal gap without loading weights or racing timers.
    worker = ui._ModelJob(manager, ("q", "g"), action="apply", bundle="A")
    results = []
    worker.completed.connect(results.append)
    worker.run()
    assert not results[0]["cancel_event"].is_set()
    worker.cancel.set()
    value._on_completed(results[0])
    assert not selections
    assert "已取消" in value.progress_label.text()


def test_real_catalog_empty_root_is_read_only_and_never_starts_download(qt_app, tmp_path, monkeypatch):
    def forbidden(*_args, **_kwargs):
        raise AssertionError("Opening a model dialog must never download a model")

    monkeypatch.setattr(ui.ModelDownloadManager, "download", forbidden)
    root = tmp_path / "FreshApplication_models"
    value = ui.ModelDownloadDialog(root)
    try:
        settle(qt_app, lambda: value._job is None and len(value._states) == len(value._model_ids))
        assert not root.exists()
        assert not value.license_checkbox.isChecked()
        assert all(not status.ready for status in value._states.values())
        assert not value.apply_buttons["A"].isEnabled()
        assert not value.apply_buttons["B"].isEnabled()
    finally:
        value.close()
        value.deleteLater()
        qt_app.processEvents()
