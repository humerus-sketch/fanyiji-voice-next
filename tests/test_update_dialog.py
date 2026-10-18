"""Exercise the updater's real GUI/backend boundary without network or installation."""
from __future__ import annotations

import os
from pathlib import Path
import threading
import time
from types import SimpleNamespace

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QTimer, Signal
from PySide6.QtWidgets import QApplication, QWidget

from fanyiji_next import __version__
from fanyiji_next import app_update
from fanyiji_next import update_dialog as ui


class UpdateWindow(QWidget):
    closing = Signal()

    def __init__(self):
        super().__init__()
        self._voice_resources_active = False
        self._update_restart_args = ["--settings-file", "D:/用户/settings.json", "--require-settings",
                                     "--model-root", "D:/模型"]
        self.close_count = 0

    def closeEvent(self, event):
        self.close_count += 1
        self.closing.emit()
        super().closeEvent(event)


@pytest.fixture(scope="module")
def qt_app():
    return QApplication.instance() or QApplication([])


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def forbidden(*_args, **_kwargs):
        raise AssertionError("An updater GUI test attempted a real network request")

    monkeypatch.setattr(app_update.UpdateClient, "_open", forbidden)


@pytest.fixture
def window(qt_app):
    parent = UpdateWindow()
    yield parent
    for dialog in parent.findChildren(ui.UpdateDialog):
        dialog.close()
    parent.close()
    parent.deleteLater()
    qt_app.processEvents()


def settle(qt_app, predicate, timeout=3):
    deadline = time.monotonic() + timeout
    while not predicate():
        qt_app.processEvents()
        if time.monotonic() >= deadline:
            raise AssertionError("Updater GUI work did not finish within the bounded test wait")
        threading.Event().wait(.002)
    qt_app.processEvents()


def info():
    return app_update.UpdateInfo("9.0.0", "program.zip",
        f"https://github.com/{app_update.DEFAULT_REPOSITORY}/releases/download/v9.0.0/program.zip",
        "0" * 64, 1024 * 1024,
        f"https://github.com/{app_update.DEFAULT_REPOSITORY}/releases/tag/v9.0.0", "修复字幕更新。")


def test_default_dialog_constructs_real_client_for_configured_repository(window):
    dialog = ui.UpdateDialog(window, frozen=True)
    assert type(dialog.client) is app_update.UpdateClient
    assert dialog.client.repository == app_update.DEFAULT_REPOSITORY
    assert app_update.DEFAULT_REPOSITORY and "/" in app_update.DEFAULT_REPOSITORY
    assert dialog.check_button.isEnabled()
    assert not dialog.download_button.isEnabled()
    assert not dialog.install_button.isEnabled()


def test_update_error_expires_after_three_seconds_without_resetting_on_duplicate(window):
    dialog = ui.UpdateDialog(window, frozen=True)
    dialog._show_error("网络错误")
    assert dialog._error_timer.interval() == 3000
    remaining = dialog._error_timer.remainingTime()
    dialog._show_error("网络错误")
    assert dialog._error_timer.remainingTime() <= remaining
    dialog._error_timer.timeout.emit()
    assert not dialog.status_label.text()
    assert not dialog._closed and dialog.check_button.isEnabled()


def test_install_uses_module_helper_and_preserves_restart_arguments(window, monkeypatch):
    # A real client intentionally has no launch_helper method: this catches the
    # previous interface mismatch that permissive fake clients concealed.
    client = app_update.UpdateClient()
    dialog = ui.UpdateDialog(window, client=client, frozen=True)
    dialog.prepared = app_update.PreparedUpdate(Path("D:/cache/job.json"), Path("D:/app"), "9.0.0")
    calls = []
    monkeypatch.setattr(ui, "launch_helper", lambda prepared, **kwargs: calls.append((prepared, kwargs)))
    dialog.install()
    dialog.install()
    assert len(calls) == 1
    assert calls[0] == (dialog.prepared, {"restart_args": window._update_restart_args})
    assert window.close_count == 1
    assert dialog._closed and dialog._cancel.is_set()


def test_helper_failure_keeps_application_open_and_allows_retry(window, monkeypatch):
    dialog = ui.UpdateDialog(window, client=SimpleNamespace(), frozen=True)
    dialog.prepared = object()

    def fail(*_args, **_kwargs):
        raise app_update.UpdateError("更新助手 fixture 失败")

    monkeypatch.setattr(ui, "launch_helper", fail)
    dialog.install()
    assert window.close_count == 0 and not dialog._closed
    assert "更新助手 fixture 失败" in dialog.status_label.text()
    dialog._refresh_controls()
    assert dialog.install_button.isEnabled()


@pytest.mark.parametrize("frozen,active", [(False, False), (False, True), (True, True)])
def test_source_or_active_audio_cannot_install(window, monkeypatch, frozen, active):
    window._voice_resources_active = active
    dialog = ui.UpdateDialog(window, client=SimpleNamespace(), frozen=frozen)
    dialog.prepared = object()
    calls = []
    monkeypatch.setattr(ui, "launch_helper", lambda *_a, **_kw: calls.append(True))
    dialog._refresh_controls()
    assert not dialog.install_button.isEnabled()
    dialog.install()
    assert calls == [] and window.close_count == 0 and not dialog._closed
    if frozen and active:
        assert "停止翻译" in dialog.status_label.text()


def test_install_control_refreshes_after_audio_resources_finish(window, qt_app):
    dialog = ui.UpdateDialog(window, client=SimpleNamespace(), frozen=True)
    dialog.prepared = object()
    window._voice_resources_active = True
    dialog._refresh_controls()
    assert not dialog.install_button.isEnabled()
    window._voice_resources_active = False
    settle(qt_app, dialog.install_button.isEnabled)
    window._voice_resources_active = True
    settle(qt_app, lambda: not dialog.install_button.isEnabled())
    dialog.close()
    assert not any(timer.isActive() for timer in dialog.findChildren(QTimer))


@pytest.mark.parametrize("result", [None, "available"])
def test_check_runs_off_gui_thread_and_displays_result(window, qt_app, result):
    thread_ids, versions = [], []
    expected = None if result is None else info()

    def check(version):
        thread_ids.append(threading.get_ident())
        versions.append(version)
        return expected

    dialog = ui.UpdateDialog(window, client=SimpleNamespace(check=check), frozen=True)
    dialog.check()
    settle(qt_app, lambda: dialog._worker is None)
    assert thread_ids == [thread_ids[0]] and thread_ids[0] != threading.get_ident()
    assert versions == [__version__]
    assert dialog.info is expected and dialog.check_button.isEnabled()
    if expected is None:
        assert "最新版本" in dialog.status_label.text()
        assert not dialog.download_button.isEnabled()
    else:
        assert "9.0.0" in dialog.version_label.text()
        assert dialog.notes.toPlainText() == expected.notes
        assert dialog.download_button.isEnabled()


def test_busy_check_ignores_duplicate_actions(window, qt_app):
    entered, release = threading.Event(), threading.Event()
    calls = []

    def check(version):
        calls.append(version)
        entered.set()
        if not release.wait(3):
            raise AssertionError("The test did not release its check worker")
        return info()

    dialog = ui.UpdateDialog(window, client=SimpleNamespace(check=check), frozen=True)
    try:
        dialog.check()
        assert entered.wait(1)
        assert not dialog.check_button.isEnabled()
        dialog.check()
        dialog.download()
        assert len(calls) == 1 and dialog._generation == 1
    finally:
        release.set()
        settle(qt_app, lambda: dialog._worker is None)


def test_network_failure_is_shown_without_stopping_application(window, qt_app):
    def check(_version):
        raise app_update.UpdateError("GitHub fixture unavailable")

    dialog = ui.UpdateDialog(window, client=SimpleNamespace(check=check), frozen=True)
    dialog.check()
    settle(qt_app, lambda: dialog._worker is None)
    assert "GitHub fixture unavailable" in dialog.status_label.text()
    assert dialog.check_button.isEnabled()
    assert window.close_count == 0 and not dialog.download_button.isEnabled()


def test_download_and_prepare_run_off_gui_thread_without_installing(window, qt_app, tmp_path, monkeypatch):
    update_info, downloaded, prepared = info(), object(), object()
    calls = []

    def download(value, *, progress, cancelled):
        assert not cancelled()
        calls.append(("download", value, threading.get_ident()))
        progress(1, 2)
        return downloaded

    def prepare(value, target):
        calls.append(("prepare", value, target, threading.get_ident()))
        return prepared

    helpers = []
    monkeypatch.setattr(ui, "launch_helper", lambda *_a, **_kw: helpers.append(True))
    dialog = ui.UpdateDialog(window, client=SimpleNamespace(download=download, prepare=prepare),
                             frozen=True, install_dir=tmp_path)
    dialog.info = update_info
    dialog.download()
    settle(qt_app, lambda: dialog._worker is None)
    assert dialog.prepared is prepared and helpers == [] and window.close_count == 0
    assert calls[0][:2] == ("download", update_info)
    assert calls[1][:3] == ("prepare", downloaded, tmp_path)
    assert calls[0][-1] == calls[1][-1] != threading.get_ident()
    assert dialog.progress_bar.value() == 1000 and dialog.install_button.isEnabled()
    assert not dialog.download_button.isEnabled()


def test_source_execution_checks_updates_but_never_downloads(window, qt_app):
    calls = []
    client = SimpleNamespace(check=lambda _version: info(), download=lambda *_a, **_kw: calls.append(True))
    dialog = ui.UpdateDialog(window, client=client, frozen=False)
    dialog.check()
    settle(qt_app, lambda: dialog._worker is None)
    assert dialog.info is not None and "源码运行" in dialog.status_label.text()
    assert not dialog.download_button.isEnabled()
    dialog.download()
    assert calls == [] and dialog.prepared is None


def test_closing_during_download_cancels_and_never_prepares(window, qt_app):
    entered, release = threading.Event(), threading.Event()
    calls = []

    def download(_info, *, progress, cancelled):
        entered.set()
        if not release.wait(3):
            raise AssertionError("The test did not release its download worker")
        assert cancelled()
        progress(1, 1)
        calls.append("download-returned")
        return object()

    dialog = ui.UpdateDialog(window,
        client=SimpleNamespace(download=download, prepare=lambda *_a: calls.append("prepare")), frozen=True)
    dialog.info = info()
    try:
        dialog.download()
        assert entered.wait(1)
        worker = dialog._worker
        dialog.close()
        text, progress = dialog.status_label.text(), dialog.progress_bar.value()
    finally:
        release.set()
        worker.join(3)
        assert not worker.is_alive()
        qt_app.processEvents()
    assert calls == ["download-returned"] and dialog.prepared is None
    assert dialog._closed and dialog._cancel.is_set()
    assert dialog.status_label.text() == text and dialog.progress_bar.value() == progress


def test_stale_or_closed_signals_do_not_mutate_dialog(window, qt_app):
    dialog = ui.UpdateDialog(window, client=SimpleNamespace(), frozen=True)
    dialog._generation = 3
    original = dialog.status_label.text()
    dialog._signals.completed.emit((2, "check", info(), None))
    dialog._signals.progress.emit((2, 1, 1))
    qt_app.processEvents()
    assert dialog.info is None and dialog.status_label.text() == original
    assert dialog.progress_bar.value() == 0
    dialog.close()
    dialog._signals.completed.emit((3, "check", info(), None))
    dialog._signals.progress.emit((3, 1, 1))
    qt_app.processEvents()
    assert dialog.info is None and dialog.status_label.text() == original
    assert dialog.progress_bar.value() == 0


def test_open_reuses_current_dialog_and_parent_close_cancels_it(window, qt_app, monkeypatch):
    versions = []
    monkeypatch.setattr(app_update.UpdateClient, "check", lambda _self, version: versions.append(version))
    first = ui.open_update_dialog(window)
    settle(qt_app, lambda: first._worker is None)
    second = ui.open_update_dialog(window)
    assert second is first and versions == [__version__]
    window.close()
    assert first._closed and first._cancel.is_set()


def test_reopen_after_close_gets_fresh_generation_and_client(window, qt_app, monkeypatch):
    monkeypatch.setattr(app_update.UpdateClient, "check", lambda _self, _version: None)
    first = ui.open_update_dialog(window)
    settle(qt_app, lambda: first._worker is None)
    first.close()
    second = ui.open_update_dialog(window)
    settle(qt_app, lambda: second._worker is None)
    assert second is not first and not second._closed
    assert first._cancel.is_set() and not second._cancel.is_set()
    assert second.client is not first.client
