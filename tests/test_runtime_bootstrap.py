"""The compatibility bootstrap never imports audio or loads weights."""
from __future__ import annotations

import importlib.util
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
import zipfile
from types import SimpleNamespace

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PySide6.QtWidgets import QApplication, QDialog
from PySide6.QtTest import QTest
from fanyiji_next import runtime_bootstrap as bootstrap

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


def descriptor():
    name = "runtime.zip.001"
    return {"schema_version": 1, "application": bootstrap.APPLICATION, "platform": "windows-x64",
            "kind": "runtime-bundle", "version": "0.5.3", "build_id": "voice-ab-0.5.3-20261019-r1",
            "name": "runtime.zip", "size": 1, "sha256": "a" * 64,
            "parts": [{"name": name, "size": 1, "sha256": "a" * 64,
                       "url": "https://github.com/humerus-sketch/fanyiji-voice-next/releases/download/v0.5.3/" + name}]}


def write_descriptor(folder):
    target = folder / bootstrap.DESCRIPTOR_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    value = descriptor()
    target.write_text(json.dumps(value), encoding="utf-8")
    return value


def settle(app, predicate, timeout=3):
    deadline = time.monotonic() + timeout
    while not predicate():
        app.processEvents()
        if time.monotonic() >= deadline:
            raise AssertionError("bootstrap worker did not finish")
        threading.Event().wait(.002)
    app.processEvents()


def test_module_import_and_source_entry_do_not_import_audio_or_numpy(tmp_path):
    code = ("import sys; sys.path.insert(0, 'src'); "
            "from fanyiji_next.runtime_bootstrap import run_runtime_bootstrap_if_needed as check; "
            "assert check(frozen=False) is None; "
            "assert not any(n == 'numpy' or n.startswith(('fanyiji_next.app', 'torch', 'funasr')) for n in sys.modules)")
    result = subprocess.run([sys.executable, "-B", "-c", code], cwd=ROOT,
                            env={**os.environ, "QT_QPA_PLATFORM": "offscreen"}, capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("changed", [{}, {"build_id": "old"}, {"application": "other"},
                                    {"version": "0.5.2"}, {"schema_version": True}, {"extra": 1}])
def test_only_exact_typed_ready_marker_bypasses_runtime_install(tmp_path, changed):
    value = write_descriptor(tmp_path)
    marker = {"schema_version": 1, "application": bootstrap.APPLICATION,
              "version": value["version"], "build_id": value["build_id"]}
    (tmp_path / bootstrap.READY_PATH).write_text(json.dumps(marker | changed), encoding="utf-8")
    assert bootstrap.runtime_is_ready(tmp_path, value) is (not changed)


def test_descriptor_is_validated_and_complete_runtime_or_source_needs_no_ui(tmp_path, monkeypatch):
    assert bootstrap.run_runtime_bootstrap_if_needed(frozen=False, install_dir=tmp_path) is None
    # Full runtime archives intentionally omit the self-referential descriptor.
    assert bootstrap.run_runtime_bootstrap_if_needed(frozen=True, install_dir=tmp_path) is None
    value = write_descriptor(tmp_path)
    assert bootstrap.read_bootstrap_descriptor(tmp_path) == value
    marker = {"schema_version": 1, "application": bootstrap.APPLICATION,
              "version": value["version"], "build_id": value["build_id"]}
    (tmp_path / bootstrap.READY_PATH).write_text(json.dumps(marker), encoding="utf-8")
    monkeypatch.setattr(bootstrap, "RuntimeBootstrapDialog", lambda *_a, **_kw: pytest.fail("ready runtime opened UI"))
    assert bootstrap.run_runtime_bootstrap_if_needed(frozen=True, install_dir=tmp_path) is None


def test_descriptor_bad_url_and_duplicate_fields_fail_closed(tmp_path):
    value = write_descriptor(tmp_path)
    value["parts"][0]["url"] = "https://evil.example/runtime.zip.001"
    (tmp_path / bootstrap.DESCRIPTOR_PATH).write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(Exception, match="地址"):
        bootstrap.read_bootstrap_descriptor(tmp_path)
    (tmp_path / bootstrap.DESCRIPTOR_PATH).write_text('{"version": "0.5.3", "version": "0.5.2"}', encoding="utf-8")
    with pytest.raises(ValueError, match="重复"):
        bootstrap.read_bootstrap_descriptor(tmp_path)


def test_run_gate_stops_before_importing_app(monkeypatch):
    spec = importlib.util.spec_from_file_location("bootstrap_test_entry", ROOT / "run.py")
    entry = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(entry)
    monkeypatch.setattr(sys, "argv", ["FanyiJiVoiceNext.exe", "--settings-file", "unchanged.json"])
    calls = []
    monkeypatch.setattr(bootstrap, "run_runtime_bootstrap_if_needed", lambda argv: calls.append(tuple(argv)) or 2)
    monkeypatch.setitem(sys.modules, "fanyiji_next.app", SimpleNamespace(main=lambda: pytest.fail("app imported before runtime ready")))
    assert entry.launch() == 2
    assert calls == [("--settings-file", "unchanged.json")]


def test_dialog_download_prepare_and_install_run_off_gui_thread_and_preserve_restart(app, tmp_path):
    calls = []
    prepared = object()
    def download(value, destination, *, progress_cb, cancel_event):
        calls.append(("download", threading.get_ident(), value, destination))
        progress_cb(1, 1)
        return "archive"
    def prepare(archive, install, *, cache_dir):
        calls.append(("prepare", threading.get_ident(), archive, install, cache_dir))
        return prepared
    def launch(result, *, restart_args, frozen):
        calls.append(("launch", threading.get_ident(), result, restart_args, frozen))
    dialog = bootstrap.RuntimeBootstrapDialog(tmp_path, descriptor(), ("--settings-file", "keep.json"),
        cache_dir=tmp_path / "cache", download=download, prepare=prepare, launcher=launch, auto_start=False)
    dialog.start_download()
    settle(app, lambda: dialog._closed)
    assert [row[0] for row in calls] == ["download", "prepare", "launch"]
    assert all(row[1] != threading.get_ident() for row in calls)
    assert calls[-1][2:] == (prepared, ("--settings-file", "keep.json"), True)
    assert dialog.result() == QDialog.DialogCode.Accepted
    assert not (tmp_path / bootstrap.READY_PATH).exists()


def test_dialog_real_multipart_and_staging_backend_preserves_models_and_profile(app, tmp_path, monkeypatch):
    from fanyiji_next import app_update as update

    value = descriptor()
    marker = {"schema_version": 1, "application": bootstrap.APPLICATION,
              "version": value["version"], "build_id": value["build_id"]}
    files = {"FanyiJiVoiceNext.exe": b"new program fixture",
             "_updater/runtime-ready.json": json.dumps(marker).encode(),
             "_internal/transformers/models/local_fixture.py": b"fixture = True\n"}
    manifest = {"schema_version": 1, "application": bootstrap.APPLICATION, "version": value["version"],
                "files": [{"path": path, "size": len(data), "sha256": hashlib.sha256(data).hexdigest()}
                          for path, data in files.items()]}
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w") as bundle:
        for path, data in files.items():
            bundle.writestr(path, data)
        bundle.writestr("update-manifest.json", json.dumps(manifest))
    body = stream.getvalue()
    value.update(size=len(body), sha256=hashlib.sha256(body).hexdigest(), parts=[])
    payloads = {}
    for index, start in enumerate(range(0, len(body), (len(body) + 2) // 3), 1):
        data = body[start:start + (len(body) + 2) // 3]
        name = f"runtime.zip.{index:03}"
        url = f"https://github.com/{update.DEFAULT_REPOSITORY}/releases/download/v0.5.3/{name}"
        value["parts"].append({"name": name, "url": url, "size": len(data), "sha256": hashlib.sha256(data).hexdigest()})
        payloads[url] = data
    opened, launched = [], []
    class Response(io.BytesIO):
        status = 200
        def __init__(self, data):
            super().__init__(data)
            self.headers = {"Content-Length": str(len(data))}
    def open_part(_client, url, **_kwargs):
        opened.append(url)
        return Response(payloads[url])
    monkeypatch.setattr(update.UpdateClient, "_open", open_part)
    install = tmp_path / "installed"
    (install / "Reader_models").mkdir(parents=True)
    (install / "settings").mkdir()
    (install / "FanyiJiVoiceNext.exe").write_bytes(b"old program")
    (install / "Reader_models/model.gguf").write_bytes(b"retained user model")
    (install / "settings/voice.settings.json").write_bytes(b"retained user settings")
    def handoff(prepared, *, restart_args, frozen):
        job, rows = update._load_job(prepared.job_path)
        assert len(rows) == 3 and job["version"] == "0.5.3"
        assert json.loads((Path(job["stage_dir"]) / bootstrap.READY_PATH).read_text()) == marker
        launched.append((tuple(restart_args), frozen))
    dialog = bootstrap.RuntimeBootstrapDialog(install, value, ("--require-settings",),
        cache_dir=tmp_path / "cache", launcher=handoff, auto_start=False)
    dialog.start_download()
    settle(app, lambda: dialog._closed)
    assert opened == list(payloads) and launched == [(("--require-settings",), True)]
    assert (install / "FanyiJiVoiceNext.exe").read_bytes() == b"old program"
    assert (install / "Reader_models/model.gguf").read_bytes() == b"retained user model"
    assert (install / "settings/voice.settings.json").read_bytes() == b"retained user settings"


def test_cancel_after_prepare_signal_before_delivery_does_not_launch(app, tmp_path):
    calls = []
    dialog = bootstrap.RuntimeBootstrapDialog(tmp_path, descriptor(), cache_dir=tmp_path / "cache",
        launcher=lambda *_a, **_kw: calls.append("launch"), auto_start=False)
    dialog._generation = 1
    cancelled = threading.Event()
    cancelled.set()
    dialog._completed((1, "prepare", object(), None, cancelled))
    assert not calls and not dialog._installing
    dialog.reject()


def test_close_cancels_download_without_preparing_or_starting_installer(app, tmp_path):
    entered, stopped = threading.Event(), threading.Event()
    calls = []
    def download(_descriptor, _destination, *, progress_cb, cancel_event):
        entered.set()
        assert cancel_event.wait(2)
        stopped.set()
        return "cancelled-archive"
    dialog = bootstrap.RuntimeBootstrapDialog(tmp_path, descriptor(), cache_dir=tmp_path / "cache",
        download=download, prepare=lambda *_a, **_kw: calls.append("prepare"),
        launcher=lambda *_a, **_kw: calls.append("launch"), auto_start=False)
    dialog.start_download()
    settle(app, entered.is_set)
    dialog.reject()
    settle(app, stopped.is_set)
    assert not calls and dialog._closed
    assert not (tmp_path / bootstrap.READY_PATH).exists()


def test_frozen_missing_ready_blocks_startup_and_preserves_restart_arguments(app, tmp_path, monkeypatch):
    write_descriptor(tmp_path)
    calls = []
    class Dialog:
        def __init__(self, install, value, restart, **kwargs):
            calls.append((install, value, tuple(restart), kwargs))
        def exec(self):
            return QDialog.DialogCode.Rejected
        def deleteLater(self):
            pass
    monkeypatch.setattr(bootstrap, "RuntimeBootstrapDialog", Dialog)
    arguments = ("--settings-file", "unchanged.json", "--require-settings")
    assert bootstrap.run_runtime_bootstrap_if_needed(arguments, install_dir=tmp_path, frozen=True) == 2
    assert calls[0][2] == arguments and calls[0][3] == {"auto_start": True}


def test_download_failure_can_retry_and_error_expires_after_three_seconds(app, tmp_path):
    attempts = []
    def download(*_a, **_kw):
        attempts.append(1)
        raise RuntimeError("本地测试网络失败")
    dialog = bootstrap.RuntimeBootstrapDialog(tmp_path, descriptor(), cache_dir=tmp_path / "cache",
        download=download, auto_start=False)
    dialog.show()
    dialog.start_download()
    settle(app, lambda: dialog._worker is None)
    assert len(attempts) == 1 and dialog.retry_button.isEnabled()
    assert dialog.notice.isVisible() and dialog._notice_timer.interval() == 3000
    QTest.qWait(1200)
    remaining = dialog._notice_timer.remainingTime()
    dialog._show_error("本地测试网络失败")
    assert dialog._notice_timer.remainingTime() <= remaining + 100
    QTest.qWait(2000)
    assert not dialog.notice.isVisible()
    report = json.loads((tmp_path / "cache/runtime-bootstrap-error.json").read_text(encoding="utf-8"))
    assert report["runtime_bootstrap_error"] == "本地测试网络失败"
    assert report["audio_capture"] is report["models_loaded"] is report["settings_written"] is False
    dialog.start_download()
    settle(app, lambda: dialog._worker is None)
    assert len(attempts) == 2
    dialog.reject()
