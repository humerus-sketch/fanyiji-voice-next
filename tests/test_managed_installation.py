"""Managed moves use tiny temporary projects; no real program or weights run."""
from dataclasses import replace
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import zipfile

import pytest

from fanyiji_next import app_update as update
from fanyiji_next import managed_installation as managed
from fanyiji_next import update_cleanup as cleanup

OLD = "0.4.8"
NEW = "0.5.0"
REPO = "example/fanyiji-voice-next"


def project(tmp_path):
    root = tmp_path / "project"
    install = root / "dist" / OLD / "FanyiJiVoiceNext"
    install.mkdir(parents=True)
    (root / "pyproject.toml").write_text('[project]\nname = "fanyiji-voice-next"\n', encoding="utf-8")
    (install / "FanyiJiVoiceNext.exe").write_bytes(b"old program")
    (install / "build-info.json").write_text(json.dumps({"application": update.APPLICATION, "version": OLD}), encoding="utf-8")
    (install / "settings").mkdir()
    (install / "settings/voice.settings.json").write_bytes(b"internal user settings")
    (root / "FanyiJiVoiceNext_models").mkdir()
    (root / "FanyiJiVoiceNext_models/model.gguf").write_bytes(b"shared user model")
    (root / "settings").mkdir()
    (root / "settings/voice.settings.json").write_bytes(b"shared B profile")
    value = {"schema_version": 1, "application": update.APPLICATION, "role": "primary", "current_version": OLD,
             "program_relative": f"dist/{OLD}/FanyiJiVoiceNext", "model_relative": "FanyiJiVoiceNext_models"}
    (root / managed.REGISTRY_NAME).write_text(json.dumps(value), encoding="utf-8")
    return root, install


def prepare(monkeypatch, tmp_path, install, *, extra_files=None, runtime=False):
    files = {"FanyiJiVoiceNext.exe": b"new program",
             "build-info.json": json.dumps({"application": update.APPLICATION, "version": NEW}).encode(),
             "_internal/new-library.dll": b"new library"}
    files.update(extra_files or {})
    files = {name: body for name, body in files.items() if body is not None}
    path = tmp_path / "program.zip"
    manifest = {"schema_version": 1, "application": update.APPLICATION, "version": NEW,
                "files": [{"path": name, "sha256": __import__("hashlib").sha256(body).hexdigest(), "size": len(body)}
                          for name, body in files.items()]}
    with zipfile.ZipFile(path, "w") as package:
        for name, body in files.items():
            package.writestr(name, body)
        package.writestr("update-manifest.json", json.dumps(manifest))
    info = update.UpdateInfo(NEW, path.name, f"https://github.com/{REPO}/releases/download/v{NEW}/{path.name}",
                             update._digest(path), path.stat().st_size, f"https://github.com/{REPO}/releases/tag/v{NEW}")
    instance = update.UpdateClient(REPO, tmp_path / "updates")
    class Response(io.BytesIO):
        status = 200
        headers = {"Content-Length": str(info.size)}
    monkeypatch.setattr(instance, "_open", lambda url: Response(path.read_bytes()))
    downloaded = instance.download(info)
    return instance.prepare(downloaded, install, _runtime_build_id="managed-test-r1" if runtime else None), downloaded


def test_registered_model_root_matches_only_exact_primary_and_filename(tmp_path):
    root, install = project(tmp_path)
    assert managed.managed_model_root(install, "FanyiJiVoiceNext.exe") == root / "FanyiJiVoiceNext_models"
    assert managed.managed_model_root(install, "Renamed.exe") is None
    assert managed.managed_model_root(root / "dist/0.5.0/FanyiJiVoiceNext") is None
    value = json.loads((root / managed.REGISTRY_NAME).read_text())
    value["role"] = "verification"
    (root / managed.REGISTRY_NAME).write_text(json.dumps(value))
    assert managed.managed_model_root(install) is None


@pytest.mark.parametrize("changes", [
    {"program_relative": "../other/FanyiJiVoiceNext"}, {"program_relative": "C:/other/FanyiJiVoiceNext"},
    {"model_relative": "../models"}, {"model_relative": "C:/models"},
    {"current_version": "0.4.9"}, {"role": []}, {"schema_version": True}, {"unexpected": True},
])
def test_malformed_registration_never_selects_or_moves_arbitrary_paths(tmp_path, changes):
    root, install = project(tmp_path)
    path = root / managed.REGISTRY_NAME
    value = json.loads(path.read_text())
    value.update(changes)
    path.write_text(json.dumps(value))
    with pytest.raises((ValueError, OSError)):
        managed.managed_model_root(install)
    assert (install / "FanyiJiVoiceNext.exe").read_bytes() == b"old program"


def test_project_metadata_name_is_required_and_missing_registry_is_unmanaged(tmp_path):
    root, install = project(tmp_path)
    (root / "pyproject.toml").write_text('[project]\nname = "another-project"\n')
    assert managed.managed_model_root(install) is None
    (root / "pyproject.toml").write_text('[project]\nname = "fanyiji-voice-next"\n')
    (root / managed.REGISTRY_NAME).rename(root / "unregistered.json")
    assert managed.managed_model_root(install) is None


def test_update_moves_only_registered_program_and_restarts_new_version_path(monkeypatch, tmp_path):
    root, previous = project(tmp_path)
    (previous.parent / "do-not-move.txt").write_bytes(b"sibling user data")
    prepared, _ = prepare(monkeypatch, tmp_path, previous)
    assert (prepared.cache_dir / managed.RELOCATION_NAME).is_file()
    assert set(json.loads(prepared.job_path.read_text())) == {
        "schema_version", "application", "version", "install_dir", "stage_dir", "backup_dir", "executable", "pid", "restart_args"}
    commands = []
    backup = update.apply_update(prepared.job_path, start=lambda command, **kw: commands.append((command, kw)))
    current = root / "dist" / NEW / "FanyiJiVoiceNext"
    assert not previous.exists()
    assert commands[0][0] == [str(current / "FanyiJiVoiceNext.exe")]
    assert commands[0][1]["cwd"] == str(current)
    assert (current / "FanyiJiVoiceNext.exe").read_bytes() == b"new program"
    assert (current / "settings/voice.settings.json").read_bytes() == b"internal user settings"
    assert (previous.parent / "do-not-move.txt").read_bytes() == b"sibling user data"
    assert (root / "FanyiJiVoiceNext_models/model.gguf").read_bytes() == b"shared user model"
    assert (root / "settings/voice.settings.json").read_bytes() == b"shared B profile"
    assert (backup / "FanyiJiVoiceNext.exe").read_bytes() == b"old program"
    assert managed.managed_model_root(current) == root / "FanyiJiVoiceNext_models"
    update._load_job(prepared.job_path)


def test_new_path_job_sidecar_and_pending_pointer_ack_clean_only_current_transaction(monkeypatch, tmp_path):
    root, previous = project(tmp_path)
    (previous / "_updater").mkdir()
    original_pointer = {"schema_version": 1, "application": update.APPLICATION, "install_dir": str(previous), "receipts": []}
    (previous / "_updater" / cleanup.PENDING_NAME).write_text(json.dumps(original_pointer))
    prepared, downloaded = prepare(monkeypatch, tmp_path, previous)
    job = json.loads(prepared.job_path.read_text())
    job["restart_args"] = ["--settings-file", str(previous / "settings/voice.settings.json"),
                           "--model-root", str(root / "FanyiJiVoiceNext_models")]
    prepared.job_path.write_text(json.dumps(job))
    commands = []
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(update.subprocess, "Popen", lambda command, **kw: commands.append(command))
    backup = update.apply_update(prepared.job_path)
    current = root / "dist" / NEW / "FanyiJiVoiceNext"
    clean, contexts = cleanup.split_cleanup_args(commands[0][1:])
    assert clean == ["--settings-file", str(current / "settings/voice.settings.json"), "--model-root", str(root / "FanyiJiVoiceNext_models")]
    assert json.loads((prepared.cache_dir / cleanup.SIDECAR_NAME).read_text())["install_dir"] == str(current)
    reports = cleanup.acknowledge_and_cleanup(contexts, current / "FanyiJiVoiceNext.exe", NEW)
    assert reports[0]["status"] == "completed"
    assert not backup.exists() and not (prepared.cache_dir / "staged").exists() and not downloaded.archive_path.exists()
    assert (current / "settings/voice.settings.json").read_bytes() == b"internal user settings"
    assert (root / "FanyiJiVoiceNext_models/model.gguf").read_bytes() == b"shared user model"
    assert managed.managed_model_root(current) == root / "FanyiJiVoiceNext_models"


@pytest.mark.parametrize("failure", ["restart", "registry-write"])
def test_failure_restores_program_directory_registration_cleanup_bindings_and_settings(monkeypatch, tmp_path, failure):
    root, previous = project(tmp_path)
    registry_before = (root / managed.REGISTRY_NAME).read_bytes()
    prepared, downloaded = prepare(monkeypatch, tmp_path, previous)
    job_before = prepared.job_path.read_bytes()
    sidecar_before = (prepared.cache_dir / cleanup.SIDECAR_NAME).read_bytes()
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    def failed_start(*args, **kwargs):
        raise OSError("fixture restart failure")
    monkeypatch.setattr(update.subprocess, "Popen", failed_start)
    if failure == "registry-write":
        original = managed._write
        def fail_write(path, body):
            if path == root / managed.REGISTRY_NAME and json.loads(body)["current_version"] == NEW:
                raise OSError("fixture registry write failure")
            return original(path, body)
        monkeypatch.setattr(managed, "_write", fail_write)
    with pytest.raises(update.UpdateError, match="已恢复原程序"):
        update.apply_update(prepared.job_path)
    assert (previous / "FanyiJiVoiceNext.exe").read_bytes() == b"old program"
    assert (previous / "settings/voice.settings.json").read_bytes() == b"internal user settings"
    assert not (root / "dist" / NEW).exists()
    assert (root / managed.REGISTRY_NAME).read_bytes() == registry_before
    assert prepared.job_path.read_bytes() == job_before
    assert (prepared.cache_dir / cleanup.SIDECAR_NAME).read_bytes() == sidecar_before
    assert downloaded.archive_path.is_file()
    assert (prepared.cache_dir / "backup/FanyiJiVoiceNext.exe").read_bytes() == b"old program"
    assert (root / "FanyiJiVoiceNext_models/model.gguf").read_bytes() == b"shared user model"
    assert not (previous / "_updater" / cleanup.PENDING_NAME).exists()


def test_existing_destination_program_is_never_overwritten(monkeypatch, tmp_path):
    root, previous = project(tmp_path)
    destination = root / "dist" / NEW / "FanyiJiVoiceNext"
    destination.mkdir(parents=True)
    (destination / "user-owned.txt").write_bytes(b"keep target")
    with pytest.raises(ValueError, match="已存在"):
        prepare(monkeypatch, tmp_path, previous)
    assert (destination / "user-owned.txt").read_bytes() == b"keep target"
    assert (previous / "FanyiJiVoiceNext.exe").read_bytes() == b"old program"


@pytest.mark.parametrize("restart_failure", [False, True])
def test_runtime_metadata_retirement_and_rollback_follow_the_moved_install(monkeypatch, tmp_path, restart_failure):
    root, previous = project(tmp_path)
    old = previous / "_internal/qa_managed-1.0.dist-info"
    old.mkdir(parents=True)
    (old / "METADATA").write_bytes(b"Name: qa-managed\nVersion: 1.0\n")
    ready = json.dumps({"schema_version": 1, "application": update.APPLICATION, "version": NEW,
                        "build_id": "managed-test-r1"}).encode()
    prepared, _ = prepare(monkeypatch, tmp_path, previous, runtime=True, extra_files={
        "_updater/runtime-ready.json": ready,
        "_internal/qa_managed-2.0.dist-info/METADATA": b"Name: qa-managed\nVersion: 2.0\n"})
    current = root / "dist" / NEW / "FanyiJiVoiceNext"
    def start(command, **kwargs):
        assert command[0] == str(current / "FanyiJiVoiceNext.exe")
        assert not (current / old.relative_to(previous)).exists()
        assert (current / "_updater/runtime-ready.json").read_bytes() == ready
        if restart_failure:
            raise OSError("fixture runtime restart failure")
    if restart_failure:
        with pytest.raises(update.UpdateError, match="已恢复原程序"):
            update.apply_update(prepared.job_path, start=start)
        assert (old / "METADATA").read_bytes() == b"Name: qa-managed\nVersion: 1.0\n"
        assert not (previous / "_internal/qa_managed-2.0.dist-info").exists()
        assert (previous / "FanyiJiVoiceNext.exe").read_bytes() == b"old program"
        assert not current.exists()
    else:
        update.apply_update(prepared.job_path, start=start)
        assert (current / "_internal/qa_managed-2.0.dist-info/METADATA").read_bytes().endswith(b"Version: 2.0\n")
        assert not previous.exists()
    assert (root / "FanyiJiVoiceNext_models/model.gguf").read_bytes() == b"shared user model"


def test_registration_changed_after_prepare_is_rejected_before_program_backup(monkeypatch, tmp_path):
    root, previous = project(tmp_path)
    prepared, _ = prepare(monkeypatch, tmp_path, previous)
    path = root / managed.REGISTRY_NAME
    path.write_text(path.read_text() + "\n")
    with pytest.raises(ValueError, match="登记在准备后发生变化"):
        update.apply_update(prepared.job_path, restart=False)
    assert (previous / "FanyiJiVoiceNext.exe").read_bytes() == b"old program"
    assert not (prepared.cache_dir / "backup").exists()


def test_missing_move_sidecar_never_downgrades_a_managed_update_to_in_place(monkeypatch, tmp_path):
    root, previous = project(tmp_path)
    prepared, _ = prepare(monkeypatch, tmp_path, previous)
    (prepared.cache_dir / managed.RELOCATION_NAME).rename(prepared.cache_dir / "held-relocation.json")
    before = (root / managed.REGISTRY_NAME).read_bytes()
    with pytest.raises(ValueError, match="缺少目录切换任务"):
        update.apply_update(prepared.job_path, restart=False)
    assert (previous / "FanyiJiVoiceNext.exe").read_bytes() == b"old program"
    assert (root / managed.REGISTRY_NAME).read_bytes() == before
    assert not (prepared.cache_dir / "backup").exists()


@pytest.mark.parametrize("body", [None, {"application": update.APPLICATION, "version": OLD},
                                    {"application": "another-product", "version": NEW}])
def test_managed_prepare_requires_manifested_matching_build_info(monkeypatch, tmp_path, body):
    _, previous = project(tmp_path)
    with pytest.raises(ValueError, match="build-info"):
        prepare(monkeypatch, tmp_path, previous,
                extra_files={"build-info.json": None if body is None else json.dumps(body).encode()})
    assert (previous / "FanyiJiVoiceNext.exe").read_bytes() == b"old program"


@pytest.mark.parametrize("remove", [False, True])
def test_helper_rechecks_matching_build_info_before_any_program_write(monkeypatch, tmp_path, remove):
    _, previous = project(tmp_path)
    prepared, _ = prepare(monkeypatch, tmp_path, previous)
    stage = prepared.cache_dir / "staged"
    path = stage / "update-manifest.json"
    manifest = json.loads(path.read_text())
    if remove:
        manifest["files"] = [row for row in manifest["files"] if row["path"] != "build-info.json"]
    else:
        body = json.dumps({"application": update.APPLICATION, "version": OLD}).encode()
        (stage / "build-info.json").write_bytes(body)
        row = next(row for row in manifest["files"] if row["path"] == "build-info.json")
        row.update(size=len(body), sha256=__import__("hashlib").sha256(body).hexdigest())
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="build-info"):
        update.apply_update(prepared.job_path, restart=False)
    assert (previous / "FanyiJiVoiceNext.exe").read_bytes() == b"old program"
    assert not (prepared.cache_dir / "backup").exists()


def test_portable_job_still_accepts_schema_one_without_build_info(monkeypatch, tmp_path):
    root, previous = project(tmp_path)
    (root / managed.REGISTRY_NAME).rename(root / "unregistered.json")
    prepared, _ = prepare(monkeypatch, tmp_path, previous, extra_files={"build-info.json": None})
    update.apply_update(prepared.job_path, restart=False)
    assert (previous / "FanyiJiVoiceNext.exe").read_bytes() == b"new program"
    assert not (root / "dist" / NEW).exists()


def test_linked_registration_is_rejected(tmp_path):
    root, install = project(tmp_path)
    external = tmp_path / "external.json"
    (root / managed.REGISTRY_NAME).rename(external)
    try:
        (root / managed.REGISTRY_NAME).symlink_to(external)
    except OSError as error:
        pytest.skip(f"OS does not permit test symlinks: {error}")
    with pytest.raises(ValueError, match="链接"):
        managed.managed_model_root(install)


def test_project_update_lock_excludes_second_helper_before_any_move(monkeypatch, tmp_path):
    root, previous = project(tmp_path)
    prepared, _ = prepare(monkeypatch, tmp_path, previous)
    script = ("from pathlib import Path;import sys;from fanyiji_next.app_update import apply_update,UpdateError;\n"
              "try: apply_update(Path(sys.argv[1]),restart=False)\n"
              "except UpdateError as e: print(str(e));sys.exit(0 if '另一更新助手' in str(e) else 3)\n"
              "sys.exit(2)\n")
    environment = dict(os.environ, PYTHONPATH=str(Path(update.__file__).resolve().parents[1]))
    with update._installation_lock(root):
        result = subprocess.run([sys.executable, "-X", "utf8", "-B", "-c", script, str(prepared.job_path)],
                                env=environment, capture_output=True, text=True, encoding="utf-8", timeout=15)
    assert result.returncode == 0, result.stdout + result.stderr
    assert (previous / "FanyiJiVoiceNext.exe").read_bytes() == b"old program"
    assert not (prepared.cache_dir / "backup").exists()
