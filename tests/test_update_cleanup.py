"""Real temporary update trees; no network, model loading or process restart."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest

from fanyiji_next import update_cleanup as cleanup


VERSION = "0.5.3"


def write(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def transaction(tmp_path, *, runtime=False, nested=False, independent=False):
    install = tmp_path / "installed"
    cache = tmp_path / "cache"
    cache.mkdir()
    download = cache / ("runtime-" + "b" * 64 if runtime else "download-" + "c" * 32)
    download.mkdir()
    archive = write(download / ("FanyiJiVoiceNext-0.5.3-runtime-r3.zip" if runtime else "program.zip"), b"verified ZIP")
    if runtime:
        write(download / "part-000.verified", b"part 0")
        write(download / "part-001.verified.part", b"part 1")
        write(download / ("merged.zip.part.bad-" + "d" * 32), b"old corrupt partial")
    parent = tmp_path / "independent-prepare" if independent else (download / "prepare" if nested else cache)
    job = parent / ("install-" + "a" * 32) / "job.json"
    stage, backup = job.parent / "staged", job.parent / "backup"
    stage.mkdir(parents=True)
    backup.mkdir()
    files = {"FanyiJiVoiceNext.exe": b"new exe", "_internal/library.dll": b"new native",
             "_internal/funasr/models/nano/model.py": b"# runtime source",
             "runtime/llama-server.exe": b"new server"}
    for name, content in files.items():
        write(stage / name, content)
        write(install / name, content)
    write(backup / "FanyiJiVoiceNext.exe", b"old exe")
    write(backup / "_internal/tokenizers-0.22.2.dist-info/METADATA", b"Name: tokenizers\nVersion: 0.22.2")
    write(job.parent / "FanyiJiUpdate.exe", b"old helper copy")
    manifest = {"schema_version": 1, "application": cleanup.APPLICATION, "version": VERSION,
                "files": [{"path": name, "size": len(content), "sha256": hashlib.sha256(content).hexdigest()}
                          for name, content in files.items()]}
    save(stage / "update-manifest.json", manifest)
    save(job, {"schema_version": 1, "application": cleanup.APPLICATION, "version": VERSION,
               "install_dir": str(install), "stage_dir": str(stage), "backup_dir": str(backup),
               "executable": "FanyiJiVoiceNext.exe", "pid": 0, "restart_args": []})
    sentinels = [write(install / "FanyiJiVoiceNext_models/model.gguf", b"model sentinel"),
                 write(install / "settings/voice.settings.json", b"settings sentinel"),
                 write(tmp_path / "recordings/sample.wav", b"recording sentinel"),
                 write(tmp_path / "source/main.py", b"source sentinel")]
    before = {str(path): path.read_bytes() for path in sentinels}
    cleanup.record_download_cleanup(job, archive, cache)
    return {"install": install, "cache": cache, "download": download, "archive": archive,
            "job": job, "stage": stage, "backup": backup, "before": before}


def acknowledge(value, context):
    return cleanup.acknowledge_and_cleanup(context, value["install"] / "FanyiJiVoiceNext.exe", VERSION)


def assert_preserved(value):
    assert all(Path(path).read_bytes() == content for path, content in value["before"].items())
    assert (value["install"] / "FanyiJiVoiceNext.exe").read_bytes() == b"new exe"


def test_no_ack_keeps_all_backups_and_downloads(tmp_path):
    value = transaction(tmp_path)
    receipt, token = cleanup.create_startup_receipt(value["job"])
    assert value["archive"].exists() and value["stage"].exists() and value["backup"].exists()
    assert load(receipt)["status"] == "awaiting_startup"
    assert token not in receipt.read_text(encoding="utf-8")
    assert len(load(value["install"] / "_updater" / cleanup.PENDING_NAME)["receipts"]) == 1
    assert_preserved(value)


@pytest.mark.parametrize("runtime,nested", [(False, False), (True, False), (True, True)])
def test_actual_startup_deletes_only_owned_program_files(tmp_path, runtime, nested):
    value = transaction(tmp_path, runtime=runtime, nested=nested)
    unknown = write(value["download"] / "my-note.txt", b"unowned sentinel")
    other_job = write(value["cache"] / ("install-" + "e" * 32) / "backup/FanyiJiVoiceNext.exe", b"other transaction")
    receipt, token = cleanup.create_startup_receipt(value["job"])
    result = acknowledge(value, [(receipt, token)])
    assert result[0]["status"] == "completed" and result[0]["startup_confirmed"]
    assert result[0]["removed_bytes"] > 0
    assert not value["stage"].exists() and not value["backup"].exists()
    assert not value["archive"].exists()
    assert not (value["job"].parent / "FanyiJiUpdate.exe").exists()
    assert value["job"].exists() and receipt.exists()
    assert unknown.read_bytes() == b"unowned sentinel" and other_job.read_bytes() == b"other transaction"
    assert load(value["install"] / "_updater" / cleanup.PENDING_NAME)["receipts"] == []
    assert load(receipt)["targets"] == []
    assert_preserved(value)
    assert acknowledge(value, [(receipt, token)])[0]["status"] == "completed"


def test_locked_file_is_pending_and_next_launch_uses_exact_pointer(tmp_path, monkeypatch):
    value = transaction(tmp_path)
    receipt, token = cleanup.create_startup_receipt(value["job"])
    locked = value["backup"] / "FanyiJiVoiceNext.exe"
    unlink = Path.unlink
    def deny_locked(path, *args, **kwargs):
        if path == locked:
            # Startup acknowledgement must reach disk before destructive work.
            assert load(receipt)["status"] == "startup_confirmed"
            assert load(receipt.parent / "cleanup-status.json")["startup_confirmed"]
            raise PermissionError("simulated locked EXE")
        return unlink(path, *args, **kwargs)
    monkeypatch.setattr(Path, "unlink", deny_locked)
    result = acknowledge(value, [(receipt, token)])
    assert result[0]["status"] == "pending" and locked.exists()
    assert len(load(value["install"] / "_updater" / cleanup.PENDING_NAME)["receipts"]) == 1
    monkeypatch.setattr(Path, "unlink", unlink)
    result = acknowledge(value, [])
    assert result[0]["status"] == "completed" and not locked.exists()
    assert_preserved(value)


@pytest.mark.parametrize("bad", ["version", "token", "executable", "job", "manifest"])
def test_wrong_startup_identity_never_deletes(tmp_path, bad):
    value = transaction(tmp_path)
    receipt, token = cleanup.create_startup_receipt(value["job"])
    executable, version = value["install"] / "FanyiJiVoiceNext.exe", VERSION
    if bad == "version": version = "0.5.4"
    elif bad == "token": token = "f" * 64
    elif bad == "executable": executable.write_bytes(b"different running executable")
    elif bad == "job":
        data = load(value["job"]); data["pid"] = 999; save(value["job"], data)
    elif bad == "manifest":
        data = load(receipt.parent / cleanup.MANIFEST_NAME); data["files"][0]["sha256"] = "f" * 64; save(receipt.parent / cleanup.MANIFEST_NAME, data)
    result = cleanup.acknowledge_and_cleanup([(receipt, token)], executable, version)
    assert result[0]["status"] == "pending"
    assert value["archive"].exists() and value["stage"].exists() and value["backup"].exists()


def test_missing_runtime_ready_defers_until_final_program_startup(tmp_path):
    from fanyiji_next.app_update import DEFAULT_REPOSITORY
    value = transaction(tmp_path, runtime=True)
    marker = {"schema_version": 1, "application": cleanup.APPLICATION, "version": VERSION, "build_id": "voice-test"}
    descriptor = {**marker, "platform": "windows-x64", "kind": "runtime-bundle", "name": "runtime.zip",
                  "size": 1, "sha256": "c" * 64, "parts": [{"name": "runtime.zip.001", "size": 1, "sha256": "b" * 64,
                  "url": f"https://github.com/{DEFAULT_REPOSITORY}/releases/download/v{VERSION}/runtime.zip.001"}]}
    (value["install"] / "_updater").mkdir()
    save(value["install"] / "_updater/runtime-bootstrap.json", descriptor)
    receipt, token = cleanup.create_startup_receipt(value["job"])
    assert acknowledge(value, [(receipt, token)])[0]["status"] == "pending"
    assert value["backup"].exists() and value["archive"].exists()
    save(value["install"] / "_updater/runtime-ready.json", marker)
    result = acknowledge(value, [])
    assert result[0]["status"] == "completed", result
    assert_preserved(value)


@pytest.mark.parametrize("relative", ["recording.wav", "_internal/example.mp3", "_internal/model.gguf",
                                     "_internal/Custom_Models/cache.json", "settings/voice.settings.json",
                                     "source/main.py", "_internal/funasr/models/weights.bin"])
def test_receipt_rejects_models_audio_settings_and_source_in_owned_tree(tmp_path, relative):
    value = transaction(tmp_path)
    sentinel = write(value["stage"] / relative, b"protected sentinel")
    with pytest.raises((ValueError, RuntimeError)):
        cleanup.create_startup_receipt(value["job"])
    assert sentinel.read_bytes() == b"protected sentinel" and value["archive"].exists()


@pytest.mark.parametrize("attack", ["parent", "other_job", "install", "model", "sidecar"])
def test_tampered_receipt_cannot_redirect_deletion(tmp_path, attack):
    value = transaction(tmp_path)
    receipt, token = cleanup.create_startup_receipt(value["job"])
    data = load(receipt)
    target = data["targets"][0]
    if attack == "parent": target["files"][0][0] = "../../installed/FanyiJiVoiceNext.exe"
    elif attack == "other_job": target["base"] = str(value["cache"] / ("install-" + "f" * 32) / "staged")
    elif attack == "install": target["base"] = str(value["install"])
    elif attack == "model": target["files"][0][0] = "_internal/A_Models/weights.gguf"
    elif attack == "sidecar":
        sidecar = load(receipt.parent / cleanup.SIDECAR_NAME)
        sidecar["download_dir"] = str(value["install"])
        save(receipt.parent / cleanup.SIDECAR_NAME, sidecar)
    save(receipt, data)
    assert acknowledge(value, [(receipt, token)])[0]["status"] == "pending"
    assert value["stage"].exists() and value["backup"].exists() and value["archive"].exists()
    assert_preserved(value)


def test_changed_file_identity_aborts_before_any_deletion(tmp_path):
    value = transaction(tmp_path)
    receipt, token = cleanup.create_startup_receipt(value["job"])
    write(value["backup"] / "FanyiJiVoiceNext.exe", b"user replaced backup")
    assert acknowledge(value, [(receipt, token)])[0]["status"] == "pending"
    assert value["archive"].exists() and value["stage"].exists()


def test_hardlinks_are_never_deleted(tmp_path):
    value = transaction(tmp_path)
    linked = value["stage"] / "_internal/library.dll"
    outside = tmp_path / "outside.dll"
    os.link(linked, outside)
    with pytest.raises(ValueError, match="hardlink"):
        cleanup.create_startup_receipt(value["job"])
    assert outside.read_bytes() == linked.read_bytes()


def test_symlink_subtree_is_never_followed(tmp_path):
    value = transaction(tmp_path)
    target = tmp_path / "outside"
    sentinel = write(target / "library.dll", b"outside sentinel")
    try:
        (value["stage"] / "_internal/linked").symlink_to(target, target_is_directory=True)
    except OSError as error:
        pytest.skip(f"Host does not permit test symlink creation: {error}")
    with pytest.raises(ValueError, match="链接"):
        cleanup.create_startup_receipt(value["job"])
    assert sentinel.read_bytes() == b"outside sentinel"


def test_external_local_archive_is_never_claimed(tmp_path):
    value = transaction(tmp_path)
    (value["job"].parent / cleanup.SIDECAR_NAME).unlink()
    external = write(tmp_path / "manual-release.zip", b"local manual archive")
    assert cleanup.record_download_cleanup(value["job"], external, value["cache"]) is None
    assert not (value["job"].parent / cleanup.SIDECAR_NAME).exists()
    receipt, token = cleanup.create_startup_receipt(value["job"])
    assert acknowledge(value, [(receipt, token)])[0]["status"] == "completed"
    assert external.read_bytes() == b"local manual archive" and value["archive"].exists()


def test_independent_prepare_cache_does_not_claim_runtime_download(tmp_path):
    value = transaction(tmp_path, runtime=True, independent=True)
    assert not (value["job"].parent / cleanup.SIDECAR_NAME).exists()
    receipt, token = cleanup.create_startup_receipt(value["job"])
    assert acknowledge(value, [(receipt, token)])[0]["status"] == "completed"
    assert not value["stage"].exists() and not value["backup"].exists()
    assert value["archive"].read_bytes() == b"verified ZIP"
    assert (value["download"] / "part-000.verified").read_bytes() == b"part 0"
    assert_preserved(value)


@pytest.mark.parametrize("which", ["pointer", "bootstrap_descriptor"])
def test_broken_metadata_symlink_cannot_bypass_guards(tmp_path, which):
    value = transaction(tmp_path)
    receipt, token = cleanup.create_startup_receipt(value["job"])
    metadata = value["install"] / "_updater" / (cleanup.PENDING_NAME if which == "pointer" else "runtime-bootstrap.json")
    if metadata.exists():
        metadata.unlink()
    try:
        metadata.symlink_to(tmp_path / "nonexistent.json")
    except OSError as error:
        pytest.skip(f"Host does not permit test symlink creation: {error}")
    assert acknowledge(value, [(receipt, token)])[0]["status"] == "pending"
    assert value["archive"].exists() and value["stage"].exists() and value["backup"].exists()


def test_parser_handles_multiple_contexts_without_polluting_user_arguments(tmp_path):
    first = tmp_path / ("install-" + "a" * 32) / cleanup.RECEIPT_NAME
    second = tmp_path / ("install-" + "b" * 32) / cleanup.RECEIPT_NAME
    clean, contexts = cleanup.split_cleanup_args(["--settings", "profile.json", "--update-receipt", str(first), "--update-token", "a" * 64,
                                                  "--update-receipt", str(second), "--update-token", "b" * 64, "--language", "auto"])
    assert clean == ["--settings", "profile.json", "--language", "auto"]
    assert contexts == [(first, "a" * 64), (second, "b" * 64)]
    clean, contexts = cleanup.split_cleanup_args(["--update-receipt", "--settings", "profile.json", "--update-token", "bad"])
    assert clean == ["--settings", "profile.json"] and contexts == []
    clean, contexts = cleanup.split_cleanup_args(["--update-receipt", "--smoke-test", "--update-token", "bad"])
    assert clean == ["--smoke-test"] and contexts == []


def test_empty_context_does_not_discover_other_transactions(tmp_path):
    value = transaction(tmp_path)
    assert acknowledge(value, []) == []
    assert value["archive"].exists() and value["stage"].exists() and value["backup"].exists()


def test_invalid_pointer_and_context_are_nonfatal_and_preserve_files(tmp_path):
    value = transaction(tmp_path)
    pointer = value["install"] / "_updater" / cleanup.PENDING_NAME
    pointer.parent.mkdir()
    save(pointer, {"receipts": [{"receipt": "../../other", "token": "a" * 64}]})
    assert acknowledge(value, [])[0]["status"] == "pending"
    pointer.unlink()
    assert acknowledge(value, ["not a context pair"])[0]["status"] == "pending"
    assert value["archive"].exists() and value["stage"].exists()
