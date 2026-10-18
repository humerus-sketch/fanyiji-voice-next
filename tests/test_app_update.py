"""Updater contract tests use synthetic archives and mocked network/processes."""
from __future__ import annotations

from dataclasses import replace
import hashlib
import io
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import threading
from urllib.request import Request
import zipfile

import pytest

from fanyiji_next import app_update as update


REPOSITORY = "example/fanyiji-voice-next"
VERSION = "0.5.0"
URL = f"https://github.com/{REPOSITORY}/releases/download/v{VERSION}/program.zip"


def archive(tmp_path, files=None, *, rows=None, extra=None):
    files = files or {"FanyiJiVoiceNext.exe": b"new executable", "_internal/library.dll": b"new library", "runtime/llama-server.exe": b"new runtime"}
    manifest = {"schema_version": 1, "application": update.APPLICATION, "version": VERSION,
                "files": rows if rows is not None else [{"path": name, "size": len(data), "sha256": hashlib.sha256(data).hexdigest()} for name, data in files.items()]}
    path = tmp_path / "program.zip"
    with zipfile.ZipFile(path, "w") as bundle:
        for name, data in files.items():
            bundle.writestr(name, data)
        for name, data in (extra or {}).items():
            bundle.writestr(name, data)
        bundle.writestr(update.ARCHIVE_MANIFEST, json.dumps(manifest).encode())
    info = update.UpdateInfo(VERSION, path.name, URL, update._digest(path), path.stat().st_size,
                             f"https://github.com/{REPOSITORY}/releases/tag/v{VERSION}", "notes")
    return update.DownloadedUpdate(info, path)


def installation(tmp_path):
    install = tmp_path / "installed"
    install.mkdir()
    (install / "FanyiJiVoiceNext.exe").write_bytes(b"old executable")
    (install / "_internal").mkdir()
    (install / "_internal/library.dll").write_bytes(b"old library")
    (install / "models").mkdir()
    (install / "models/model.gguf").write_bytes(b"user model")
    (install / "settings").mkdir()
    (install / "settings/voice.settings.json").write_bytes(b"user settings")
    (install / "custom.txt").write_bytes(b"user text")
    return install


def client(tmp_path):
    return update.UpdateClient(REPOSITORY, tmp_path / "cache")


def release_and_feed(info):
    feed = {"schema_version": 1, "application": update.APPLICATION, "platform": update.PLATFORM,
            "version": info.version, "program": {key: getattr(info, key) for key in ("name", "url", "sha256", "size")}}
    release = {"draft": False, "prerelease": False, "tag_name": "v" + info.version,
               "html_url": info.release_url, "body": info.notes,
               "assets": [{"name": update.FEED_NAME, "size": 400, "browser_download_url": URL.replace("program.zip", update.FEED_NAME)},
                          {"name": info.name, "size": info.size, "browser_download_url": info.url}]}
    return release, feed


def mock_release(monkeypatch, instance, release, feed):
    monkeypatch.setattr(instance, "_read_json", lambda url, **kw: release if kw.get("api") else feed)


def test_check_release_and_no_downgrade(monkeypatch, tmp_path):
    info = archive(tmp_path).info
    instance = client(tmp_path)
    release, feed = release_and_feed(info)
    mock_release(monkeypatch, instance, release, feed)
    assert instance.check("0.4.8") == info
    assert instance.check("0.5.0") is None
    assert instance.check("0.6.0") is None
    assert instance.check("0.5.0-rc.1") == info


@pytest.mark.parametrize("mutation", [
    lambda r, f: f.update(schema_version=True),
    lambda r, f: f.update(application="other"),
    lambda r, f: f.update(platform="linux-x64"),
    lambda r, f: f.update(version="0.5"),
    lambda r, f: f.update(extra="unknown"),
    lambda r, f: f["program"].update(size=True),
    lambda r, f: f["program"].update(sha256="bad"),
    lambda r, f: f["program"].update(url="https://github.com/other/repo/releases/download/v0.5.0/program.zip"),
    lambda r, f: r.update(tag_name="v0.5.1"),
    lambda r, f: r.update(prerelease=True),
    lambda r, f: r["assets"].append(dict(r["assets"][1])),
    lambda r, f: r["assets"][1].update(size=123),
])
def test_rejects_inconsistent_or_untrusted_feed(monkeypatch, tmp_path, mutation):
    info = archive(tmp_path).info
    instance = client(tmp_path)
    release, feed = release_and_feed(info)
    mutation(release, feed)
    mock_release(monkeypatch, instance, release, feed)
    with pytest.raises(update.UpdateError):
        instance.check("0.4.8")


@pytest.mark.parametrize("url", ["http://github.com/example/fanyiji-voice-next/releases/download/v0.5.0/a.zip",
    "https://evil.test/file.zip", "https://github.com.evil.test/example/fanyiji-voice-next/releases/download/v0.5.0/a.zip",
    "https://user:pass@github.com/example/fanyiji-voice-next/releases/download/v0.5.0/a.zip",
    "https://github.com:444/example/fanyiji-voice-next/releases/download/v0.5.0/a.zip",
    "https://github.com/example/fanyiji-voice-next/releases/download/../other/a.zip",
    "https://github.com/example/fanyiji-voice-next/releases/download/%2e%2e/other/a.zip",
    "https://github.com/example/fanyiji-voice-next/releases/download/v0.5.0/a.zip#x"])
def test_https_and_repository_boundary(url):
    with pytest.raises(update.UpdateError):
        update._trusted_url(url, REPOSITORY)


def test_only_official_asset_redirect_domains():
    redirect = update._TrustedRedirect(REPOSITORY, False)
    request = Request(URL)
    allowed = "https://release-assets.githubusercontent.com/github-production-release-asset/a?token=x"
    assert redirect.redirect_request(request, None, 302, "Found", {}, allowed).full_url == allowed
    with pytest.raises(update.UpdateError):
        redirect.redirect_request(request, None, 302, "Found", {}, "https://example.org/a")
    with pytest.raises(update.UpdateError):
        update._trusted_url(allowed, REPOSITORY)  # A feed cannot directly select a CDN object.


class Response(io.BytesIO):
    def __init__(self, body, length=None):
        super().__init__(body)
        self.headers = {} if length is None else {"Content-Length": str(length)}


def test_verified_download_progress(monkeypatch, tmp_path):
    data = b"a" * (300 * 1024)
    info = replace(archive(tmp_path).info, size=len(data), sha256=hashlib.sha256(data).hexdigest())
    instance = client(tmp_path)
    monkeypatch.setattr(instance, "_open", lambda url: Response(data, len(data)))
    observed = []
    downloaded = instance.download(info, lambda done, total: observed.append((done, total)))
    assert downloaded.archive_path.read_bytes() == data
    assert observed[-1] == (len(data), len(data))
    assert len(observed) == 2


@pytest.mark.parametrize("body,length,digest", [(b"wrong!", 6, "0" * 64), (b"too long", 8, None), (b"short", None, None)])
def test_download_rejects_hash_and_length(monkeypatch, tmp_path, body, length, digest):
    info = replace(archive(tmp_path).info, size=6, sha256=digest or hashlib.sha256(body).hexdigest())
    instance = client(tmp_path)
    monkeypatch.setattr(instance, "_open", lambda url: Response(body, length))
    with pytest.raises(update.UpdateError):
        instance.download(info)
    assert not list(instance.cache_dir.glob("*/program.zip"))


def test_cancel_download_preserves_existing_files(monkeypatch, tmp_path):
    data = b"x" * (300 * 1024)
    info = replace(archive(tmp_path).info, size=len(data), sha256=hashlib.sha256(data).hexdigest())
    install = installation(tmp_path)
    instance = client(tmp_path)
    monkeypatch.setattr(instance, "_open", lambda url: Response(data))
    progress = []
    with pytest.raises(update.UpdateCancelled):
        instance.download(info, lambda done, total: progress.append(done), lambda: bool(progress))
    assert progress == [256 * 1024]
    assert (install / "FanyiJiVoiceNext.exe").read_bytes() == b"old executable"
    assert not list(instance.cache_dir.glob("*/program.zip"))


def test_prepare_does_not_install_and_apply_preserves_models_settings(tmp_path):
    downloaded, install = archive(tmp_path), installation(tmp_path)
    prepared = client(tmp_path).prepare(downloaded, install)
    assert (install / "FanyiJiVoiceNext.exe").read_bytes() == b"old executable"
    started = []
    backup = update.apply_update(prepared.job_path, start=lambda *args, **kwargs: started.append((args, kwargs)))
    assert (install / "FanyiJiVoiceNext.exe").read_bytes() == b"new executable"
    assert (install / "runtime/llama-server.exe").read_bytes() == b"new runtime"
    assert (backup / "FanyiJiVoiceNext.exe").read_bytes() == b"old executable"
    assert (install / "models/model.gguf").read_bytes() == b"user model"
    assert (install / "settings/voice.settings.json").read_bytes() == b"user settings"
    assert (install / "custom.txt").read_bytes() == b"user text"
    assert started[0][0][0] == [str(install / "FanyiJiVoiceNext.exe")]
    assert started[0][1]["cwd"] == str(install)
    assert started[0][1]["env"]["PYINSTALLER_RESET_ENVIRONMENT"] == "1"
    with pytest.raises(update.UpdateError, match="不重复应用"):
        update.apply_update(prepared.job_path, restart=False)


@pytest.mark.parametrize("bad_path", ["../escape.exe", "/absolute.exe", "C:/escape.exe", "\\server\\a.exe", "_internal/../escape.exe", "_internal//x.dll", "_internal/CON.dll", "_internal/ads:stream", "_internal/dot.", "_internal/trail ", "models/model.gguf", "_internal/models/model.onnx", "settings/voice.settings.json", "_internal/voice.settings.json", "settings.json", "private.txt", "src/other.py"])
def test_unsafe_paths_and_user_files_rejected(tmp_path, bad_path):
    downloaded = archive(tmp_path, {"FanyiJiVoiceNext.exe": b"new", bad_path: b"bad"})
    with pytest.raises(update.UpdateError):
        client(tmp_path).prepare(downloaded, installation(tmp_path))
    assert not (tmp_path / "escape.exe").exists()


def test_case_collisions_rejected_in_implicit_parent(tmp_path):
    downloaded = archive(tmp_path, {"FanyiJiVoiceNext.exe": b"new", "_internal/Foo/a.dll": b"a", "_internal/foo/b.dll": b"b"})
    with pytest.raises(update.UpdateError, match="大小写"):
        client(tmp_path).prepare(downloaded, installation(tmp_path))


def test_zip_symlink_rejected(tmp_path):
    downloaded = archive(tmp_path)
    with zipfile.ZipFile(downloaded.archive_path, "a") as bundle:
        entry = zipfile.ZipInfo("_internal/link.dll")
        entry.create_system = 3
        entry.external_attr = (stat.S_IFLNK | 0o777) << 16
        bundle.writestr(entry, "../../outside")
    downloaded = replace(downloaded, info=replace(downloaded.info, size=downloaded.archive_path.stat().st_size, sha256=update._digest(downloaded.archive_path)))
    with pytest.raises(update.UpdateError, match="符号链接"):
        client(tmp_path).prepare(downloaded, installation(tmp_path))


def test_zip_hash_and_unlisted_file_tampering(tmp_path):
    files = {"FanyiJiVoiceNext.exe": b"new"}
    rows = [{"path": "FanyiJiVoiceNext.exe", "size": 3, "sha256": "0" * 64}]
    downloaded = archive(tmp_path, files, rows=rows)
    install = installation(tmp_path)
    with pytest.raises(update.UpdateError, match="SHA256"):
        client(tmp_path).prepare(downloaded, install)
    downloaded = archive(tmp_path, extra={"_internal/unlisted.dll": b"bad"})
    with pytest.raises(update.UpdateError, match="清单不一致"):
        client(tmp_path).prepare(downloaded, install)


def test_stage_tampering_prevents_any_overwrite(tmp_path):
    prepared = client(tmp_path).prepare(archive(tmp_path), installation(tmp_path))
    (prepared.cache_dir / "staged/FanyiJiVoiceNext.exe").write_bytes(b"tampered")
    with pytest.raises(update.UpdateError, match="再次校验失败"):
        update.apply_update(prepared.job_path, restart=False)
    assert (prepared.install_dir / "FanyiJiVoiceNext.exe").read_bytes() == b"old executable"
    assert not (prepared.cache_dir / "backup").exists()


def test_apply_failure_rolls_back_and_retains_backup(monkeypatch, tmp_path):
    files = {"FanyiJiVoiceNext.exe": b"new executable", "_internal/new.dll": b"new file", "_internal/library.dll": b"new library"}
    prepared = client(tmp_path).prepare(archive(tmp_path, files), installation(tmp_path))
    original, failures = update._atomic_copy, []
    def fail_library(source, target):
        if source.parent.name == "_internal" and source.parent.parent.name == "staged" and target.name == "library.dll":
            failures.append(target)
            raise OSError("simulated sharing violation")
        return original(source, target)
    monkeypatch.setattr(update, "_atomic_copy", fail_library)
    with pytest.raises(update.UpdateError, match="已恢复原程序"):
        update.apply_update(prepared.job_path, restart=False)
    assert len(failures) == 1
    assert (prepared.install_dir / "FanyiJiVoiceNext.exe").read_bytes() == b"old executable"
    assert (prepared.install_dir / "_internal/library.dll").read_bytes() == b"old library"
    assert not (prepared.install_dir / "_internal/new.dll").exists()
    assert (prepared.cache_dir / "backup/FanyiJiVoiceNext.exe").read_bytes() == b"old executable"
    assert (prepared.install_dir / "models/model.gguf").read_bytes() == b"user model"


def test_restart_failure_rolls_back(tmp_path):
    prepared = client(tmp_path).prepare(archive(tmp_path), installation(tmp_path))
    def fail(*args, **kwargs):
        raise OSError("cannot start")
    with pytest.raises(update.UpdateError, match="已恢复原程序"):
        update.apply_update(prepared.job_path, start=fail)
    assert (prepared.install_dir / "FanyiJiVoiceNext.exe").read_bytes() == b"old executable"


def test_wait_for_running_process_is_bounded(tmp_path):
    prepared = client(tmp_path).prepare(archive(tmp_path), installation(tmp_path))
    job = json.loads(prepared.job_path.read_text(encoding="utf-8"))
    job["pid"] = 123
    prepared.job_path.write_text(json.dumps(job), encoding="utf-8")
    with pytest.raises(update.UpdateError, match="退出超时"):
        update.apply_update(prepared.job_path, wait_timeout=0, pid_alive=lambda pid: True, restart=False)
    assert not (prepared.cache_dir / "backup").exists()


def test_job_directory_escape_rejected(tmp_path):
    prepared = client(tmp_path).prepare(archive(tmp_path), installation(tmp_path))
    job = json.loads(prepared.job_path.read_text(encoding="utf-8"))
    job["backup_dir"] = str(tmp_path / "other")
    prepared.job_path.write_text(json.dumps(job), encoding="utf-8")
    with pytest.raises(update.UpdateError, match="目录无效"):
        update.apply_update(prepared.job_path, restart=False)


def test_source_mode_cannot_launch_helper(tmp_path):
    prepared = client(tmp_path).prepare(archive(tmp_path), installation(tmp_path))
    with pytest.raises(update.UpdateError, match="源码运行"):
        update.launch_helper(prepared, frozen=False)


def test_helper_is_copied_outside_installation_and_launched_hidden(monkeypatch, tmp_path):
    monkeypatch.setenv("PYINSTALLER_RESET_ENVIRONMENT", "0")
    prepared = client(tmp_path).prepare(archive(tmp_path), installation(tmp_path))
    helpers = prepared.install_dir / "_updater"
    helpers.mkdir()
    (helpers / "FanyiJiUpdate.exe").write_bytes(b"standalone helper")
    launched = []
    monkeypatch.setattr(update.subprocess, "Popen", lambda *args, **kw: launched.append((args, kw)) or "process")
    assert update.launch_helper(prepared, pid=4321, restart_args=["--settings-file", "D:/用户/settings.json"], frozen=True) == "process"
    command = launched[0][0][0]
    assert Path(command[0]).parent == prepared.cache_dir
    assert Path(command[0]).read_bytes() == b"standalone helper"
    assert launched[0][1]["creationflags"] == update.subprocess.CREATE_NO_WINDOW
    assert launched[0][1]["close_fds"] is True
    assert launched[0][1]["cwd"] == str(prepared.cache_dir)
    assert launched[0][1]["env"]["PYINSTALLER_RESET_ENVIRONMENT"] == "1"
    assert os.environ["PYINSTALLER_RESET_ENVIRONMENT"] == "0"
    job = json.loads(prepared.job_path.read_text(encoding="utf-8"))
    assert job["pid"] == 4321
    assert job["restart_args"] == ["--settings-file", "D:/用户/settings.json"]


def test_in_tree_link_cannot_redirect_program_update_into_model(tmp_path):
    install = installation(tmp_path)
    link = install / "_internal/innocent.dll"
    try:
        link.symlink_to(install / "models/model.gguf")
    except OSError as exc:
        pytest.skip(f"OS does not permit test symlinks: {exc}")
    downloaded = archive(tmp_path, {"FanyiJiVoiceNext.exe": b"new", "_internal/innocent.dll": b"overwrite model"})
    with pytest.raises(update.UpdateError, match="符号链接"):
        client(tmp_path).prepare(downloaded, install)
        # Preparation only stages. Existing targets must be rejected again by the
        # actual installer; this is deliberately not a substring/path-only test.
        prepared = next((tmp_path / "cache").glob("install-*/job.json"))
        update.apply_update(prepared, restart=False)
    assert (install / "models/model.gguf").read_bytes() == b"user model"
    assert (install / "FanyiJiVoiceNext.exe").read_bytes() == b"old executable"


def test_existing_symlink_install_root_rejected(tmp_path):
    install = installation(tmp_path)
    alias = tmp_path / "alias"
    try:
        alias.symlink_to(install, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"OS does not permit test symlinks: {exc}")
    with pytest.raises(update.UpdateError, match="符号链接"):
        client(tmp_path).prepare(archive(tmp_path), alias)


def test_ctranslate_dependency_source_allowed_but_model_data_rejected(tmp_path):
    files = {"FanyiJiVoiceNext.exe": b"new", "_internal/ctranslate2/models/__init__.py": b"# dependency"}
    prepared = client(tmp_path).prepare(archive(tmp_path, files), installation(tmp_path))
    assert (prepared.cache_dir / "staged/_internal/ctranslate2/models/__init__.py").read_bytes() == b"# dependency"
    for path in ("_internal/ctranslate2/models/model.bin", "_internal/ctranslate2/models/voice.onnx", "runtime/model.gguf"):
        with pytest.raises(update.UpdateError):
            update._safe_path(path)


def test_windows_temporary_sharing_lock_is_retried(monkeypatch, tmp_path):
    source, target = tmp_path / "source", tmp_path / "target"
    source.write_bytes(b"new")
    target.write_bytes(b"old")
    original, attempts = update.os.replace, []
    def transient_error(left, right):
        attempts.append((left, right))
        if len(attempts) == 1:
            error = OSError("sharing violation")
            error.winerror = 32
            raise error
        return original(left, right)
    monkeypatch.setattr(update.os, "replace", transient_error)
    monkeypatch.setattr(update.time, "sleep", lambda value: None)
    update._atomic_copy(source, target)
    assert len(attempts) == 2
    assert target.read_bytes() == b"new"


def test_sharing_lock_retry_has_deadline(monkeypatch, tmp_path):
    source, target = tmp_path / "source", tmp_path / "target"
    source.write_bytes(b"new")
    target.write_bytes(b"old")
    times = iter([0, 0, 11])
    monkeypatch.setattr(update.time, "monotonic", lambda: next(times))
    monkeypatch.setattr(update.time, "sleep", lambda value: None)
    def locked(*args):
        error = OSError("sharing violation")
        error.winerror = 32
        raise error
    monkeypatch.setattr(update.os, "replace", locked)
    with pytest.raises(OSError, match="sharing violation"):
        update._atomic_copy(source, target)
    assert target.read_bytes() == b"old"
    assert not list(tmp_path.glob("target.update-*"))


def test_duplicate_json_fields_rejected():
    with pytest.raises(update.UpdateError, match="重复 JSON"):
        update._json(b'{"schema_version":1,"schema_version":2}')


@pytest.mark.parametrize("value", ["", "1.0", "01.0.0", "1.0.0-01", "1.0.0-rc..1", "main"])
def test_version_format_rejects_invalid(value):
    with pytest.raises(update.UpdateError):
        update._version_key(value)


def test_empty_repository_is_not_a_fake_update_service(tmp_path):
    with pytest.raises(update.UpdateError, match="尚未配置"):
        update.UpdateClient("", tmp_path)


def test_second_helper_for_same_installation_is_rejected_across_processes(tmp_path):
    prepared = client(tmp_path).prepare(archive(tmp_path), installation(tmp_path))
    script = ("from pathlib import Path; import sys; from fanyiji_next.app_update import apply_update,UpdateError; "
              "job=Path(sys.argv[1]);\n"
              "try: apply_update(job,restart=False)\n"
              "except UpdateError as exc:\n"
              " print(str(exc)); sys.exit(0 if '另一更新助手' in str(exc) else 3)\n"
              "sys.exit(2)\n")
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(Path(update.__file__).resolve().parents[1])
    with update._installation_lock(prepared.install_dir):
        completed = subprocess.run([sys.executable, "-X", "utf8", "-B", "-c", script, str(prepared.job_path)],
                                   env=environment, capture_output=True, text=True, encoding="utf-8", timeout=15)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "另一更新助手" in completed.stdout
    assert (prepared.install_dir / "FanyiJiVoiceNext.exe").read_bytes() == b"old executable"
    assert not (prepared.cache_dir / "backup").exists()
    # The first helper releases its lock without leaving a stale lock owner.
    update.apply_update(prepared.job_path, restart=False)
    assert (prepared.install_dir / "FanyiJiVoiceNext.exe").read_bytes() == b"new executable"


@pytest.mark.parametrize("bad_path", [
    "FanyiJiVoiceNext_models/catalog.json", "字幕_models/speech/model.py", "Renamed_MODELS/file.dll",
    "_internal/any_models/catalog.json", "runtime/直播_models/config.json",
    "_internal/transformers/models/user_models/modeling.py", "_internal/funasr/models/saved_MODELS/entry.py",
    "_internal/ctranslate2/models/extra_models/dependency.py",
])
def test_external_model_suffix_is_protected_at_every_depth(tmp_path, bad_path):
    downloaded = archive(tmp_path, {"FanyiJiVoiceNext.exe": b"new", bad_path: b"bad"})
    install = installation(tmp_path)
    with pytest.raises(update.UpdateError, match="模型"):
        client(tmp_path).prepare(downloaded, install)
    assert (install / "FanyiJiVoiceNext.exe").read_bytes() == b"old executable"


@pytest.mark.parametrize("suffix", [".onnx", ".gguf", ".safetensors", ".pt", ".pth", ".ckpt", ".tflite", ".bin", ".model"])
def test_any_weight_name_or_case_is_rejected_even_outside_models(suffix):
    with pytest.raises(update.UpdateError, match="模型"):
        update._safe_path("_internal/runtime/innocent" + suffix.upper())


@pytest.mark.parametrize("package", ["ctranslate2", "transformers", "funasr", "torchaudio", "torchaudio/prototype"])
@pytest.mark.parametrize("suffix", [".py", ".pyc", ".pyd"])
def test_dependency_models_permits_only_code_in_exact_prefix(tmp_path, package, suffix):
    relative = f"_internal/{package}/models/qwen3_asr/implementation{suffix}"
    prepared = client(tmp_path).prepare(archive(tmp_path, {"FanyiJiVoiceNext.exe": b"new", relative: b"# public dependency"}),
                                        installation(tmp_path))
    assert (prepared.cache_dir / "staged" / relative).read_bytes() == b"# public dependency"
    for name in ("config.json", "model.bin", "model.safetensors", "weights.onnx", "runtime.dll", "untyped"):
        with pytest.raises(update.UpdateError, match="模型"):
            update._safe_path(f"_internal/{package}/models/qwen3_asr/{name}")
    with pytest.raises(update.UpdateError, match="模型"):
        update._safe_path(f"_internal/not-{package}/models/implementation.py")


def test_dependency_nested_directory_entries_are_valid_but_not_manifest_files(tmp_path):
    relative = "_internal/transformers/models/qwen3_asr/modeling.py"
    downloaded = archive(tmp_path, {"FanyiJiVoiceNext.exe": b"new", relative: b"# source"},
                         extra={"_internal/transformers/models/": b"", "_internal/transformers/models/qwen3_asr/": b""})
    prepared = client(tmp_path).prepare(downloaded, installation(tmp_path))
    assert (prepared.cache_dir / "staged" / relative).is_file()
    with pytest.raises(update.UpdateError, match="模型"):
        update._safe_path("_internal/transformers/models/qwen3_asr")


def test_exact_sentencepiece_tables_are_runtime_data_and_cannot_allow_other_bins(tmp_path):
    files = {"FanyiJiVoiceNext.exe": b"new"}
    files.update({path: b"Unicode normalization runtime data" for path in update._NORMALIZATION_TABLES})
    prepared = client(tmp_path).prepare(archive(tmp_path, files), installation(tmp_path))
    for relative in update._NORMALIZATION_TABLES:
        assert (prepared.cache_dir / "staged" / relative).read_bytes() == b"Unicode normalization runtime data"
    for path in ("_internal/sentencepiece/package_data/weights.bin", "_internal/vendor/nfkc.bin",
                 "_internal/sentencepiece/package_data/nfkc.BIN", "runtime/nfkc.bin"):
        with pytest.raises(update.UpdateError, match="模型"):
            update._safe_path(path)


@pytest.mark.parametrize("restart_fails", [False, True])
def test_install_success_and_rollback_preserve_external_models_and_user_settings(tmp_path, restart_fails):
    install = installation(tmp_path)
    weights = {}
    for directory in ("FanyiJiVoiceNext_models", "直播字幕_models", "Another_MODELS"):
        relative = directory + "/speech/model.safetensors"
        target = install / relative
        target.parent.mkdir(parents=True)
        weights[relative] = ("original " + directory).encode()
        target.write_bytes(weights[relative])
    prepared = client(tmp_path).prepare(archive(tmp_path), install)
    def start(*args, **kwargs):
        if restart_fails:
            raise OSError("simulated restart error")
    if restart_fails:
        with pytest.raises(update.UpdateError, match="已恢复原程序"):
            update.apply_update(prepared.job_path, start=start)
        assert (install / "FanyiJiVoiceNext.exe").read_bytes() == b"old executable"
    else:
        update.apply_update(prepared.job_path, start=start)
        assert (install / "FanyiJiVoiceNext.exe").read_bytes() == b"new executable"
    for relative, content in weights.items():
        assert (install / relative).read_bytes() == content
        assert not (prepared.cache_dir / "backup" / relative).exists()
    assert (install / "settings/voice.settings.json").read_bytes() == b"user settings"
    assert (install / "models/model.gguf").read_bytes() == b"user model"


def test_shipped_torch_heuristic_code_can_be_prepared_and_installed(tmp_path):
    names = ("__init__", "_MixedMMA100", "_MixedMMH100", "_MMRankingA100", "_MMRankingH100", "_PadMMA100")
    directory = "_internal/torch/_inductor/autoheuristic/artifacts"
    files = {"FanyiJiVoiceNext.exe": b"new executable"}
    files.update({f"{directory}/{name}.py": ("# torch heuristic " + name).encode() for name in names})
    downloaded = archive(tmp_path, files, extra={directory + "/": b""})
    install = installation(tmp_path)
    prepared = client(tmp_path).prepare(downloaded, install)
    update.apply_update(prepared.job_path, restart=False)
    for relative, content in files.items():
        assert (install / relative).read_bytes() == content
    assert (install / "models/model.gguf").read_bytes() == b"user model"
    assert (install / "settings/voice.settings.json").read_bytes() == b"user settings"


@pytest.mark.parametrize("relative", [
    "artifacts/_MMRankingA100.py", "_internal/other/artifacts/_MMRankingA100.py",
    "_internal/torch/_inductor/autoheuristic/artifacts/private.py",
    "_internal/torch/_inductor/autoheuristic/artifacts/_MMRankingA100.pyc",
    "_internal/torch/_inductor/autoheuristic/artifacts/_MMRankingA100.pyd",
    "_internal/torch/_inductor/autoheuristic/artifacts/model.gguf",
    "_internal/torch/_inductor/autoheuristic/artifacts/model.safetensors",
    "_internal/torch/_inductor/autoheuristic/artifacts/nested/_MMRankingA100.py",
    "_internal/torch/_inductor/autoheuristic/artifacts/settings.json",
])
def test_torch_artifacts_exception_rejects_other_files_without_install(tmp_path, relative):
    install = installation(tmp_path)
    downloaded = archive(tmp_path, {"FanyiJiVoiceNext.exe": b"new", relative: b"excluded"})
    with pytest.raises(update.UpdateError):
        client(tmp_path).prepare(downloaded, install)
    assert (install / "FanyiJiVoiceNext.exe").read_bytes() == b"old executable"


@pytest.mark.parametrize("relative", [
    "artifacts", "_internal/other/artifacts",
    "_internal/torch/_inductor/autoheuristic/artifacts/nested",
    "_internal/torch/_inductor/autoheuristic/artifacts/_MMRankingA100.py",
])
def test_torch_artifacts_exception_does_not_allow_unlisted_directories(relative):
    with pytest.raises(update.UpdateError):
        update._safe_path(relative, directory=True)


def runtime_descriptor(body, *, count=3):
    chunks = [body[index * len(body) // count:(index + 1) * len(body) // count] for index in range(count)]
    parts = [{"name": f"runtime.zip.{index:03d}", "url": URL.replace("program.zip", f"runtime.zip.{index:03d}"),
              "size": len(chunk), "sha256": hashlib.sha256(chunk).hexdigest()}
             for index, chunk in enumerate(chunks, 1)]
    descriptor = {"schema_version": 1, "application": update.APPLICATION, "platform": update.PLATFORM,
                  "kind": "runtime-bundle", "version": VERSION, "build_id": "voice-ab-test-r1",
                  "name": "runtime.zip", "size": len(body), "sha256": hashlib.sha256(body).hexdigest(), "parts": parts}
    return descriptor, dict(zip((part["url"] for part in parts), chunks))


class RuntimeResponse(Response):
    def __init__(self, body, *, offset=0, total=None, status=None):
        super().__init__(body, len(body))
        self.status = status if status is not None else (206 if offset else 200)
        if self.status == 206:
            self.headers["Content-Range"] = f"bytes {offset}-{(total or len(body)) - 1}/{total or len(body)}"


def runtime_network(monkeypatch, bodies, calls=None):
    def open_part(self, url, **kwargs):
        headers = kwargs.get("headers", {})
        if calls is not None:
            calls.append((url, dict(headers)))
        offset = int(headers.get("Range", "bytes=0-")[6:-1])
        return RuntimeResponse(bodies[url][offset:], offset=offset, total=len(bodies[url]))
    monkeypatch.setattr(update.UpdateClient, "_open", open_part)


def test_runtime_bundle_download_order_hash_cache_and_progress(monkeypatch, tmp_path):
    body = b"first second third" * 30000
    descriptor, bodies = runtime_descriptor(body)
    calls, observed = [], []
    runtime_network(monkeypatch, bodies, calls)
    downloaded = update.download_runtime_bundle(descriptor, tmp_path / "downloads", lambda *row: observed.append(row), repository=REPOSITORY)
    assert downloaded.archive_path.read_bytes() == body
    assert downloaded.descriptor.build_id == descriptor["build_id"]
    assert [url for url, _ in calls] == list(bodies)
    assert observed[-1] == (len(body), len(body))
    assert update.parse_runtime_bundle_descriptor(downloaded.descriptor, repository=REPOSITORY).as_dict() == descriptor
    calls.clear()
    assert update.download_runtime_bundle(descriptor, tmp_path / "downloads", repository=REPOSITORY) == downloaded
    assert not calls


@pytest.mark.parametrize("mutate", [
    lambda d: d.update(extra=True), lambda d: d.update(schema_version=True), lambda d: d.update(application="other"),
    lambda d: d.update(platform="linux"), lambda d: d.update(kind="program"), lambda d: d.update(version="latest"),
    lambda d: d.update(build_id="../outside"), lambda d: d.update(name="../runtime.zip"),
    lambda d: d.update(size=update.MAX_RUNTIME_DOWNLOAD_BYTES + 1), lambda d: d.update(sha256="A" * 64),
    lambda d: d.update(size=d["size"] + 1), lambda d: d["parts"][0].update(size=update.MAX_DOWNLOAD_BYTES),
    lambda d: d["parts"][0].update(sha256="invalid"),
    lambda d: d["parts"][0].update(url=d["parts"][0]["url"].replace("/v0.5.0/", "/v0.4.8/")),
    lambda d: d["parts"][0].update(url=d["parts"][0]["url"].replace(REPOSITORY, "other/repository")),
    lambda d: d["parts"][0].update(url=d["parts"][0]["url"] + "?token=SECRET"),
    lambda d: d["parts"][0].update(url=d["parts"][0]["url"].replace("https://github.com", "https://@github.com")),
    lambda d: d["parts"][0].update(url=d["parts"][1]["url"]),
    lambda d: d["parts"].append(dict(d["parts"][0])), lambda d: d.update(parts=[]),
])
def test_runtime_descriptor_is_strict_and_bound_to_repo_version_asset(mutate):
    descriptor, _ = runtime_descriptor(b"abcdefghij")
    mutate(descriptor)
    with pytest.raises(update.UpdateError):
        update.parse_runtime_bundle_descriptor(descriptor, repository=REPOSITORY)


def test_runtime_partial_interruption_retries_with_range(monkeypatch, tmp_path):
    body = b"abcdefgh" * 50000
    descriptor, bodies = runtime_descriptor(body, count=1)
    calls = []
    class Interrupted(RuntimeResponse):
        def read(self, size=-1):
            if self.tell():
                raise OSError("injected connection reset")
            return super().read(size)
    def open_part(self, url, **kwargs):
        headers = kwargs["headers"]
        calls.append(dict(headers))
        offset = int(headers.get("Range", "bytes=0-")[6:-1])
        return (Interrupted(body) if len(calls) == 1
                else RuntimeResponse(body[offset:],offset=offset,total=len(body)))
    monkeypatch.setattr(update.UpdateClient, "_open", open_part)
    result = update.download_runtime_bundle(descriptor, tmp_path, repository=REPOSITORY)
    assert result.archive_path.read_bytes() == body
    assert "Range" not in calls[0]
    assert calls[1]["Range"] == "bytes=262144-"


def test_runtime_cancel_preserves_verified_parts_and_resumes_other_part(monkeypatch, tmp_path):
    body = b"abcdefgh" * 100000
    descriptor, bodies = runtime_descriptor(body, count=2)
    calls, event = [], threading.Event()
    runtime_network(monkeypatch, bodies, calls)
    first_size = descriptor["parts"][0]["size"]
    def cancel_after_second_prefix(done, total):
        if first_size < done < total:
            event.set()
    with pytest.raises(update.UpdateCancelled):
        update.download_runtime_bundle(descriptor, tmp_path, cancel_after_second_prefix, event, repository=REPOSITORY)
    assert len(list(tmp_path.glob("runtime-*/part-000.verified"))) == 1
    assert len(list(tmp_path.glob("runtime-*/part-001.verified.part"))) == 1
    event.clear()
    calls.clear()
    result = update.download_runtime_bundle(descriptor,tmp_path,cancel_event=event,repository=REPOSITORY)
    assert result.archive_path.read_bytes() == body
    assert len(calls) == 1 and calls[0][1]["Range"] == "bytes=262144-"


def test_runtime_bad_sha_is_quarantined_and_clean_retry_succeeds(monkeypatch, tmp_path):
    body = b"correct bytes"
    descriptor, _ = runtime_descriptor(body, count=1)
    calls = []
    def open_part(self, url, **kwargs):
        calls.append(kwargs)
        return RuntimeResponse(b"!" * len(body) if len(calls) == 1 else body)
    monkeypatch.setattr(update.UpdateClient, "_open", open_part)
    result = update.download_runtime_bundle(descriptor, tmp_path, repository=REPOSITORY)
    assert result.archive_path.read_bytes() == body
    assert len(calls) == 2 and all("Range" not in call["headers"] for call in calls)
    assert [path.read_bytes() for path in tmp_path.glob("runtime-*/*.bad-*")] == [b"!" * len(body)]


@pytest.mark.parametrize("ignore_range", [True, False])
def test_runtime_range_response_cannot_append_wrong_bytes(monkeypatch, tmp_path, ignore_range):
    body = b"correct bytes" * 30000
    descriptor, _ = runtime_descriptor(body, count=1)
    event = threading.Event()
    runtime_network(monkeypatch, {descriptor["parts"][0]["url"]:body})
    with pytest.raises(update.UpdateCancelled):
        update.download_runtime_bundle(descriptor,tmp_path,lambda *_:event.set(),event,repository=REPOSITORY)
    part = next(tmp_path.glob("runtime-*/*.part"))
    part.write_bytes(b"wrong")
    def open_part(self,url,**kwargs):
        assert kwargs["headers"]["Range"] == "bytes=5-"
        response = RuntimeResponse(body,status=200 if ignore_range else 206)
        if not ignore_range:
            response.headers["Content-Range"] = f"bytes 0-{len(body)-1}/{len(body)}"
        return response
    monkeypatch.setattr(update.UpdateClient,"_open",open_part)
    if ignore_range:
        assert update.download_runtime_bundle(descriptor,tmp_path,repository=REPOSITORY).archive_path.read_bytes() == body
    else:
        with pytest.raises(update.UpdateError):
            update.download_runtime_bundle(descriptor,tmp_path,repository=REPOSITORY)
        assert part.read_bytes() == b"wrong"


def test_runtime_full_merge_sha_blocks_incorrect_part_order(monkeypatch, tmp_path):
    body = b"111122223333"
    descriptor,bodies = runtime_descriptor(body)
    descriptor["parts"] = list(reversed(descriptor["parts"]))
    runtime_network(monkeypatch,bodies)
    with pytest.raises(update.UpdateError,match="合并 SHA256"):
        update.download_runtime_bundle(descriptor,tmp_path,repository=REPOSITORY)
    assert not list(tmp_path.glob("runtime-*/runtime.zip"))


def test_runtime_cache_hardlink_rejected_without_modifying_external_file(monkeypatch,tmp_path):
    body = b"correct bytes"
    descriptor,bodies = runtime_descriptor(body,count=1)
    runtime_network(monkeypatch,bodies)
    downloaded = update.download_runtime_bundle(descriptor,tmp_path / "cache",repository=REPOSITORY)
    outside = tmp_path / "outside"
    os.link(downloaded.archive_path,outside)
    with pytest.raises(update.UpdateError,match="普通文件"):
        update.download_runtime_bundle(descriptor,tmp_path / "cache",repository=REPOSITORY)
    assert outside.read_bytes() == body


@pytest.mark.parametrize("ready_build", [None,"wrong-build","voice-ab-test-r1"])
def test_runtime_prepare_checks_build_marker_and_preserves_user_data(monkeypatch,tmp_path,ready_build):
    files = {"FanyiJiVoiceNext.exe":b"new runtime executable","_internal/library.dll":b"new runtime library"}
    if ready_build:
        files["_updater/runtime-ready.json"] = json.dumps({"schema_version":1,"application":update.APPLICATION,
                                                        "version":VERSION,"build_id":ready_build}).encode()
    package = archive(tmp_path,files)
    descriptor,bodies = runtime_descriptor(package.archive_path.read_bytes())
    runtime_network(monkeypatch,bodies)
    downloaded = update.download_runtime_bundle(descriptor,tmp_path / "downloads",repository=REPOSITORY)
    install = installation(tmp_path)
    if ready_build != "voice-ab-test-r1":
        with pytest.raises(update.UpdateError,match="标识|runtime-ready"):
            update.prepare_runtime_bundle(downloaded,install,repository=REPOSITORY)
        assert not list(downloaded.archive_path.parent.glob("prepare/install-*/job.json"))
    else:
        prepared = update.prepare_runtime_bundle(downloaded,install,repository=REPOSITORY)
        assert prepared.version == VERSION
        backup = update.apply_update(prepared.job_path,restart=False)
        assert (backup / "FanyiJiVoiceNext.exe").read_bytes() == b"old executable"
        assert (install / "FanyiJiVoiceNext.exe").read_bytes() == files["FanyiJiVoiceNext.exe"]
    assert (install / "models/model.gguf").read_bytes() == b"user model"
    assert (install / "settings/voice.settings.json").read_bytes() == b"user settings"


@pytest.mark.parametrize("path", ["_internal/jieba/lac_small/model_baseline/params",
    "_internal/Jieba/LAC_SMALL/model_baseline/model", "_internal/example/__pycache__/module.pyc"])
def test_hidden_dependency_weights_and_generated_cache_are_rejected(tmp_path,path):
    package = archive(tmp_path,{"FanyiJiVoiceNext.exe":b"new",path:b"hidden weight or generated cache"})
    with pytest.raises(update.UpdateError):
        client(tmp_path).prepare(package,installation(tmp_path))


def test_runtime_cancel_before_network_and_disk_error_do_not_retry(monkeypatch,tmp_path):
    descriptor,bodies = runtime_descriptor(b"correct bytes",count=1)
    calls = []
    runtime_network(monkeypatch,bodies,calls)
    event = threading.Event()
    event.set()
    with pytest.raises(update.UpdateCancelled):
        update.download_runtime_bundle(descriptor,tmp_path,cancel_event=event,repository=REPOSITORY)
    assert not calls and not list(tmp_path.iterdir())
    event.clear()
    monkeypatch.setattr(update.os,"fsync",lambda _: (_ for _ in ()).throw(OSError("injected disk failure")))
    with pytest.raises(update.UpdateError,match="缓存读写"):
        update.download_runtime_bundle(descriptor,tmp_path,repository=REPOSITORY)
    assert len(calls)==1 and not list(tmp_path.glob("runtime-*/*.verified"))


def test_runtime_part_mutation_during_merge_is_detected(monkeypatch,tmp_path):
    descriptor,bodies = runtime_descriptor(b"111122223333")
    runtime_network(monkeypatch,bodies)
    def mutate(done,total):
        if done==total:
            first = next(tmp_path.glob("runtime-*/part-000.verified"))
            first.write_bytes(b"!" * first.stat().st_size)
    with pytest.raises(update.UpdateError,match="合并期间"):
        update.download_runtime_bundle(descriptor,tmp_path,mutate,repository=REPOSITORY)
    assert not list(tmp_path.glob("runtime-*/runtime.zip"))


def test_runtime_rejects_recursive_bootstrap_and_applies_ready_marker_last(monkeypatch,tmp_path):
    ready = json.dumps({"schema_version":1,"application":update.APPLICATION,"version":VERSION,
                        "build_id":"voice-ab-test-r1"}).encode()
    files = {"_updater/runtime-ready.json":ready,"FanyiJiVoiceNext.exe":b"new exe",
             "runtime/llama-server.exe":b"new runtime"}
    package = archive(tmp_path,files)
    descriptor,bodies = runtime_descriptor(package.archive_path.read_bytes())
    runtime_network(monkeypatch,bodies)
    downloaded = update.download_runtime_bundle(descriptor,tmp_path / "downloads",repository=REPOSITORY)
    install = installation(tmp_path)
    prepared = update.prepare_runtime_bundle(downloaded,install,repository=REPOSITORY)
    original,order = update._atomic_copy,[]
    def copy(source,target):
        order.append(target.relative_to(install).as_posix())
        original(source,target)
    monkeypatch.setattr(update,"_atomic_copy",copy)
    update.apply_update(prepared.job_path,restart=False)
    assert order[-1] == "_updater/runtime-ready.json"
    files["_updater/runtime-bootstrap.json"] = b"{}"
    package = archive(tmp_path,files)
    descriptor,bodies = runtime_descriptor(package.archive_path.read_bytes())
    runtime_network(monkeypatch,bodies)
    downloaded = update.download_runtime_bundle(descriptor,tmp_path / "other-downloads",repository=REPOSITORY)
    with pytest.raises(update.UpdateError,match="bootstrap"):
        update.prepare_runtime_bundle(downloaded,install,repository=REPOSITORY)
    assert not list(downloaded.archive_path.parent.glob("prepare/install-*/job.json"))
