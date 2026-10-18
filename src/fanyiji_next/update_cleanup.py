"""Remove only one updater transaction after the new GUI confirms startup.

This module imports no Qt, audio or model code. A readiness marker proves files
were installed; the caller must separately wait for the real GUI event loop.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import os
from pathlib import Path, PurePosixPath
import re
import secrets
import stat
import uuid


APPLICATION = "fanyiji-voice-next"
RECEIPT_NAME = "startup-receipt.json"
PENDING_NAME = "update-cleanup-pending.json"
SIDECAR_NAME = "cleanup-downloads.json"
MANIFEST_NAME = "cleanup-program-manifest.json"
MAX_JSON_BYTES = 32 * 1024**2
_INSTALL = re.compile(r"install-[0-9a-f]{32}\Z")
_DOWNLOAD = re.compile(r"download-[0-9a-f]{32}\Z")
_RUNTIME = re.compile(r"runtime-[0-9a-f]{64}\Z")
_TOKEN = re.compile(r"[0-9a-f]{64}\Z")
_ARCHIVE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,195}\.zip\Z")
_RUNTIME_FILE = re.compile(r"(?:part-\d{3}\.verified(?:\.part)?|merged\.zip\.part|[A-Za-z0-9][A-Za-z0-9._-]{0,195}\.zip)(?:\.bad-[0-9a-f]{32})?\Z")
_PROTECTED = {"models", "settings", "user-data", "source", "src", "tests", "tools", "audio", "recordings"}
_AUDIO = {".wav", ".flac", ".mp3", ".ogg", ".m4a", ".mp4", ".webm", ".aac", ".opus"}


def _plain(path, *, directory=False, missing=False):
    path = Path(path)
    if not path.is_absolute() or ".." in path.parts:
        raise ValueError("清理路径必须是绝对路径")
    for ancestor in (path, *path.parents):
        if ancestor.is_symlink() or ancestor.is_junction():
            raise ValueError("清理拒绝链接或 junction")
    try:
        info = path.lstat()
    except FileNotFoundError:
        if missing:
            return None
        raise
    if getattr(info, "st_file_attributes", 0) & 0x400:
        raise ValueError("清理拒绝 reparse 路径")
    if directory:
        if not stat.S_ISDIR(info.st_mode):
            raise ValueError("清理目录不是普通目录")
    elif not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
        raise ValueError("清理文件不是无 hardlink 的普通文件")
    return info


def _protected_base(path):
    if any(part.casefold() in _PROTECTED or part.casefold().endswith("_models") for part in Path(path).parts):
        raise ValueError("清理路径涉及模型、设置、源码或录音")


def _identity(path):
    value = _plain(path)
    return [value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns]


def _digest(path):
    _plain(path)
    checksum = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024**2), b""):
            checksum.update(block)
    return checksum.hexdigest()


def _read(path):
    _plain(path)
    with Path(path).open("rb") as stream:
        data = stream.read(MAX_JSON_BYTES + 1)
    if len(data) > MAX_JSON_BYTES:
        raise ValueError("更新清理元数据超过限制")
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("更新清理元数据含重复字段")
            result[key] = value
        return result
    value = json.loads(data.decode("utf-8"), object_pairs_hook=unique)
    if not isinstance(value, dict):
        raise ValueError("更新清理元数据不是对象")
    return value


def _write(path, value):
    path = Path(path)
    _plain(path.parent, directory=True)
    _plain(path, missing=True)
    temporary = path.with_name(path.name + ".tmp-" + uuid.uuid4().hex)
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _job(job_path):
    from .app_update import _version_key
    path = Path(job_path).absolute()
    if path.name != "job.json" or not _INSTALL.fullmatch(path.parent.name):
        raise ValueError("清理任务不是 updater 创建的 install 事务")
    _protected_base(path.parent)
    value = _read(path)
    required = {"schema_version", "application", "version", "install_dir", "stage_dir", "backup_dir", "executable", "pid", "restart_args"}
    if set(value) not in (required, required | {"runtime_build_id"}):
        raise ValueError("清理安装任务字段无效")
    if type(value["schema_version"]) is not int or value["schema_version"] != 1 or value["application"] != APPLICATION or value["executable"] != "FanyiJiVoiceNext.exe":
        raise ValueError("清理安装任务产品无效")
    if (type(value["pid"]) is not int or value["pid"] < 0 or not isinstance(value["restart_args"], list)
            or any(not isinstance(item, str) or "\x00" in item for item in value["restart_args"])
            or ("runtime_build_id" in value and (not isinstance(value["runtime_build_id"], str)
                or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", value["runtime_build_id"])))):
        raise ValueError("清理安装任务进程或运行库身份无效")
    _version_key(value["version"])
    install = Path(value["install_dir"])
    _plain(install, directory=True)
    _protected_base(install)
    if Path(value["stage_dir"]) != path.parent / "staged" or Path(value["backup_dir"]) != path.parent / "backup":
        raise ValueError("清理 staged/backup 不属于本事务")
    if install.resolve() == path.parent.resolve() or install.resolve().is_relative_to(path.parent.resolve()) or path.parent.resolve().is_relative_to(install.resolve()):
        raise ValueError("清理事务不能与安装目录重叠")
    return path, value, install


def _cache_relation(root, job_dir, install):
    root = Path(root)
    _plain(root, directory=True)
    _protected_base(root)
    root = root.resolve()
    relative = job_dir.resolve().relative_to(root)
    valid = (len(relative.parts) == 1 and _INSTALL.fullmatch(relative.name)) or (
        len(relative.parts) == 3 and _RUNTIME.fullmatch(relative.parts[0])
        and relative.parts[1] == "prepare" and _INSTALL.fullmatch(relative.parts[2]))
    if not valid or root == install.resolve() or root.is_relative_to(install.resolve()) or install.resolve().is_relative_to(root):
        raise ValueError("清理缓存根不属于本安装任务")
    return root


def _download_name(folder, name):
    return (name in {"program.zip", "program.zip.part"} if _DOWNLOAD.fullmatch(folder.name)
            else bool(_RUNTIME.fullmatch(folder.name) and _RUNTIME_FILE.fullmatch(name)))


def record_download_cleanup(job_path, archive_path, cache_root):
    """Record verified updater-owned downloads; external local ZIPs are untouched."""
    archive = Path(archive_path).absolute()
    root = Path(cache_root).absolute()
    # Local QA/manual update archives outside the downloader cache are valid,
    # but being a valid archive never gives the updater ownership of them.
    if not archive.is_relative_to(root):
        return None
    path, job, install = _job(job_path)
    _plain(root, directory=True)
    _protected_base(root)
    _plain(archive)
    # prepare_runtime_bundle may stage into an independent custom cache. Both
    # locations remain valid, but that job does not own the download cache.
    if not path.parent.resolve().is_relative_to(root.resolve()):
        return None
    relative = path.parent.resolve().relative_to(root.resolve())
    if not ((len(relative.parts) == 1 and _INSTALL.fullmatch(relative.name)) or (
            len(relative.parts) == 3 and _RUNTIME.fullmatch(relative.parts[0])
            and relative.parts[1] == "prepare" and _INSTALL.fullmatch(relative.parts[2]))):
        return None
    root = _cache_relation(root, path.parent, install)
    folder = archive.parent
    if folder.parent != root or not (_DOWNLOAD.fullmatch(folder.name) or _RUNTIME.fullmatch(folder.name)):
        return None
    if not _download_name(folder, archive.name) or not _ARCHIVE.fullmatch(archive.name):
        return None
    _plain(folder, directory=True)
    _plain(archive)
    entries = []
    for item in folder.iterdir():
        if _download_name(folder, item.name):
            entries.append([item.name, _identity(item)])
    _write(path.parent / SIDECAR_NAME, {"schema_version": 1, "application": APPLICATION,
           "job_dir": str(path.parent), "install_dir": str(install), "cache_root": str(root),
           "download_dir": str(folder), "files": sorted(entries)})


def _program_name(relative, *, directory=False):
    from .app_update import _safe_path, _program_path, PROGRAM_DIRECTORIES
    if relative == "update-manifest.json" and not directory:
        return
    path = _safe_path(relative, directory=directory)
    if not (directory and len(path.parts) == 1 and path.name.casefold() in PROGRAM_DIRECTORIES):
        _program_path(path)
    if path.suffix.casefold() in _AUDIO:
        raise ValueError("清理拒绝录音或媒体文件")


def _snapshot_tree(base, kind):
    _plain(base, directory=True)
    files, directories, pending = [], ["."], [base]
    while pending:
        current = pending.pop()
        _plain(current, directory=True)
        for item in current.iterdir():
            relative = item.relative_to(base).as_posix()
            info = item.lstat()
            directory = stat.S_ISDIR(info.st_mode)
            _plain(item, directory=directory)
            _program_name(relative, directory=directory)
            if directory:
                directories.append(relative)
                pending.append(item)
            else:
                files.append([relative, _identity(item)])
            if len(files) + len(directories) > 200000:
                raise ValueError("清理文件清单超过限制")
    return {"base": str(base), "kind": kind, "files": sorted(files), "directories": sorted(directories)}


def _download_target(path, job, install):
    sidecar = path.parent / SIDECAR_NAME
    if not os.path.lexists(sidecar):
        return None
    value = _read(sidecar)
    if (set(value) != {"schema_version", "application", "job_dir", "install_dir", "cache_root", "download_dir", "files"}
            or type(value["schema_version"]) is not int or value["schema_version"] != 1
            or value["application"] != APPLICATION or value["job_dir"] != str(path.parent)
            or value["install_dir"] != str(install)):
        raise ValueError("下载清理 sidecar 与安装事务不匹配")
    root = _cache_relation(Path(value["cache_root"]), path.parent, install)
    folder = Path(value["download_dir"])
    if folder.parent != root or not (_DOWNLOAD.fullmatch(folder.name) or _RUNTIME.fullmatch(folder.name)):
        raise ValueError("下载清理 sidecar 越界")
    _plain(folder, directory=True)
    files = value["files"]
    if not isinstance(files, list) or len(files) > 100:
        raise ValueError("下载清理文件清单无效")
    for name, identity in files:
        if not isinstance(name, str) or not _download_name(folder, name) or Path(name).name != name:
            raise ValueError("下载清理文件名越界")
        if _identity(folder / name) != identity:
            raise ValueError("下载缓存在登记后发生变化")
    return {"base": str(folder), "kind": "download", "files": files, "directories": ["."]}


def _pointer_path(install):
    updater = install / "_updater"
    if not updater.exists():
        updater.mkdir()
    _plain(updater, directory=True)
    return updater / PENDING_NAME


def _pointers(install):
    path = install / "_updater" / PENDING_NAME
    if not os.path.lexists(path):
        return []
    value = _read(path)
    if (set(value) != {"schema_version", "application", "install_dir", "receipts"}
            or type(value["schema_version"]) is not int or value["schema_version"] != 1
            or value["application"] != APPLICATION or value["install_dir"] != str(install)
            or not isinstance(value["receipts"], list) or len(value["receipts"]) > 128):
        raise ValueError("待清理指针格式或安装身份不匹配")
    result = []
    for row in value["receipts"]:
        if not isinstance(row, dict) or set(row) != {"receipt", "token"} or not isinstance(row["token"], str) or not _TOKEN.fullmatch(row["token"]):
            raise ValueError("待清理指针项无效")
        receipt = Path(row["receipt"])
        if not receipt.is_absolute() or receipt.name != RECEIPT_NAME or not _INSTALL.fullmatch(receipt.parent.name):
            raise ValueError("待清理指针不是 updater 事务")
        result.append((receipt, row["token"]))
    return result


def _save_pointers(install, contexts):
    path = _pointer_path(install)
    _write(path, {"schema_version": 1, "application": APPLICATION, "install_dir": str(install),
                  "receipts": [{"receipt": str(path), "token": token} for path, token in contexts]})


def create_startup_receipt(job_path):
    """Called by the new helper after copying files, before starting the new EXE."""
    from .app_update import _manifest
    path, job, install = _job(job_path)
    manifest_path = Path(job["stage_dir"]) / "update-manifest.json"
    _plain(manifest_path)
    manifest_bytes = manifest_path.read_bytes()
    rows = _manifest(manifest_bytes, job["version"])
    executable_sha = next(row["sha256"] for row in rows if row["path"].casefold() == "fanyijivoicenext.exe")
    if _digest(install / job["executable"]) != executable_sha:
        raise ValueError("新 EXE 尚未安装完成，不能创建启动凭据")
    targets = [_snapshot_tree(Path(job["stage_dir"]), "stage"), _snapshot_tree(Path(job["backup_dir"]), "backup")]
    download = _download_target(path, job, install)
    if download:
        targets.append(download)
    helper = path.parent / "FanyiJiUpdate.exe"
    if os.path.lexists(helper):
        targets.append({"base": str(path.parent), "kind": "helper", "files": [[helper.name, _identity(helper)]], "directories": []})
    receipt_path = path.parent / RECEIPT_NAME
    if receipt_path.exists():
        raise ValueError("本事务已有启动凭据，不覆盖")
    # Retain the small signed-by-feed manifest after staged files are removed.
    _write(path.parent / MANIFEST_NAME, json.loads(manifest_bytes))
    token = secrets.token_hex(32)
    receipt = {"schema_version": 1, "application": APPLICATION, "version": job["version"],
               "job_dir": str(path.parent), "install_dir": str(install), "job_sha256": _digest(path),
               "manifest_sha256": _digest(path.parent / MANIFEST_NAME), "executable_sha256": executable_sha,
               "runtime_build_id": job.get("runtime_build_id"), "token_sha256": hashlib.sha256(token.encode()).hexdigest(),
               "status": "awaiting_startup", "targets": targets}
    _write(receipt_path, receipt)
    contexts = _pointers(install)
    contexts.append((receipt_path, token))
    _save_pointers(install, contexts)
    return receipt_path, token


def split_cleanup_args(argv):
    """Strip internal restart arguments; malformed contexts never block startup."""
    clean, contexts, index = [], [], 0
    argv = list(argv)
    while index < len(argv):
        item = argv[index]
        if item == "--update-receipt":
            if (index + 3 < len(argv) and argv[index + 2] == "--update-token"
                    and not argv[index + 1].startswith("--") and not argv[index + 3].startswith("--")):
                path, token = Path(argv[index + 1]), argv[index + 3]
                if path.is_absolute() and path.name == RECEIPT_NAME and _INSTALL.fullmatch(path.parent.name) and _TOKEN.fullmatch(token):
                    contexts.append((path, token))
                index += 4
                continue
            index += 1
            if index < len(argv) and not argv[index].startswith("--"):
                index += 1
            continue
        if item == "--update-token":
            index += 1
            if index < len(argv) and not argv[index].startswith("--"):
                index += 1
            continue
        clean.append(item)
        index += 1
    return clean, contexts


def _validate_receipt(receipt_path, token, executable, version):
    from .app_update import _manifest, parse_runtime_bundle_descriptor
    path, job, install = _job(Path(receipt_path).parent / "job.json")
    if Path(receipt_path) != path.parent / RECEIPT_NAME or executable != install / job["executable"]:
        raise ValueError("启动凭据不属于当前 EXE")
    value = _read(receipt_path)
    required = {"schema_version", "application", "version", "job_dir", "install_dir", "job_sha256", "manifest_sha256", "executable_sha256", "runtime_build_id", "token_sha256", "status", "targets"}
    if (set(value) != required or type(value["schema_version"]) is not int or value["schema_version"] != 1
            or value["application"] != APPLICATION or value["version"] != version or job["version"] != version
            or value["job_dir"] != str(path.parent) or value["install_dir"] != str(install)
            or value["runtime_build_id"] != job.get("runtime_build_id")
            or value["status"] not in {"awaiting_startup", "startup_confirmed", "completed"}
            or not isinstance(token, str) or not _TOKEN.fullmatch(token)
            or not hmac.compare_digest(value["token_sha256"], hashlib.sha256(token.encode()).hexdigest())
            or value["job_sha256"] != _digest(path)):
        raise ValueError("启动凭据版本、令牌或任务身份不匹配")
    manifest_path = path.parent / MANIFEST_NAME
    if value["manifest_sha256"] != _digest(manifest_path):
        raise ValueError("保留的程序清单已变化")
    rows = _manifest(manifest_path.read_bytes(), version)
    expected = next(row["sha256"] for row in rows if row["path"].casefold() == "fanyijivoicenext.exe")
    if expected != value["executable_sha256"] or _digest(executable) != expected:
        raise ValueError("当前 EXE 不是本次安装的新程序")
    def ready(expected_version, build_id):
        marker = _read(install / "_updater/runtime-ready.json")
        if (type(marker.get("schema_version")) is not int or marker != {
                "schema_version": 1, "application": APPLICATION, "version": expected_version, "build_id": build_id}):
            raise ValueError("已安装运行库完成标识与启动凭据不匹配")
    if value["runtime_build_id"] is not None:
        ready(version, value["runtime_build_id"])
    descriptor = install / "_updater/runtime-bootstrap.json"
    if os.path.lexists(descriptor):
        spec = parse_runtime_bundle_descriptor(_read(descriptor))
        # Existing installs retain the first-stage descriptor; unlike staging
        # validation, the installed tree may contain both descriptor and marker.
        ready(spec.version, spec.build_id)
    return value, path, install


def _validate_targets(value, path, install):
    targets = value["targets"]
    if not isinstance(targets, list) or len(targets) > 4:
        raise ValueError("清理目标清单无效")
    seen = set()
    for target in targets:
        if not isinstance(target, dict) or set(target) != {"base", "kind", "files", "directories"}:
            raise ValueError("清理目标字段无效")
        base, kind = Path(target["base"]), target["kind"]
        _protected_base(base)
        if kind in {"stage", "backup"}:
            if base != path.parent / {"stage": "staged", "backup": "backup"}[kind]:
                raise ValueError("清理目标不是本事务 stage/backup")
        elif kind == "helper":
            if base != path.parent or target["directories"] or [row[0] for row in target["files"]] != ["FanyiJiUpdate.exe"]:
                raise ValueError("清理助手目标越界")
        elif kind == "download":
            sidecar = _read(path.parent / SIDECAR_NAME)
            root = _cache_relation(Path(sidecar["cache_root"]), path.parent, install)
            if base != Path(sidecar["download_dir"]) or base.parent != root or target["files"] != sidecar["files"] or target["directories"] != ["."]:
                raise ValueError("下载清理目标与 ownership 不一致")
        else:
            raise ValueError("未知清理目标类型")
        key = str(base).casefold()
        if key in seen or base == install or base.is_relative_to(install) or install.is_relative_to(base):
            raise ValueError("清理目标重复或与程序目录重叠")
        seen.add(key)
        _plain(base, directory=True, missing=True)
        if not isinstance(target["files"], list) or not isinstance(target["directories"], list):
            raise ValueError("清理快照无效")
        for relative, identity in target["files"]:
            if (not isinstance(relative, str) or PurePosixPath(relative).as_posix() != relative
                    or PurePosixPath(relative).is_absolute() or ".." in PurePosixPath(relative).parts
                    or "\\" in relative or not isinstance(identity, list) or len(identity) != 4
                    or any(type(number) is not int or number < 0 for number in identity)):
                raise ValueError("清理文件快照路径或身份无效")
            if kind in {"stage", "backup"}:
                _program_name(relative)
            elif kind == "download" and (Path(relative).name != relative or not _download_name(base, relative)):
                raise ValueError("下载清理文件名无效")
            current = _plain(base / relative, missing=True)
            if current is not None and _identity(base / relative) != identity:
                raise ValueError("清理文件已被替换，保留待核验")
        for relative in target["directories"]:
            if relative != ".":
                if kind not in {"stage", "backup"}:
                    raise ValueError("下载缓存清理不能递归其它目录")
                _program_name(relative, directory=True)
            _plain(base / relative, directory=True, missing=True)
    return targets


def acknowledge_and_cleanup(context, executable, version):
    """Best-effort worker API. Errors are persisted, never raised to the GUI."""
    executable = Path(executable).absolute()
    install = executable.parent
    reports = []
    try:
        _plain(executable)
        _protected_base(install)
        if isinstance(context, tuple) and len(context) == 2 and isinstance(context[1], str):
            context = [context]
        contexts = list(context or [])
        if any(not isinstance(row, (tuple, list)) or len(row) != 2 for row in contexts):
            raise ValueError("启动清理上下文格式无效")
        supplied = {str(Path(row[0])).casefold() for row in contexts}
        for row in _pointers(install):
            if str(row[0]).casefold() not in supplied:
                contexts.append(row)
    except Exception as error:
        return [{"status": "pending", "error": str(error)}]
    for receipt_path, token in contexts:
        report = {"application": APPLICATION, "version": version, "status": "pending", "removed_files": 0, "removed_bytes": 0}
        trusted = None
        try:
            candidate = Path(receipt_path)
            if candidate.name == RECEIPT_NAME:
                candidate_job, _, candidate_install = _job(candidate.parent / "job.json")
                if candidate_install == install:
                    trusted = candidate_job.parent
            value, path, owner = _validate_receipt(Path(receipt_path), token, executable, version)
            trusted = path.parent
            from .app_update import _installation_lock
            with _installation_lock(owner):
                targets = _validate_targets(value, path, owner)
                # Startup is confirmed before deletion, not inferred from Popen.
                report["startup_confirmed"] = True
                _write(path.parent / "cleanup-status.json", report)
                value["status"] = "startup_confirmed"
                _write(Path(receipt_path), value)
                failures = []
                for target in targets:
                    base = Path(target["base"])
                    for relative, identity in target["files"]:
                        candidate = base / relative
                        try:
                            current = _plain(candidate, missing=True)
                            if current is not None:
                                if _identity(candidate) != identity:
                                    raise ValueError("清理文件在执行前变化")
                                candidate.unlink()
                                report["removed_files"] += 1
                                report["removed_bytes"] += identity[2]
                        except Exception as error:
                            failures.append(f"{relative}: {error}")
                    # Empty-directory removal never discovers or removes files.
                    for relative in sorted(target["directories"], key=lambda item: len(PurePosixPath(item).parts), reverse=True):
                        directory = base / relative
                        try:
                            if _plain(directory, directory=True, missing=True) is not None:
                                directory.rmdir()
                        except OSError:
                            pass  # Unknown files/another prepare job remain protected.
                if failures:
                    report["errors"] = failures[:100]
                else:
                    report["status"] = "completed"
                    value["status"], value["targets"] = "completed", []
                    _write(Path(receipt_path), value)
                    pending = [(p, t) for p, t in _pointers(owner) if p != Path(receipt_path)]
                    _save_pointers(owner, pending)
        except Exception as error:
            report["error"] = str(error)
        if trusted is not None:
            try:
                _write(trusted / "cleanup-status.json", report)
            except Exception:
                pass
        reports.append(report)
    return reports
