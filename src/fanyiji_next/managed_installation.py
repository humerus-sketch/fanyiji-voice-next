"""Explicit project-owned primary installs; ordinary portable installs are unchanged."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import stat
import time
import tomllib
import uuid

APPLICATION = "fanyiji-voice-next"
REGISTRY_NAME = "local-installation.json"
RELOCATION_NAME = "managed-relocation.json"
PENDING_NAME = "update-cleanup-pending.json"
DOWNLOAD_NAME = "cleanup-downloads.json"


def _plain(path: Path):
    path = Path(path)
    if not path.is_absolute() or ".." in path.parts:
        raise ValueError("管理安装路径必须是无越界的绝对路径")
    for part in (path, *path.parents):
        if part.is_symlink() or (hasattr(part, "is_junction") and part.is_junction()):
            raise ValueError("管理安装路径不得经过链接或 junction")
        if part.exists() and getattr(part.lstat(), "st_file_attributes", 0) & 0x400:
            raise ValueError("管理安装路径不得经过 reparse point")
    if path.is_file() and (not stat.S_ISREG(path.stat().st_mode) or path.stat().st_nlink != 1):
        raise ValueError("管理安装登记必须是无硬链接的普通文件")
    return path


def _json(data):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("管理安装 JSON 含重复字段")
            result[key] = value
        return result
    if len(data) > 65536:
        raise ValueError("管理安装登记过大")
    value = json.loads(data.decode("utf-8-sig"), object_pairs_hook=pairs)
    if not isinstance(value, dict):
        raise ValueError("管理安装登记必须是 JSON object")
    return value


def _read(path):
    return _plain(path).read_bytes()


def _write(path, data):
    path = _plain(Path(path))
    temporary = path.with_name(path.name + ".managed-" + uuid.uuid4().hex)
    try:
        with temporary.open("xb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _encoded(value):
    return (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8")


def _rename(source, destination):
    deadline = time.monotonic() + 10
    while True:
        _plain(source)
        _plain(destination)
        if os.path.lexists(destination):
            raise ValueError("目标程序目录已存在，未覆盖")
        try:
            source.rename(destination)
            return
        except OSError as error:
            if getattr(error, "winerror", None) not in {5, 32, 33} or time.monotonic() >= deadline:
                raise
            time.sleep(.2)


@dataclass(frozen=True)
class ManagedInstallation:
    project_root: Path
    registry_path: Path
    registry_bytes: bytes
    value: dict
    program_dir: Path
    model_dir: Path


def read_managed_installation(install_dir, exe_name="FanyiJiVoiceNext.exe"):
    """Inspect exactly project/dist/version/app; never search arbitrary ancestors."""
    from .app_update import _version_key
    install = Path(install_dir).absolute()
    if (exe_name != "FanyiJiVoiceNext.exe" or install.name != "FanyiJiVoiceNext"
            or install.parent.parent.name != "dist"):
        return None
    try:
        _version_key(install.parent.name)
    except ValueError:
        return None
    except RuntimeError:
        return None
    root = install.parents[2]
    metadata, registry = root / "pyproject.toml", root / REGISTRY_NAME
    if not metadata.exists() or not os.path.lexists(registry):
        return None
    _plain(install)
    project = tomllib.loads(_read(metadata).decode("utf-8")).get("project")
    if not isinstance(project, dict) or project.get("name") != APPLICATION:
        return None
    raw = _read(registry)
    value = _json(raw)
    if (set(value) != {"schema_version", "application", "role", "current_version", "program_relative", "model_relative"}
            or type(value["schema_version"]) is not int or value["schema_version"] != 1
            or value["application"] != APPLICATION or not isinstance(value["role"], str)
            or value["role"] not in {"primary", "verification"}):
        raise ValueError("项目主用安装登记格式无效")
    try:
        _version_key(value["current_version"])
    except (ValueError, RuntimeError) as error:
        raise ValueError("项目主用安装登记版本无效") from error
    relative = f"dist/{value['current_version']}/FanyiJiVoiceNext"
    if value["program_relative"] != relative or value["model_relative"] != "FanyiJiVoiceNext_models":
        raise ValueError("项目登记只能指向明确版本程序和项目共享模型目录")
    registered = root / relative
    _plain(registered)
    if value["role"] != "primary" or registered != install:
        return None
    _plain(install / exe_name)
    if not (install / exe_name).is_file():
        raise ValueError("登记的主用 EXE 不存在")
    build = _json(_read(install / "build-info.json"))
    if build.get("application") != APPLICATION or build.get("version") != value["current_version"]:
        raise ValueError("登记版本与主用程序 build-info 不一致")
    return ManagedInstallation(root, registry, raw, value, install, root / value["model_relative"])


def managed_model_root(install_dir, exe_name="FanyiJiVoiceNext.exe"):
    registered = read_managed_installation(install_dir, exe_name)
    return registered.model_dir if registered is not None else None


def _record(registered, version):
    from .app_update import _version_key
    _version_key(version)
    return {"schema_version": 1, "application": APPLICATION, "project_root": str(registered.project_root),
            "registry_sha256": hashlib.sha256(registered.registry_bytes).hexdigest(),
            "source_relative": registered.value["program_relative"], "source_version": registered.value["current_version"],
            "target_relative": f"dist/{version}/FanyiJiVoiceNext", "target_version": version}


def _check_staged_build_info(job):
    from .app_update import ARCHIVE_MANIFEST, _manifest
    stage = Path(job["stage_dir"])
    rows = _manifest(_read(stage / ARCHIVE_MANIFEST), job["version"])
    if not any(row["path"] == "build-info.json" for row in rows):
        raise ValueError("主用跨版本更新必须包含 build-info.json")
    build = _json(_read(stage / "build-info.json"))
    if build.get("application") != APPLICATION or build.get("version") != job["version"]:
        raise ValueError("主用跨版本更新的 build-info 与目标版本不一致")


def record_managed_update(job_path):
    job = _json(_read(Path(job_path)))
    registered = read_managed_installation(Path(job["install_dir"]), job["executable"])
    if registered is None or registered.value["current_version"] == job["version"]:
        return
    _check_staged_build_info(job)
    value = _record(registered, job["version"])
    destination = _plain(registered.project_root / value["target_relative"])
    if os.path.lexists(destination):
        raise ValueError("目标版本程序目录已存在，未覆盖任何程序")
    _write(Path(job_path).parent / RELOCATION_NAME, _encoded(value))


def load_managed_update(job_path):
    path = Path(job_path).absolute()
    sidecar = path.parent / RELOCATION_NAME
    job = _json(_read(path))
    registered = read_managed_installation(Path(job["install_dir"]), job["executable"])
    if not os.path.lexists(sidecar):
        if registered is not None and registered.value["current_version"] != job["version"]:
            raise ValueError("主用跨版本更新缺少目录切换任务；未覆盖程序")
        return None
    value = _json(_read(sidecar))
    if registered is None or value != _record(registered, job["version"]):
        raise ValueError("主用安装登记在准备后发生变化，未移动程序")
    _check_staged_build_info(job)
    destination = _plain(registered.project_root / value["target_relative"])
    if os.path.lexists(destination):
        raise ValueError("目标版本程序目录已存在，未覆盖任何程序")
    return ManagedUpdate(path, registered, destination, job["version"])


def _restart_args(args, previous, current):
    result = list(args)
    options = {"--settings-file", "--model-root", "--report", "--screenshot"}
    for index, item in enumerate(result):
        flag, equal, inline = item.partition("=")
        if flag not in options:
            continue
        value_index = index if equal else index + 1
        if value_index >= len(result):
            continue
        value = inline if equal else result[value_index]
        path = Path(value)
        if path.is_absolute() and path.is_relative_to(previous):
            changed = str(current / path.relative_to(previous))
            result[value_index] = flag + "=" + changed if equal else changed
    return result


class ManagedUpdate:
    def __init__(self, job_path, registered, destination, version):
        self.job_path, self.registered, self.destination, self.version = job_path, registered, destination, version
        self.source = registered.program_dir
        self.moved = False
        self.created_parent = False
        self.saved = {}

    @property
    def current_install(self):
        return self.destination if self.moved else self.source

    def activate(self):
        if _read(self.registered.registry_path) != self.registered.registry_bytes:
            raise ValueError("主用安装登记在安装时发生变化")
        _plain(self.destination)
        if os.path.lexists(self.destination):
            raise ValueError("目标版本程序目录已存在，未覆盖")
        # Validate cleanup's current bindings before moving, then change only
        # install_dir. Cache paths and ownership snapshots remain unchanged.
        from .update_cleanup import _download_target, _pointers
        job = _json(_read(self.job_path))
        _download_target(self.job_path, job, self.source)
        _pointers(self.source)
        for path in (self.job_path, self.job_path.parent / DOWNLOAD_NAME, self.source / "_updater" / PENDING_NAME):
            self.saved[path] = _read(path) if os.path.lexists(path) else None
        if not self.destination.parent.exists():
            self.destination.parent.mkdir()
            self.created_parent = True
        _plain(self.destination.parent)
        _rename(self.source, self.destination)
        self.moved = True
        job["install_dir"] = str(self.destination)
        job["restart_args"] = _restart_args(job["restart_args"], self.source, self.destination)
        _write(self.job_path, _encoded(job))
        for previous in (self.job_path.parent / DOWNLOAD_NAME, self.source / "_updater" / PENDING_NAME):
            raw = self.saved[previous]
            if raw is not None:
                value = _json(raw)
                if value.get("install_dir") != str(self.source):
                    raise ValueError("清理绑定不属于原主用安装")
                value["install_dir"] = str(self.destination)
                target = self.destination / previous.relative_to(self.source) if previous.is_relative_to(self.source) else previous
                _write(target, _encoded(value))
        value = dict(self.registered.value, current_version=self.version,
                     program_relative=f"dist/{self.version}/FanyiJiVoiceNext")
        _write(self.registered.registry_path, _encoded(value))
        return self.destination, job

    def rollback(self):
        if not self.moved:
            if self.created_parent:
                self.destination.parent.rmdir()
            return
        if os.path.lexists(self.source):
            raise ValueError("原程序路径已被占用；未覆盖，保留新路径与备份")
        _rename(self.destination, self.source)
        self.moved = False
        for path, raw in self.saved.items():
            if raw is not None:
                _write(path, raw)
            elif path.name == PENDING_NAME and os.path.lexists(path):
                value = _json(_read(path))
                receipts = value.get("receipts")
                if (value.get("install_dir") != str(self.destination) or not isinstance(receipts, list)
                        or any(Path(row.get("receipt", "")).parent != self.job_path.parent for row in receipts)):
                    raise ValueError("新增清理指针身份变化，未删除")
                path.unlink()
        current = _read(self.registered.registry_path)
        updated = _encoded(dict(self.registered.value, current_version=self.version,
                                program_relative=f"dist/{self.version}/FanyiJiVoiceNext"))
        if current not in (self.registered.registry_bytes, updated):
            raise ValueError("项目登记被外部修改，未覆盖新登记")
        _write(self.registered.registry_path, self.registered.registry_bytes)
        if self.created_parent:
            self.destination.parent.rmdir()
