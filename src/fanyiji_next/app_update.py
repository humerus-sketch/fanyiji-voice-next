"""Explicit, program-only GitHub Release updates; no audio or model imports.

The separate frozen helper is copied to the update cache before the application
exits. It never deletes an installation tree and retains its transaction backup.
"""
from __future__ import annotations

import ctypes
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import Callable
from urllib.error import HTTPError, URLError
from urllib.parse import unquote, urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener
import uuid
import zipfile


APPLICATION = "fanyiji-voice-next"
PLATFORM = "windows-x64"
DEFAULT_REPOSITORY = "humerus-sketch/fanyiji-voice-next"
FEED_NAME = "update-feed.json"
ARCHIVE_MANIFEST = "update-manifest.json"
MAX_JSON_BYTES = 2 * 1024 * 1024
MAX_DOWNLOAD_BYTES = 2 * 1024**3
MAX_EXTRACT_BYTES = 8 * 1024**3
MAX_ARCHIVE_FILES = 100000
PROTECTED_PARTS = frozenset({"models", "settings", "user-data", "artifacts", "logs", "cache"})
PROGRAM_DIRECTORIES = frozenset({"_internal", "licenses", "_updater", "runtime"})
PROGRAM_FILES = frozenset({"fanyijivoicenext.exe", "release-defaults.json", "build-info.json", "readme.md", "readme.txt", "license", "license.txt", "使用说明.txt", "使用说明.md", "公开包说明.md", "说明.txt", "说明.md"})
_REPOSITORY = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*/[A-Za-z0-9][A-Za-z0-9_.-]*\Z")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_VERSION = re.compile(r"(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)(?:-([0-9A-Za-z.-]+))?(?:\+[0-9A-Za-z.-]+)?\Z")
_DEVICE_NAME = re.compile(r"(?:CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9])(?:\..*)?\Z", re.I)
_MODEL_SUFFIXES = frozenset({".onnx", ".gguf", ".safetensors", ".pt", ".pth", ".ckpt", ".tflite"})


class UpdateError(RuntimeError):
    """A network, release, archive, or update transaction failed safely."""


class UpdateCancelled(UpdateError):
    """The user cancelled before installation began."""


@dataclass(frozen=True)
class UpdateInfo:
    version: str
    name: str
    url: str
    sha256: str
    size: int
    release_url: str
    notes: str = ""


@dataclass(frozen=True)
class DownloadedUpdate:
    info: UpdateInfo
    archive_path: Path


@dataclass(frozen=True)
class PreparedUpdate:
    job_path: Path
    install_dir: Path
    version: str

    @property
    def cache_dir(self) -> Path:
        return self.job_path.parent


def _version_key(value: str):
    if not isinstance(value, str) or not (match := _VERSION.fullmatch(value)):
        raise UpdateError(f"无效版本号：{value!r}")
    major, minor, patch = (int(match.group(i)) for i in (1, 2, 3))
    prerelease = match.group(4)
    if prerelease is None:
        return major, minor, patch, 1, ()
    identifiers = prerelease.split(".")
    if any(not p or (p.isdigit() and len(p) > 1 and p.startswith("0")) for p in identifiers):
        raise UpdateError("无效 prerelease 版本号")
    return major, minor, patch, 0, tuple((0, int(p)) if p.isdigit() else (1, p) for p in identifiers)


def _object(value, keys: set[str], label: str):
    if not isinstance(value, dict) or set(value) != keys:
        raise UpdateError(f"{label} 字段不符合更新格式")
    return value


def _positive_size(value, limit: int, label: str) -> int:
    if type(value) is not int or not 0 < value <= limit:
        raise UpdateError(f"{label} 大小无效或超过限制")
    return value


def _sha(value) -> str:
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise UpdateError("SHA256 格式无效")
    return value


def _json(data: bytes):
    try:
        return json.loads(data.decode("utf-8"), object_pairs_hook=_unique_json_keys)
    except (UnicodeError, ValueError) as exc:
        raise UpdateError(f"更新 JSON 无法解析：{exc}") from exc


def _unique_json_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"重复 JSON 字段：{key}")
        result[key] = value
    return result


def _trusted_url(url: str, repository: str, *, api: bool = False, redirected: bool = False) -> str:
    if not isinstance(url, str):
        raise UpdateError("下载地址格式无效")
    parts = urlsplit(url)
    try:
        port = parts.port
    except ValueError as exc:
        raise UpdateError("下载端口无效") from exc
    if (parts.scheme != "https" or parts.username or parts.password or parts.fragment
            or port not in (None, 443) or "\\" in url or any(ord(c) < 32 for c in url)):
        raise UpdateError("更新只接受可信 HTTPS 地址")
    host = (parts.hostname or "").casefold()
    path = unquote(parts.path)
    if ".." in PurePosixPath(path).parts or "\\" in path:
        raise UpdateError("下载地址包含非法路径")
    if api and host == "api.github.com" and path == f"/repos/{repository}/releases/latest":
        return url
    if not api and host == "github.com" and path.startswith(f"/{repository}/releases/download/"):
        return url
    if not api and redirected and host in {"release-assets.githubusercontent.com", "objects.githubusercontent.com"}:
        return url
    raise UpdateError("更新地址不属于当前 GitHub 仓库或官方 Release 资源域")


class _TrustedRedirect(HTTPRedirectHandler):
    def __init__(self, repository: str, api: bool):
        self.repository, self.api = repository, api

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        _trusted_url(newurl, self.repository, api=self.api, redirected=True)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def _default_cache() -> Path:
    base = os.environ.get("LOCALAPPDATA")
    return (Path(base) if base else Path.home() / "AppData" / "Local") / "FanyiJiVoiceNext" / "updates"


def _digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


class UpdateClient:
    def __init__(self, repository: str = DEFAULT_REPOSITORY, cache_dir: Path | None = None, timeout: float = 15):
        if not repository or not _REPOSITORY.fullmatch(repository) or repository.endswith((".", "/")):
            raise UpdateError("此版本尚未配置有效的 GitHub 更新仓库")
        if not 0 < timeout <= 15:
            raise UpdateError("更新网络超时必须在 0 到 15 秒之间")
        self.repository = repository
        self.cache_dir = Path(cache_dir) if cache_dir else _default_cache()
        self.timeout = timeout

    def _open(self, url: str, *, api: bool = False):
        _trusted_url(url, self.repository, api=api)
        opener = build_opener(_TrustedRedirect(self.repository, api))
        request = Request(url, headers={"User-Agent": "FanyiJiVoiceNext-Updater", "Accept": "application/vnd.github+json" if api else "application/octet-stream"})
        try:
            response = opener.open(request, timeout=self.timeout)
            try:
                _trusted_url(response.geturl(), self.repository, api=api, redirected=True)
            except Exception:
                response.close()
                raise
            return response
        except HTTPError as exc:
            raise UpdateError(f"GitHub 返回 HTTP {exc.code}") from exc
        except (URLError, OSError) as exc:
            raise UpdateError(f"更新网络请求失败：{exc}") from exc

    def _read_json(self, url: str, *, api: bool = False):
        try:
            with self._open(url, api=api) as response:
                data = response.read(MAX_JSON_BYTES + 1)
        except OSError as exc:
            raise UpdateError(f"读取更新描述失败：{exc}") from exc
        if len(data) > MAX_JSON_BYTES:
            raise UpdateError("更新说明超过大小限制")
        return _json(data)

    def check(self, current_version: str) -> UpdateInfo | None:
        current_key = _version_key(current_version)
        release = self._read_json(f"https://api.github.com/repos/{self.repository}/releases/latest", api=True)
        if not isinstance(release, dict) or release.get("draft") is not False or release.get("prerelease") is not False:
            raise UpdateError("GitHub 最新版本不是正式发布")
        assets = release.get("assets")
        if not isinstance(assets, list):
            raise UpdateError("GitHub Release 缺少资源列表")
        feeds = [a for a in assets if isinstance(a, dict) and a.get("name") == FEED_NAME]
        if len(feeds) != 1:
            raise UpdateError("此 Release 缺少唯一的 update-feed.json，暂不支持软件内更新")
        _positive_size(feeds[0].get("size"), MAX_JSON_BYTES, "更新描述")
        feed = _object(self._read_json(feeds[0].get("browser_download_url")),
                       {"schema_version", "application", "platform", "version", "program"}, "更新描述")
        if type(feed["schema_version"]) is not int or feed["schema_version"] != 1 or feed["application"] != APPLICATION or feed["platform"] != PLATFORM:
            raise UpdateError("更新描述的 schema、产品或平台不匹配")
        version = feed["version"]
        latest_key = _version_key(version)
        if release.get("tag_name") != "v" + version:
            raise UpdateError("更新版本与 GitHub Release tag 不一致")
        program = _object(feed["program"], {"name", "url", "sha256", "size"}, "程序资源")
        name = program["name"]
        if not isinstance(name, str) or not name.endswith(".zip") or _safe_path(name).name != name:
            raise UpdateError("更新 ZIP 文件名无效")
        _trusted_url(program["url"], self.repository)
        _sha(program["sha256"])
        size = _positive_size(program["size"], MAX_DOWNLOAD_BYTES, "程序资源")
        matching = [a for a in assets if isinstance(a, dict) and a.get("name") == name]
        if len(matching) != 1 or matching[0].get("size") != size or matching[0].get("browser_download_url") != program["url"]:
            raise UpdateError("程序资源与 GitHub Release 列表不一致")
        expected_prefix = f"https://github.com/{self.repository}/releases/tag/"
        release_url = release.get("html_url")
        if not isinstance(release_url, str) or release_url != expected_prefix + "v" + version:
            raise UpdateError("GitHub Release 页面与版本不匹配")
        if latest_key <= current_key:
            return None
        notes = release.get("body") or ""
        if not isinstance(notes, str):
            raise UpdateError("GitHub Release 说明格式无效")
        return UpdateInfo(version, name, program["url"], program["sha256"], size, release_url, notes[:20000])

    def download(self, info: UpdateInfo, progress: Callable[[int, int], None] | None = None,
                 cancelled: Callable[[], bool] | None = None) -> DownloadedUpdate:
        _trusted_url(info.url, self.repository)
        _sha(info.sha256)
        _positive_size(info.size, MAX_DOWNLOAD_BYTES, "程序资源")
        _version_key(info.version)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        folder = self.cache_dir / ("download-" + uuid.uuid4().hex)
        folder.mkdir()
        partial = folder / "program.zip.part"
        hasher, received = hashlib.sha256(), 0
        try:
            _cancel(cancelled)
            with self._open(info.url) as response, partial.open("xb") as output:
                header = response.headers.get("Content-Length")
                if header is not None and (not header.isdecimal() or int(header) != info.size):
                    raise UpdateError("下载 Content-Length 与发布描述不一致")
                while True:
                    _cancel(cancelled)
                    block = response.read(256 * 1024)
                    if not block:
                        break
                    received += len(block)
                    if received > info.size:
                        raise UpdateError("下载大小超过发布描述")
                    hasher.update(block)
                    output.write(block)
                    if progress:
                        progress(received, info.size)
            _cancel(cancelled)
            if received != info.size or hasher.hexdigest() != info.sha256:
                raise UpdateError("更新包长度或 SHA256 不匹配；未安装")
            archive = folder / "program.zip"
            partial.replace(archive)
            return DownloadedUpdate(info, archive)
        except UpdateError:
            raise
        except (OSError, URLError) as exc:
            raise UpdateError(f"下载更新失败：{exc}") from exc

    def prepare(self, downloaded: DownloadedUpdate, install_dir: Path, executable: str = "FanyiJiVoiceNext.exe") -> PreparedUpdate:
        if executable != "FanyiJiVoiceNext.exe":
            raise UpdateError("更新目标不是翻译姬语音版程序")
        _reject_link_tree(Path(install_dir))
        install = Path(install_dir).resolve(strict=True)
        if not (install / executable).is_file():
            raise UpdateError("目标目录中没有已安装的翻译姬 EXE")
        archive = Path(downloaded.archive_path).resolve(strict=True)
        if archive.stat().st_size != downloaded.info.size or _digest(archive) != downloaded.info.sha256:
            raise UpdateError("准备安装前更新包校验失败")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        cache = self.cache_dir.resolve()
        if cache == install or cache.is_relative_to(install) or install.is_relative_to(cache):
            raise UpdateError("更新缓存不得与程序目录重叠")
        job_dir = cache / ("install-" + uuid.uuid4().hex)
        job_dir.mkdir()
        stage = job_dir / "staged"
        stage.mkdir()
        _extract_program(archive, stage, downloaded.info.version)
        payload = {"schema_version": 1, "application": APPLICATION, "version": downloaded.info.version,
                   "install_dir": str(install), "stage_dir": str(stage), "backup_dir": str(job_dir / "backup"),
                   "executable": executable, "pid": 0, "restart_args": []}
        job = job_dir / "job.json"
        job.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return PreparedUpdate(job, install, downloaded.info.version)


def _cancel(cancelled):
    if cancelled and cancelled():
        raise UpdateCancelled("更新下载已取消；现有程序和模型未更改")


def _safe_path(value: str) -> PurePosixPath:
    if (not isinstance(value, str) or not value or "\\" in value or ":" in value
            or "\x00" in value or value.startswith("/") or any(ord(c) < 32 for c in value)):
        raise UpdateError("更新包包含非法路径")
    path = PurePosixPath(value)
    if (path.as_posix() != value or any(part in {".", "..", ""} or part.endswith((".", " "))
            or _DEVICE_NAME.fullmatch(part) or any(c in part for c in '<>"|?*') for part in path.parts)):
        raise UpdateError(f"更新包路径不安全：{value}")
    # ctranslate2/models/__init__.py is dependency source, not model data.
    dependency_models = tuple(p.casefold() for p in path.parts[:3]) == ("_internal", "ctranslate2", "models")
    protected = any(part.casefold() in PROTECTED_PARTS and not (index == 2 and dependency_models
                    and (len(path.parts) == 3 or path.suffix.casefold() in {".py", ".pyc", ".pyd"}))
                    for index, part in enumerate(path.parts))
    if (protected or path.name.casefold().endswith(".settings.json") or path.name.casefold() == "settings.json"
            or path.suffix.casefold() in _MODEL_SUFFIXES or path.name.casefold() == "model.bin"):
        raise UpdateError(f"程序更新包不得包含模型、设置或用户资料：{value}")
    return path


def _manifest(data: bytes, version: str) -> list[dict]:
    if len(data) > MAX_JSON_BYTES:
        raise UpdateError("ZIP 文件清单超过限制")
    value = _object(_json(data), {"schema_version", "application", "version", "files"}, "ZIP 文件清单")
    if type(value["schema_version"]) is not int or value["schema_version"] != 1 or value["application"] != APPLICATION or value["version"] != version:
        raise UpdateError("ZIP 文件清单的产品或版本不匹配")
    rows = value["files"]
    if not isinstance(rows, list) or not 1 <= len(rows) <= MAX_ARCHIVE_FILES:
        raise UpdateError("ZIP 文件数量无效")
    seen, total = set(), 0
    for row in rows:
        _object(row, {"path", "sha256", "size"}, "ZIP 文件项")
        path = _safe_path(row["path"])
        _program_path(path)
        key = path.as_posix().casefold()
        if key in seen or key == ARCHIVE_MANIFEST.casefold():
            raise UpdateError("ZIP 文件清单包含重复路径")
        seen.add(key)
        _sha(row["sha256"])
        if type(row["size"]) is not int or not 0 <= row["size"] <= MAX_EXTRACT_BYTES:
            raise UpdateError("ZIP 文件大小无效")
        total += row["size"]
    if total > MAX_EXTRACT_BYTES:
        raise UpdateError("ZIP 解压大小超过限制")
    if "fanyijivoicenext.exe" not in seen:
        raise UpdateError("ZIP 文件清单缺少翻译姬 EXE")
    return rows


def _program_path(path: PurePosixPath):
    if len(path.parts) == 1:
        if path.name.casefold() not in PROGRAM_FILES and path.name != ARCHIVE_MANIFEST:
            raise UpdateError(f"更新包含未知的程序根文件：{path}")
    elif path.parts[0].casefold() not in PROGRAM_DIRECTORIES:
        raise UpdateError(f"更新包含未知的程序目录：{path}")


def _case_path(path: PurePosixPath, spellings: dict):
    for size in range(1, len(path.parts) + 1):
        value = "/".join(path.parts[:size])
        key = value.casefold()
        if key in spellings and spellings[key] != value:
            raise UpdateError("ZIP 包含大小写冲突路径")
        spellings[key] = value


def _extract_program(archive: Path, stage: Path, version: str):
    try:
        with zipfile.ZipFile(archive) as bundle:
            infos = bundle.infolist()
            if len(infos) > MAX_ARCHIVE_FILES * 2:
                raise UpdateError("ZIP 条目数量超过限制")
            paths, files, spellings = {}, {}, {}
            for entry in infos:
                value = entry.filename[:-1] if entry.is_dir() else entry.filename
                path = _safe_path(value)
                _case_path(path, spellings)
                if entry.is_dir():
                    if path.parts[0].casefold() not in PROGRAM_DIRECTORIES:
                        raise UpdateError("ZIP 包含未知的程序根目录")
                else:
                    _program_path(path)
                key = path.as_posix().casefold()
                if key in paths:
                    raise UpdateError("ZIP 包含大小写冲突或重复路径")
                paths[key] = entry
                if stat.S_ISLNK(entry.external_attr >> 16) or entry.flag_bits & 1:
                    raise UpdateError("ZIP 不得包含符号链接或加密条目")
                if not entry.is_dir():
                    files[entry.filename] = entry
            if ARCHIVE_MANIFEST not in files or files[ARCHIVE_MANIFEST].file_size > MAX_JSON_BYTES:
                raise UpdateError("ZIP 缺少有效文件清单")
            rows = _manifest(bundle.read(ARCHIVE_MANIFEST), version)
            if set(files) != {row["path"] for row in rows} | {ARCHIVE_MANIFEST}:
                raise UpdateError("ZIP 内容与文件清单不一致")
            for row in rows:
                entry = files[row["path"]]
                if entry.file_size != row["size"]:
                    raise UpdateError("ZIP 文件大小与清单不一致")
                target = _inside(stage / row["path"], stage)
                target.parent.mkdir(parents=True, exist_ok=True)
                checksum, received = hashlib.sha256(), 0
                with bundle.open(entry) as source, target.open("xb") as output:
                    while block := source.read(256 * 1024):
                        received += len(block)
                        if received > row["size"]:
                            raise UpdateError("ZIP 实际解压大小超出清单")
                        checksum.update(block)
                        output.write(block)
                if received != row["size"] or checksum.hexdigest() != row["sha256"]:
                    raise UpdateError("ZIP 文件 SHA256 不匹配")
            (stage / ARCHIVE_MANIFEST).write_bytes(bundle.read(ARCHIVE_MANIFEST))
    except (OSError, zipfile.BadZipFile, RuntimeError) as exc:
        if isinstance(exc, UpdateError):
            raise
        raise UpdateError(f"更新包无法安全解压：{exc}") from exc


def _inside(path: Path, root: Path) -> Path:
    # Check the lexical path before resolve(): an in-tree symlink could otherwise
    # turn a harmless DLL destination into a protected model/settings file.
    _reject_link_tree(path)
    resolved, base = path.resolve(), root.resolve()
    if resolved == base or not resolved.is_relative_to(base):
        raise UpdateError("更新路径越出限定目录")
    return resolved


def _reject_link_tree(path: Path):
    for part in (path, *path.parents):
        if part.is_symlink() or (hasattr(part, "is_junction") and part.is_junction()):
            raise UpdateError("更新路径不得经过符号链接或 junction")


def _load_job(job_path: Path):
    job_path = Path(job_path).resolve(strict=True)
    if job_path.name != "job.json":
        raise UpdateError("安装任务文件名无效")
    data = job_path.read_bytes()
    if len(data) > MAX_JSON_BYTES:
        raise UpdateError("安装任务超过限制")
    job = _object(_json(data), {"schema_version", "application", "version", "install_dir", "stage_dir", "backup_dir", "executable", "pid", "restart_args"}, "安装任务")
    if type(job["schema_version"]) is not int or job["schema_version"] != 1 or job["application"] != APPLICATION or job["executable"] != "FanyiJiVoiceNext.exe":
        raise UpdateError("安装任务产品或格式无效")
    _version_key(job["version"])
    if type(job["pid"]) is not int or job["pid"] < 0:
        raise UpdateError("等待退出的进程 ID 无效")
    if not isinstance(job["restart_args"], list) or any(not isinstance(a, str) or "\x00" in a for a in job["restart_args"]):
        raise UpdateError("重启参数格式无效")
    folder = job_path.parent
    stage, backup, install = (Path(job[key]) for key in ("stage_dir", "backup_dir", "install_dir"))
    if not all(p.is_absolute() for p in (stage, backup, install)) or stage != folder / "staged" or backup != folder / "backup":
        raise UpdateError("安装任务目录无效")
    for target in (folder, stage, backup, install):
        _reject_link_tree(target)
    if install.resolve() == folder or folder.is_relative_to(install.resolve()) or install.resolve().is_relative_to(folder):
        raise UpdateError("更新缓存不得与程序目录重叠")
    if not (install / job["executable"]).is_file():
        raise UpdateError("安装目标 EXE 已不存在")
    rows = _manifest((stage / ARCHIVE_MANIFEST).read_bytes(), job["version"])
    for row in rows:
        source = _inside(stage / row["path"], stage)
        target = _inside(install / row["path"], install)
        _reject_link_tree(source)
        _reject_link_tree(target)
        if not source.is_file() or source.stat().st_size != row["size"] or _digest(source) != row["sha256"]:
            raise UpdateError(f"待安装文件再次校验失败：{row['path']}")
        if target.exists() and not target.is_file():
            raise UpdateError(f"程序文件路径已被目录占用：{row['path']}")
    return job, rows


def launch_helper(prepared: PreparedUpdate, pid: int | None = None, restart_args=(), *, frozen: bool | None = None):
    if frozen is None:
        frozen = bool(getattr(sys, "frozen", False))
    if not frozen:
        raise UpdateError("源码运行仅支持检查更新；请使用打包版安装程序更新")
    if os.name != "nt":
        raise UpdateError("自动安装目前仅支持 Windows")
    job, _ = _load_job(prepared.job_path)
    helper_source = Path(job["install_dir"]) / "_updater" / "FanyiJiUpdate.exe"
    _reject_link_tree(helper_source)
    if not helper_source.is_file():
        raise UpdateError("此程序包缺少独立更新助手，请先下载完整版本")
    helper = prepared.cache_dir / "FanyiJiUpdate.exe"
    shutil.copy2(helper_source, helper)
    job["pid"] = os.getpid() if pid is None else pid
    job["restart_args"] = list(restart_args)
    prepared.job_path.write_text(json.dumps(job, ensure_ascii=False, indent=2), encoding="utf-8")
    _load_job(prepared.job_path)
    try:
        return subprocess.Popen([str(helper), "--job", str(prepared.job_path)], cwd=str(prepared.cache_dir),
                                creationflags=subprocess.CREATE_NO_WINDOW, close_fds=True,
                                env=_fresh_process_environment())
    except OSError as exc:
        raise UpdateError(f"无法启动更新助手：{exc}") from exc


def _pid_alive(pid: int) -> bool:
    if not pid:
        return False
    if os.name == "nt":
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.OpenProcess.restype = ctypes.c_void_p
        handle = kernel.OpenProcess(0x100000, False, pid)
        if not handle:
            return ctypes.get_last_error() != 87
        try:
            # Only WAIT_OBJECT_0 proves termination. WAIT_FAILED is not a licence
            # to touch a possibly running application's files.
            return kernel.WaitForSingleObject(ctypes.c_void_p(handle), 0) != 0
        finally:
            kernel.CloseHandle(ctypes.c_void_p(handle))
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def _fresh_process_environment():
    # Both helpers and restarted programs outlive their parent. PyInstaller must
    # allocate an independent extraction directory rather than reuse _MEIPASS.
    return {**os.environ, "PYINSTALLER_RESET_ENVIRONMENT": "1"}


def _atomic_copy(source: Path, target: Path):
    _reject_link_tree(source)
    _reject_link_tree(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".update-" + uuid.uuid4().hex)
    try:
        shutil.copy2(source, temporary)
        deadline = time.monotonic() + 10
        while True:
            try:
                os.replace(temporary, target)
                return
            except OSError as exc:
                # A PyInstaller parent or antivirus can retain a short-lived file
                # handle after the Qt child exits. Never kill it or wait forever.
                if getattr(exc, "winerror", None) not in {5, 32, 33} or time.monotonic() >= deadline:
                    raise
                time.sleep(.2)
    finally:
        # This exact transaction-owned temporary file is the only cleanup here.
        # Installation directories, backups and existing files are never pruned.
        if temporary.is_file():
            temporary.unlink()


@contextmanager
def _installation_lock(install_dir: Path):
    """Serialize helpers for the same installed directory across processes."""
    install_dir = Path(install_dir).resolve(strict=True)
    key = hashlib.sha256(os.path.normcase(str(install_dir)).encode("utf-8")).hexdigest()
    if os.name == "nt":
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.CreateMutexW.argtypes = (ctypes.c_void_p, ctypes.c_int, ctypes.c_wchar_p)
        kernel.CreateMutexW.restype = ctypes.c_void_p
        kernel.WaitForSingleObject.argtypes = (ctypes.c_void_p, ctypes.c_uint32)
        kernel.ReleaseMutex.argtypes = (ctypes.c_void_p,)
        kernel.CloseHandle.argtypes = (ctypes.c_void_p,)
        handle = kernel.CreateMutexW(None, False, "Global\\FanyiJiVoiceNext.Update." + key)
        if not handle:
            raise UpdateError("无法取得程序更新独占锁；未修改程序")
        owned = False
        try:
            result = kernel.WaitForSingleObject(handle, 0)
            if result not in (0, 0x80):
                raise UpdateError("已有另一更新助手正在处理此程序目录；未修改程序")
            owned = True
            yield
        finally:
            if owned:
                kernel.ReleaseMutex(handle)
            kernel.CloseHandle(handle)
    else:
        # The shipped helper is Windows-only. flock keeps local tests meaningful
        # on other build hosts without weakening the Windows mutex contract.
        import fcntl
        lock_path = install_dir / ".fanyiji-update.lock"
        _reject_link_tree(lock_path)
        with lock_path.open("a+b") as stream:
            try:
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise UpdateError("已有另一更新助手正在处理此程序目录；未修改程序") from exc
            try:
                yield
            finally:
                fcntl.flock(stream, fcntl.LOCK_UN)


def apply_update(job_path: Path, *, wait_timeout: float = 120, pid_alive=None, restart: bool = True, start=None):
    job, _ = _load_job(Path(job_path))
    with _installation_lock(Path(job["install_dir"])):
        return _apply_update_locked(job_path, wait_timeout=wait_timeout, pid_alive=pid_alive, restart=restart, start=start)


def _apply_update_locked(job_path: Path, *, wait_timeout: float = 120, pid_alive=None, restart: bool = True, start=None):
    """Apply one verified transaction. Exposed for local file-system tests only."""
    job_path = Path(job_path)
    job, rows = _load_job(job_path)
    if not 0 <= wait_timeout <= 120:
        raise UpdateError("退出等待时间无效")
    is_alive = pid_alive or _pid_alive
    deadline = time.monotonic() + wait_timeout
    while is_alive(job["pid"]):
        if time.monotonic() >= deadline:
            raise UpdateError("等待主程序退出超时；未覆盖程序")
        time.sleep(min(.2, max(0, deadline - time.monotonic())))
    # Recheck staged hashes and every existing target after the finite wait.
    job, rows = _load_job(job_path)
    stage, install, backup = (Path(job[key]) for key in ("stage_dir", "install_dir", "backup_dir"))
    if backup.exists():
        raise UpdateError("此安装任务已创建备份，不重复应用；请保留日志和备份")
    backup.mkdir()
    touched = []
    try:
        for row in rows:
            relative = row["path"]
            target = _inside(install / relative, install)
            previous = _inside(backup / relative, backup)
            existed = target.exists()
            if existed:
                previous.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(target, previous)
            touched.append((relative, existed))
            _atomic_copy(stage / relative, target)
        if restart:
            command = [str(install / job["executable"]), *job["restart_args"]]
            (start or subprocess.Popen)(command, cwd=str(install), close_fds=True, env=_fresh_process_environment(),
                                       **({"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {}))
    except Exception as exc:
        rollback_errors = []
        for relative, existed in reversed(touched):
            try:
                target = _inside(install / relative, install)
                if existed:
                    _atomic_copy(backup / relative, target)
                elif target.is_file():
                    target.unlink()
            except Exception as rollback_exc:
                rollback_errors.append(f"{relative}: {rollback_exc}")
        suffix = "；回滚失败：" + "; ".join(rollback_errors) if rollback_errors else "；已恢复原程序文件"
        raise UpdateError(f"安装失败：{exc}{suffix}。备份保留在 {backup}") from exc
    return backup


def helper_main(argv=None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="翻译姬程序更新助手")
    parser.add_argument("--job", type=Path, required=True)
    args = parser.parse_args(argv)
    log_path = args.job.resolve().parent / "update.log"
    try:
        if os.name != "nt":
            raise UpdateError("更新助手仅用于 Windows")
        backup = apply_update(args.job)
        log_path.write_text(f"更新完成。旧程序备份：{backup}\n", encoding="utf-8")
        return 0
    except Exception as exc:
        message = f"翻译姬更新未完成：{exc}\n日志和备份保留在 {log_path.parent}"
        try:
            log_path.write_text(message + "\n", encoding="utf-8")
        except OSError:
            pass
        if os.name == "nt":
            ctypes.windll.user32.MessageBoxW(None, message, "翻译姬 · 更新失败", 0x10)
        return 1
