"""Resumable, verified model downloads into a caller-selected external root."""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
from http.client import HTTPException
import json
import math
import os
from pathlib import Path
import re
import stat
import threading
import time
from typing import Mapping
from urllib.error import HTTPError
from urllib.request import HTTPRedirectHandler, Request, build_opener
from urllib.parse import urlsplit
import uuid

from .model_catalog import MODEL_CATALOG, ModelSpec, validate_download_url


class ModelDownloadError(RuntimeError):
    """Safe user-facing failure; never includes signed URLs or response bodies."""


class DownloadCancelled(ModelDownloadError):
    pass


class ModelLicenseRequired(ModelDownloadError):
    pass


class ModelDownloadBusy(ModelDownloadError):
    pass


class _NetworkFailure(ModelDownloadError):
    pass


class _IntegrityFailure(ModelDownloadError):
    pass


@dataclass(frozen=True)
class DownloadProgress:
    model_id: str
    file_path: str
    downloaded_bytes: int
    total_bytes: int
    status: str
    source_name: str = ""
    source_index: int = 0
    attempt: int = 0


@dataclass(frozen=True)
class ModelStatus:
    model_id: str
    ready: bool
    verified_bytes: int
    partial_bytes: int
    total_bytes: int

    @property
    def remaining_bytes(self):
        """Transfer estimate only: partial bytes have not passed their full SHA."""
        return max(0, self.total_bytes - self.verified_bytes - self.partial_bytes)


class _TrustedRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        try:
            validate_download_url(newurl, redirect=True)
        except ValueError:
            raise ModelDownloadError("模型服务器重定向到了未允许的地址。") from None
        return super().redirect_request(req, fp, code, msg, headers, newurl)


class _HttpTransport:
    def __init__(self):
        self._opener = build_opener(_TrustedRedirect())

    def open(self, url, *, headers, timeout):
        validate_download_url(url)
        return self._opener.open(Request(url, headers=headers), timeout=timeout)


_LOCKS = {}
_LOCKS_GUARD = threading.Lock()
_MARKER = ".model-ready.json"


def _cancel(event):
    if event is not None and event.is_set():
        raise DownloadCancelled("模型下载已取消，已保留可续传的部分文件。")


def _linked(path):
    try:
        value = path.lstat()
    except FileNotFoundError:
        return False
    return stat.S_ISLNK(value.st_mode) or bool(getattr(value, "st_file_attributes", 0) & 0x400)


class ModelDownloadManager:
    def __init__(self, model_root, *, catalog: Mapping[str, ModelSpec] = MODEL_CATALOG,
                 transport=None, timeout=10.0, chunk_size=256 * 1024,
                 attempts_per_source=2, retry_backoff=0.25):
        self.model_root = Path(os.path.abspath(model_root))
        self.catalog = dict(catalog)
        if (any(not isinstance(v, ModelSpec) or k != v.model_id for k, v in self.catalog.items())
                or len({s.directory_name.casefold() for s in self.catalog.values()}) != len(self.catalog)):
            raise ModelDownloadError("模型清单无效。")
        if (isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not math.isfinite(timeout)
                or not 0 < timeout <= 60 or type(chunk_size) is not int or not 1 <= chunk_size <= 4 * 1024 * 1024
                or type(attempts_per_source) is not int or not 1 <= attempts_per_source <= 3
                or isinstance(retry_backoff, bool) or not isinstance(retry_backoff, (int, float))
                or not math.isfinite(retry_backoff) or not 0 <= retry_backoff <= 2):
            raise ValueError("Invalid download timeout or block size")
        self.transport = transport if transport is not None else _HttpTransport()
        self.timeout, self.chunk_size = float(timeout), chunk_size
        self.attempts_per_source, self.retry_backoff = attempts_per_source, float(retry_backoff)

    def _spec(self, model_id):
        try:
            return self.catalog[model_id]
        except (KeyError, TypeError):
            raise ModelDownloadError("未知模型，请从内置模型列表中选择。") from None

    def _safe(self, path, *, directory=False, writing=False):
        path = Path(path)
        if not path.is_relative_to(self.model_root):
            raise ModelDownloadError("模型路径超出了指定目录。")
        # Reject reparse points before resolving, including a linked root.
        for current in (*reversed(self.model_root.parents), self.model_root,
                        *[self.model_root.joinpath(*path.relative_to(self.model_root).parts[:i])
                          for i in range(1, len(path.relative_to(self.model_root).parts) + 1)]):
            if _linked(current):
                raise ModelDownloadError("模型目录含符号链接或重解析点，不能安全下载。")
        try:
            info = path.stat()
        except FileNotFoundError:
            return path
        if directory:
            if not stat.S_ISDIR(info.st_mode):
                raise ModelDownloadError("模型目录被其他文件占用。")
        elif not stat.S_ISREG(info.st_mode) or (writing and info.st_nlink != 1):
            raise ModelDownloadError("模型文件类型无效或存在不安全的硬链接。")
        return path

    def _mkdir(self, path):
        self._safe(path, directory=True)
        path.mkdir(parents=True, exist_ok=True)
        self._safe(path, directory=True)

    def _paths(self, spec, file):
        folder = self.model_root / spec.directory_name
        target = folder.joinpath(*file.relative_path.split("/"))
        return self._safe(target), self._safe(target.with_name(target.name + ".part"), writing=True)

    def _digest(self, path, event=None):
        self._safe(path)
        result = hashlib.sha256()
        with path.open("rb") as stream:
            while True:
                _cancel(event)
                block = stream.read(1024 * 1024)
                if not block:
                    break
                result.update(block)
        return result.hexdigest()

    def _verified(self, path, file, event=None):
        self._safe(path)
        return path.is_file() and path.stat().st_size == file.size_bytes and self._digest(path, event) == file.sha256

    def inspect(self, model_id):
        spec = self._spec(model_id)
        try:
            verified = partial = 0
            for file in spec.files:
                target, part = self._paths(spec, file)
                if self._verified(target, file):
                    verified += file.size_bytes
                elif part.is_file() and 0 < part.stat().st_size <= file.size_bytes:
                    partial += part.stat().st_size
            marker = self._safe(self.model_root / spec.directory_name / _MARKER)
            valid_marker = False
            if marker.is_file() and marker.stat().st_size <= 4096:
                try:
                    value = json.loads(marker.read_text(encoding="utf-8"))
                    valid_marker = any(value == {"schema_version": 1, "model_id": spec.model_id,
                                       "manifest_sha256": fingerprint}
                                       for fingerprint in (spec.fingerprint, spec.legacy_fingerprint))
                except (ValueError, UnicodeError):
                    pass
            return ModelStatus(model_id, verified == spec.total_bytes and valid_marker, verified, partial, spec.total_bytes)
        except ModelDownloadError:
            raise
        except OSError:
            raise ModelDownloadError("无法检查模型目录，请检查文件权限和磁盘状态。") from None

    @contextmanager
    def _lock(self, spec):
        key = (os.path.normcase(str(self.model_root)), spec.model_id)
        with _LOCKS_GUARD:
            lock = _LOCKS.setdefault(key, threading.Lock())
        if not lock.acquire(blocking=False):
            raise ModelDownloadBusy("这个模型已有下载任务，请等待该任务结束。")
        stream = None
        acquired = False
        try:
            folder = self.model_root / spec.directory_name
            self._mkdir(folder)
            path = self._safe(folder / ".download.lock", writing=True)
            flags = os.O_CREAT | os.O_RDWR | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
            descriptor = os.open(path, flags, 0o600)
            stream = os.fdopen(descriptor, "r+b")
            info = os.fstat(stream.fileno())
            if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
                raise ModelDownloadError("模型下载锁文件无效。")
            if info.st_size == 0:
                stream.write(b"0")
                stream.flush()
            stream.seek(0)
            try:
                if os.name == "nt":
                    import msvcrt
                    msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                acquired = True
            except OSError:
                raise ModelDownloadBusy("这个模型已有下载任务，请等待该任务结束。") from None
            yield
        finally:
            if stream is not None:
                if acquired:
                    stream.seek(0)
                    if os.name == "nt":
                        import msvcrt
                        msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
                    else:
                        import fcntl
                        fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
                stream.close()
            lock.release()

    def download(self, model_id, *, progress=None, cancel_event=None, license_acknowledged=False):
        spec = self._spec(model_id)
        _cancel(cancel_event)
        if spec.requires_license_ack and license_acknowledged is not True:
            raise ModelLicenseRequired("下载此模型前，请先阅读并确认相应的 Gemma 使用条款。")
        try:
            with self._lock(spec):
                verified = {file.relative_path: self._verified(self._paths(spec, file)[0], file, cancel_event)
                            for file in spec.files}
                completed = sum(f.size_bytes for f in spec.files if verified[f.relative_path])

                def emit(file_path, received, status, **source):
                    if progress is not None:
                        progress(DownloadProgress(model_id, file_path, completed + received, spec.total_bytes, status, **source))

                emit("", 0, "checking")
                for file in spec.files:
                    _cancel(cancel_event)
                    if verified[file.relative_path]:
                        emit(file.relative_path, 0, "skipped")
                        continue
                    target, part = self._paths(spec, file)
                    self._mkdir(target.parent)
                    if self._verified(part, file, cancel_event):
                        self._safe(target, writing=True)
                        os.replace(part, target)
                    else:
                        self._download_sources(file, part, lambda n, s, **meta: emit(file.relative_path, n, s, **meta), cancel_event)
                        self._safe(target)
                        self._safe(part, writing=True)
                        os.replace(part, target)
                    completed += file.size_bytes
                    emit(file.relative_path, 0, "verified")
                _cancel(cancel_event)
                # A progress callback or another process may alter a previously checked file.
                if not all(self._verified(self._paths(spec, f)[0], f, cancel_event) for f in spec.files):
                    raise ModelDownloadError("模型文件在校验期间发生变化，请重试。")
                self._write_marker(spec)
                emit("", 0, "complete")
                return self.model_root / spec.directory_name
        except ModelDownloadError:
            raise
        except HTTPError as exc:
            if exc.code in {401, 403}:
                raise ModelDownloadError("模型服务器拒绝访问，请检查模型许可或访问权限。") from None
            raise ModelDownloadError("模型服务器返回错误，请稍后重试；部分文件已保留。") from None
        except Exception:
            raise ModelDownloadError("模型下载中断，请检查网络、磁盘空间和权限后重试；部分文件已保留。") from None

    def _quarantine_partial(self, part):
        self._safe(part, writing=True)
        if part.is_file():
            quarantine = self._safe(part.with_name(part.name + ".bad-" + uuid.uuid4().hex), writing=True)
            os.replace(part, quarantine)

    def _wait_retry(self, event):
        if event is not None:
            if event.wait(self.retry_backoff):
                _cancel(event)
        elif self.retry_backoff:
            time.sleep(self.retry_backoff)
        _cancel(event)

    def _download_sources(self, file, part, emit, event):
        # A full but invalid previous .part must never keep every retry pinned to
        # the same bad bytes. Preserve it for diagnosis; never delete user models.
        if part.is_file() and part.stat().st_size >= file.size_bytes:
            self._quarantine_partial(part)
            emit(0, "quarantined")
        last_error = "所有可用模型下载源均失败；已保留可续传部分文件。"
        for index, url in enumerate(file.download_urls, 1):
            host = urlsplit(url).hostname
            label = {"hf-mirror.com": "HF-Mirror（第三方镜像）", "modelscope.cn": "ModelScope（国内）",
                     "www.modelscope.cn": "ModelScope（国内）", "huggingface.co": "Hugging Face",
                     "github.com": "GitHub"}.get(host, "模型下载源")
            for attempt in range(1, self.attempts_per_source + 1):
                _cancel(event)
                meta = {"source_name": label, "source_index": index, "attempt": attempt}
                def update(received, status):
                    emit(received, status, **meta)
                if index > 1 and attempt == 1:
                    update(part.stat().st_size if part.is_file() else 0, "switching_source")
                initial_partial = part.stat().st_size if part.is_file() else 0
                try:
                    self._download_file(file, part, update, event, url=url)
                    update(file.size_bytes, "verifying")
                    if not self._verified(part, file, event):
                        raise _IntegrityFailure("模型文件大小或 SHA256 校验失败。")
                    return
                except DownloadCancelled:
                    raise
                except _IntegrityFailure:
                    self._quarantine_partial(part)
                    update(0, "quarantined")
                    last_error = "下载源返回的模型 SHA256 不匹配；坏部分文件已隔离，请重试或使用其他来源。"
                    if initial_partial and attempt < self.attempts_per_source:
                        # The inherited prefix may have come from a bad source.
                        # Give this source one bounded clean attempt before leaving.
                        update(0, "retrying")
                        self._wait_retry(event)
                        continue
                    break
                except HTTPError as error:
                    retryable = error.code == 429 or 500 <= error.code <= 599
                    last_error = ("模型服务器拒绝访问，请检查模型许可或访问权限。" if error.code in {401, 403}
                                  else "模型服务器返回错误；部分文件已保留，可重试其他来源。")
                    error.close()
                    if not retryable:
                        break
                except _NetworkFailure:
                    last_error = "模型下载网络中断；已自动重试并保留可续传部分文件。"
                except ModelDownloadError as error:
                    # Invalid range/length/redirect: do not trust this source, but
                    # a previously verified prefix can still be tried elsewhere.
                    last_error = str(error)
                    break
                if attempt < self.attempts_per_source:
                    update(part.stat().st_size if part.is_file() else 0, "retrying")
                    self._wait_retry(event)
        raise ModelDownloadError(last_error)

    def _download_file(self, file, part, emit, event, *, url=None):
        self._safe(part, writing=True)
        offset = part.stat().st_size if part.is_file() else 0
        if offset >= file.size_bytes:
            offset = 0
        headers = {"Accept-Encoding": "identity", "User-Agent": "FanyiJiVoiceNext-ModelDownloader"}
        if offset:
            headers["Range"] = f"bytes={offset}-"
        _cancel(event)
        try:
            response = self.transport.open(url or file.url, headers=headers, timeout=self.timeout)
        except (HTTPError, ModelDownloadError):
            raise
        except (OSError, HTTPException):
            raise _NetworkFailure("模型下载连接失败。") from None
        with response:
            status = getattr(response, "status", None)
            if status == 429 or (isinstance(status, int) and 500 <= status <= 599):
                raise _NetworkFailure("模型下载服务暂时不可用。")
            if status not in {200, 206}:
                raise ModelDownloadError("模型服务器没有返回可验证的下载内容。")
            encoding = response.headers.get("Content-Encoding", "identity").lower()
            if encoding not in {"", "identity"}:
                raise ModelDownloadError("模型服务器返回了不支持的压缩内容。")
            if status == 206:
                match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", response.headers.get("Content-Range", ""))
                if not match or tuple(map(int, match.groups())) != (offset, file.size_bytes - 1, file.size_bytes):
                    raise ModelDownloadError("服务器续传范围不匹配，已保留原部分文件。")
            else:
                offset = 0  # Range ignored: replace the partial stream, never append the full file.
            length = response.headers.get("Content-Length")
            if length is not None and (not length.isdigit() or int(length) != file.size_bytes - offset):
                raise ModelDownloadError("模型服务器返回的文件长度不匹配。")
            self._safe(part, writing=True)
            flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
            descriptor = os.open(part, flags, 0o600)
            with os.fdopen(descriptor, "r+b") as stream:
                if os.fstat(stream.fileno()).st_nlink != 1:
                    raise ModelDownloadError("部分文件存在不安全的硬链接。")
                if not offset:
                    stream.truncate(0)
                elif os.fstat(stream.fileno()).st_size != offset:
                    raise ModelDownloadError("部分文件在续传前发生变化。")
                stream.seek(offset)
                received = offset
                emit(received, "downloading")
                try:
                    while True:
                        _cancel(event)
                        try:
                            block = response.read(self.chunk_size)
                        except (OSError, HTTPException):
                            raise _NetworkFailure("模型下载连接中断。") from None
                        if not block:
                            break
                        if received + len(block) > file.size_bytes:
                            raise _IntegrityFailure("模型服务器返回了超出清单大小的内容。")
                        stream.write(block)
                        received += len(block)
                        emit(received, "downloading")
                    _cancel(event)
                    if received != file.size_bytes:
                        raise _NetworkFailure("模型下载未完成，可重试续传。")
                finally:
                    stream.flush()
                    os.fsync(stream.fileno())

    def _write_marker(self, spec):
        folder = self.model_root / spec.directory_name
        target = self._safe(folder / _MARKER, writing=True)
        temporary = folder / f".model-ready-{uuid.uuid4().hex}.tmp"
        payload = {"schema_version": 1, "model_id": spec.model_id, "manifest_sha256": spec.fingerprint}
        try:
            with temporary.open("x", encoding="utf-8") as stream:
                json.dump(payload, stream, sort_keys=True)
                stream.flush()
                os.fsync(stream.fileno())
            self._safe(target, writing=True)
            os.replace(temporary, target)
        finally:
            temporary.unlink(missing_ok=True)
