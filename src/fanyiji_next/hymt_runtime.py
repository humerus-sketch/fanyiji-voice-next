"""Own one local llama.cpp server; importing this starts no process or network IO."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
import secrets
import socket
import subprocess
import tempfile
import threading
import time
import urllib.error
import urllib.parse
import urllib.request


class HyRuntimeCancelled(RuntimeError):
    """The owning voice session requested cancellation."""


class HyRuntimeError(RuntimeError):
    """A selected local server could not be started or contacted."""


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        # Even a local server must not redirect a model request onto the network.
        return None


def _local_file(value: str | Path, label: str, suffix: str) -> Path:
    if not isinstance(value, (str, Path)) or not str(value).strip() or "\0" in str(value):
        raise ValueError(f"尚未选择{label}。")
    path = Path(os.path.expandvars(str(value))).expanduser().resolve()
    if not path.is_file() or path.suffix.lower() != suffix:
        raise ValueError(f"{label}不存在或文件类型无效：{path}")
    return path


def _positive_timeout(value: float, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{label}必须是有限的正数。")
    return float(value)


class LlamaServerRuntime:
    """Explicitly start and close a private local server on a worker thread.

    close() has bounded process waits and belongs on the session worker. A GUI
    only sets stop_event; the watchdog interrupts startup or HTTP by stopping
    this instance's child. Closed instances cannot be restarted.
    """

    def __init__(self, executable: str | Path, model_path: str | Path, *,
                 device: str = "cuda", stop_event=None, startup_timeout: float = 120.0,
                 request_timeout: float = 30.0, ctx_size: int = 2048) -> None:
        self.executable = _local_file(executable, "llama-server 可执行文件", ".exe")
        self.model_path = _local_file(model_path, "Hy-MT GGUF 模型", ".gguf")
        try:
            with self.model_path.open("rb") as model:
                if model.read(4) != b"GGUF":
                    raise ValueError(f"Hy-MT 模型不是有效的 GGUF 文件：{self.model_path}")
        except OSError as exc:
            raise ValueError(f"无法读取 Hy-MT GGUF 模型：{self.model_path}") from exc
        if device not in {"cpu", "cuda"}:
            raise ValueError(f"不支持的 Hy-MT 运行设备：{device}")
        if type(ctx_size) is not int or ctx_size <= 0:
            raise ValueError("Hy-MT context window 必须是正整数。")
        self.device = device
        self._stop_event = stop_event
        self._startup_timeout = _positive_timeout(startup_timeout, "模型加载超时")
        self._request_timeout = _positive_timeout(request_timeout, "翻译请求超时")
        self._ctx_size = ctx_size
        self._lock = threading.RLock()
        self._start_lock = threading.Lock()
        self._close_lock = threading.Lock()
        self._request_lock = threading.Lock()
        self._watcher_shutdown = threading.Event()
        self._watcher = None
        self._process = None
        self._temporary_directory = None
        self._log_file = None
        self._log_path = None
        self._base_url = None
        self._headers = {}
        self._started = False
        self._closed = False
        self._request_deadline = None
        self._timeout_message = ""
        self._cleanup_error = None
        self._opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), _NoRedirect())
        self.backend_info = {
            "runtime": "llama.cpp", "device": device, "server_exe": str(self.executable),
            "model_path": str(self.model_path), "ctx_size": ctx_size, "parallel": 1,
        }

    @property
    def base_url(self) -> str:
        with self._lock:
            self._check_available()
            return self._base_url

    @property
    def log_path(self) -> Path | None:
        """Private temporary diagnostic log, removed after the child exits."""
        return self._log_path

    def _check_cancelled(self) -> None:
        if self._stop_event is not None and self._stop_event.is_set():
            raise HyRuntimeCancelled("Hy-MT 翻译已取消。")
        if self._timeout_message:
            raise HyRuntimeError(self._timeout_message)

    def _check_available(self, *, loading: bool = False) -> None:
        self._check_cancelled()
        if self._closed:
            raise HyRuntimeError("Hy-MT 本地运行库已关闭，请重新开始语音会话。")
        if self._process is None or (not self._started and not loading):
            raise HyRuntimeError("Hy-MT 本地运行库尚未启动。")
        code = self._process.poll()
        if code is not None:
            raise HyRuntimeError(f"Hy-MT 本地服务已退出（code={code}）：{self.executable}")

    def _child_environment(self) -> dict[str, str]:
        env = {key: value for key, value in os.environ.items()
               if not key.upper().startswith("LLAMA_ARG_") and key.upper() != "LLAMA_API_KEY"}
        # CTranslate2 may have prepended another CUDA version to the parent PATH.
        # The child must use its matching bundled DLLs, never that implicit mix.
        windows = env.get("SystemRoot", env.get("WINDIR", r"C:\Windows"))
        env["PATH"] = os.pathsep.join((str(self.executable.parent), str(Path(windows) / "System32"), windows))
        return env

    def start(self) -> None:
        with self._start_lock:
            with self._lock:
                self._check_cancelled()
                if self._closed:
                    raise HyRuntimeError("Hy-MT 本地运行库已关闭，不能重复使用。")
                if self._started:
                    self._check_available()
                    return
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as bound:
                    bound.bind(("127.0.0.1", 0))
                    port = bound.getsockname()[1]
                with self._lock:
                    self._check_cancelled()
                    if self._closed:
                        raise HyRuntimeError("Hy-MT 本地运行库已关闭。")
                    self._temporary_directory = tempfile.TemporaryDirectory(prefix="fanyiji-hymt-")
                    self._log_path = Path(self._temporary_directory.name) / "server.log"
                    self._log_file = self._log_path.open("w+b")
                    self._base_url = f"http://127.0.0.1:{port}"
                    token = secrets.token_urlsafe(32)
                    self._headers = {"Content-Type": "application/json", "Authorization": f"Bearer {token}"}
                    command = [str(self.executable), "--model", str(self.model_path),
                               "--host", "127.0.0.1", "--port", str(port), "--ctx-size", str(self._ctx_size),
                               "--device", "CUDA0" if self.device == "cuda" else "none",
                               "--n-gpu-layers", "99" if self.device == "cuda" else "0",
                               "--parallel", "1", "--fit", "off", "--offline", "--no-webui",
                               "-lv", "2", "--api-key", token]
                    self._process = subprocess.Popen(
                        command, cwd=str(self.executable.parent), env=self._child_environment(),
                        stdin=subprocess.DEVNULL, stdout=self._log_file, stderr=subprocess.STDOUT,
                        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                    )
                    self.backend_info.update(pid=self._process.pid, base_url=self._base_url)
                    self._watcher = threading.Thread(target=self._watch, name="fanyiji-hymt-owner", daemon=True)
                    self._watcher.start()
                deadline = time.monotonic() + self._startup_timeout
                while True:
                    with self._lock:
                        self._check_available(loading=True)
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise HyRuntimeError(f"Hy-MT 模型加载超过 {self._startup_timeout:g} 秒：{self.model_path}")
                    try:
                        health = self._json_request("/health", None, timeout=min(0.5, remaining), loading=True)
                    except urllib.error.HTTPError as exc:
                        if exc.code != 503:
                            raise HyRuntimeError(f"Hy-MT 本地服务健康检查返回 HTTP {exc.code}：{self.executable}") from exc
                        self._check_cancelled()
                        self._watcher_shutdown.wait(min(0.05, max(0, remaining)))
                        continue
                    except (urllib.error.URLError, OSError, json.JSONDecodeError):
                        self._check_cancelled()
                        self._watcher_shutdown.wait(min(0.05, max(0, remaining)))
                        continue
                    if isinstance(health, dict) and health.get("status") == "ok":
                        break
                    self._watcher_shutdown.wait(min(0.05, max(0, remaining)))
                props = self._json_request("/props", None, timeout=min(self._request_timeout, max(0.01, deadline-time.monotonic())), loading=True)
                if not isinstance(props, dict):
                    raise HyRuntimeError("Hy-MT 本地服务返回了无效的模型信息。")
                if not isinstance(props.get("model_path"), str) or Path(props["model_path"]).resolve() != self.model_path:
                    raise HyRuntimeError(f"Hy-MT 本地服务返回的模型路径与所选文件不一致：{self.model_path}")
                with self._lock:
                    self._check_available(loading=True)
                    self.backend_info["server_props"] = props
                    self._started = True
            except BaseException as exc:
                try:
                    self.close()
                except Exception as cleanup:
                    exc.add_note(f"本地服务收尾失败：{type(cleanup).__name__}")
                self._check_cancelled()
                if isinstance(exc, (HyRuntimeError, HyRuntimeCancelled, ValueError)):
                    raise
                raise HyRuntimeError(f"无法启动 Hy-MT 本地服务：{self.executable}；模型：{self.model_path}") from exc

    def _watch(self) -> None:
        while not self._watcher_shutdown.wait(0.05):
            with self._lock:
                cancelled = self._stop_event is not None and self._stop_event.is_set()
                expired = self._request_deadline is not None and time.monotonic() >= self._request_deadline
                if expired and not cancelled:
                    self._timeout_message = f"Hy-MT 本地请求超过 {self._request_timeout:g} 秒，服务已停止；请重新开始会话：{self.executable}"
            if cancelled or expired:
                try:
                    self.close()
                except Exception as exc:
                    self._cleanup_error = exc
                return

    def _request(self, route: str, body, *, timeout: float, loading: bool = False):
        if not isinstance(route, str) or not route.startswith("/") or route.startswith("//") or "\\" in route:
            raise ValueError("Hy-MT 请求必须使用本地相对 API 路径。")
        parsed = urllib.parse.urlsplit(route)
        if parsed.scheme or parsed.netloc or parsed.fragment:
            raise ValueError("Hy-MT 请求不允许外部 URL 或重定向。")
        with self._lock:
            self._check_available(loading=loading)
            request = urllib.request.Request(
                self._base_url + route,
                data=None if body is None else json.dumps(body, ensure_ascii=False).encode("utf-8"),
                headers=self._headers,
            )
        return self._opener.open(request, timeout=timeout)

    def _json_request(self, route, body, *, timeout, loading=False):
        with self._request(route, body, timeout=timeout, loading=loading) as response:
            return json.load(response)

    def _begin_request(self) -> None:
        with self._lock:
            self._check_available()
            self._request_deadline = time.monotonic() + self._request_timeout

    def _finish_request(self) -> None:
        with self._lock:
            self._request_deadline = None

    def _request_error(self, route: str, exc: Exception) -> HyRuntimeError:
        if isinstance(exc, TimeoutError) or isinstance(getattr(exc, "reason", None), TimeoutError):
            with self._lock:
                self._timeout_message = f"Hy-MT 本地请求超过 {self._request_timeout:g} 秒，服务已停止；请重新开始会话：{self.executable}"
            self.close()
        self._check_cancelled()
        if isinstance(exc, urllib.error.HTTPError):
            return HyRuntimeError(f"Hy-MT 本地 API {route} 返回 HTTP {exc.code}：{self.executable}")
        return HyRuntimeError(f"Hy-MT 本地 API {route} 请求失败：{self.executable}")

    def post_json(self, route: str, body) -> dict:
        with self._request_lock:
            self._begin_request()
            try:
                result = self._json_request(route, body, timeout=self._request_timeout)
                with self._lock:
                    self._check_available()
                if not isinstance(result, dict):
                    raise HyRuntimeError(f"Hy-MT 本地 API {route} 返回了无效的 JSON object。")
                return result
            except json.JSONDecodeError as exc:
                raise HyRuntimeError(f"Hy-MT 本地 API {route} 返回了无效的 JSON。") from exc
            except (HyRuntimeCancelled, HyRuntimeError, ValueError):
                raise
            except Exception as exc:
                raise self._request_error(route, exc) from exc
            finally:
                self._finish_request()

    def iter_completion(self, body: dict):
        """Yield native SSE objects, including the final stop event; no text policy."""
        if not isinstance(body, dict):
            raise ValueError("Hy-MT completion 请求必须是 JSON object。")
        with self._request_lock:
            self._begin_request()
            completed = False
            try:
                with self._request("/completion", {**body, "stream": True}, timeout=self._request_timeout) as response:
                    for line in response:
                        with self._lock:
                            self._check_available()
                        if not line.startswith(b"data:"):
                            continue
                        data = line[5:].strip()
                        if not data or data == b"[DONE]":
                            continue
                        event = json.loads(data)
                        if not isinstance(event, dict) or "error" in event:
                            raise HyRuntimeError("Hy-MT 本地流式请求返回错误。")
                        completed = event.get("stop") is True
                        yield event
                        if completed:
                            return
                self._check_cancelled()
                raise HyRuntimeError("Hy-MT 流式请求结束但缺少完成标志。")
            except json.JSONDecodeError as exc:
                raise HyRuntimeError("Hy-MT 本地流式请求返回了无效的 JSON。") from exc
            except (HyRuntimeCancelled, HyRuntimeError, ValueError):
                raise
            except Exception as exc:
                raise self._request_error("/completion", exc) from exc
            finally:
                self._finish_request()
                # Closing an abandoned HTTP response is not a server-side cancel
                # guarantee. Retire our child so it cannot keep decoding unseen.
                if not completed:
                    self.close()

    def close(self) -> None:
        self._watcher_shutdown.set()
        with self._close_lock:
            with self._lock:
                self._closed = True
                self._started = False
                process = self._process
            if process is not None and process.poll() is None:
                try:
                    process.terminate()
                    try:
                        process.wait(timeout=3.0)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait(timeout=3.0)
                except Exception as exc:
                    if process.poll() is None:
                        raise HyRuntimeError(f"无法结束本会话的 Hy-MT 服务（pid={process.pid}）：{self.executable}") from exc
            with self._lock:
                self._process = None
                if self._log_file is not None:
                    self._log_file.close()
                    self._log_file = None
                if self._temporary_directory is not None:
                    self._temporary_directory.cleanup()
                    self._temporary_directory = None
        watcher = self._watcher
        if watcher is not None and watcher is not threading.current_thread() and watcher.is_alive():
            watcher.join(timeout=1.0)
