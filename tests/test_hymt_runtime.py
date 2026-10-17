"""Owned-process and local HTTP contracts; no real server, audio or GPU is used."""

from __future__ import annotations

import io
import json
from pathlib import Path
import subprocess
import threading
from types import SimpleNamespace
import urllib.error
import urllib.parse

import pytest

from fanyiji_next import hymt_runtime as module


class FakeProcess:
    def __init__(self, pid, *, returncode=None, terminate_timeout=False):
        self.pid = pid
        self.returncode = returncode
        self.terminate_timeout = terminate_timeout
        self.terminated = 0
        self.killed = 0
        self.waits = []
        self.exited = threading.Event()

    def poll(self):
        return self.returncode

    def terminate(self):
        self.terminated += 1
        if not self.terminate_timeout:
            self.returncode = 0
            self.exited.set()

    def kill(self):
        self.killed += 1
        self.returncode = -9
        self.exited.set()

    def wait(self, timeout):
        self.waits.append(timeout)
        if self.returncode is None:
            raise subprocess.TimeoutExpired("fake-server", timeout)
        return self.returncode


def response(value):
    return io.BytesIO(json.dumps(value).encode("utf-8"))


@pytest.fixture
def fake(tmp_path, monkeypatch):
    executable = tmp_path / "runtime" / "llama-server.exe"
    executable.parent.mkdir()
    executable.write_bytes(b"MZ fixture, never executed")
    model = tmp_path / "model.gguf"
    model.write_bytes(b"GGUF fixture, never loaded")
    state = SimpleNamespace(executable=executable, model=model, processes=[], calls=[], requests=[],
                            health_failures=0, handler=None, process_options={})

    class BoundSocket:
        port = 41000

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def bind(self, address):
            assert address == ("127.0.0.1", 0)
            BoundSocket.port += 1
            self.port = BoundSocket.port

        def getsockname(self):
            return "127.0.0.1", self.port

    def popen(command, **kwargs):
        process = FakeProcess(100 + len(state.processes), **state.process_options)
        state.processes.append(process)
        state.calls.append((command, kwargs))
        return process

    def open_request(request, timeout):
        state.requests.append((request, timeout))
        path = urllib.parse.urlsplit(request.full_url).path
        if state.handler is not None:
            custom = state.handler(path, request, timeout)
            if custom is not None:
                return custom
        if path == "/health":
            if state.health_failures:
                state.health_failures -= 1
                raise urllib.error.HTTPError(request.full_url, 503, "loading", {}, None)
            return response({"status": "ok"})
        if path == "/props":
            return response({"model_path": str(model), "build_info": "fake-build"})
        if path == "/completion":
            return io.BytesIO(b': keepalive\n\ndata: {"content":"fixture","stop":false}\n\n'
                              b'data: {"content":"","stop":true,"stop_type":"eos"}\n\n')
        return response({"prompt": "fixture rendered prompt"})

    state.opener = SimpleNamespace(open=open_request)
    monkeypatch.setattr(module.socket, "socket", lambda *args: BoundSocket())
    monkeypatch.setattr(module.subprocess, "Popen", popen)
    monkeypatch.setattr(module.urllib.request, "build_opener", lambda *args: state.opener)
    state.runtime = lambda **kwargs: module.LlamaServerRuntime(executable, model, **kwargs)
    yield state
    for process in state.processes:
        process.kill()


def test_constructor_validates_without_starting(fake):
    runtime = fake.runtime()
    assert fake.calls == [] and fake.requests == []
    assert runtime.log_path is None
    with pytest.raises(module.HyRuntimeError, match="尚未启动"):
        _ = runtime.base_url
    runtime.close()


@pytest.mark.parametrize("which", ["executable", "model"])
def test_missing_local_path_is_clear(fake, which):
    values = {"executable": fake.executable, "model_path": fake.model}
    missing = fake.model.parent / "missing.gguf" if which == "model" else fake.executable.parent / "missing.exe"
    values["model_path" if which == "model" else "executable"] = missing
    with pytest.raises(ValueError, match="不存在") as error:
        module.LlamaServerRuntime(**values)
    assert str(missing) in str(error.value)
    assert not fake.calls


def test_non_gguf_model_does_not_start(fake):
    fake.model.write_bytes(b"not a model")
    with pytest.raises(ValueError, match="GGUF"):
        fake.runtime()
    assert not fake.calls


@pytest.mark.parametrize("kwargs", [{"device": "auto"}, {"ctx_size": True}, {"ctx_size": 0},
                                   {"startup_timeout": 0}, {"request_timeout": float("inf")}])
def test_invalid_runtime_options_fail_before_process(fake, kwargs):
    with pytest.raises(ValueError):
        fake.runtime(**kwargs)
    assert not fake.calls


def test_start_is_idempotent_private_hidden_and_does_not_modify_environment(fake, monkeypatch):
    monkeypatch.setenv("PATH", "parent-nvidia-wheel")
    monkeypatch.setenv("LLAMA_ARG_MODEL_URL", "https://invalid.example/model")
    runtime = fake.runtime()
    runtime.start()
    runtime.start()
    assert len(fake.calls) == 1
    command, kwargs = fake.calls[0]
    assert command[command.index("--host") + 1] == "127.0.0.1"
    assert command[command.index("--n-gpu-layers") + 1] == "99"
    assert command[command.index("--device") + 1] == "CUDA0"
    assert "--offline" in command and "--no-webui" in command
    assert command[command.index("--fit") + 1] == "off"
    assert kwargs["stdin"] == subprocess.DEVNULL
    assert kwargs["stderr"] == subprocess.STDOUT
    assert kwargs["stdout"] is not subprocess.PIPE
    assert kwargs["creationflags"] == getattr(subprocess, "CREATE_NO_WINDOW", 0)
    assert "parent-nvidia-wheel" not in kwargs["env"]["PATH"]
    assert "LLAMA_ARG_MODEL_URL" not in kwargs["env"]
    assert module.os.environ["PATH"] == "parent-nvidia-wheel"
    token = command[command.index("--api-key") + 1]
    assert token and token not in repr(runtime.backend_info)
    assert all(req.headers.get("Authorization") == f"Bearer {token}" for req, _ in fake.requests)
    log = runtime.log_path
    assert log.is_file()
    runtime.close()
    runtime.close()
    assert fake.processes[0].terminated == 1
    assert not log.exists() and not log.parent.exists()
    with pytest.raises(module.HyRuntimeError, match="已关闭"):
        runtime.start()


def test_cpu_is_explicit_and_no_fallback(fake):
    runtime = fake.runtime(device="cpu")
    runtime.start()
    command = fake.calls[0][0]
    assert command[command.index("--n-gpu-layers") + 1] == "0"
    assert command[command.index("--device") + 1] == "none"
    assert runtime.backend_info["device"] == "cpu"
    runtime.close()


def test_close_only_terminates_own_process_and_ports_differ(fake):
    first, second = fake.runtime(), fake.runtime()
    first.start()
    second.start()
    assert first.base_url != second.base_url
    first.close()
    assert fake.processes[0].terminated == 1 and fake.processes[1].terminated == 0
    assert second.post_json("/props", None)["build_info"] == "fake-build"
    second.close()


def test_slow_termination_uses_kill_and_wait(fake):
    fake.process_options = {"terminate_timeout": True}
    runtime = fake.runtime()
    runtime.start()
    runtime.close()
    process = fake.processes[0]
    assert (process.terminated, process.killed) == (1, 1)
    assert process.waits == [3.0, 3.0]


def test_loading_retries_503_then_returns_ready(fake):
    fake.health_failures = 2
    runtime = fake.runtime(startup_timeout=1)
    runtime.start()
    assert len([r for r, _ in fake.requests if r.full_url.endswith("/health")]) == 3
    runtime.close()


def test_startup_timeout_closes_child_and_logs(fake):
    fake.health_failures = 1000
    runtime = fake.runtime(startup_timeout=.01)
    with pytest.raises(module.HyRuntimeError, match="加载超过"):
        runtime.start()
    assert fake.processes[0].terminated == 1
    assert not runtime.log_path.exists()


def test_exited_child_reports_selected_path(fake):
    fake.process_options = {"returncode": 23}
    runtime = fake.runtime()
    with pytest.raises(module.HyRuntimeError, match="code=23") as error:
        runtime.start()
    assert str(fake.executable) in str(error.value)
    assert not runtime.log_path.exists()


def test_spawn_failure_cleans_temporary_log_and_reports_paths(fake, monkeypatch):
    def fail(*args, **kwargs):
        raise OSError("fixture executable unavailable")

    monkeypatch.setattr(module.subprocess, "Popen", fail)
    runtime = fake.runtime()
    with pytest.raises(module.HyRuntimeError, match="无法启动") as error:
        runtime.start()
    assert str(fake.executable) in str(error.value) and str(fake.model) in str(error.value)
    assert not runtime.log_path.exists()


def test_wrong_service_model_is_not_accepted_as_ready(fake):
    fake.handler = lambda path, *_: response({"model_path": str(fake.model.parent / "other.gguf")}) if path == "/props" else None
    runtime = fake.runtime()
    with pytest.raises(module.HyRuntimeError, match="不一致"):
        runtime.start()
    assert fake.processes[0].terminated == 1


def test_unauthorized_health_fails_immediately_instead_of_loading_timeout(fake):
    def handler(path, request, timeout):
        if path == "/health":
            raise urllib.error.HTTPError(request.full_url, 401, "unauthorized", {}, None)

    fake.handler = handler
    runtime = fake.runtime(startup_timeout=120)
    with pytest.raises(module.HyRuntimeError, match="HTTP 401"):
        runtime.start()
    assert len(fake.requests) == 1 and fake.processes[0].terminated == 1


def test_cancel_before_start_does_not_spawn(fake):
    stop = threading.Event()
    stop.set()
    runtime = fake.runtime(stop_event=stop)
    with pytest.raises(module.HyRuntimeCancelled):
        runtime.start()
    assert not fake.calls
    runtime.close()


def run_in_thread(call):
    errors = []

    def run():
        try:
            call()
        except BaseException as exc:
            errors.append(exc)

    thread = threading.Thread(target=run)
    thread.start()
    return thread, errors


def test_stop_interrupts_blocked_startup_instead_of_waiting_for_timeout(fake):
    entered = threading.Event()
    stop = threading.Event()

    def handler(path, request, timeout):
        if path == "/health":
            entered.set()
            assert fake.processes[0].exited.wait(2)
            raise urllib.error.URLError("fixture process closed")

    fake.handler = handler
    runtime = fake.runtime(stop_event=stop, startup_timeout=120)
    thread, errors = run_in_thread(runtime.start)
    assert entered.wait(1)
    stop.set()
    thread.join(1)
    assert not thread.is_alive()
    assert len(errors) == 1 and isinstance(errors[0], module.HyRuntimeCancelled)
    assert fake.processes[0].terminated == 1
    runtime.close()


def test_post_json_and_stream_are_plain_local_transport(fake):
    runtime = fake.runtime()
    runtime.start()
    assert runtime.post_json("/apply-template", {"messages": []}) == {"prompt": "fixture rendered prompt"}
    events = list(runtime.iter_completion({"prompt": "public fixture", "temperature": 0, "cache_prompt": False}))
    assert [x["content"] for x in events] == ["fixture", ""]
    assert events[-1]["stop_type"] == "eos"
    sent = json.loads(fake.requests[-1][0].data)
    assert sent == {"prompt": "public fixture", "temperature": 0, "cache_prompt": False, "stream": True}
    runtime.close()


@pytest.mark.parametrize("route", ["https://external.invalid/api", "//external.invalid/api", "/\\external", "/props#fragment"])
def test_external_or_ambiguous_routes_never_reach_opener(fake, route):
    runtime = fake.runtime()
    runtime.start()
    before = len(fake.requests)
    with pytest.raises(ValueError):
        runtime.post_json(route, {})
    assert len(fake.requests) == before
    runtime.close()


def test_real_opener_configuration_disables_environment_proxy_and_redirects(tmp_path, monkeypatch):
    # Inspect handlers without making a request or resolving a hostname.
    executable, model = tmp_path / "server.exe", tmp_path / "model.gguf"
    executable.write_bytes(b"MZ")
    model.write_bytes(b"GGUF")
    monkeypatch.setenv("HTTPS_PROXY", "http://invalid.example:1234")
    runtime = module.LlamaServerRuntime(executable, model)
    redirects = [h for h in runtime._opener.handlers if isinstance(h, module._NoRedirect)]
    assert len(redirects) == 1
    assert redirects[0].redirect_request(None, None, 302, "found", {}, "https://external.invalid") is None
    assert not any(isinstance(h, module.urllib.request.ProxyHandler) and h.proxies for h in runtime._opener.handlers)
    runtime.close()


def test_http_error_does_not_echo_server_body_or_api_key(fake):
    runtime = fake.runtime()
    runtime.start()

    def handler(path, request, timeout):
        if path == "/tokenize":
            raise urllib.error.HTTPError(request.full_url, 400, "PRIVATE INPUT MUST NOT BE SHOWN", {}, None)

    fake.handler = handler
    with pytest.raises(module.HyRuntimeError, match="HTTP 400") as error:
        runtime.post_json("/tokenize", {})
    assert "PRIVATE INPUT" not in str(error.value)
    assert fake.processes[0].terminated == 0
    runtime.close()


def test_missing_stream_stop_retires_server(fake):
    runtime = fake.runtime()
    runtime.start()
    fake.handler = lambda path, *_: io.BytesIO(b'data: {"content":"partial","stop":false}\n\n') if path == "/completion" else None
    with pytest.raises(module.HyRuntimeError, match="缺少完成"):
        list(runtime.iter_completion({"prompt": "fixture"}))
    assert fake.processes[0].terminated == 1


def test_abandoning_stream_cancels_owned_native_work(fake):
    runtime = fake.runtime()
    runtime.start()
    stream = runtime.iter_completion({"prompt": "fixture"})
    assert next(stream)["content"] == "fixture"
    stream.close()
    assert fake.processes[0].terminated == 1


class BlockedResponse:
    def __init__(self, process, entered):
        self.process, self.entered = process, entered

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self, *args):
        self.entered.set()
        assert self.process.exited.wait(2)
        raise OSError("fixture server closed")


def test_stop_interrupts_blocked_json_request(fake):
    entered, stop = threading.Event(), threading.Event()
    runtime = fake.runtime(stop_event=stop)
    runtime.start()
    fake.handler = lambda path, *_: BlockedResponse(fake.processes[0], entered) if path == "/apply-template" else None
    thread, errors = run_in_thread(lambda: runtime.post_json("/apply-template", {}))
    assert entered.wait(1)
    stop.set()
    thread.join(1)
    assert not thread.is_alive()
    assert len(errors) == 1 and isinstance(errors[0], module.HyRuntimeCancelled)
    runtime.close()


class ContinuousStream(BlockedResponse):
    def __iter__(self):
        self.entered.set()
        while not self.process.exited.wait(.01):
            yield b'data: {"content":"fixture","stop":false}\n\n'
        raise OSError("fixture server stopped")


def test_stop_interrupts_active_stream_and_does_not_yield_late_output(fake):
    entered, stop = threading.Event(), threading.Event()
    runtime = fake.runtime(stop_event=stop)
    runtime.start()
    fake.handler = lambda path, *_: ContinuousStream(fake.processes[0], entered) if path == "/completion" else None
    thread, errors = run_in_thread(lambda: list(runtime.iter_completion({"prompt": "fixture"})))
    assert entered.wait(1)
    stop.set()
    thread.join(1)
    assert not thread.is_alive()
    assert len(errors) == 1 and isinstance(errors[0], module.HyRuntimeCancelled)
    assert fake.processes[0].terminated == 1
    runtime.close()


def test_wall_timeout_still_applies_while_stream_keeps_sending_chunks(fake):
    entered = threading.Event()
    runtime = fake.runtime(request_timeout=.06)
    runtime.start()
    fake.handler = lambda path, *_: ContinuousStream(fake.processes[0], entered) if path == "/completion" else None
    with pytest.raises(module.HyRuntimeError, match="超过"):
        list(runtime.iter_completion({"prompt": "fixture"}))
    assert entered.is_set() and fake.processes[0].terminated == 1
    runtime.close()


def test_wall_timeout_interrupts_blocked_request(fake):
    entered = threading.Event()
    runtime = fake.runtime(request_timeout=.06)
    runtime.start()
    fake.handler = lambda path, *_: BlockedResponse(fake.processes[0], entered) if path == "/apply-template" else None
    with pytest.raises(module.HyRuntimeError, match="超过"):
        runtime.post_json("/apply-template", {})
    assert entered.is_set() and fake.processes[0].terminated == 1
    runtime.close()


def test_socket_timeout_also_retires_native_generation(fake):
    runtime = fake.runtime(request_timeout=1)
    runtime.start()

    def handler(path, *args):
        if path == "/apply-template":
            raise TimeoutError("fixture timeout")

    fake.handler = handler
    with pytest.raises(module.HyRuntimeError, match="超过"):
        runtime.post_json("/apply-template", {})
    assert fake.processes[0].terminated == 1

