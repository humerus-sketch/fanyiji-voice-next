"""Windows DLL setup tests fake every native load and package metadata lookup."""

from concurrent.futures import ThreadPoolExecutor
from importlib import metadata
from pathlib import Path
import os
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from fanyiji_next import native_runtime as module


@pytest.fixture
def runtime(tmp_path, monkeypatch):
    monkeypatch.setattr(module.sys, "platform", "win32")
    monkeypatch.setattr(module.sys, "frozen", False, raising=False)
    monkeypatch.setattr(module, "_CONFIGURED_DIRS", None)
    monkeypatch.setattr(module, "_DLL_DIRECTORY_HANDLES", [])
    monkeypatch.setattr(module, "_PRELOADED_LIBRARIES", [])
    monkeypatch.setenv("PATH", os.pathsep.join(["existing_first", "existing_second"]))
    distributions = {}
    directories = []
    for distribution, component, required in module._COMPONENTS:
        relative = Path("nvidia") / component / "bin"
        folder = tmp_path / relative
        folder.mkdir(parents=True)
        for name in required:
            (folder / name).write_bytes(b"fake DLL: must never be loaded")
        distributions[distribution] = SimpleNamespace(
            files=[(relative / name).as_posix() for name in required],
            locate_file=Mock(side_effect=lambda value: tmp_path / value),
        )
        directories.append(folder.resolve())
    provider = Mock(side_effect=distributions.__getitem__)
    monkeypatch.setattr(metadata, "distribution", provider)
    handles = []

    def add_directory(path):
        handle = SimpleNamespace(close=Mock())
        handles.append(handle)
        return handle

    add = Mock(side_effect=add_directory)
    preload = Mock(side_effect=lambda path: SimpleNamespace(path=path))
    monkeypatch.setattr(module.os, "add_dll_directory", add, raising=False)
    monkeypatch.setattr(module, "_preload_library", preload)
    return SimpleNamespace(base=tmp_path, distributions=distributions, provider=provider,
                           directories=directories, handles=handles, add=add, preload=preload)


def test_cpu_and_non_windows_paths_do_not_probe_or_mutate_environment(runtime, monkeypatch):
    old = os.environ["PATH"]
    assert module.configure_cuda_runtime("cpu") == ()
    monkeypatch.setattr(module.sys, "platform", "linux")
    assert module.configure_cuda_runtime("cuda") == ()
    assert os.environ["PATH"] == old
    runtime.provider.assert_not_called()
    runtime.add.assert_not_called()
    runtime.preload.assert_not_called()


def test_only_distribution_directories_are_added_and_absolute_core_dlls_are_preloaded(runtime):
    old = os.environ["PATH"]
    paths = module.configure_cuda_runtime()
    assert paths == tuple(str(path) for path in runtime.directories)
    assert os.environ["PATH"].split(os.pathsep) == [*paths, *old.split(os.pathsep)]
    assert [call.args[0] for call in runtime.provider.call_args_list] == [
        "nvidia-cublas-cu12", "nvidia-cudnn-cu12",
    ]
    assert [call.args[0] for call in runtime.preload.call_args_list] == [
        runtime.directories[0] / "cublasLt64_12.dll",
        runtime.directories[0] / "cublas64_12.dll",
        runtime.directories[1] / "cudnn64_9.dll",
    ]
    assert all(call.args[0].is_absolute() for call in runtime.preload.call_args_list)
    assert module._DLL_DIRECTORY_HANDLES == runtime.handles
    assert len(module._PRELOADED_LIBRARIES) == 3
    assert all(not handle.close.called for handle in runtime.handles)


def test_setup_is_idempotent_even_when_called_by_parallel_backend_constructors(runtime):
    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(lambda _: module.configure_cuda_runtime(), range(24)))
    assert all(result == results[0] for result in results)
    assert runtime.provider.call_count == 2
    assert runtime.add.call_count == 2
    assert runtime.preload.call_count == 3
    assert len(os.environ["PATH"].split(os.pathsep)) == 4


def test_existing_runtime_path_entries_are_promoted_without_duplicates(runtime, monkeypatch):
    monkeypatch.setenv("PATH", os.pathsep.join(["existing_first", str(runtime.directories[1]), str(runtime.directories[0])]))
    paths = module.configure_cuda_runtime()
    assert os.environ["PATH"].split(os.pathsep) == [*paths, "existing_first"]


def test_frozen_runtime_uses_only_bundle_locations_without_installed_package_fallback(runtime, monkeypatch):
    monkeypatch.setattr(module.sys, "frozen", True, raising=False)
    monkeypatch.setattr(module.sys, "_MEIPASS", str(runtime.base), raising=False)
    paths = module.configure_cuda_runtime()
    assert paths == tuple(str(path) for path in runtime.directories)
    runtime.provider.assert_not_called()


def test_frozen_missing_library_does_not_fall_back_to_installed_wheel_or_cwd(runtime, monkeypatch):
    monkeypatch.setattr(module.sys, "frozen", True, raising=False)
    monkeypatch.setattr(module.sys, "_MEIPASS", str(runtime.base / "missing_bundle"), raising=False)
    old = os.environ["PATH"]
    with pytest.raises(module.CudaRuntimeError, match="目录不存在"):
        module.configure_cuda_runtime()
    assert os.environ["PATH"] == old
    runtime.provider.assert_not_called()
    runtime.preload.assert_not_called()


def test_missing_distribution_is_an_actionable_error_before_environment_mutation(runtime):
    runtime.provider.side_effect = metadata.PackageNotFoundError("nvidia-cublas-cu12")
    old = os.environ["PATH"]
    with pytest.raises(module.CudaRuntimeError, match="nvidia-cublas-cu12") as error:
        module.configure_cuda_runtime()
    assert isinstance(error.value.__cause__, metadata.PackageNotFoundError)
    assert os.environ["PATH"] == old
    runtime.add.assert_not_called()
    runtime.preload.assert_not_called()


def test_incomplete_wheel_record_does_not_accept_an_ambient_dll(runtime):
    runtime.distributions["nvidia-cublas-cu12"].files = []
    with pytest.raises(module.CudaRuntimeError, match="安装记录缺少 cublasLt64_12.dll"):
        module.configure_cuda_runtime()
    runtime.add.assert_not_called()
    runtime.preload.assert_not_called()


def test_missing_cudnn_child_library_is_detected_before_loading_any_dll(runtime):
    (runtime.directories[1] / "cudnn_graph64_9.dll").unlink()
    with pytest.raises(module.CudaRuntimeError, match="cudnn_graph64_9.dll"):
        module.configure_cuda_runtime()
    runtime.add.assert_not_called()
    runtime.preload.assert_not_called()


def test_invalid_dll_error_is_preserved_and_failed_setup_can_be_retried(runtime):
    old = os.environ["PATH"]
    runtime.preload.side_effect = [SimpleNamespace(), OSError("invalid PE image")]
    with pytest.raises(module.CudaRuntimeError, match="cublas64_12.dll.*invalid PE image") as error:
        module.configure_cuda_runtime()
    assert isinstance(error.value.__cause__, OSError)
    assert module._CONFIGURED_DIRS is None
    assert os.environ["PATH"] == old
    assert all(handle.close.call_count == 1 for handle in runtime.handles)
    assert module._DLL_DIRECTORY_HANDLES == []
    assert len(module._PRELOADED_LIBRARIES) == 1
    runtime.preload.side_effect = lambda path: SimpleNamespace(path=path)
    assert module.configure_cuda_runtime() == tuple(str(path) for path in runtime.directories)


def test_absolute_preloader_uses_restricted_windows_search_flags(monkeypatch, tmp_path):
    import ctypes
    library = tmp_path / "cublas64_12.dll"
    load = Mock(return_value=object())
    monkeypatch.setattr(ctypes, "WinDLL", load, raising=False)
    result = module._preload_library(library)
    assert result is load.return_value
    load.assert_called_once_with(str(library), winmode=0x00000100 | 0x00001000)
