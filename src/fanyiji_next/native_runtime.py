"""Configure this process's Windows CUDA libraries from pinned package locations."""

from __future__ import annotations

import os
from pathlib import Path, PurePosixPath
import sys
import threading


_LOCK = threading.RLock()
_CONFIGURED_DIRS: tuple[str, ...] | None = None
_DLL_DIRECTORY_HANDLES: list[object] = []
_PRELOADED_LIBRARIES: list[object] = []
_COMPONENTS = (
    ("nvidia-cublas-cu12", "cublas", ("cublasLt64_12.dll", "cublas64_12.dll")),
    ("nvidia-cudnn-cu12", "cudnn", (
        "cudnn64_9.dll", "cudnn_adv64_9.dll", "cudnn_cnn64_9.dll", "cudnn_graph64_9.dll",
        "cudnn_ops64_9.dll", "cudnn_engines_precompiled64_9.dll",
        "cudnn_engines_runtime_compiled64_9.dll", "cudnn_heuristic64_9.dll",
    )),
)


class CudaRuntimeError(RuntimeError):
    """The selected CUDA runtime is missing or cannot be loaded safely."""


def _runtime_directories() -> tuple[Path, ...]:
    frozen = bool(getattr(sys, "frozen", False))
    if frozen:
        value = getattr(sys, "_MEIPASS", None)
        if not value or not Path(value).is_absolute():
            raise CudaRuntimeError("冻结程序缺少有效的 CUDA 运行库根目录 sys._MEIPASS。")
        bundle = Path(value).resolve()
    else:
        from importlib import metadata

    directories = []
    for distribution_name, component, required in _COMPONENTS:
        relative = PurePosixPath("nvidia", component, "bin")
        if frozen:
            directory = (bundle / Path(relative)).resolve()
            if not directory.is_relative_to(bundle):
                raise CudaRuntimeError(f"CUDA 运行库目录超出冻结程序目录：{directory}")
        else:
            try:
                distribution = metadata.distribution(distribution_name)
            except metadata.PackageNotFoundError as exc:
                raise CudaRuntimeError(f"CUDA 缺少本项目依赖 {distribution_name}；请使用完整的 GPU 运行环境。") from exc
            recorded = {str(item).replace("\\", "/") for item in (distribution.files or ())}
            for filename in required:
                if str(relative / filename) not in recorded:
                    raise CudaRuntimeError(f"{distribution_name} 的安装记录缺少 {filename}。")
            located = Path(distribution.locate_file(str(relative)))
            if not located.is_absolute():
                raise CudaRuntimeError(f"{distribution_name} 返回了非绝对运行库目录。")
            directory = located.resolve()
        if not directory.is_dir():
            raise CudaRuntimeError(f"CUDA 运行库目录不存在：{directory}")
        for filename in required:
            path = directory / filename
            if not path.is_file() or path.resolve().parent != directory:
                raise CudaRuntimeError(f"CUDA 运行库缺失或位置无效：{path}")
        directories.append(directory)
    return tuple(directories)


def _preload_library(path: Path):
    import ctypes

    # Absolute path plus DLL-load/default directories avoids a bare-name lookup
    # from the working directory. Keep these objects alive for the process.
    return ctypes.WinDLL(str(path), winmode=0x00000100 | 0x00001000)


def configure_cuda_runtime(device: str = "cuda") -> tuple[str, ...]:
    """Register NVIDIA wheel/frozen directories once, only for Windows CUDA.

    PATH changes affect this process only. They also cover cuDNN's own dynamic
    child-library loading, which is outside Python's extension import machinery.
    CPU callers never inspect package metadata or touch native libraries/PATH.
    """
    global _CONFIGURED_DIRS
    if device == "cpu" or sys.platform != "win32":
        return ()
    if device != "cuda":
        raise ValueError(f"不支持的运行设备：{device}")
    with _LOCK:
        if _CONFIGURED_DIRS is not None:
            return _CONFIGURED_DIRS
        directories = _runtime_directories()
        paths = tuple(str(directory) for directory in directories)
        old_path = os.environ.get("PATH")
        normal = lambda value: os.path.normcase(os.path.normpath(value.strip('"')))
        expected = {normal(value) for value in paths}
        retained = [value for value in (old_path or "").split(os.pathsep)
                    if value and normal(value) not in expected]
        process_path = os.pathsep.join([*paths, *retained])
        handles = []
        try:
            for path in paths:
                try:
                    handles.append(os.add_dll_directory(path))
                except OSError as exc:
                    raise CudaRuntimeError(f"无法注册 CUDA DLL 目录：{path}；{exc}") from exc
            os.environ["PATH"] = process_path
            for path in (directories[0] / "cublasLt64_12.dll", directories[0] / "cublas64_12.dll",
                         directories[1] / "cudnn64_9.dll"):
                try:
                    _PRELOADED_LIBRARIES.append(_preload_library(path))
                except OSError as exc:
                    raise CudaRuntimeError(f"CUDA DLL 加载失败：{path}；{exc}") from exc
        except Exception as error:
            # A failed setup must not leave closed directory handles pretending
            # configuration succeeded. Partial native loads stay referenced.
            if os.environ.get("PATH") == process_path:
                if old_path is None:
                    os.environ.pop("PATH", None)
                else:
                    os.environ["PATH"] = old_path
            for handle in reversed(handles):
                try:
                    handle.close()
                except Exception as cleanup_error:
                    error.add_note(f"CUDA DLL 目录句柄清理失败：{cleanup_error}")
            raise
        _DLL_DIRECTORY_HANDLES.extend(handles)
        _CONFIGURED_DIRS = paths
        return paths
