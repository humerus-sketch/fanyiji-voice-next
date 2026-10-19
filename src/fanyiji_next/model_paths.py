"""External weights always follow the main program filename, never CWD."""
from __future__ import annotations

import sys
from pathlib import Path

MODEL_DIRECTORY_SUFFIX = "_models"


def external_model_root(*, main_file: str | Path | None = None, frozen: bool | None = None) -> Path:
    if main_file is None:
        packaged = getattr(sys, "frozen", False) if frozen is None else frozen
        main_file = sys.executable if packaged else Path(__file__).resolve().parents[2] / "run.py"
    main = Path(main_file).absolute()
    return main.parent / (main.stem + MODEL_DIRECTORY_SUFFIX)


def application_model_root(explicit_root: str | Path | None = None) -> Path:
    """Use a caller override or a registered installation; keep EXE defaults."""
    if explicit_root is not None:
        return Path(explicit_root).absolute()
    if getattr(sys, "frozen", False):
        from .managed_installation import managed_model_root
        executable = Path(sys.executable).absolute()
        shared = managed_model_root(executable.parent, executable.name)
        if shared is not None:
            return shared
    return external_model_root()


def bundled_llama_server() -> Path:
    """Runtime binaries are application resources; weights are never resources."""
    source = Path(__file__).resolve().parents[2]
    base = Path(sys.executable).absolute().parent if getattr(sys, "frozen", False) else source
    relative = Path("runtime/llama-b11696-cuda12.4/llama-server.exe")
    normal = base / relative
    internal = Path(getattr(sys, "_MEIPASS", base)) / relative
    return normal if normal.is_file() or not internal.is_file() else internal
