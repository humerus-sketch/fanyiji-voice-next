"""Keep verification installations and remote artifacts out of local builds."""
from __future__ import annotations

import json
import os
from pathlib import Path


_VERIFICATION_DIRECTORIES = {"fanyijinextreleasechecks", "verification", "validation", "qa"}
_VERIFICATION_MARKERS = ("verification-only.json", ".verification-only.json")
_VERIFICATION_ROLES = {"verification", "validation", "qa", "public-update-qa"}


def assert_build_input(path: Path, *, purpose: str = "program input") -> None:
    """Check provenance before enumerating/copying/hashing a program input.

    This does not inspect executable or model contents. Local, hash-pinned
    build seeds remain usable; a QA installation is never a build seed.
    """
    path = Path(path).absolute()
    if ".." in path.parts:
        raise ValueError(f"构建输入不得含父路径：{purpose}")
    if any(part.casefold() in _VERIFICATION_DIRECTORIES for part in path.parts):
        raise ValueError(f"验证程序/目录不得作为构建输入：{purpose}")
    for node in (path, *path.parents):
        if node.is_symlink() or node.is_junction():
            raise ValueError(f"构建输入不得经过链接：{purpose}")
    start = path if path.is_dir() else path.parent
    if any(os.path.lexists(directory / name)
           for directory in (start, *start.parents) for name in _VERIFICATION_MARKERS):
        raise ValueError(f"已标记的验证程序不得作为构建输入：{purpose}")
    for directory in (start, *start.parents):
        info_path = directory / "build-info.json"
        if not os.path.lexists(info_path):
            continue
        if info_path.is_symlink() or info_path.is_junction() or info_path.stat().st_size > 2 * 1024**2:
            raise ValueError(f"构建来源元数据无效：{purpose}")
        try:
            info = json.loads(info_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, ValueError) as error:
            raise ValueError(f"构建来源元数据不可读：{purpose}") from error
        if not isinstance(info, dict):
            raise ValueError(f"构建来源元数据不是对象：{purpose}")
        if (any(info.get(name) is True for name in ("verification_only", "validation_only", "qa_only"))
                or any(isinstance(info.get(name), str) and info[name].casefold() in _VERIFICATION_ROLES
                       for name in ("build_role", "artifact_role", "role"))):
            raise ValueError(f"验证程序不得作为构建输入：{purpose}")
        if os.environ.get("GITHUB_ACTIONS") != "true" and info.get("ci_source_commit"):
            raise ValueError(f"远程构建产物不得作为本地构建输入：{purpose}")
        break
