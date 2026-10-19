"""Source builds switch only an explicitly registered, stopped primary install."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from fanyiji_next import app_update as update
from fanyiji_next import managed_installation as managed


@pytest.fixture
def build_case(tmp_path, monkeypatch):
    path = Path(__file__).resolve().parents[1] / "tools/build.py"
    spec = importlib.util.spec_from_file_location("managed_source_builder", path)
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    root = tmp_path / "project"
    old = root / "dist/0.4.8/FanyiJiVoiceNext"
    old.mkdir(parents=True)
    (old / "FanyiJiVoiceNext.exe").write_bytes(b"old source build")
    (old / "build-info.json").write_text(json.dumps({"application": update.APPLICATION, "version": "0.4.8"}))
    (root / "pyproject.toml").write_text('[project]\nname="fanyiji-voice-next"\nversion="0.5.0"\n')
    for name in ("run.py", "run_prototype.py", "README.md"):
        (root / name).write_text("fixture")
    (root / "FanyiJiVoiceNext_models").mkdir()
    (root / "FanyiJiVoiceNext_models/user.gguf").write_bytes(b"shared model sentinel")
    value = {"schema_version": 1, "application": update.APPLICATION, "role": "primary", "current_version": "0.4.8",
             "program_relative": "dist/0.4.8/FanyiJiVoiceNext", "model_relative": "FanyiJiVoiceNext_models"}
    registry = root / managed.REGISTRY_NAME
    registry.write_text(json.dumps(value))
    monkeypatch.setattr(builder, "ROOT", root)
    monkeypatch.delenv("GITHUB_ACTIONS", raising=False)
    monkeypatch.setattr(builder.importlib.metadata, "distribution", lambda name: object())
    monkeypatch.setattr(builder, "collect_licenses", lambda *args: None)
    monkeypatch.setattr(builder, "run_frozen_checks", lambda *args: None)
    monkeypatch.setattr(builder, "primary_program_running", lambda paths: False)
    monkeypatch.setattr(builder, "build_command", lambda output, *args: [str(output)])
    def package(command, **kwargs):
        folder = Path(command[0]) / "FanyiJiVoiceNext"
        folder.mkdir()
        (folder / "FanyiJiVoiceNext.exe").write_bytes(b"new source build")
    monkeypatch.setattr(builder.subprocess, "run", package)
    return builder, root, old, registry


def test_default_voice_build_switches_after_checks_and_preserves_existing_program_and_model(build_case):
    builder, root, old, registry = build_case
    builder.main(["--edition", "voice"])
    value = json.loads(registry.read_text())
    assert value["current_version"] == "0.5.0"
    assert value["program_relative"] == "dist/0.5.0/FanyiJiVoiceNext"
    assert managed.managed_model_root(root / value["program_relative"]) == root / "FanyiJiVoiceNext_models"
    assert (old / "FanyiJiVoiceNext.exe").read_bytes() == b"old source build"
    assert (root / "FanyiJiVoiceNext_models/user.gguf").read_bytes() == b"shared model sentinel"


@pytest.mark.parametrize("case", ["ci", "unregistered", "custom", "prototype"])
def test_ci_unregistered_custom_and_prototype_builds_never_change_primary(build_case, monkeypatch, case):
    builder, root, _, registry = build_case
    before = registry.read_bytes()
    args = ["--edition", "prototype" if case == "prototype" else "voice"]
    if case == "ci":
        monkeypatch.setenv("GITHUB_ACTIONS", "true")
    elif case == "unregistered":
        registry.rename(root / "held-registry.json")
    elif case == "custom":
        args.extend(("--output", str(root / "custom-build")))
    builder.main(args)
    if case == "unregistered":
        assert not registry.exists()
    else:
        assert registry.read_bytes() == before


@pytest.mark.parametrize("failure", ["frozen-check", "changed-registry", "running", "wrong-build-info"])
def test_failed_checks_changed_registration_and_running_program_keep_new_output_unselected(build_case, monkeypatch, failure):
    builder, root, _, registry = build_case
    before = registry.read_bytes()
    def checks(folder, *args):
        if failure == "frozen-check":
            raise RuntimeError("fixture frozen check failure")
        if failure == "changed-registry":
            registry.write_bytes(before + b"\n")
        if failure == "wrong-build-info":
            value = json.loads((folder / "build-info.json").read_text())
            value["version"] = "0.4.8"
            (folder / "build-info.json").write_text(json.dumps(value))
    monkeypatch.setattr(builder, "run_frozen_checks", checks)
    if failure == "running":
        monkeypatch.setattr(builder, "primary_program_running", lambda paths: True)
    with pytest.raises(RuntimeError):
        builder.main(["--edition", "voice"])
    assert registry.read_bytes() == before + (b"\n" if failure == "changed-registry" else b"")
    assert (root / "dist/0.5.0/FanyiJiVoiceNext/FanyiJiVoiceNext.exe").read_bytes() == b"new source build"
    assert (root / "FanyiJiVoiceNext_models/user.gguf").read_bytes() == b"shared model sentinel"


def test_process_check_rechecks_registry_before_atomic_write(build_case, monkeypatch):
    builder, _, _, registry = build_case
    before = registry.read_bytes()
    def running(paths):
        registry.write_bytes(before + b"\n")
        return False
    monkeypatch.setattr(builder, "primary_program_running", running)
    with pytest.raises(RuntimeError, match="进程检查期间"):
        builder.main(["--edition", "voice"])
    assert registry.read_bytes() == before + b"\n"


def test_build_activation_uses_same_project_mutex_as_update_helper(build_case, monkeypatch):
    builder, root, _, registry = build_case
    # Save the real subprocess implementation before the build-only fake.
    monkeypatch.undo()
    before = registry.read_bytes()
    target = root / "dist/0.5.0/FanyiJiVoiceNext"
    target.mkdir(parents=True)
    (target / "FanyiJiVoiceNext.exe").write_bytes(b"new source build")
    (target / "build-info.json").write_text(json.dumps({"application": update.APPLICATION, "version": "0.5.0", "edition": "voice"}))
    source = Path(update.__file__).resolve().parents[2]
    code = ("from pathlib import Path;import sys;sys.path.insert(0,sys.argv[1]);import build;"
            "from fanyiji_next.managed_installation import read_managed_installation;"
            "build.primary_program_running=lambda paths:False;\n"
            "try: build.activate_built_primary(read_managed_installation(Path(sys.argv[2])),Path(sys.argv[3]),'0.5.0')\n"
            "except RuntimeError as e: print(str(e));sys.exit(0 if '另一更新助手' in str(e) else 3)\n"
            "sys.exit(2)\n")
    with update._installation_lock(root):
        result = subprocess.run([sys.executable, "-X", "utf8", "-B", "-c", code, str(source / "tools"),
                                 str(root / "dist/0.4.8/FanyiJiVoiceNext"), str(target)],
                                env=dict(os.environ, PYTHONPATH=str(source / "src")),
                                capture_output=True, text=True, encoding="utf-8", timeout=15)
    assert result.returncode == 0, result.stdout + result.stderr
    assert registry.read_bytes() == before
