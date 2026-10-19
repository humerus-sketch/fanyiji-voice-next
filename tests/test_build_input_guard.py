"""Verification output never flows back into a build or a release package."""
import importlib.util
import json
from pathlib import Path

import pytest


def load_tool(name):
    path = Path(__file__).resolve().parents[1] / "tools" / (name + ".py")
    spec = importlib.util.spec_from_file_location("guard_test_" + name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def guard():
    return load_tool("build_input_guard")


@pytest.mark.parametrize("directory", ["FanyiJiNextReleaseChecks", "fanyijinextreleasechecks", "verification", "validation", "qa"])
def test_qa_tree_is_rejected_before_reading_any_binary(guard, tmp_path, monkeypatch, directory):
    path = tmp_path / directory / "installation/FanyiJiVoiceNext.exe"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"QA executable must not be read")
    monkeypatch.setattr(Path, "read_bytes", lambda path: pytest.fail("Binary content was accessed"))
    with pytest.raises(ValueError, match="验证"):
        guard.assert_build_input(path)


@pytest.mark.parametrize("info", [{"verification_only": True}, {"qa_only": True}, {"validation_only": True},
                                 {"artifact_role": "Verification"}, {"build_role": "public-update-qa"}])
def test_metadata_blocks_a_renamed_qa_program(guard, tmp_path, info):
    folder = tmp_path / "renamed-program"
    folder.mkdir()
    (folder / "build-info.json").write_text(json.dumps(info), encoding="utf-8")
    with pytest.raises(ValueError, match="验证"):
        guard.assert_build_input(folder / "renamed.exe")


def test_explicit_verification_marker_blocks_an_otherwise_normal_folder(guard, tmp_path):
    (tmp_path / "verification-only.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="验证"):
        guard.assert_build_input(tmp_path / "FanyiJiVoiceNext.exe")


def test_parent_role_marker_is_not_hidden_by_child_build_metadata(guard, tmp_path):
    (tmp_path / "verification-only.json").write_text("{}", encoding="utf-8")
    folder = tmp_path / "renamed-copy"
    folder.mkdir()
    (folder / "build-info.json").write_text('{"version":"0.5.3"}', encoding="utf-8")
    with pytest.raises(ValueError, match="验证"):
        guard.assert_build_input(folder)


def test_remote_build_output_cannot_seed_local_packaging(guard, tmp_path, monkeypatch):
    (tmp_path / "build-info.json").write_text(json.dumps({"ci_source_commit": "a" * 40}), encoding="utf-8")
    monkeypatch.delenv("GITHUB_ACTIONS", raising=False)
    with pytest.raises(ValueError, match="远程构建"):
        guard.assert_build_input(tmp_path)
    monkeypatch.setenv("GITHUB_ACTIONS", "true")
    guard.assert_build_input(tmp_path)


def test_local_source_build_and_pinned_local_seed_are_allowed_without_binary_reads(guard, tmp_path, monkeypatch):
    folder = tmp_path / "artifacts/local-seed"
    folder.mkdir(parents=True)
    seed = folder / "pre-font-main.exe"
    seed.write_bytes(b"local seed")
    monkeypatch.setattr(Path, "read_bytes", lambda path: pytest.fail("Binary content was accessed"))
    guard.assert_build_input(seed)
    guard.assert_build_input(tmp_path / "src")


def test_update_zip_rejects_qa_before_scanning_or_hashing(tmp_path, monkeypatch):
    packager = load_tool("package_release")
    folder = tmp_path / "FanyiJiNextReleaseChecks/candidate"
    folder.mkdir(parents=True)
    monkeypatch.setattr(packager, "_program_files", lambda *args: pytest.fail("QA files enumerated"))
    with pytest.raises(ValueError, match="验证"):
        packager.create_update_zip(folder, tmp_path / "out.zip", "0.5.3")
    assert not (tmp_path / "out.zip").exists()


def test_full_packaging_rejects_qa_before_creating_output(tmp_path):
    packager = load_tool("package_release")
    output = tmp_path / "new-release"
    with pytest.raises(ValueError, match="验证"):
        packager.package_release(program_folder=tmp_path / "qa/installation", runtime=tmp_path / "runtime", output=output, version="0.5.3")
    assert not output.exists()


def test_patch_rejects_a_qa_baseline_before_reading_or_copying(tmp_path):
    patcher = load_tool("package_patch")
    output = tmp_path / "patch"
    with pytest.raises(ValueError, match="验证"):
        patcher.package_patch(program_folder=tmp_path / "fresh", baseline_folder=tmp_path / "qa/installation",
                              baseline_manifest=tmp_path / "report.json", output=output, version="0.5.3")
    assert not output.exists()


def test_runtime_assembler_rejects_a_qa_archive_before_hashing(tmp_path):
    assembler = load_tool("assemble_runtime_bundle")
    output = tmp_path / "runtime.zip"
    with pytest.raises(ValueError, match="验证"):
        assembler.assemble_runtime_bundle(source_zip=tmp_path / "qa/candidate.zip", source_manifest=tmp_path / "manifest.json",
            source_sha256="a" * 64, source_manifest_sha256="b" * 64, overlay=tmp_path / "fresh-overlay", output_zip=output,
            version="0.5.3", build_id="local-fixture", repository="example/repo")
    assert not output.exists()


def test_source_builder_has_no_qa_exe_or_spec_seed_input(tmp_path, monkeypatch):
    builder = load_tool("build")
    monkeypatch.setattr(builder, "nvidia_binaries", lambda: [])
    command = builder.build_command(tmp_path / "dist", tmp_path / "build", "voice")
    assert command[-1] == str(builder.ROOT / "run.py")
    assert not any(Path(item).suffix in {".exe", ".spec"} for item in command[2:])
    monkeypatch.setattr(builder, "ROOT", tmp_path / "FanyiJiNextReleaseChecks/source-copy")
    with pytest.raises(ValueError, match="验证"):
        builder.build_command(tmp_path / "dist", tmp_path / "build", "voice")


def test_source_builder_rejects_a_frozen_verification_interpreter(tmp_path, monkeypatch):
    builder = load_tool("build")
    monkeypatch.setattr(builder.sys, "frozen", True, raising=False)
    with pytest.raises(ValueError, match="独立 Python"):
        builder.build_command(tmp_path / "dist", tmp_path / "build", "prototype")
