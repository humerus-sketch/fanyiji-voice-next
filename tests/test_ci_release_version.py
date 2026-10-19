"""CI version/tag contracts; synthetic sources only, no Git or build process."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def ci():
    path = Path(__file__).resolve().parents[1] / "tools/ci_release.py"
    spec = importlib.util.spec_from_file_location("ci_release_version_contract", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("version,staging", [
    ("0.5.4", "v0.5.4-build-20261019"),
    ("0.5.4", "v0.5.4-build-run.39157112410-2"),
    ("0.5.3", "v0.5.3-r3-build-20261019"),
    ("1.20.0", "v1.20.0-build"),
    ("2.0.1", "v2.0.1-r12-build-new"),
])
def test_staging_tags_are_bound_to_source_version(ci, version, staging):
    ci.validate_release_tags(version, staging, "v" + version)


@pytest.mark.parametrize("staging,final", [
    ("v0.5.4", "v0.5.4"),
    ("v0.5.3-build-20261019", "v0.5.4"),
    ("v0.5.4-build-20261019", "v0.5.3"),
    ("v0.5.4-build-20261019", "v0.5.4-build"),
    ("v0.5.4-build-../other", "v0.5.4"),
    ("v0.5.4-build-a..b", "v0.5.4"),
    ("v0.5.4-build-a/branch", "v0.5.4"),
    ("v0.5.4-build-a\\branch", "v0.5.4"),
    ("v0.5.4-build-a.lock", "v0.5.4"),
    ("v0.5.4-build-a.LOCK", "v0.5.4"),
    ("v0.5.4-build-", "v0.5.4"),
    ("v0.5.4-build-a.", "v0.5.4"),
    ("v0.5.4-build-a@{1}", "v0.5.4"),
    ("v0.5.4-build-a:bad", "v0.5.4"),
    ("v0.5.4-build-a?", "v0.5.4"),
    ("v0.5.4-build-a\n", "v0.5.4"),
    ("v0.5.4-r0-build", "v0.5.4"),
    (None, "v0.5.4"),
])
def test_unsafe_cross_version_or_stable_staging_tags_are_rejected(ci, staging, final):
    with pytest.raises(ValueError):
        ci.validate_release_tags("0.5.4", staging, final)


@pytest.mark.parametrize("version", ["0.5", "0.5.4-rc1", "0.5.4/local", "01.5.4", "０.5.4", "0.5.4\n", None])
def test_source_version_must_be_stable_and_path_safe(ci, version):
    with pytest.raises(ValueError):
        ci.validate_release_tags(version, "v0.5.4-build-20261019", "v0.5.4")


@pytest.fixture
def prepared_ci(ci, tmp_path, monkeypatch):
    source = tmp_path / "checkout"
    source.mkdir()
    (source / "pyproject.toml").write_text('[project]\nversion="0.5.4"\n', encoding="utf-8")
    (source / "README.md").write_text("synthetic immutable source", encoding="utf-8")
    runner = tmp_path / "runner"
    runner.mkdir()
    commit = "1" * 40
    commands, disk_checks = [], []

    def run(command, *, cwd=None, capture=False, timeout=None):
        commands.append(command)
        assert command == ["git", "rev-parse", "HEAD"] and cwd == source and capture
        return commit

    def tracked(command, *, cwd=None):
        commands.append(command)
        assert command == ["git", "ls-files", "-z"] and cwd == source
        return b"pyproject.toml\0README.md\0"

    monkeypatch.setattr(ci, "ROOT", source)
    monkeypatch.setattr(ci, "sys", SimpleNamespace(version_info=(3, 12, 10), version="3.12.10 synthetic"))
    monkeypatch.setattr(ci, "os", SimpleNamespace(name="nt", environ={
        "GITHUB_ACTIONS": "true", "GITHUB_RUN_ID": "123456", "RUNNER_TEMP": str(runner),
    }))
    monkeypatch.setattr(ci, "run", run)
    monkeypatch.setattr(ci.subprocess, "check_output", tracked)
    monkeypatch.setattr(ci, "disk", lambda root, phase, required: disk_checks.append((phase, required)))
    return ci, source, runner, commit, commands, disk_checks


def test_prepare_accepts_new_version_and_copies_only_its_tracked_source(prepared_ci):
    ci, source, runner, commit, commands, disk_checks = prepared_ci
    output = runner / "fanyiji-release-123456-1"
    ci.prepare(output, commit, "v0.5.4-build-20261019", "v0.5.4")
    state = json.loads((output / "state.json").read_text(encoding="utf-8"))
    assert state["version"] == "0.5.4" and state["source_commit"] == commit
    assert state["staging_tag"] == "v0.5.4-build-20261019" and state["final_tag"] == "v0.5.4"
    assert state["model_weights_bundled"] is False and state["public_download_or_install_verified"] is False
    assert "0.5.4" in state["build_id"]
    assert (output / "source/README.md").read_bytes() == (source / "README.md").read_bytes()
    assert disk_checks == [("before-dependencies", 45 * ci.GIB)]
    assert commands == [["git", "rev-parse", "HEAD"], ["git", "ls-files", "-z"]]


@pytest.mark.parametrize("commit", ["main", "a" * 39, "A" * 40, "a" * 41, None])
def test_prepare_rejects_nonexact_commit_before_git_or_output(prepared_ci, commit):
    ci, source, runner, _, commands, disk_checks = prepared_ci
    output = runner / "fanyiji-release-123456-1"
    with pytest.raises(ValueError, match="immutable commit"):
        ci.prepare(output, commit, "v0.5.4-build-20261019", "v0.5.4")
    assert not output.exists() and commands == [] and disk_checks == []


def test_prepare_rejects_mismatched_checkout_before_output(prepared_ci):
    ci, source, runner, _, commands, disk_checks = prepared_ci
    output = runner / "fanyiji-release-123456-1"
    with pytest.raises(ValueError, match="immutable commit"):
        ci.prepare(output, "2" * 40, "v0.5.4-build-20261019", "v0.5.4")
    assert not output.exists() and commands == [["git", "rev-parse", "HEAD"]] and disk_checks == []


@pytest.mark.parametrize("staging,final", [("v0.5.4", "v0.5.4"), ("v0.5.3-build", "v0.5.3")])
def test_prepare_rejects_invalid_release_identity_before_output(prepared_ci, staging, final):
    ci, source, runner, commit, commands, disk_checks = prepared_ci
    output = runner / "fanyiji-release-123456-1"
    with pytest.raises(ValueError, match="independent"):
        ci.prepare(output, commit, staging, final)
    assert not output.exists() and disk_checks == []


def test_prepare_keeps_owned_runner_directory_boundary(prepared_ci):
    ci, source, runner, commit, commands, disk_checks = prepared_ci
    output = runner.parent / "fanyiji-release-123456-1"
    with pytest.raises(ValueError, match="temporary directory"):
        ci.prepare(output, commit, "v0.5.4-build-20261019", "v0.5.4")
    assert not output.exists() and disk_checks == []


def test_workflow_retains_no_compile_draft_only_and_immutable_source_contract(ci):
    workflow = (ci.ROOT / ".github/workflows/build-release.yml").read_text(encoding="utf-8")
    assert "default: 'v0.5.4-build-20261019'" in workflow and "default: 'v0.5.4'" in workflow
    assert "ref: ${{ inputs.source_commit || github.sha }}" in workflow
    assert "CI_SOURCE_COMMIT: ${{ inputs.source_commit || github.sha }}" in workflow
    assert "--no-compile" in workflow and "PYTHONDONTWRITEBYTECODE: '1'" in workflow
    assert "default: false" in workflow and "cancel-in-progress: false" in workflow
    assert "upload-draft --work-root" in workflow and "gh release edit" not in workflow
