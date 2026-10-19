from pathlib import Path

from fanyiji_next.model_paths import external_model_root, application_model_root


def test_main_filename_controls_prefix_and_not_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    program = tmp_path / "portable" / "翻译姬.exe"
    assert external_model_root(main_file=program) == program.parent / "翻译姬_models"
    assert external_model_root(main_file=program.with_name("Live.exe")) == program.parent / "Live_models"


def test_source_follows_source_entry_filename(tmp_path):
    assert external_model_root(main_file=tmp_path / "main.py") == tmp_path / "main_models"


def test_frozen_does_not_follow_meipass(tmp_path, monkeypatch):
    import sys
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "executable", str(tmp_path / "Reader.exe"))
    monkeypatch.setattr(sys, "_MEIPASS", str(tmp_path / "temporary"), raising=False)
    assert external_model_root() == tmp_path / "Reader_models"


def test_explicit_root_precedes_registry_and_does_not_create_or_read_models(tmp_path, monkeypatch):
    import sys
    from types import SimpleNamespace
    def forbidden(*args):
        raise AssertionError("Explicit root must not consult managed registration")
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setitem(sys.modules, "fanyiji_next.managed_installation", SimpleNamespace(managed_model_root=forbidden))
    monkeypatch.chdir(tmp_path)
    assert application_model_root("shared") == tmp_path / "shared"
    assert not (tmp_path / "shared").exists()


def test_registered_root_only_changes_application_resolution_not_exe_default(tmp_path, monkeypatch):
    import sys
    from types import SimpleNamespace
    calls = []
    root = tmp_path / "FanyiJiVoiceNext_models"
    def registered(directory, name):
        calls.append((directory, name))
        return root
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "executable", str(tmp_path / "dist/0.5.3/FanyiJiVoiceNext/FanyiJiVoiceNext.exe"))
    monkeypatch.setitem(sys.modules, "fanyiji_next.managed_installation", SimpleNamespace(managed_model_root=registered))
    assert application_model_root() == root
    assert external_model_root() == tmp_path / "dist/0.5.3/FanyiJiVoiceNext/FanyiJiVoiceNext_models"
    assert calls == [(Path(sys.executable).parent, "FanyiJiVoiceNext.exe")]
    assert not root.exists()


def test_unregistered_frozen_and_source_keep_normal_filename_rule(tmp_path, monkeypatch):
    import sys
    from types import SimpleNamespace
    calls = []
    def absent(directory, name):
        calls.append(name)
        return None
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "executable", str(tmp_path / "renamed.exe"))
    monkeypatch.setitem(sys.modules, "fanyiji_next.managed_installation", SimpleNamespace(managed_model_root=absent))
    assert application_model_root() == tmp_path / "renamed_models"
    assert calls == ["renamed.exe"]
    monkeypatch.setattr(sys, "frozen", False)
    assert application_model_root() == external_model_root()
    assert calls == ["renamed.exe"]
