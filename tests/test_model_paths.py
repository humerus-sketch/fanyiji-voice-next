from pathlib import Path

from fanyiji_next.model_paths import external_model_root


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
