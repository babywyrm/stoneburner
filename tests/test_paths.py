"""Tests for cross-platform default paths."""

from pathlib import Path

from atomics.paths import default_data_dir, default_db_path


def test_default_paths_non_linux(monkeypatch):
    monkeypatch.setattr("atomics.paths.platform.system", lambda: "Darwin")
    assert default_data_dir() == Path("data")
    assert default_db_path() == Path("data") / "atomics.db"


def test_default_paths_linux_xdg(monkeypatch, tmp_path):
    monkeypatch.setattr("atomics.paths.platform.system", lambda: "Linux")
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "xdg"))
    assert default_data_dir() == tmp_path / "xdg" / "atomics"
    assert default_db_path() == tmp_path / "xdg" / "atomics" / "atomics.db"


def test_default_paths_linux_fallback(monkeypatch, tmp_path):
    monkeypatch.setattr("atomics.paths.platform.system", lambda: "Linux")
    monkeypatch.delenv("XDG_DATA_HOME", raising=False)
    monkeypatch.setattr("atomics.paths.Path.home", lambda: tmp_path)
    assert default_data_dir() == tmp_path / ".local" / "share" / "atomics"
