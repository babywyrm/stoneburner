"""Tests for `python -m atomics` entry point."""

import importlib
import subprocess
import sys


def test_main_module_invokes_cli(monkeypatch):
    calls: list[bool] = []

    def fake_cli():
        calls.append(True)

    import atomics.cli as cli_module

    monkeypatch.setattr(cli_module, "cli", fake_cli)
    sys.modules.pop("atomics.__main__", None)
    importlib.import_module("atomics.__main__")
    assert calls == [True]


def test_main_subprocess_help():
    result = subprocess.run(
        [sys.executable, "-m", "atomics", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert "Atomics" in result.stdout
