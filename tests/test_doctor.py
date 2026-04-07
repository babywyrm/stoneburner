"""Tests for atomics doctor."""

import sys

from atomics.doctor import run_doctor


def test_doctor_exits_zero(monkeypatch, tmp_path):
    monkeypatch.setenv("ATOMICS_DB_PATH", str(tmp_path / "doc.db"))
    assert run_doctor() == 0


def test_doctor_fails_old_python(monkeypatch, tmp_path):
    monkeypatch.setenv("ATOMICS_DB_PATH", str(tmp_path / "doc.db"))
    monkeypatch.setattr(sys, "version_info", (3, 10, 0))
    assert run_doctor() == 1
