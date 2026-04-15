"""Tests for atomics doctor."""

import sys

from atomics.config import AtomicsSettings
from atomics.doctor import run_doctor


def test_doctor_exits_zero(monkeypatch, tmp_path):
    monkeypatch.setenv("ATOMICS_DB_PATH", str(tmp_path / "doc.db"))
    assert run_doctor() == 0


def test_doctor_fails_old_python(monkeypatch, tmp_path):
    monkeypatch.setenv("ATOMICS_DB_PATH", str(tmp_path / "doc.db"))
    monkeypatch.setattr(sys, "version_info", (3, 10, 0))
    assert run_doctor() == 1


def test_doctor_shows_openai_key_set(capsys, tmp_path):
    settings = AtomicsSettings(
        db_path=tmp_path / "doc.db",
        openai_api_key="sk-test",
    )
    run_doctor(settings=settings)
    captured = capsys.readouterr()
    assert "OPENAI_API_KEY" in captured.out


def test_doctor_shows_openai_key_missing(capsys, tmp_path):
    settings = AtomicsSettings(
        db_path=tmp_path / "doc.db",
        openai_api_key="",
    )
    run_doctor(settings=settings)
    captured = capsys.readouterr()
    assert "OPENAI_API_KEY" in captured.out
    assert "not set" in captured.out


def test_doctor_checks_boto3_creds(capsys, monkeypatch, tmp_path):
    """Doctor should attempt AWS credential validation when boto3 is installed."""
    settings = AtomicsSettings(db_path=tmp_path / "doc.db")
    rc = run_doctor(settings=settings)
    captured = capsys.readouterr()
    assert "boto3" in captured.out
    assert rc == 0
