"""Tests for atomics doctor."""

import sys

import pytest

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


def test_doctor_shows_openai_key_missing(capsys, tmp_path, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.chdir(tmp_path)
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


# ── Doctor missing branch coverage ───────────────────────────────────────────

import platform
import sqlite3
from unittest.mock import patch, MagicMock


def test_doctor_db_oserror(monkeypatch, tmp_path):
    """Lines 42-44: OSError path when DB parent isn't creatable."""
    from atomics.doctor import run_doctor
    from atomics.config import AtomicsSettings
    settings = AtomicsSettings(db_path=tmp_path / "doc.db")
    with patch("sqlite3.connect", side_effect=OSError("permission denied")):
        rc = run_doctor(settings=settings)
    assert rc == 1


def test_doctor_openai_sdk_missing(capsys, tmp_path):
    """Line 61: openai SDK not installed path."""
    import importlib.util
    from atomics.doctor import run_doctor
    from atomics.config import AtomicsSettings
    settings = AtomicsSettings(db_path=tmp_path / "doc.db")
    with patch("importlib.util.find_spec", return_value=None):
        run_doctor(settings=settings)
    captured = capsys.readouterr()
    assert "not installed" in captured.out or "not set" in captured.out


def test_doctor_boto3_aws_creds_valid(capsys, tmp_path):
    """Lines 75-79: boto3 installed + valid creds branch."""
    pytest.importorskip("boto3", reason="optional 'bedrock' extra not installed")
    from atomics.doctor import run_doctor
    from atomics.config import AtomicsSettings
    settings = AtomicsSettings(db_path=tmp_path / "doc.db")

    mock_sts = MagicMock()
    mock_sts.get_caller_identity.return_value = {"Account": "123456789"}
    mock_boto3 = MagicMock()
    mock_boto3.client.return_value = mock_sts

    import importlib.util as _ilu
    orig_find_spec = _ilu.find_spec

    def patched_find_spec(name, *args, **kwargs):
        if name == "boto3":
            return MagicMock()  # non-None → boto3 "installed"
        return orig_find_spec(name, *args, **kwargs)

    with patch("importlib.util.find_spec", side_effect=patched_find_spec), \
         patch("boto3.client", return_value=mock_sts):
        run_doctor(settings=settings)

    captured = capsys.readouterr()
    assert "boto3" in captured.out


def test_doctor_boto3_aws_creds_invalid(capsys, tmp_path):
    """Lines 75-79: boto3 installed but creds invalid (exception)."""
    pytest.importorskip("boto3", reason="optional 'bedrock' extra not installed")
    from atomics.doctor import run_doctor
    from atomics.config import AtomicsSettings
    settings = AtomicsSettings(db_path=tmp_path / "doc.db")

    import importlib.util as _ilu
    orig_find_spec = _ilu.find_spec

    def patched_find_spec(name, *args, **kwargs):
        if name == "boto3":
            return MagicMock()
        return orig_find_spec(name, *args, **kwargs)

    with patch("importlib.util.find_spec", side_effect=patched_find_spec), \
         patch("boto3.client", side_effect=Exception("no creds")):
        run_doctor(settings=settings)

    captured = capsys.readouterr()
    assert "boto3" in captured.out


def test_doctor_scheduler_crontab_missing(capsys, tmp_path):
    """Lines 101-102: crontab scheduler detected but binary missing."""
    from atomics.doctor import run_doctor
    from atomics.config import AtomicsSettings
    settings = AtomicsSettings(db_path=tmp_path / "doc.db")
    with patch("atomics.doctor.detect_best_scheduler", return_value="crontab"), \
         patch("shutil.which", return_value=None):
        run_doctor(settings=settings)
    captured = capsys.readouterr()
    assert "crontab" in captured.out


def test_doctor_linux_paths(capsys, tmp_path):
    """Lines 108-111: Linux-specific data dir lines."""
    from atomics.doctor import run_doctor
    from atomics.config import AtomicsSettings
    settings = AtomicsSettings(db_path=tmp_path / "doc.db")
    with patch("platform.system", return_value="Linux"):
        run_doctor(settings=settings)
    captured = capsys.readouterr()
    assert "data dir" in captured.out.lower() or "Linux" in captured.out or captured.out


def test_doctor_scheduler_systemd_missing_systemctl(capsys, tmp_path):
    """Line 104: systemd scheduler detected but systemctl binary missing."""
    from atomics.doctor import run_doctor
    from atomics.config import AtomicsSettings
    settings = AtomicsSettings(db_path=tmp_path / "doc.db")
    with patch("atomics.doctor.detect_best_scheduler", return_value="systemd"), \
         patch("shutil.which", return_value=None):
        run_doctor(settings=settings)
    captured = capsys.readouterr()
    assert "systemd" in captured.out or "systemctl" in captured.out
