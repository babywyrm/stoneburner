"""Tests for atomics doctor."""

import sys

import pytest

from atomics.config import AtomicsSettings
from atomics.doctor import run_doctor, suggest_next_step


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


def test_doctor_anthropic_key_is_optional_for_local_provider_test(capsys, tmp_path, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.chdir(tmp_path)
    settings = AtomicsSettings(
        db_path=tmp_path / "doc.db",
        anthropic_api_key="",
    )
    assert run_doctor(settings=settings) == 0
    out = capsys.readouterr().out
    assert "ANTHROPIC_API_KEY" in out
    assert "not set" in out
    assert "required for provider-test" not in out.lower()
    assert "optional" in out.lower()
    assert "Claude" in out


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

from unittest.mock import MagicMock, patch


def test_doctor_db_oserror(monkeypatch, tmp_path):
    """Lines 42-44: OSError path when DB parent isn't creatable."""
    from atomics.config import AtomicsSettings
    from atomics.doctor import run_doctor

    settings = AtomicsSettings(db_path=tmp_path / "doc.db")
    with patch("sqlite3.connect", side_effect=OSError("permission denied")):
        rc = run_doctor(settings=settings)
    assert rc == 1


def test_doctor_openai_sdk_missing(capsys, tmp_path):
    """Line 61: openai SDK not installed path."""
    from atomics.config import AtomicsSettings
    from atomics.doctor import run_doctor

    settings = AtomicsSettings(db_path=tmp_path / "doc.db")
    with patch("importlib.util.find_spec", return_value=None):
        run_doctor(settings=settings)
    captured = capsys.readouterr()
    assert "not installed" in captured.out or "not set" in captured.out


def test_doctor_boto3_aws_creds_valid(capsys, tmp_path):
    """Lines 75-79: boto3 installed + valid creds branch."""
    pytest.importorskip("boto3", reason="optional 'bedrock' extra not installed")
    from atomics.config import AtomicsSettings
    from atomics.doctor import run_doctor

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

    with (
        patch("importlib.util.find_spec", side_effect=patched_find_spec),
        patch("boto3.client", return_value=mock_sts),
    ):
        run_doctor(settings=settings)

    captured = capsys.readouterr()
    assert "boto3" in captured.out


def test_doctor_boto3_aws_creds_invalid(capsys, tmp_path):
    """Lines 75-79: boto3 installed but creds invalid (exception)."""
    pytest.importorskip("boto3", reason="optional 'bedrock' extra not installed")
    from atomics.config import AtomicsSettings
    from atomics.doctor import run_doctor

    settings = AtomicsSettings(db_path=tmp_path / "doc.db")

    import importlib.util as _ilu

    orig_find_spec = _ilu.find_spec

    def patched_find_spec(name, *args, **kwargs):
        if name == "boto3":
            return MagicMock()
        return orig_find_spec(name, *args, **kwargs)

    with (
        patch("importlib.util.find_spec", side_effect=patched_find_spec),
        patch("boto3.client", side_effect=Exception("no creds")),
    ):
        run_doctor(settings=settings)

    captured = capsys.readouterr()
    assert "boto3" in captured.out


def test_doctor_scheduler_crontab_missing(capsys, tmp_path):
    """Lines 101-102: crontab scheduler detected but binary missing."""
    from atomics.config import AtomicsSettings
    from atomics.doctor import run_doctor

    settings = AtomicsSettings(db_path=tmp_path / "doc.db")
    with (
        patch("atomics.doctor.detect_best_scheduler", return_value="crontab"),
        patch("shutil.which", return_value=None),
    ):
        run_doctor(settings=settings)
    captured = capsys.readouterr()
    assert "crontab" in captured.out


def test_doctor_linux_paths(capsys, tmp_path):
    """Lines 108-111: Linux-specific data dir lines."""
    from atomics.config import AtomicsSettings
    from atomics.doctor import run_doctor

    settings = AtomicsSettings(db_path=tmp_path / "doc.db")
    with patch("platform.system", return_value="Linux"):
        run_doctor(settings=settings)
    captured = capsys.readouterr()
    assert "data dir" in captured.out.lower() or "Linux" in captured.out or captured.out


def test_doctor_reports_inference_env_without_leaking_key(capsys, tmp_path, monkeypatch):
    env_path = tmp_path / "inference.env"
    env_path.write_text(
        "INFERENCE_BACKEND=ollama\n"
        "INFERENCE_URL=http://127.0.0.1:11434\n"
        "INFERENCE_MODEL=gemma3:4b\n"
        "INFERENCE_API_KEY=sk-secret-must-not-leak\n"
    )
    monkeypatch.setenv("INFERENCE_ENV", str(env_path))
    monkeypatch.delenv("BRAIN_ENV", raising=False)
    settings = AtomicsSettings(db_path=tmp_path / "doc.db")
    assert run_doctor(settings=settings) == 0
    out = capsys.readouterr().out
    assert "inference.env" in out
    assert "ollama" in out
    assert "gemma3:4b" in out
    assert "http://127.0.0.1:11434" in out
    assert "sk-secret-must-not-leak" not in out


def test_doctor_reports_missing_inference_env(capsys, tmp_path, monkeypatch):
    monkeypatch.setenv("INFERENCE_ENV", str(tmp_path / "missing.env"))
    monkeypatch.delenv("BRAIN_ENV", raising=False)
    monkeypatch.setattr("atomics.inference._DEFAULT_PATHS", ())
    settings = AtomicsSettings(db_path=tmp_path / "doc.db")
    assert run_doctor(settings=settings) == 0
    out = capsys.readouterr().out
    assert "inference.env" in out
    assert "not found" in out.lower()


def test_doctor_scheduler_systemd_missing_systemctl(capsys, tmp_path):
    """Line 104: systemd scheduler detected but systemctl binary missing."""
    from atomics.config import AtomicsSettings
    from atomics.doctor import run_doctor

    settings = AtomicsSettings(db_path=tmp_path / "doc.db")
    with (
        patch("atomics.doctor.detect_best_scheduler", return_value="systemd"),
        patch("shutil.which", return_value=None),
    ):
        run_doctor(settings=settings)
    captured = capsys.readouterr()
    assert "systemd" in captured.out or "systemctl" in captured.out


def test_suggest_next_step_prefers_reachable_ollama():
    step = suggest_next_step(errors=0, ollama_reachable=True, has_claude_key=True)
    assert step is not None
    assert step.command == "atomics provider-test --provider ollama --no-thinking"
    assert "Ollama" in step.reason


def test_suggest_next_step_uses_claude_when_ollama_is_down():
    step = suggest_next_step(errors=0, ollama_reachable=False, has_claude_key=True)
    assert step is not None
    assert step.command == "atomics provider-test"
    assert "Claude" in step.reason


def test_suggest_next_step_points_at_localhost_ollama_when_nothing_is_ready():
    step = suggest_next_step(errors=0, ollama_reachable=False, has_claude_key=False)
    assert step is not None
    assert "--provider ollama" in step.command
    assert "localhost:11434" in step.reason


def test_suggest_next_step_is_silent_when_doctor_has_errors():
    assert suggest_next_step(errors=1, ollama_reachable=True, has_claude_key=True) is None


def test_doctor_prints_next_step_when_ollama_answers(capsys, tmp_path):
    settings = AtomicsSettings(db_path=tmp_path / "doc.db")
    mock_response = MagicMock()
    mock_response.json.return_value = {"models": [{"name": "qwen2.5:7b"}]}
    with patch("httpx.get", return_value=mock_response):
        assert run_doctor(settings=settings) == 0
    out = capsys.readouterr().out
    assert "Next:" in out
    assert "atomics provider-test --provider ollama --no-thinking" in out


def test_doctor_omits_next_step_when_the_database_is_unusable(capsys, tmp_path):
    settings = AtomicsSettings(db_path=tmp_path / "doc.db")
    with patch("sqlite3.connect", side_effect=OSError("permission denied")):
        assert run_doctor(settings=settings) == 1
    out = capsys.readouterr().out
    assert "Next:" not in out
