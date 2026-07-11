"""Tests for atomics.secrets — OS keychain secrets management."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

# ── Module tests ──────────────────────────────────────────────────────────────


def test_set_secret_calls_keyring():
    with patch("keyring.set_password") as mock_set:
        from atomics.secrets import set_secret

        set_secret("ANTHROPIC_API_KEY", "sk-ant-test123")
        mock_set.assert_called_once_with("atomics", "ANTHROPIC_API_KEY", "sk-ant-test123")


def test_get_secret_returns_value():
    with patch("keyring.get_password", return_value="sk-ant-test123"):
        from atomics.secrets import get_secret

        assert get_secret("ANTHROPIC_API_KEY") == "sk-ant-test123"


def test_get_secret_returns_none_on_missing():
    with patch("keyring.get_password", return_value=None):
        from atomics.secrets import get_secret

        assert get_secret("NONEXISTENT_KEY") is None


def test_get_secret_returns_none_on_error():
    with patch("keyring.get_password", side_effect=Exception("backend unavailable")):
        from atomics.secrets import get_secret

        assert get_secret("ANTHROPIC_API_KEY") is None


def test_delete_secret_calls_keyring():
    with patch("keyring.delete_password") as mock_del:
        from atomics.secrets import delete_secret

        assert delete_secret("ANTHROPIC_API_KEY") is True
        mock_del.assert_called_once_with("atomics", "ANTHROPIC_API_KEY")


def test_delete_secret_returns_false_on_error():
    with patch("keyring.delete_password", side_effect=Exception("not found")):
        from atomics.secrets import delete_secret

        assert delete_secret("NONEXISTENT") is False


def test_list_secrets_returns_stored_keys():
    def mock_get(service, key):
        return "value" if key == "ANTHROPIC_API_KEY" else None

    with patch("keyring.get_password", side_effect=mock_get):
        from atomics.secrets import list_secrets

        stored = list_secrets()
        assert "ANTHROPIC_API_KEY" in stored
        assert "OPENAI_API_KEY" not in stored


def test_list_secrets_never_exposes_values():
    with patch("keyring.get_password", return_value="super-secret-value"):
        from atomics.secrets import list_secrets

        stored = list_secrets()
        for key in stored:
            assert "super-secret" not in key
            assert key == key.upper()


def test_keychain_available_true():
    mock_keyring = MagicMock()
    mock_keyring.name = "macOS Keyring"
    with patch("keyring.get_keyring", return_value=mock_keyring):
        from atomics.secrets import keychain_available

        assert keychain_available() is True


def test_keychain_available_false_on_fail_backend():
    mock_keyring = MagicMock()
    mock_keyring.name = "keyring.backends.fail.Keyring"
    with patch("keyring.get_keyring", return_value=mock_keyring):
        from atomics.secrets import keychain_available

        assert keychain_available() is False


# ── Resolution order tests ────────────────────────────────────────────────────


def test_env_var_wins_over_keychain(monkeypatch):
    """Env var takes priority — keychain is not consulted when env is set."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "from-env")
    with patch("keyring.get_password", return_value="from-keychain"):
        from atomics.config import load_settings

        settings = load_settings()
        assert settings.anthropic_api_key == "from-env"


def test_keychain_fills_empty_key(monkeypatch):
    """Keychain provides value when env var is empty."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with patch("keyring.get_password", return_value="from-keychain"):
        from atomics.config import load_settings

        settings = load_settings()
        assert settings.anthropic_api_key == "from-keychain"


def test_empty_when_neither_env_nor_keychain(monkeypatch):
    """Empty string when neither env nor keychain has the key."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with patch("keyring.get_password", return_value=None):
        from atomics.config import load_settings

        settings = load_settings()
        assert settings.anthropic_api_key == ""


# ── CLI tests ─────────────────────────────────────────────────────────────────


def test_secrets_list_cli():
    from click.testing import CliRunner

    from atomics.cli import cli

    with (
        patch("atomics.secrets.keychain_available", return_value=True),
        patch("keyring.get_password", return_value=None),
    ):
        runner = CliRunner()
        result = runner.invoke(cli, ["secrets", "list"])
        assert result.exit_code == 0
        assert "No secrets stored" in result.output


def test_secrets_get_not_found():
    from click.testing import CliRunner

    from atomics.cli import cli

    with patch("keyring.get_password", return_value=None):
        runner = CliRunner()
        result = runner.invoke(cli, ["secrets", "get", "NONEXISTENT"])
        assert result.exit_code == 1
        assert "Not found" in result.output


def test_secrets_get_masks_by_default():
    """Without --show, the value must not be printed (secure by default)."""
    from click.testing import CliRunner

    from atomics.cli import cli

    secret = "sk-ant-supersecretvalue12345"
    with patch("keyring.get_password", return_value=secret):
        runner = CliRunner()
        result = runner.invoke(cli, ["secrets", "get", "ANTHROPIC_API_KEY"])
        assert result.exit_code == 0
        assert secret not in result.output
        assert "set" in result.output
        assert "--show" in result.output


def test_secrets_get_show_reveals_value():
    """With --show, the raw value is printed for piping."""
    from click.testing import CliRunner

    from atomics.cli import cli

    secret = "sk-ant-supersecretvalue12345"
    with patch("keyring.get_password", return_value=secret):
        runner = CliRunner()
        result = runner.invoke(cli, ["secrets", "get", "ANTHROPIC_API_KEY", "--show"])
        assert result.exit_code == 0
        assert secret in result.output


def test_secrets_get_mask_never_reveals_short_secret():
    """Short values are fully masked (no tail preview)."""
    from click.testing import CliRunner

    from atomics.cli import cli

    with patch("keyring.get_password", return_value="short"):
        runner = CliRunner()
        result = runner.invoke(cli, ["secrets", "get", "ANTHROPIC_API_KEY"])
        assert result.exit_code == 0
        assert "short" not in result.output
        assert "****" in result.output


def test_secrets_delete_success():
    from click.testing import CliRunner

    from atomics.cli import cli

    with patch("keyring.delete_password"):
        runner = CliRunner()
        result = runner.invoke(cli, ["secrets", "delete", "ANTHROPIC_API_KEY"])
        assert result.exit_code == 0
        assert "Deleted" in result.output
