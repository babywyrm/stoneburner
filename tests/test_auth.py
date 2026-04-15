"""Tests for the auth module — strategies, store, profiles, auto-detect."""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from atomics.auth import auto_detect_auth
from atomics.auth.apikey import ApiKeyAuth
from atomics.auth.codex import CodexTokenAuth
from atomics.auth.oauth import OAuthPKCEAuth, _generate_pkce
from atomics.auth.profiles import OPENAI_PROFILE, get_profile, list_profiles
from atomics.auth.store import CachedTokens, TokenStore


# ── ApiKeyAuth ──────────────────────────────────────────


@pytest.mark.asyncio
async def test_apikey_auth_headers():
    auth = ApiKeyAuth("sk-test-key")
    headers = await auth.get_headers()
    assert headers["Authorization"] == "Bearer sk-test-key"


@pytest.mark.asyncio
async def test_apikey_auth_validate():
    assert await ApiKeyAuth("sk-test").validate() is True
    assert await ApiKeyAuth("").validate() is False


def test_apikey_description():
    auth = ApiKeyAuth("sk-test-1234567890")
    assert "sk-test-" in auth.description
    assert "..." in auth.description


# ── TokenStore ──────────────────────────────────────────


def test_store_save_and_load(tmp_path: Path):
    store = TokenStore(tmp_path / "auth.json")
    tokens = CachedTokens(
        access_token="acc_123",
        refresh_token="ref_456",
        expires_at=time.time() + 3600,
        profile_name="openai",
    )
    store.save(tokens)
    loaded = store.load()
    assert loaded.access_token == "acc_123"
    assert loaded.refresh_token == "ref_456"
    assert loaded.profile_name == "openai"
    assert not loaded.expired


def test_store_load_nonexistent(tmp_path: Path):
    store = TokenStore(tmp_path / "nope.json")
    tokens = store.load()
    assert tokens.access_token == ""
    assert tokens.expired


def test_store_clear(tmp_path: Path):
    store = TokenStore(tmp_path / "auth.json")
    store.save(CachedTokens(access_token="x", expires_at=time.time() + 3600))
    assert store.has_valid_tokens()
    store.clear()
    assert not store.has_valid_tokens()


def test_store_has_valid_tokens_expired(tmp_path: Path):
    store = TokenStore(tmp_path / "auth.json")
    store.save(CachedTokens(access_token="x", expires_at=time.time() - 100))
    assert not store.has_valid_tokens()


def test_store_file_permissions(tmp_path: Path):
    store = TokenStore(tmp_path / "auth.json")
    store.save(CachedTokens(access_token="secret", expires_at=time.time() + 3600))
    stat = store.path.stat()
    assert oct(stat.st_mode & 0o777) == "0o600"


def test_cached_tokens_needs_refresh():
    fresh = CachedTokens(access_token="x", expires_at=time.time() + 3600)
    assert not fresh.needs_refresh

    stale = CachedTokens(access_token="x", expires_at=time.time() + 60)
    assert stale.needs_refresh

    expired = CachedTokens(access_token="x", expires_at=time.time() - 10)
    assert expired.needs_refresh


def test_store_corrupt_json(tmp_path: Path):
    path = tmp_path / "auth.json"
    path.write_text("{bad json")
    store = TokenStore(path)
    tokens = store.load()
    assert tokens.access_token == ""


# ── Profiles ────────────────────────────────────────────


def test_openai_profile_endpoints():
    p = OPENAI_PROFILE
    assert p.name == "openai"
    assert "auth.openai.com" in p.authorization_endpoint
    assert "auth0.openai.com" in p.token_endpoint
    assert p.client_id.startswith("app_")
    assert "offline_access" in p.scopes


def test_get_profile_known():
    p = get_profile("openai")
    assert p.name == "openai"


def test_get_profile_unknown():
    with pytest.raises(ValueError, match="Unknown OIDC profile"):
        get_profile("nonexistent")


def test_list_profiles():
    profiles = list_profiles()
    assert "openai" in profiles


# ── PKCE ────────────────────────────────────────────────


def test_pkce_generation():
    verifier, challenge = _generate_pkce()
    assert len(verifier) > 20
    assert len(challenge) > 20
    assert verifier != challenge


def test_pkce_verifier_challenge_relationship():
    import base64
    import hashlib

    verifier, challenge = _generate_pkce()
    digest = hashlib.sha256(verifier.encode()).digest()
    expected = base64.urlsafe_b64encode(digest).rstrip(b"=").decode()
    assert challenge == expected


# ── CodexTokenAuth ──────────────────────────────────────


def test_codex_tokens_available_no_file(tmp_path: Path):
    auth = CodexTokenAuth(path=tmp_path / "nonexistent.json")
    assert auth.tokens_available() is False


def test_codex_tokens_available_with_api_key(tmp_path: Path):
    auth_file = tmp_path / "auth.json"
    auth_file.write_text(json.dumps({
        "OPENAI_API_KEY": "sk-real-key-123",
        "tokens": {"access_token": "eyJ...", "refresh_token": "rt_..."},
    }))
    auth = CodexTokenAuth(path=auth_file)
    assert auth.tokens_available() is True


def test_codex_tokens_available_null_api_key(tmp_path: Path):
    """ChatGPT OAuth tokens without exchanged API key are NOT usable."""
    auth_file = tmp_path / "auth.json"
    auth_file.write_text(json.dumps({
        "OPENAI_API_KEY": None,
        "tokens": {"access_token": "eyJ...", "refresh_token": "rt_..."},
    }))
    auth = CodexTokenAuth(path=auth_file)
    assert auth.tokens_available() is False


def test_codex_installed_but_no_key(tmp_path: Path):
    auth_file = tmp_path / "auth.json"
    auth_file.write_text(json.dumps({"OPENAI_API_KEY": None, "tokens": {}}))
    auth = CodexTokenAuth(path=auth_file)
    assert auth.tokens_available() is False
    assert auth.codex_installed() is True


@pytest.mark.asyncio
async def test_codex_get_headers_with_api_key(tmp_path: Path):
    auth_file = tmp_path / "auth.json"
    auth_file.write_text(json.dumps({
        "OPENAI_API_KEY": "sk-real-key",
        "tokens": {"access_token": "eyJ...", "refresh_token": "rt_..."},
    }))
    auth = CodexTokenAuth(path=auth_file)
    headers = await auth.get_headers()
    assert headers["Authorization"] == "Bearer sk-real-key"


@pytest.mark.asyncio
async def test_codex_get_headers_raises_without_api_key(tmp_path: Path):
    auth_file = tmp_path / "auth.json"
    auth_file.write_text(json.dumps({
        "OPENAI_API_KEY": None,
        "tokens": {"access_token": "eyJ...", "refresh_token": "rt_..."},
    }))
    auth = CodexTokenAuth(path=auth_file)
    with pytest.raises(RuntimeError, match="ChatGPT OAuth tokens lack"):
        await auth.get_headers()


@pytest.mark.asyncio
async def test_codex_validate_true(tmp_path: Path):
    auth_file = tmp_path / "auth.json"
    auth_file.write_text(json.dumps({"OPENAI_API_KEY": "sk-test"}))
    auth = CodexTokenAuth(path=auth_file)
    assert await auth.validate() is True


@pytest.mark.asyncio
async def test_codex_validate_false_no_file(tmp_path: Path):
    auth = CodexTokenAuth(path=tmp_path / "nope.json")
    assert await auth.validate() is False


@pytest.mark.asyncio
async def test_codex_validate_false_null_key(tmp_path: Path):
    auth_file = tmp_path / "auth.json"
    auth_file.write_text(json.dumps({"OPENAI_API_KEY": None}))
    auth = CodexTokenAuth(path=auth_file)
    assert await auth.validate() is False


def test_codex_description():
    auth = CodexTokenAuth()
    assert "Codex CLI" in auth.description


# ── OAuthPKCEAuth ───────────────────────────────────────


def test_oauth_description():
    auth = OAuthPKCEAuth(profile=OPENAI_PROFILE)
    assert "OAuth" in auth.description
    assert "openai" in auth.description


@pytest.mark.asyncio
async def test_oauth_validate_no_tokens(tmp_path: Path):
    store = TokenStore(tmp_path / "empty.json")
    auth = OAuthPKCEAuth(profile=OPENAI_PROFILE, store=store)
    assert await auth.validate() is False


@pytest.mark.asyncio
async def test_oauth_validate_with_cached_tokens(tmp_path: Path):
    store = TokenStore(tmp_path / "auth.json")
    store.save(CachedTokens(
        access_token="valid",
        expires_at=time.time() + 3600,
        profile_name="openai",
    ))
    auth = OAuthPKCEAuth(profile=OPENAI_PROFILE, store=store)
    assert await auth.validate() is True


@pytest.mark.asyncio
async def test_oauth_get_headers_from_cache(tmp_path: Path):
    store = TokenStore(tmp_path / "auth.json")
    store.save(CachedTokens(
        access_token="cached-token",
        expires_at=time.time() + 3600,
        profile_name="openai",
    ))
    auth = OAuthPKCEAuth(profile=OPENAI_PROFILE, store=store)
    headers = await auth.get_headers()
    assert headers["Authorization"] == "Bearer cached-token"


def test_oauth_logout_clears_store(tmp_path: Path):
    store = TokenStore(tmp_path / "auth.json")
    store.save(CachedTokens(access_token="x", expires_at=time.time() + 3600))
    auth = OAuthPKCEAuth(profile=OPENAI_PROFILE, store=store)
    auth.logout()
    assert not store.has_valid_tokens()


# ── auto_detect_auth ────────────────────────────────────


def test_auto_detect_apikey():
    auth = auto_detect_auth(api_key="sk-hello")
    assert isinstance(auth, ApiKeyAuth)


def test_auto_detect_codex_api_key(tmp_path: Path, monkeypatch):
    auth_file = tmp_path / "auth.json"
    auth_file.write_text(json.dumps({
        "OPENAI_API_KEY": "sk-from-codex",
        "tokens": {"access_token": "eyJ...", "refresh_token": "rt"},
    }))
    monkeypatch.setattr(
        "atomics.auth.codex._default_codex_auth_path", lambda: auth_file
    )
    auth = auto_detect_auth()
    assert isinstance(auth, CodexTokenAuth)


def test_auto_detect_cached_oauth(tmp_path: Path, monkeypatch):
    store_path = tmp_path / "auth.json"
    store = TokenStore(store_path)
    store.save(CachedTokens(
        access_token="cached",
        expires_at=time.time() + 3600,
        profile_name="openai",
    ))
    monkeypatch.setattr(
        "atomics.auth.codex._default_codex_auth_path",
        lambda: tmp_path / "nonexistent.json",
    )
    monkeypatch.setattr(
        "atomics.auth.store._default_auth_dir", lambda: tmp_path
    )
    auth = auto_detect_auth()
    assert isinstance(auth, OAuthPKCEAuth)


def test_auto_detect_nothing_raises(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(
        "atomics.auth.codex._default_codex_auth_path",
        lambda: tmp_path / "nonexistent.json",
    )
    monkeypatch.setattr(
        "atomics.auth.store._default_auth_dir", lambda: tmp_path
    )
    with pytest.raises(RuntimeError, match="No OpenAI credentials"):
        auto_detect_auth()


def test_auto_detect_codex_installed_but_no_key_shows_hint(tmp_path: Path, monkeypatch):
    """When Codex is installed but has no API key, error message explains why."""
    auth_file = tmp_path / "codex_auth.json"
    auth_file.write_text(json.dumps({"OPENAI_API_KEY": None, "tokens": {}}))
    monkeypatch.setattr(
        "atomics.auth.codex._default_codex_auth_path", lambda: auth_file
    )
    monkeypatch.setattr(
        "atomics.auth.store._default_auth_dir", lambda: tmp_path
    )
    with pytest.raises(RuntimeError, match="Codex CLI is installed"):
        auto_detect_auth()


# ── CLI commands ────────────────────────────────────────


def test_cli_whoami_no_auth(monkeypatch, tmp_path):
    from click.testing import CliRunner

    from atomics.cli import cli

    monkeypatch.setenv("OPENAI_API_KEY", "")
    monkeypatch.setattr(
        "atomics.auth.codex._default_codex_auth_path",
        lambda: tmp_path / "nonexistent.json",
    )
    monkeypatch.setattr(
        "atomics.auth.store._default_auth_dir", lambda: tmp_path
    )
    runner = CliRunner()
    result = runner.invoke(cli, ["whoami"])
    assert result.exit_code == 0
    assert "Not authenticated" in result.output


def test_cli_whoami_apikey():
    from click.testing import CliRunner

    from atomics.cli import cli

    runner = CliRunner(env={"OPENAI_API_KEY": "sk-test-1234567890"})
    result = runner.invoke(cli, ["whoami"])
    assert result.exit_code == 0
    assert "API key" in result.output


def test_cli_logout(tmp_path, monkeypatch):
    from click.testing import CliRunner

    from atomics.cli import cli

    monkeypatch.setattr("atomics.auth.store._default_auth_dir", lambda: tmp_path)
    store = TokenStore(tmp_path / "auth.json")
    store.save(CachedTokens(access_token="x", expires_at=time.time() + 3600))

    runner = CliRunner()
    result = runner.invoke(cli, ["logout"])
    assert result.exit_code == 0
    assert "Logged out" in result.output


def test_cli_login_help():
    from click.testing import CliRunner

    from atomics.cli import cli

    runner = CliRunner()
    result = runner.invoke(cli, ["login", "--help"])
    assert result.exit_code == 0
    assert "--headless" in result.output
    assert "--issuer" in result.output
