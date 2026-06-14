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


# ── OAuthPKCEAuth — flow coverage ───────────────────────────────────────────


_DEVICE_PROFILE = OPENAI_PROFILE  # has device_authorization_endpoint


@pytest.mark.asyncio
async def test_oauth_validate_exception_returns_false(tmp_path: Path):
    """validate() returns False when _get_tokens raises."""
    from unittest.mock import patch

    store = TokenStore(tmp_path / "auth.json")
    auth = OAuthPKCEAuth(profile=OPENAI_PROFILE, store=store)
    with patch.object(auth, "_get_tokens", side_effect=RuntimeError("store broken")):
        result = await auth.validate()
    assert result is False


@pytest.mark.asyncio
async def test_oauth_get_headers_triggers_refresh(tmp_path: Path):
    """get_headers() calls _refresh when token needs_refresh."""
    import asyncio as aio
    from unittest.mock import AsyncMock, patch

    store = TokenStore(tmp_path / "auth.json")
    # Save a token that needs_refresh (expires soon but not yet expired)
    near_expiry = CachedTokens(
        access_token="old-token",
        refresh_token="rt-abc",
        expires_at=time.time() + 30,  # needs_refresh threshold is typically 60s
        profile_name="openai",
    )
    store.save(near_expiry)
    auth = OAuthPKCEAuth(profile=OPENAI_PROFILE, store=store)

    refreshed = CachedTokens(
        access_token="new-token",
        refresh_token="rt-new",
        expires_at=time.time() + 3600,
        profile_name="openai",
    )
    with patch.object(auth, "_refresh", new=AsyncMock(return_value=refreshed)):
        headers = await auth.get_headers()
    assert headers["Authorization"] == "Bearer new-token"


@pytest.mark.asyncio
async def test_oauth_parse_token_response():
    """_parse_token_response correctly maps body fields."""
    store = TokenStore.__new__(TokenStore)
    store._path = Path("/dev/null")
    auth = OAuthPKCEAuth(profile=OPENAI_PROFILE, store=store)
    body = {
        "access_token": "at-xyz",
        "refresh_token": "rt-xyz",
        "id_token": "id-xyz",
        "expires_in": 7200,
    }
    tokens = auth._parse_token_response(body)
    assert tokens.access_token == "at-xyz"
    assert tokens.refresh_token == "rt-xyz"
    assert tokens.id_token == "id-xyz"
    assert tokens.expires_at > time.time() + 7100


@pytest.mark.asyncio
async def test_oauth_parse_token_response_default_expiry():
    """_parse_token_response uses 3600s when expires_in is absent."""
    store = TokenStore.__new__(TokenStore)
    store._path = Path("/dev/null")
    auth = OAuthPKCEAuth(profile=OPENAI_PROFILE, store=store)
    body = {"access_token": "at", "refresh_token": ""}
    tokens = auth._parse_token_response(body)
    assert tokens.expires_at > time.time() + 3590


@pytest.mark.asyncio
async def test_oauth_exchange_code(tmp_path: Path):
    """_exchange_code posts to token_endpoint and parses the response."""
    from unittest.mock import AsyncMock, MagicMock, patch

    mock_resp = MagicMock()
    mock_resp.raise_for_status = MagicMock()
    mock_resp.json = MagicMock(return_value={
        "access_token": "at-from-exchange",
        "refresh_token": "rt-exchange",
        "expires_in": 3600,
    })

    mock_client = AsyncMock()
    mock_client.post = AsyncMock(return_value=mock_resp)
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)

    store = TokenStore(tmp_path / "auth.json")
    auth = OAuthPKCEAuth(profile=OPENAI_PROFILE, store=store)

    with patch("httpx.AsyncClient", return_value=mock_client):
        tokens = await auth._exchange_code(
            code="auth-code", verifier="verifier-123",
            redirect_uri="http://localhost:19274/callback",
        )
    assert tokens.access_token == "at-from-exchange"
    assert tokens.refresh_token == "rt-exchange"


@pytest.mark.asyncio
async def test_oauth_refresh(tmp_path: Path):
    """_refresh posts with refresh_token and returns new tokens."""
    from unittest.mock import AsyncMock, MagicMock, patch

    mock_resp = MagicMock()
    mock_resp.raise_for_status = MagicMock()
    mock_resp.json = MagicMock(return_value={
        "access_token": "refreshed-at",
        "refresh_token": "refreshed-rt",
        "expires_in": 3600,
    })

    mock_client = AsyncMock()
    mock_client.post = AsyncMock(return_value=mock_resp)
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)

    store = TokenStore(tmp_path / "auth.json")
    auth = OAuthPKCEAuth(profile=OPENAI_PROFILE, store=store)
    old_tokens = CachedTokens(
        access_token="old", refresh_token="old-rt", expires_at=time.time() + 30
    )

    with patch("httpx.AsyncClient", return_value=mock_client):
        new_tokens = await auth._refresh(old_tokens)
    assert new_tokens.access_token == "refreshed-at"
    assert new_tokens.refresh_token == "refreshed-rt"
    mock_client.post.assert_awaited_once()


@pytest.mark.asyncio
async def test_oauth_refresh_preserves_old_refresh_token(tmp_path: Path):
    """_refresh reuses old refresh_token when new response omits it."""
    from unittest.mock import AsyncMock, MagicMock, patch

    mock_resp = MagicMock()
    mock_resp.raise_for_status = MagicMock()
    mock_resp.json = MagicMock(return_value={
        "access_token": "fresh-at",
        "expires_in": 3600,
        # no refresh_token in response
    })

    mock_client = AsyncMock()
    mock_client.post = AsyncMock(return_value=mock_resp)
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)

    store = TokenStore(tmp_path / "auth.json")
    auth = OAuthPKCEAuth(profile=OPENAI_PROFILE, store=store)
    old_tokens = CachedTokens(
        access_token="old", refresh_token="keep-me", expires_at=time.time() + 30
    )

    with patch("httpx.AsyncClient", return_value=mock_client):
        new_tokens = await auth._refresh(old_tokens)
    assert new_tokens.refresh_token == "keep-me"


@pytest.mark.asyncio
async def test_oauth_device_code_flow_success(tmp_path: Path):
    """_device_code_flow polls until 200 and returns tokens."""
    from unittest.mock import AsyncMock, MagicMock, patch

    device_resp = MagicMock()
    device_resp.raise_for_status = MagicMock()
    device_resp.json = MagicMock(return_value={
        "device_code": "dc-123",
        "user_code": "ABCD-1234",
        "verification_uri_complete": "https://example.com/activate?code=ABCD-1234",
        "interval": 0,
    })

    # First poll: authorization_pending; second poll: success
    pending_resp = MagicMock()
    pending_resp.status_code = 400
    pending_resp.raise_for_status = MagicMock()
    pending_resp.json = MagicMock(return_value={"error": "authorization_pending"})

    token_resp = MagicMock()
    token_resp.status_code = 200
    token_resp.raise_for_status = MagicMock()
    token_resp.json = MagicMock(return_value={
        "access_token": "at-device",
        "refresh_token": "rt-device",
        "expires_in": 3600,
    })

    call_count = 0

    async def fake_post(url, **_kw):
        nonlocal call_count
        if "device/code" in url:
            return device_resp
        call_count += 1
        return pending_resp if call_count == 1 else token_resp

    mock_client = AsyncMock()
    mock_client.post = AsyncMock(side_effect=fake_post)
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)

    store = TokenStore(tmp_path / "auth.json")
    auth = OAuthPKCEAuth(profile=_DEVICE_PROFILE, store=store)

    with patch("httpx.AsyncClient", return_value=mock_client), \
         patch("asyncio.sleep", new=AsyncMock()):
        tokens = await auth._device_code_flow()

    assert tokens.access_token == "at-device"


@pytest.mark.asyncio
async def test_oauth_device_code_flow_slow_down(tmp_path: Path):
    """_device_code_flow increments poll_interval on slow_down and eventually succeeds."""
    from unittest.mock import AsyncMock, MagicMock, patch

    device_resp = MagicMock()
    device_resp.raise_for_status = MagicMock()
    device_resp.json = MagicMock(return_value={
        "device_code": "dc-slow",
        "user_code": "",
        "verification_uri": "https://example.com/activate",
        "interval": 1,
    })

    slow_resp = MagicMock()
    slow_resp.status_code = 400
    slow_resp.raise_for_status = MagicMock()
    slow_resp.json = MagicMock(return_value={"error": "slow_down"})

    token_resp = MagicMock()
    token_resp.status_code = 200
    token_resp.raise_for_status = MagicMock()
    token_resp.json = MagicMock(return_value={
        "access_token": "at-slow", "refresh_token": "", "expires_in": 3600,
    })

    call_count = 0

    async def fake_post(url, **_kw):
        nonlocal call_count
        if "device/code" in url:
            return device_resp
        call_count += 1
        return slow_resp if call_count == 1 else token_resp

    mock_client = AsyncMock()
    mock_client.post = AsyncMock(side_effect=fake_post)
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)

    store = TokenStore(tmp_path / "auth.json")
    auth = OAuthPKCEAuth(profile=_DEVICE_PROFILE, store=store)

    with patch("httpx.AsyncClient", return_value=mock_client), \
         patch("asyncio.sleep", new=AsyncMock()):
        tokens = await auth._device_code_flow()

    assert tokens.access_token == "at-slow"


@pytest.mark.asyncio
async def test_oauth_device_code_flow_unknown_error(tmp_path: Path):
    """_device_code_flow calls raise_for_status on unrecognised error codes."""
    import httpx
    from unittest.mock import AsyncMock, MagicMock, patch

    device_resp = MagicMock()
    device_resp.raise_for_status = MagicMock()
    device_resp.json = MagicMock(return_value={
        "device_code": "dc-err", "user_code": "", "verification_uri": "https://x.com", "interval": 0,
    })

    error_resp = MagicMock()
    error_resp.status_code = 400
    error_resp.json = MagicMock(return_value={"error": "expired_token"})
    error_resp.raise_for_status = MagicMock(
        side_effect=httpx.HTTPStatusError("400", request=MagicMock(), response=error_resp)
    )

    async def fake_post(url, **_kw):
        if "device/code" in url:
            return device_resp
        return error_resp

    mock_client = AsyncMock()
    mock_client.post = AsyncMock(side_effect=fake_post)
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)

    store = TokenStore(tmp_path / "auth.json")
    auth = OAuthPKCEAuth(profile=_DEVICE_PROFILE, store=store)

    with patch("httpx.AsyncClient", return_value=mock_client), \
         patch("asyncio.sleep", new=AsyncMock()):
        with pytest.raises(httpx.HTTPStatusError):
            await auth._device_code_flow()


@pytest.mark.asyncio
async def test_oauth_device_code_flow_no_endpoint(tmp_path: Path):
    """_device_code_flow raises when profile has no device_authorization_endpoint."""
    from atomics.auth.profiles import OIDCProfile

    no_device_profile = OIDCProfile(
        name="no-device", issuer="https://example.com",
        token_endpoint="https://example.com/token",
        client_id="client-x",
    )
    store = TokenStore(tmp_path / "auth.json")
    auth = OAuthPKCEAuth(profile=no_device_profile, store=store)

    with pytest.raises(RuntimeError, match="does not support device code flow"):
        await auth._device_code_flow()


@pytest.mark.asyncio
async def test_oauth_login_headless_delegates(tmp_path: Path):
    """login(headless=True) calls _device_code_flow and saves tokens."""
    from unittest.mock import AsyncMock, patch

    store = TokenStore(tmp_path / "auth.json")
    auth = OAuthPKCEAuth(profile=_DEVICE_PROFILE, store=store)
    expected = CachedTokens(
        access_token="hd-token", refresh_token="hd-rt",
        expires_at=time.time() + 3600, profile_name="openai",
    )
    with patch.object(auth, "_device_code_flow", new=AsyncMock(return_value=expected)):
        tokens = await auth.login(headless=True)
    assert tokens.access_token == "hd-token"
    assert store.has_valid_tokens()


@pytest.mark.asyncio
async def test_oauth_login_browser_delegates(tmp_path: Path):
    """login(headless=False) calls _browser_flow and saves tokens."""
    from unittest.mock import AsyncMock, patch

    store = TokenStore(tmp_path / "auth.json")
    auth = OAuthPKCEAuth(profile=_DEVICE_PROFILE, store=store)
    expected = CachedTokens(
        access_token="br-token", refresh_token="br-rt",
        expires_at=time.time() + 3600, profile_name="openai",
    )
    with patch.object(auth, "_browser_flow", new=AsyncMock(return_value=expected)):
        tokens = await auth.login(headless=False)
    assert tokens.access_token == "br-token"
    assert store.has_valid_tokens()


@pytest.mark.asyncio
async def test_oauth_browser_flow(tmp_path: Path):
    """_browser_flow opens browser, awaits callback code, and exchanges it."""
    import asyncio as aio
    from unittest.mock import AsyncMock, MagicMock, patch

    store = TokenStore(tmp_path / "auth.json")
    auth = OAuthPKCEAuth(profile=OPENAI_PROFILE, store=store)
    expected_tokens = CachedTokens(
        access_token="browser-at", refresh_token="browser-rt",
        expires_at=time.time() + 3600, profile_name="openai",
    )

    def fake_start_server(port, state, code_future, loop):
        loop.call_soon(code_future.set_result, "auth-code-from-browser")
        mock_srv = MagicMock()
        mock_srv.shutdown = MagicMock()
        return mock_srv

    with patch("webbrowser.open") as mock_browser, \
         patch("atomics.auth.oauth._start_callback_server", side_effect=fake_start_server), \
         patch.object(auth, "_exchange_code", new=AsyncMock(return_value=expected_tokens)):
        tokens = await auth._browser_flow()

    assert tokens.access_token == "browser-at"
    mock_browser.assert_called_once()


# ── _start_callback_server Handler ──────────────────────────────────────────


def _run_until_future_done(loop, future, timeout=3.0):
    """Tick the loop until the handler thread's call_soon_threadsafe callback
    lands and resolves the future. A single sleep(0) raced the cross-thread
    scheduling (the HTTP response can return before the callback runs), which
    flaked on Linux/3.12 with InvalidStateError. Polling until done is
    deterministic across platforms.
    """
    import asyncio as aio
    import time

    deadline = time.monotonic() + timeout
    while not future.done() and time.monotonic() < deadline:
        loop.run_until_complete(aio.sleep(0.01))


def test_callback_handler_success():
    """Handler.do_GET with valid state and code sets the future result."""
    import asyncio as aio
    import threading
    from io import BytesIO
    from http.server import HTTPServer
    from atomics.auth.oauth import _start_callback_server

    loop = aio.new_event_loop()
    future: aio.Future[str] = loop.create_future()

    server = _start_callback_server(
        port=0,  # OS picks a free port
        expected_state="st-abc",
        code_future=future,
        loop=loop,
    )
    port = server.server_address[1]

    import urllib.request
    url = f"http://127.0.0.1:{port}/callback?state=st-abc&code=code-xyz"
    resp = urllib.request.urlopen(url, timeout=3)
    assert resp.status == 200

    _run_until_future_done(loop, future)
    assert future.result() == "code-xyz"
    server.shutdown()
    loop.close()


def test_callback_handler_state_mismatch():
    """Handler.do_GET with wrong state sets RuntimeError on the future."""
    import asyncio as aio
    from atomics.auth.oauth import _start_callback_server
    import urllib.request, urllib.error

    loop = aio.new_event_loop()
    future: aio.Future[str] = loop.create_future()

    server = _start_callback_server(
        port=0, expected_state="expected-state",
        code_future=future, loop=loop,
    )
    port = server.server_address[1]

    try:
        urllib.request.urlopen(
            f"http://127.0.0.1:{port}/callback?state=WRONG&code=c",
            timeout=3,
        )
    except urllib.error.HTTPError as e:
        assert e.code == 400

    _run_until_future_done(loop, future)
    with pytest.raises(RuntimeError, match="state mismatch"):
        future.result()
    server.shutdown()
    loop.close()


def test_callback_handler_oauth_error():
    """Handler.do_GET with error param sets RuntimeError on the future."""
    import asyncio as aio
    from atomics.auth.oauth import _start_callback_server
    import urllib.request, urllib.error

    loop = aio.new_event_loop()
    future: aio.Future[str] = loop.create_future()

    server = _start_callback_server(
        port=0, expected_state="st",
        code_future=future, loop=loop,
    )
    port = server.server_address[1]

    try:
        urllib.request.urlopen(
            f"http://127.0.0.1:{port}/callback?error=access_denied",
            timeout=3,
        )
    except urllib.error.HTTPError as e:
        assert e.code == 400

    _run_until_future_done(loop, future)
    with pytest.raises(RuntimeError, match="OAuth error"):
        future.result()
    server.shutdown()
    loop.close()
