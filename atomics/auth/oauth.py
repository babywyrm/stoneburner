"""OAuth 2.0 PKCE auth strategy with browser and device-code flows."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import logging
import secrets
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
from threading import Thread
from urllib.parse import parse_qs, urlencode, urlparse

import httpx

from atomics.auth import AuthStrategy
from atomics.auth.profiles import OIDCProfile
from atomics.auth.store import CachedTokens, TokenStore

logger = logging.getLogger("atomics.auth.oauth")

_LOOPBACK_OAUTH_HOSTS: frozenset[str] = frozenset({"localhost", "127.0.0.1", "::1"})


def _validate_oauth_uri(uri: str, *, label: str) -> str:
    """Validate an OAuth endpoint or verification URI before opening or printing it."""
    if not uri or any(ord(ch) <= 32 or ord(ch) == 127 for ch in uri):
        raise ValueError(
            f"{label}: URI must be non-empty and free of whitespace or control characters"
        )
    parsed = urlparse(uri)
    if parsed.username or parsed.password:
        raise ValueError(f"{label}: embedded credentials not allowed")
    host = (parsed.hostname or "").lower()
    is_loopback_http = parsed.scheme == "http" and host in _LOOPBACK_OAUTH_HOSTS
    if parsed.scheme != "https" and not is_loopback_http:
        raise ValueError(f"{label}: must use https (or loopback http), got {parsed.scheme!r}")
    if not host:
        raise ValueError(f"{label}: missing hostname")
    return uri


def _validate_terminal_safe(text: str, *, label: str) -> str:
    if any(ord(ch) < 32 or ord(ch) == 127 for ch in text):
        raise ValueError(f"{label}: control characters not allowed")
    return text


def _generate_pkce() -> tuple[str, str]:
    """Generate PKCE code_verifier and code_challenge (S256)."""
    verifier = base64.urlsafe_b64encode(secrets.token_bytes(32)).rstrip(b"=").decode()
    digest = hashlib.sha256(verifier.encode()).digest()
    challenge = base64.urlsafe_b64encode(digest).rstrip(b"=").decode()
    return verifier, challenge


class OAuthPKCEAuth(AuthStrategy):
    """Full PKCE browser flow with token caching and refresh."""

    def __init__(
        self,
        profile: OIDCProfile,
        store: TokenStore | None = None,
    ) -> None:
        self._profile = profile
        self._store = store or TokenStore()
        self._tokens: CachedTokens | None = None

    async def get_headers(self) -> dict[str, str]:
        tokens = self._get_tokens()
        if tokens.needs_refresh and tokens.refresh_token:
            tokens = await self._refresh(tokens)
        return {"Authorization": f"Bearer {tokens.access_token}"}

    async def validate(self) -> bool:
        try:
            tokens = self._get_tokens()
            return bool(tokens.access_token) and not tokens.expired
        except Exception:
            return False

    @property
    def description(self) -> str:
        return f"OAuth ({self._profile.name})"

    def _get_tokens(self) -> CachedTokens:
        if self._tokens is None:
            self._tokens = self._store.load()
        return self._tokens

    async def login(self, *, headless: bool = False) -> CachedTokens:
        """Run the interactive login flow. Returns cached tokens."""
        if headless:
            tokens = await self._device_code_flow()
        else:
            tokens = await self._browser_flow()
        self._tokens = tokens
        self._store.save(tokens)
        return tokens

    def logout(self) -> None:
        self._store.clear()
        self._tokens = None

    async def _browser_flow(self) -> CachedTokens:
        """PKCE authorization code flow with local redirect."""
        auth_endpoint = _validate_oauth_uri(
            self._profile.authorization_endpoint, label="authorization_endpoint"
        )
        _validate_oauth_uri(self._profile.token_endpoint, label="token_endpoint")
        verifier, challenge = _generate_pkce()
        state = secrets.token_urlsafe(16)

        redirect_uri = f"http://localhost:{self._profile.callback_port}/callback"
        params = {
            "response_type": "code",
            "client_id": self._profile.client_id,
            "redirect_uri": redirect_uri,
            "scope": " ".join(self._profile.scopes),
            "state": state,
            "code_challenge": challenge,
            "code_challenge_method": "S256",
        }
        auth_url = f"{auth_endpoint}?{urlencode(params)}"

        code_future: asyncio.Future[str] = asyncio.get_running_loop().create_future()
        server = _start_callback_server(
            self._profile.callback_port, state, code_future, asyncio.get_running_loop()
        )

        try:
            logger.info("Opening browser for login...")
            webbrowser.open(auth_url)
            code = await asyncio.wait_for(code_future, timeout=300)
        finally:
            server.shutdown()
            server.server_close()

        return await self._exchange_code(code, verifier, redirect_uri)

    async def _device_code_flow(self) -> CachedTokens:
        """Device authorization grant for headless environments."""
        if not self._profile.device_authorization_endpoint:
            raise RuntimeError(f"Profile {self._profile.name!r} does not support device code flow")
        device_endpoint = _validate_oauth_uri(
            self._profile.device_authorization_endpoint, label="device_authorization_endpoint"
        )
        token_endpoint = _validate_oauth_uri(self._profile.token_endpoint, label="token_endpoint")

        async with httpx.AsyncClient() as client:
            resp = await client.post(
                device_endpoint,
                data={
                    "client_id": self._profile.client_id,
                    "scope": " ".join(self._profile.scopes),
                },
            )
            resp.raise_for_status()
            body = resp.json()

        raw_uri = body.get("verification_uri_complete") or body.get("verification_uri", "")
        verification_uri = _validate_oauth_uri(str(raw_uri), label="verification_uri")
        user_code = _validate_terminal_safe(str(body.get("user_code", "")), label="user_code")
        device_code = str(body["device_code"])
        raw_interval = body.get("interval", 5)
        poll_interval = (
            max(0.0, float(raw_interval))
            if isinstance(raw_interval, int | float) and not isinstance(raw_interval, bool)
            else 5.0
        )
        raw_expires = body.get("expires_in", 600)
        expires_in = (
            min(3600.0, float(raw_expires))
            if isinstance(raw_expires, int | float)
            and not isinstance(raw_expires, bool)
            and raw_expires > 0
            else 600.0
        )
        deadline = time.monotonic() + expires_in

        print(f"\nVisit: {verification_uri}")
        if user_code:
            print(f"Code:  {user_code}")
        print("Waiting for authorization...\n")

        async with httpx.AsyncClient() as client:
            while time.monotonic() < deadline:
                await asyncio.sleep(poll_interval)
                if time.monotonic() >= deadline:
                    break
                resp = await client.post(
                    token_endpoint,
                    data={
                        "grant_type": "urn:ietf:params:oauth:grant-type:device_code",
                        "device_code": device_code,
                        "client_id": self._profile.client_id,
                    },
                )
                if resp.status_code == 200:
                    return self._parse_token_response(resp.json())
                body = resp.json()
                error = body.get("error", "")
                if error == "authorization_pending":
                    continue
                if error == "slow_down":
                    poll_interval += 5.0
                    continue
                resp.raise_for_status()
        raise TimeoutError("Device authorization timed out")

    async def _exchange_code(self, code: str, verifier: str, redirect_uri: str) -> CachedTokens:
        token_endpoint = _validate_oauth_uri(self._profile.token_endpoint, label="token_endpoint")
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                token_endpoint,
                data={
                    "grant_type": "authorization_code",
                    "code": code,
                    "redirect_uri": redirect_uri,
                    "client_id": self._profile.client_id,
                    "code_verifier": verifier,
                },
            )
            resp.raise_for_status()
            return self._parse_token_response(resp.json())

    async def _refresh(self, tokens: CachedTokens) -> CachedTokens:
        token_endpoint = _validate_oauth_uri(self._profile.token_endpoint, label="token_endpoint")
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                token_endpoint,
                data={
                    "grant_type": "refresh_token",
                    "refresh_token": tokens.refresh_token,
                    "client_id": self._profile.client_id,
                },
            )
            resp.raise_for_status()
            new_tokens = self._parse_token_response(resp.json())
            if not new_tokens.refresh_token:
                new_tokens.refresh_token = tokens.refresh_token
            self._tokens = new_tokens
            self._store.save(new_tokens)
            return new_tokens

    def _parse_token_response(self, body: dict[str, object]) -> CachedTokens:
        raw_exp = body.get("expires_in", 3600)
        expires_in = float(raw_exp) if isinstance(raw_exp, (int, float)) else 3600.0
        return CachedTokens(
            access_token=str(body["access_token"]),
            refresh_token=str(body.get("refresh_token", "")),
            id_token=str(body.get("id_token", "")),
            expires_at=time.time() + expires_in,
            profile_name=self._profile.name,
        )


def _start_callback_server(
    port: int,
    expected_state: str,
    code_future: asyncio.Future[str],
    loop: asyncio.AbstractEventLoop,
) -> HTTPServer:
    """Start a one-shot HTTP server to catch the OAuth redirect."""

    def _resolve_error(exc: Exception) -> None:
        if not code_future.done():
            code_future.set_exception(exc)

    def _resolve_code(value: str) -> None:
        if not code_future.done():
            code_future.set_result(value)

    class Handler(BaseHTTPRequestHandler):
        def _send_plain(self, status_code: int, message: str) -> None:
            self.send_response(status_code)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(message.encode("utf-8", errors="replace"))

        def do_GET(self) -> None:
            qs = parse_qs(urlparse(self.path).query)
            state = qs.get("state", [""])[0]
            code = qs.get("code", [""])[0]
            error = qs.get("error", [""])[0]

            if error:
                self._send_plain(400, f"Authorization error: {error}")
                loop.call_soon_threadsafe(_resolve_error, RuntimeError(f"OAuth error: {error}"))
                return

            if state != expected_state:
                self._send_plain(400, "State mismatch")
                loop.call_soon_threadsafe(_resolve_error, RuntimeError("OAuth state mismatch"))
                return

            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(
                b"<html><body><h2>Logged in to Atomics!</h2>"
                b"<p>You can close this tab.</p></body></html>"
            )
            loop.call_soon_threadsafe(_resolve_code, code)

        def log_message(self, format: str, *args: object) -> None:
            pass

    server = HTTPServer(("127.0.0.1", port), Handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server
