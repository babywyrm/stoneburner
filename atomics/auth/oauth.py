"""OAuth 2.0 PKCE auth strategy with browser and device-code flows."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import logging
import os
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
        auth_url = f"{self._profile.authorization_endpoint}?{urlencode(params)}"

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

        return await self._exchange_code(code, verifier, redirect_uri)

    async def _device_code_flow(self) -> CachedTokens:
        """Device authorization grant for headless environments."""
        if not self._profile.device_authorization_endpoint:
            raise RuntimeError(
                f"Profile {self._profile.name!r} does not support device code flow"
            )

        async with httpx.AsyncClient() as client:
            resp = await client.post(
                self._profile.device_authorization_endpoint,
                data={
                    "client_id": self._profile.client_id,
                    "scope": " ".join(self._profile.scopes),
                },
            )
            resp.raise_for_status()
            body = resp.json()

        verification_uri = body.get("verification_uri_complete") or body.get("verification_uri", "")
        user_code = body.get("user_code", "")
        device_code = body["device_code"]
        poll_interval = body.get("interval", 5)

        print(f"\nVisit: {verification_uri}")
        if user_code:
            print(f"Code:  {user_code}")
        print("Waiting for authorization...\n")

        async with httpx.AsyncClient() as client:
            while True:
                await asyncio.sleep(poll_interval)
                resp = await client.post(
                    self._profile.token_endpoint,
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
                    poll_interval += 5
                    continue
                resp.raise_for_status()

    async def _exchange_code(
        self, code: str, verifier: str, redirect_uri: str
    ) -> CachedTokens:
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                self._profile.token_endpoint,
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
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                self._profile.token_endpoint,
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

    def _parse_token_response(self, body: dict) -> CachedTokens:
        expires_in = body.get("expires_in", 3600)
        return CachedTokens(
            access_token=body["access_token"],
            refresh_token=body.get("refresh_token", ""),
            id_token=body.get("id_token", ""),
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

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            qs = parse_qs(urlparse(self.path).query)
            state = qs.get("state", [""])[0]
            code = qs.get("code", [""])[0]
            error = qs.get("error", [""])[0]

            if error:
                self.send_response(400)
                self.end_headers()
                self.wfile.write(f"Authorization error: {error}".encode())
                loop.call_soon_threadsafe(
                    code_future.set_exception, RuntimeError(f"OAuth error: {error}")
                )
                return

            if state != expected_state:
                self.send_response(400)
                self.end_headers()
                self.wfile.write(b"State mismatch")
                loop.call_soon_threadsafe(
                    code_future.set_exception, RuntimeError("OAuth state mismatch")
                )
                return

            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(
                b"<html><body><h2>Logged in to Atomics!</h2>"
                b"<p>You can close this tab.</p></body></html>"
            )
            loop.call_soon_threadsafe(code_future.set_result, code)

        def log_message(self, format, *args):
            pass

    server = HTTPServer(("127.0.0.1", port), Handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server
