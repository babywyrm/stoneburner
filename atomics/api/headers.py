"""Response security headers for the atomics API server.

The API returns JSON to programmatic callers, but the optional dashboard returns
HTML to a browser, and that browser holds an API key. These headers are cheap
and narrow the blast radius if something ever does reach the page as markup.
"""

from __future__ import annotations

import secrets
from collections.abc import Awaitable, Callable
from urllib.parse import urlsplit

from fastapi import HTTPException
from starlette.requests import Request
from starlette.responses import JSONResponse, PlainTextResponse, Response
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from atomics.api.config import is_loopback_host

# Applies to every response. `default-src 'none'` is safe for JSON endpoints and
# is overridden for the dashboard, which is the only route that renders.
_BASE_HEADERS = {
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Referrer-Policy": "no-referrer",
    "Cross-Origin-Opener-Policy": "same-origin",
    "Cache-Control": "no-store",
}

_JSON_CSP = "default-src 'none'; frame-ancestors 'none'"


def dashboard_csp(nonce: str) -> str:
    """CSP for the dashboard page.

    The page's script and style are inline, so they are allowed by nonce rather
    than by `unsafe-inline` — an injected tag has no way to guess the nonce, so
    this stays a real control instead of a decorative header.
    """
    return (
        "default-src 'none'; "
        f"script-src 'nonce-{nonce}'; "
        f"style-src 'nonce-{nonce}'; "
        "connect-src 'self'; "
        "frame-ancestors 'none'; "
        "base-uri 'none'; "
        "form-action 'none'"
    )


def new_nonce() -> str:
    return secrets.token_urlsafe(16)


async def security_headers_middleware(
    request: Request,
    call_next: Callable[[Request], Awaitable[Response]],
) -> Response:
    """Attach baseline security headers to every response."""
    response = await call_next(request)
    for header, value in _BASE_HEADERS.items():
        response.headers.setdefault(header, value)
    # The dashboard sets its own nonce-based policy; do not clobber it.
    response.headers.setdefault("Content-Security-Policy", _JSON_CSP)
    return response


async def loopback_host_middleware(
    request: Request,
    call_next: Callable[[Request], Awaitable[Response]],
) -> Response:
    """Refuse a Host header that does not name this machine.

    Installed only under --no-auth. A page that DNS-rebinds its own name to
    127.0.0.1 is same-origin with the server, so without this it could submit
    evals and read the results back.
    """
    host = urlsplit(f"//{request.headers.get('host', '')}").hostname or ""
    if not is_loopback_host(host):
        return PlainTextResponse("Host header must name a loopback address", status_code=421)
    return await call_next(request)


class RequestBodyLimitMiddleware:
    """Pure ASGI body-size guard enforcing `max_bytes` on Content-Length and chunked streams."""

    def __init__(self, app: ASGIApp, *, max_bytes: int) -> None:
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        detail = f"Request body exceeds maximum of {self.max_bytes} bytes"
        for raw_name, raw_value in scope.get("headers", ()):
            if raw_name.lower() == b"content-length":
                try:
                    declared = int(raw_value.decode("ascii"))
                except ValueError:
                    bad = JSONResponse({"detail": "Invalid Content-Length header"}, status_code=400)
                    await bad(scope, receive, send)
                    return
                if declared < 0:
                    bad = JSONResponse({"detail": "Invalid Content-Length header"}, status_code=400)
                    await bad(scope, receive, send)
                    return
                if declared > self.max_bytes:
                    too_big = JSONResponse({"detail": detail}, status_code=413)
                    await too_big(scope, receive, send)
                    return

        seen_bytes = 0
        response_started = False

        async def limited_receive() -> Message:
            nonlocal seen_bytes
            message = await receive()
            if message["type"] == "http.request":
                chunk = message.get("body", b"")
                seen_bytes += len(chunk)
                if seen_bytes > self.max_bytes:
                    raise HTTPException(status_code=413, detail=detail)
            return message

        async def tracked_send(message: Message) -> None:
            nonlocal response_started
            if message["type"] == "http.response.start":
                response_started = True
            await send(message)

        try:
            await self.app(scope, limited_receive, tracked_send)
        except HTTPException as exc:
            if response_started or exc.status_code != 413:
                raise
            too_big = JSONResponse({"detail": exc.detail}, status_code=413)
            await too_big(scope, receive, send)
