"""Response security headers for the atomics API server.

The API returns JSON to programmatic callers, but the optional dashboard returns
HTML to a browser, and that browser holds an API key. These headers are cheap
and narrow the blast radius if something ever does reach the page as markup.
"""

from __future__ import annotations

import secrets
from collections.abc import Awaitable, Callable

from starlette.requests import Request
from starlette.responses import Response

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
