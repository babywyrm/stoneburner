"""Request correlation IDs and structured access logging.

Every request gets an identifier that appears in its access log line, in its
response headers, and in the logs of any job it starts. That last part is the
point: runs and evals are async jobs, so the request that submitted one has
returned long before the work finishes, and without a shared identifier there
is nothing tying a failure hours later back to who asked for it.

The identifier propagates into job tasks through a context variable, which
`asyncio.create_task` copies automatically at creation time.

Two things are deliberately absent from log lines: query strings and bodies.
Both have historically carried API keys — the dashboard once passed one as
`?api_key=` — and an access log is the wrong place to find out.
"""

from __future__ import annotations

import logging
import re
import time
import uuid
from collections.abc import Awaitable, Callable
from contextvars import ContextVar

from starlette.requests import Request
from starlette.responses import Response

from atomics.api.callers import ANONYMOUS_CALLER

logger = logging.getLogger(__name__)

REQUEST_ID_HEADER = "X-Request-ID"

# An inbound identifier is attacker-controlled and goes straight into a log
# file, so it is accepted only in a shape that cannot forge a log line: no
# newlines, no control characters, no unbounded length.
_SAFE_REQUEST_ID = re.compile(r"\A[A-Za-z0-9._-]{1,64}\Z")

_request_id: ContextVar[str] = ContextVar("atomics_request_id", default="")


def sanitize_request_id(raw: str | None) -> str | None:
    """Accept a caller-supplied correlation ID only if it is safe to log."""
    if not raw or not _SAFE_REQUEST_ID.match(raw):
        return None
    return raw


def new_request_id() -> str:
    return uuid.uuid4().hex[:16]


def current_request_id() -> str:
    """The correlation ID of the request in scope, or empty outside one."""
    return _request_id.get()


async def request_log_middleware(
    request: Request,
    call_next: Callable[[Request], Awaitable[Response]],
) -> Response:
    """Assign a correlation ID, then log one line describing the request."""
    request_id = sanitize_request_id(
        request.headers.get(REQUEST_ID_HEADER)
    ) or new_request_id()
    token = _request_id.set(request_id)
    request.state.request_id = request_id

    started = time.perf_counter()
    status = 500
    try:
        response = await call_next(request)
        status = response.status_code
        response.headers[REQUEST_ID_HEADER] = request_id
        return response
    finally:
        duration_ms = (time.perf_counter() - started) * 1000
        # Set by require_auth; absent on unauthenticated routes like /health.
        caller = getattr(request.state, "caller_id", ANONYMOUS_CALLER)
        logger.info(
            "request_id=%s caller=%s method=%s path=%s status=%d duration_ms=%.1f",
            request_id,
            caller,
            request.method,
            request.url.path,
            status,
            duration_ms,
        )
        _request_id.reset(token)
