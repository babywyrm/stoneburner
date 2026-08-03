"""Authentication backends for the atomics API server."""

from __future__ import annotations

import hmac
from typing import Protocol

from fastapi import Request


class AuthBackend(Protocol):
    """Protocol for pluggable API authentication."""

    async def authenticate(self, request: Request) -> bool: ...


def key_matches(candidate: str, keys: set[str]) -> bool:
    """Compare a presented key against every accepted key in constant time.

    Set membership would compare by hash and short-circuit, leaking key content
    through timing. Every key is checked with no early return so the work done
    depends only on how many keys are configured, not on which one matched.
    """
    matched = False
    for key in keys:
        if hmac.compare_digest(candidate, key):
            matched = True
    return matched


class ApiKeyAuth:
    """API key authentication via the X-API-Key header."""

    def __init__(self, keys: set[str]) -> None:
        self._keys = keys

    async def authenticate(self, request: Request) -> bool:
        header = request.headers.get("x-api-key", "")
        return key_matches(header, self._keys)


class NoAuth:
    """Allow all requests. Intended for local development only."""

    async def authenticate(self, request: Request) -> bool:
        return True
