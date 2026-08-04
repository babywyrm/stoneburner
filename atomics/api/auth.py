"""Authentication backends for the atomics API server."""

from __future__ import annotations

import hmac
from typing import Protocol

from fastapi import Request

from atomics.api.callers import ANONYMOUS_CALLER, caller_id_from_key


class AuthBackend(Protocol):
    """Protocol for pluggable API authentication."""

    async def authenticate(self, request: Request) -> bool: ...

    def identify(self, request: Request) -> str:
        """Return a stable, non-secret identifier for the caller."""
        ...


def matched_key(candidate: str, keys: set[str]) -> str | None:
    """Return the configured key equal to `candidate`, comparing all of them.

    Set membership would compare by hash and short-circuit, leaking key content
    through timing. Every key is checked with no early return so the work done
    depends only on how many keys are configured, not on which one matched.
    """
    found: str | None = None
    for key in keys:
        if hmac.compare_digest(candidate, key):
            found = key
    return found


def key_matches(candidate: str, keys: set[str]) -> bool:
    """Whether `candidate` is one of the accepted keys, in constant time."""
    return matched_key(candidate, keys) is not None


class ApiKeyAuth:
    """API key authentication via the X-API-Key header."""

    def __init__(self, keys: set[str]) -> None:
        self._keys = keys

    async def authenticate(self, request: Request) -> bool:
        header = request.headers.get("x-api-key", "")
        return key_matches(header, self._keys)

    def identify(self, request: Request) -> str:
        """Identify the caller by which configured key they presented.

        Returns the anonymous identifier for an unrecognized key rather than
        raising: identification is not authorization, and callers of this have
        already been authenticated.
        """
        key = matched_key(request.headers.get("x-api-key", ""), self._keys)
        return caller_id_from_key(key) if key is not None else ANONYMOUS_CALLER


class NoAuth:
    """Allow all requests. Intended for local development only."""

    async def authenticate(self, request: Request) -> bool:
        return True

    def identify(self, request: Request) -> str:
        # No credential means no way to tell callers apart, so per-caller
        # quotas collapse to the global one. That is a consequence of running
        # without auth, not something identification can recover.
        return ANONYMOUS_CALLER
