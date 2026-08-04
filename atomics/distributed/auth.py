"""Authentication for distributed worker endpoints.

Workers authenticate with their own key set, distinct from the keys that
authorize submitting runs and evals. A worker runs on a host the coordinator
does not control, so its credential must not also open the submitter surface.
"""

from __future__ import annotations

from fastapi import Request

from atomics.api.auth import AuthBackend, key_matches, matched_key
from atomics.api.callers import ANONYMOUS_CALLER, caller_id_from_key


class WorkerAuth(AuthBackend):
    """API key authentication for worker lifecycle endpoints."""

    def __init__(self, keys: set[str]) -> None:
        self._keys = keys

    async def authenticate(self, request: Request) -> bool:
        key = request.headers.get("x-api-key", "")
        return key_matches(key, self._keys)

    def identify(self, request: Request) -> str:
        key = matched_key(request.headers.get("x-api-key", ""), self._keys)
        return caller_id_from_key(key) if key is not None else ANONYMOUS_CALLER
