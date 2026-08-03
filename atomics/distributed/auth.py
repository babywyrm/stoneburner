"""Authentication for distributed worker endpoints.

Workers authenticate with their own key set, distinct from the keys that
authorize submitting runs and evals. A worker runs on a host the coordinator
does not control, so its credential must not also open the submitter surface.
"""

from __future__ import annotations

from fastapi import Request

from atomics.api.auth import AuthBackend, key_matches


class WorkerAuth(AuthBackend):
    """API key authentication for worker lifecycle endpoints."""

    def __init__(self, keys: set[str]) -> None:
        self._keys = keys

    async def authenticate(self, request: Request) -> bool:
        key = request.headers.get("x-api-key", "")
        return key_matches(key, self._keys)
