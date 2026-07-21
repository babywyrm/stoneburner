from __future__ import annotations

from fastapi import Request

from atomics.api.auth import AuthBackend


class WorkerAuth(AuthBackend):
    def __init__(self, keys: set[str]) -> None:
        self._keys = keys

    async def authenticate(self, request: Request) -> bool:
        key = request.headers.get("x-api-key")
        return key in self._keys
