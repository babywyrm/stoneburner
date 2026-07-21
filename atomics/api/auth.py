"""Authentication backends for the atomics API server."""

from __future__ import annotations

from typing import Protocol

from fastapi import Request


class AuthBackend(Protocol):
    """Protocol for pluggable API authentication."""

    async def authenticate(self, request: Request) -> bool: ...


class ApiKeyAuth:
    """API key authentication via the X-API-Key header."""

    def __init__(self, keys: set[str]) -> None:
        self._keys = keys

    async def authenticate(self, request: Request) -> bool:
        header = request.headers.get("x-api-key", "")
        return header in self._keys


class NoAuth:
    """Allow all requests. Intended for local development only."""

    async def authenticate(self, request: Request) -> bool:
        return True
