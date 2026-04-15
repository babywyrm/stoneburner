"""Static API key auth strategy."""

from __future__ import annotations

from atomics.auth import AuthStrategy


class ApiKeyAuth(AuthStrategy):
    """Authenticate with a static API key (current default path)."""

    def __init__(self, api_key: str) -> None:
        self._api_key = api_key

    async def get_headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self._api_key}"}

    async def validate(self) -> bool:
        return bool(self._api_key)

    @property
    def description(self) -> str:
        masked = self._api_key[:8] + "..." if len(self._api_key) > 8 else "***"
        return f"API key ({masked})"
