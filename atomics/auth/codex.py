"""Auth strategy that detects Codex CLI tokens from ~/.codex/auth.json.

NOTE: Codex CLI authenticates via ChatGPT OAuth, which produces tokens that
only work with ChatGPT's internal backend API (chatgpt.com/backend-api).
These tokens lack the scopes (model.request, api.responses.write) needed
for the public OpenAI developer API at api.openai.com.

If ~/.codex/auth.json contains an exchanged OPENAI_API_KEY (non-null), that
key IS usable. Otherwise, this strategy signals that the user should create
an API key at platform.openai.com/api-keys or use `atomics login`.
"""

from __future__ import annotations

import json
from pathlib import Path

from atomics.auth import AuthStrategy


def _default_codex_auth_path() -> Path:
    return Path.home() / ".codex" / "auth.json"


class CodexTokenAuth(AuthStrategy):
    """Reuse an API key from Codex CLI's auth.json, if one was exchanged.

    The Codex CLI sometimes exchanges its id_token for a real API key and
    stores it in auth.json as ``OPENAI_API_KEY``.  When that key is present,
    this strategy works like ``ApiKeyAuth``.  When only ChatGPT session
    tokens are present (the common case), ``tokens_available()`` returns
    False so auto-detect skips to the next strategy.
    """

    def __init__(self, path: Path | None = None) -> None:
        self._path = path or _default_codex_auth_path()
        self._api_key: str = ""
        self._loaded = False

    def tokens_available(self) -> bool:
        """True only if auth.json has a usable OPENAI_API_KEY."""
        if not self._path.exists():
            return False
        try:
            data = self._read_file()
            key = data.get("OPENAI_API_KEY") or ""
            return bool(key)
        except (json.JSONDecodeError, OSError):
            return False

    def codex_installed(self) -> bool:
        """True if auth.json exists (even without a usable API key)."""
        return self._path.exists()

    async def get_headers(self) -> dict[str, str]:
        self._ensure_loaded()
        return {"Authorization": f"Bearer {self._api_key}"}

    async def validate(self) -> bool:
        try:
            self._ensure_loaded()
            return bool(self._api_key)
        except Exception:
            return False

    @property
    def description(self) -> str:
        return "Codex CLI API key (~/.codex/auth.json)"

    def _read_file(self) -> dict:
        return json.loads(self._path.read_text())

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        data = self._read_file()
        self._api_key = data.get("OPENAI_API_KEY") or ""
        if not self._api_key:
            raise RuntimeError(
                "Codex CLI is installed but its auth tokens cannot access the "
                "OpenAI developer API (ChatGPT OAuth tokens lack the required "
                "scopes). Create an API key at https://platform.openai.com/api-keys "
                "and export OPENAI_API_KEY, or run: atomics login"
            )
        self._loaded = True
