"""Persistent token storage with XDG / macOS conventions."""

from __future__ import annotations

import json
import os
import platform
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class CachedTokens:
    access_token: str = ""
    refresh_token: str = ""
    id_token: str = ""
    expires_at: float = 0.0
    profile_name: str = ""
    extra: dict = field(default_factory=dict)

    @property
    def expired(self) -> bool:
        if self.expires_at <= 0:
            return True
        return time.time() >= self.expires_at

    @property
    def needs_refresh(self) -> bool:
        """True if token expires within 5 minutes."""
        if self.expires_at <= 0:
            return True
        return time.time() >= (self.expires_at - 300)


def _default_auth_dir() -> Path:
    system = platform.system()
    if system == "Darwin":
        return Path.home() / "Library" / "Application Support" / "atomics"
    xdg = os.environ.get("XDG_DATA_HOME", "").strip()
    base = Path(xdg).expanduser() if xdg else Path.home() / ".local" / "share"
    return base / "atomics"


class TokenStore:
    """Read/write cached OAuth tokens to disk."""

    def __init__(self, path: Path | None = None) -> None:
        self._path = path or (_default_auth_dir() / "auth.json")

    @property
    def path(self) -> Path:
        return self._path

    def load(self) -> CachedTokens:
        if not self._path.exists():
            return CachedTokens()
        try:
            data = json.loads(self._path.read_text())
            return CachedTokens(
                access_token=data.get("access_token", ""),
                refresh_token=data.get("refresh_token", ""),
                id_token=data.get("id_token", ""),
                expires_at=float(data.get("expires_at", 0)),
                profile_name=data.get("profile_name", ""),
                extra=data.get("extra", {}),
            )
        except (json.JSONDecodeError, OSError):
            return CachedTokens()

    def save(self, tokens: CachedTokens) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(json.dumps(asdict(tokens), indent=2))
        self._path.chmod(0o600)

    def clear(self) -> None:
        if self._path.exists():
            self._path.unlink()

    def has_valid_tokens(self) -> bool:
        tokens = self.load()
        return bool(tokens.access_token) and not tokens.expired
