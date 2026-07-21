"""Server configuration for atomics API mode."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from atomics.paths import default_db_path


@dataclass
class ServerSettings:
    """Settings for the atomics API server."""

    host: str = "127.0.0.1"
    port: int = 8000
    api_keys: set[str] = field(default_factory=set)
    no_auth: bool = False
    log_level: str = "info"
    db_path: Path = field(default_factory=default_db_path)

    def __post_init__(self) -> None:
        if self.port < 1 or self.port > 65535:
            raise ValueError(f"port must be in range 1-65535, got {self.port}")
