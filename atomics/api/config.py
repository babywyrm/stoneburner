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
    # Roughly four missed heartbeats at the worker's default 30s interval. It has
    # to be settable because that interval is: `atomics worker
    # --heartbeat-interval 300` against a fixed 120s threshold would have every
    # worker declared absent, and its pinned fleet work failed, while behaving
    # exactly as configured.
    worker_absent_after_seconds: float = 120.0

    def __post_init__(self) -> None:
        if self.port < 1 or self.port > 65535:
            raise ValueError(f"port must be in range 1-65535, got {self.port}")
        if self.worker_absent_after_seconds <= 0:
            raise ValueError(
                "worker_absent_after_seconds must be positive, got "
                f"{self.worker_absent_after_seconds}"
            )
