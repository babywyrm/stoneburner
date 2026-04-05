"""Central configuration loaded from env vars / .env files."""

from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings


class AtomicsSettings(BaseSettings):
    model_config = {"env_prefix": "ATOMICS_", "env_file": ".env", "extra": "ignore"}

    db_path: Path = Field(default=Path("data/atomics.db"))
    log_level: str = Field(default="INFO")
    default_model: str = Field(default="claude-sonnet-4-20250514")

    max_tokens_per_hour: int = Field(default=100_000)
    max_requests_per_minute: int = Field(default=30)
    loop_interval_seconds: int = Field(default=120)
    loop_jitter_seconds: int = Field(default=15)
    budget_limit_usd: float = Field(default=50.0)

    max_retries: int = Field(default=3)
    retry_backoff_base: float = Field(default=2.0)
    circuit_breaker_threshold: int = Field(default=10)

    anthropic_api_key: str = Field(default="")


def load_settings() -> AtomicsSettings:
    return AtomicsSettings()
