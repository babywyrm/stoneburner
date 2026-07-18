"""Central configuration loaded from env vars / .env files."""

from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings

from atomics.paths import default_db_path


class AtomicsSettings(BaseSettings):
    model_config = {"env_prefix": "ATOMICS_", "env_file": ".env", "extra": "ignore"}

    db_path: Path = Field(default_factory=default_db_path)
    log_level: str = Field(default="INFO")
    default_model: str = Field(default="claude-sonnet-4-6")

    max_tokens_per_hour: int = Field(default=100_000)
    max_requests_per_minute: int = Field(default=30)
    loop_interval_seconds: int = Field(default=120)
    loop_jitter_seconds: int = Field(default=15)
    budget_limit_usd: float = Field(default=50.0)

    max_retries: int = Field(default=3)
    retry_backoff_base: float = Field(default=2.0)
    circuit_breaker_threshold: int = Field(default=10)

    anthropic_api_key: str = Field(default="", validation_alias="ANTHROPIC_API_KEY")
    openai_api_key: str = Field(default="", validation_alias="OPENAI_API_KEY")
    groq_api_key: str = Field(default="", validation_alias="GROQ_API_KEY")
    together_api_key: str = Field(default="", validation_alias="TOGETHER_API_KEY")
    gemini_api_key: str = Field(default="", validation_alias="GEMINI_API_KEY")

    ollama_host: str = Field(default="http://localhost:11434", validation_alias="ATOMICS_OLLAMA_HOST")
    ollama_model: str = Field(default="qwen2.5:7b", validation_alias="ATOMICS_OLLAMA_MODEL")
    # Per-request timeout (seconds) for local generation. Thinking models on hard
    # prompts can reason well past the old hard-coded 120s, so default high.
    ollama_timeout: float = Field(default=300.0, validation_alias="ATOMICS_OLLAMA_TIMEOUT")

    # vLLM / OpenAI-compatible gateway (e.g. a LiteLLM gateway at :8000/v1).
    # "vllm" refers to the wire-format dialect (POST /v1/chat/completions),
    # not the OpenAI company. Nothing leaves the LAN.
    vllm_host: str = Field(default="http://localhost:8000/v1", validation_alias="ATOMICS_VLLM_HOST")
    vllm_model: str = Field(default="qwen2.5:3b", validation_alias="ATOMICS_VLLM_MODEL")
    vllm_timeout: float = Field(default=300.0, validation_alias="ATOMICS_VLLM_TIMEOUT")

    brain_gateway_url: str = Field(default="http://localhost:8080", validation_alias="ATOMICS_BRAIN_GATEWAY_URL")

    auth_mode: str = Field(default="auto")
    oidc_issuer: str = Field(default="")
    oidc_client_id: str = Field(default="")
    oidc_scopes: str = Field(default="")

    webhook_url: str = Field(default="", validation_alias="ATOMICS_WEBHOOK_URL")
    post_run_hook: str = Field(default="")
    notify_on_finish: bool = Field(default=False)


def load_settings() -> AtomicsSettings:
    """Load settings with layered secret resolution: env -> .env -> keychain."""
    settings = AtomicsSettings()

    # Backfill empty API keys from OS keychain (layer 3)
    from atomics.secrets import get_secret

    if not settings.anthropic_api_key:
        settings.anthropic_api_key = get_secret("ANTHROPIC_API_KEY") or ""
    if not settings.openai_api_key:
        settings.openai_api_key = get_secret("OPENAI_API_KEY") or ""

    return settings
