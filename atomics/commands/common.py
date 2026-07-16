"""Shared command-layer primitives."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Protocol

import click
from rich.console import Console
from rich.status import Status

from atomics.config import AtomicsSettings
from atomics.eval.outcomes import RunIntegrity
from atomics.providers.base import BaseProvider
from atomics.validation import sanitize_error, validate_endpoint_url

PROVIDER_CHOICES = click.Choice(
    ["claude", "bedrock", "openai", "ollama", "vllm", "brain-gateway"],
    case_sensitive=False,
)


class SerializableSummary(Protocol):
    """Summary objects accepted by the shared JSON writer."""

    def to_dict(self) -> dict[str, object]: ...


class FixtureProgress:
    """Real-time progress tracker for long-running fixture-based evals."""

    def __init__(self, total: int, console: Console, label: str = "fixture"):
        self.total = total
        self.console = console
        self.label = label
        self._start = time.monotonic()
        self._fixture_times: list[float] = []
        self._current_start: float | None = None
        self._status: Status | None = None

    def on_start(self, index: int, fixture_id: str, category: str) -> None:
        self._current_start = time.monotonic()
        eta = self._estimate_remaining(index)
        eta_str = (
            f" | ETA remaining: {self._fmt_duration(eta)}"
            if eta is not None
            else ""
        )
        status_msg = (
            f"[{index + 1}/{self.total}] {fixture_id} ({category}) "
            f"— generating...{eta_str}"
        )
        self._status = self.console.status(status_msg, spinner="dots")
        self._status.start()

    def on_done(self, index: int) -> None:
        if self._status:
            self._status.stop()
            self._status = None
        if self._current_start is not None:
            self._fixture_times.append(time.monotonic() - self._current_start)
            self._current_start = None

    def _estimate_remaining(self, current_index: int) -> float | None:
        if not self._fixture_times:
            return None
        average = sum(self._fixture_times) / len(self._fixture_times)
        return average * (self.total - current_index)

    @staticmethod
    def _fmt_duration(seconds: float) -> str:
        if seconds < 60:
            return f"{seconds:.0f}s"
        minutes, remaining_seconds = divmod(int(seconds), 60)
        return f"{minutes}m{remaining_seconds:02d}s"


def effective_model(requested_model: str | None, provider: object) -> str:
    """Resolve requested, provider-default, or generic model attribution."""
    if requested_model:
        return requested_model
    provider_default = getattr(provider, "default_model", None)
    if isinstance(provider_default, str) and provider_default:
        return provider_default
    return "default"


def _attribution_model(provider: object, requested_model: str | None) -> str:
    """Compatibility form of effective-model resolution."""
    return effective_model(requested_model, provider)


def write_summary_json(summary: SerializableSummary, path: Path) -> None:
    """Write one summary through its canonical serializer."""
    try:
        with path.open("w", encoding="utf-8") as handle:
            json.dump(summary.to_dict(), handle, indent=2)
    except (OSError, TypeError, ValueError) as exc:
        raise click.ClickException(
            f"Unable to write JSON output: {sanitize_error(exc)}"
        ) from exc


def integrity_exit_code(
    integrity: RunIntegrity,
    *,
    allow_partial: bool,
) -> int:
    """Return the command exit code for one run-integrity result."""
    return int(integrity.should_exit_nonzero and not allow_partial)


def _make_provider(
    name: str,
    mdl: str | None,
    host: str | None,
    settings: AtomicsSettings,
    *,
    vllm_host: str | None = None,
    region: str = "us-east-1",
    context_tokens: int | None = None,
    inference_timeout: int | None = None,
) -> BaseProvider:
    """Build a provider instance for any command."""
    if host:
        try:
            host = validate_endpoint_url(
                host,
                label="--ollama-host/--judge-host",
            )
        except ValueError as exc:
            raise click.ClickException(str(exc)) from exc
    if vllm_host:
        try:
            vllm_host = validate_endpoint_url(vllm_host, label="--vllm-host")
        except ValueError as exc:
            raise click.ClickException(str(exc)) from exc

    if name == "claude":
        if not settings.anthropic_api_key:
            raise click.ClickException(
                "ANTHROPIC_API_KEY not set. Export it or add to .env"
            )
        from atomics.providers.claude import ClaudeProvider

        return ClaudeProvider(
            api_key=settings.anthropic_api_key,
            default_model=mdl or settings.default_model,
        )
    if name == "bedrock":
        from atomics.providers.bedrock import BedrockProvider

        return BedrockProvider(
            region=region,
            model_id=mdl or "us.anthropic.claude-sonnet-4-6",
        )
    if name == "openai":
        if not settings.openai_api_key:
            raise click.ClickException(
                "OPENAI_API_KEY not set. Export it or install with: "
                "uv sync --extra openai"
            )
        from atomics.providers.openai import OpenAIProvider

        return OpenAIProvider(
            api_key=settings.openai_api_key,
            default_model=mdl or "gpt-4o",
        )
    if name == "vllm":
        from atomics.providers.vllm import VllmProvider

        return VllmProvider(
            base_url=vllm_host or settings.vllm_host,
            default_model=mdl or settings.vllm_model,
            timeout=inference_timeout or settings.vllm_timeout,
        )
    if name == "brain-gateway":
        from atomics.providers.brain_gateway import BrainGatewayProvider

        return BrainGatewayProvider(
            url=host or settings.brain_gateway_url,
            default_model=mdl,
        )
    if name == "ollama":
        from atomics.providers.ollama import OllamaProvider

        return OllamaProvider(
            host=host or settings.ollama_host,
            default_model=mdl or settings.ollama_model,
            timeout=inference_timeout or settings.ollama_timeout,
            context_tokens=context_tokens,
        )
    raise click.ClickException(
        f"Unknown provider: {name!r}. "
        "Valid: claude, bedrock, openai, ollama, vllm, brain-gateway"
    )
