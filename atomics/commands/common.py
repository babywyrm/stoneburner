"""Shared command-layer primitives."""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable
from pathlib import Path
from typing import Protocol, TypeVar, cast

import click
from rich.console import Console
from rich.logging import RichHandler
from rich.status import Status

from atomics.config import AtomicsSettings
from atomics.eval.budget import EvalBudget
from atomics.eval.outcomes import RunIntegrity
from atomics.providers.base import BaseProvider
from atomics.providers.factory import (
    PROVIDER_NAMES,
    ProviderConfigError,
    make_provider,
)
from atomics.storage.records import EvaluationResultRecord
from atomics.validation import sanitize_error

PROVIDER_CHOICES = click.Choice(list(PROVIDER_NAMES), case_sensitive=False)

_P = TypeVar("_P")


def _effort_callback(_ctx: click.Context, _param: click.Parameter, value: str | None) -> str | None:
    if value is None:
        return None
    from atomics.providers.effort import EffortError, normalize_effort

    try:
        return normalize_effort(value)
    except EffortError as exc:
        raise click.BadParameter(str(exc), param_hint="--effort") from exc


def _reasoning_mode_callback(
    _ctx: click.Context, _param: click.Parameter, value: str | None
) -> str | None:
    if value is None:
        return None
    from atomics.providers.effort import EffortError, normalize_reasoning_mode

    try:
        return normalize_reasoning_mode(value)
    except EffortError as exc:
        raise click.BadParameter(str(exc), param_hint="--reasoning-mode") from exc


def effort_options(fn: Callable) -> Callable:
    """Add ``--effort`` and ``--reasoning-mode`` next to thinking flags."""
    fn = click.option(
        "--reasoning-mode",
        callback=_reasoning_mode_callback,
        default=None,
        help="OpenAI reasoning.mode: standard or pro. Ignored by other providers.",
    )(fn)
    fn = click.option(
        "--effort",
        callback=_effort_callback,
        default=None,
        help="Shared reasoning effort: none, minimal, low, medium, high, "
        "xhigh (alias xl), max (alias ultra). Mapped per provider.",
    )(fn)
    return fn


def extra_judges_option(fn: Callable) -> Callable:
    """Add `--extra-judges` for consensus scoring."""
    return click.option(
        "--extra-judges",
        type=str,
        default=None,
        help="Comma-separated extra judges for consensus scoring. "
        "Format: provider:model[@host] "
        "(e.g. claude:claude-sonnet-4-6,ollama:deepseek-r1:14b@http://gpu-host:11434).",
    )(fn)


def parse_extra_judges(
    spec: str | None,
    *,
    build: Callable[[str, str | None, str | None], _P],
    default_host: str | None = None,
) -> list[tuple[_P, str | None]]:
    """Parse the --extra-judges string into (provider, model) pairs."""
    if not spec:
        return []
    pairs: list[tuple[_P, str | None]] = []
    for raw in spec.split(","):
        item = raw.strip()
        if not item:
            continue
        host = default_host
        if "@" in item:
            item, host = item.rsplit("@", 1)
        name, _, model = item.partition(":")
        model_or_none = model or None
        pairs.append((build(name, model_or_none, host), model_or_none))
    return pairs


def budget_option(fn: Callable) -> Callable:
    """Add `--budget` to an eval command.

    Defaults to None — no ceiling — so every existing invocation behaves
    exactly as before. Opt-in on the CLI is deliberate: a local operator is
    spending their own money on a run they chose to start, and imposing a
    default would break anyone doing a large sweep today. The API takes the
    opposite default, because there the caller is remote.
    """
    return click.option(
        "--budget",
        "budget_usd",
        type=float,
        default=None,
        help="Stop the run once this many USD has been spent across the model "
        "under test and every judge. Default: no ceiling.",
    )(fn)


def eval_budget_from(budget_usd: float | None) -> EvalBudget | None:
    """Build an `EvalBudget` from a `--budget` value, or None when unset."""
    if budget_usd is None:
        return None
    try:
        return EvalBudget(budget_limit_usd=budget_usd)
    except ValueError as exc:
        raise click.BadParameter(str(exc), param_hint="--budget") from exc


def setup_logging(level: str, *, rich_tracebacks: bool = False, plain: bool = False) -> None:
    """Configure logging for the atomics logger.

    `plain` swaps Rich for one unwrapped line per record, and is what long-lived
    processes want. Rich wraps to the console width — 80 columns when output is
    redirected to a file — which splits a structured access log entry across
    four lines and leaves it unparseable by grep, journald, or any aggregator.
    That is fine for an interactive run someone is watching and wrong for a
    server whose logs are read by machines.
    """
    numeric = getattr(logging, level.upper(), logging.INFO)
    handler: logging.Handler
    if plain:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s %(levelname)s %(name)s %(message)s",
                datefmt="%Y-%m-%dT%H:%M:%S%z",
            )
        )
    else:
        handler = RichHandler(rich_tracebacks=rich_tracebacks, markup=True)
    logging.basicConfig(
        level=logging.WARNING,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[handler],
        force=True,
    )
    # Only our own loggers get the requested level; third-party stays quiet.
    logging.getLogger("atomics").setLevel(numeric)


# Backward-compatible alias for earlier command modules.
_setup_logging = setup_logging


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
        eta_str = f" | ETA remaining: {self._fmt_duration(eta)}" if eta is not None else ""
        status_msg = (
            f"[{index + 1}/{self.total}] {fixture_id} ({category}) — generating...{eta_str}"
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
        raise click.ClickException(f"Unable to write JSON output: {sanitize_error(exc)}") from exc


def integrity_exit_code(
    integrity: RunIntegrity,
    *,
    allow_partial: bool,
) -> int:
    """Return the command exit code for one run-integrity result."""
    return int(integrity.should_exit_nonzero and not allow_partial)


def evaluation_record_from_fixture(
    *,
    run_id: str,
    suite: str,
    provider: str,
    model: str,
    payload: dict[str, object],
) -> EvaluationResultRecord:
    """Convert canonical fixture JSON into a storage-owned record."""
    attempts_value = payload.get("attempts")
    if not isinstance(attempts_value, list):
        raise ValueError("fixture payload attempts must be a list")
    attempts = cast(list[dict[str, object]], attempts_value)
    input_tokens = sum(_as_int(attempt["input_tokens"]) for attempt in attempts)
    output_tokens = sum(_as_int(attempt["output_tokens"]) for attempt in attempts)
    thinking_tokens = sum(_as_int(attempt["thinking_tokens"]) for attempt in attempts)
    score_value = payload.get("score")
    score = None if score_value is None else _as_float(score_value)
    agreement_value = payload.get("judge_agreement")
    judge_agreement = None if agreement_value is None else _as_float(agreement_value)
    return EvaluationResultRecord(
        run_id=run_id,
        suite=suite,
        fixture_id=str(payload["id"]),
        status=str(payload["status"]),
        score=score,
        generation_status=str(payload["generation_status"]),
        judge_status=str(payload["judge_status"]),
        latency_ms=_as_float(payload["latency_ms"]),
        estimated_cost_usd=_as_float(payload["estimated_cost_usd"]),
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=input_tokens + output_tokens,
        thinking_tokens=thinking_tokens,
        attempt_count=_as_int(payload["attempt_count"]),
        generation_failures=_as_int(payload["generation_failures"]),
        infrastructure_failures=_as_int(payload["infrastructure_failures"]),
        judge_failures=_as_int(payload["judge_failures"]),
        parse_failed=bool(payload["parse_failed"]),
        provider=provider,
        model=model,
        error_class=str(payload.get("error_class") or ""),
        error_message=str(payload.get("error_message") or ""),
        result_json=payload,
        judge_agreement=judge_agreement,
    )


def _as_int(value: object) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float, str)):
        return int(value)
    raise ValueError(f"expected integer-compatible value, got {type(value).__name__}")


def _as_float(value: object) -> float:
    if isinstance(value, (int, float, str)):
        return float(value)
    raise ValueError(f"expected numeric value, got {type(value).__name__}")


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
    """Build a provider, reporting configuration problems as CLI errors.

    The construction itself lives in `atomics.providers.factory` so the API
    server and distributed workers can call it without importing the command
    layer. This wrapper exists to translate the domain error into Click's, and
    to label endpoint failures with the flag names a CLI user actually typed.
    """
    try:
        return make_provider(
            name,
            mdl,
            host,
            settings,
            vllm_host=vllm_host,
            region=region,
            context_tokens=context_tokens,
            inference_timeout=inference_timeout,
            host_label="--ollama-host/--judge-host",
            vllm_host_label="--vllm-host",
        )
    except ProviderConfigError as exc:
        raise click.ClickException(str(exc)) from exc
