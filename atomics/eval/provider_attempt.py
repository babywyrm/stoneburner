"""Helpers for converting provider responses into immutable attempts."""

from __future__ import annotations

from atomics.eval.outcomes import (
    AttemptResult,
    JudgeOutcome,
    ProviderOutcome,
    ProviderOutcomeKind,
)
from atomics.providers.base import ProviderResponse


def provider_outcome_from_response(response: ProviderResponse) -> ProviderOutcome:
    """Use an adapter outcome when present, otherwise infer text completion."""
    if response.outcome is not None:
        return response.outcome
    kind = (
        ProviderOutcomeKind.COMPLETED
        if response.text.strip()
        else ProviderOutcomeKind.EMPTY
    )
    return ProviderOutcome(kind=kind, finish_reason=response.finish_reason)


def build_attempt(
    *,
    attempt_index: int,
    outcome: ProviderOutcome,
    response: ProviderResponse | None,
    judge: JudgeOutcome | None,
) -> AttemptResult:
    """Build one attempt while retaining known provider and judge costs."""
    provider_cost = response.estimated_cost_usd if response is not None else 0.0
    judge_cost = judge.judge_cost_usd if judge is not None else 0.0
    return AttemptResult(
        attempt_index=attempt_index,
        provider=outcome,
        response_text=response.text if response is not None else "",
        latency_ms=response.latency_ms if response is not None else 0.0,
        estimated_cost_usd=provider_cost + judge_cost,
        input_tokens=response.input_tokens if response is not None else 0,
        output_tokens=response.output_tokens if response is not None else 0,
        thinking_tokens=response.thinking_tokens if response is not None else 0,
        judge=judge,
    )
