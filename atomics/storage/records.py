"""Storage-owned typed records."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EvaluationResultRecord:
    """One generic fixture result prepared for durable persistence."""

    run_id: str
    suite: str
    fixture_id: str
    status: str
    generation_status: str
    judge_status: str
    latency_ms: float
    input_tokens: int
    output_tokens: int
    total_tokens: int
    result_json: dict[str, object]
    score: float | None = None
    estimated_cost_usd: float = 0.0
    thinking_tokens: int = 0
    attempt_count: int = 0
    generation_failures: int = 0
    infrastructure_failures: int = 0
    judge_failures: int = 0
    parse_failed: bool = False
    provider: str = ""
    model: str = ""
    error_class: str = ""
    error_message: str = ""
