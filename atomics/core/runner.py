"""Task runner — executes a single task against a provider and records the result."""

from __future__ import annotations

import logging
from datetime import UTC, datetime

from atomics.models import TaskDefinition, TaskResult, TaskStatus
from atomics.providers.base import BaseProvider
from atomics.validation import sanitize_error

logger = logging.getLogger("atomics.runner")


async def execute_task(
    task: TaskDefinition,
    topic: str,
    *,
    provider: BaseProvider,
    run_id: str,
    model: str | None = None,
    thinking: bool | None = None,
    thinking_budget: int | None = None,
) -> TaskResult:
    """Run a single benchmark task and return the result (never raises)."""
    if "{prompt}" in task.prompt_template:
        prompt = topic
    else:
        prompt = task.prompt_template.format(topic=topic)
    result = TaskResult(
        run_id=run_id,
        category=task.category,
        task_name=task.name,
        provider=provider.name,
        model=model or "default",
        prompt=prompt,
        thinking_enabled=thinking is True,
    )

    try:
        result.status = TaskStatus.RUNNING
        resp = await provider.generate(
            prompt,
            system="You are a knowledgeable technical assistant.",
            model=model,
            max_tokens=task.max_output_tokens,
            thinking=thinking,
            thinking_budget=thinking_budget,
        )
        result.status = TaskStatus.SUCCESS
        result.response = resp.text
        result.input_tokens = resp.input_tokens
        result.output_tokens = resp.output_tokens
        result.total_tokens = resp.total_tokens
        result.thinking_tokens = resp.thinking_tokens
        result.cache_read_tokens = resp.cache_read_tokens
        result.cache_write_tokens = resp.cache_write_tokens
        result.model = resp.model
        result.latency_ms = resp.latency_ms
        result.estimated_cost_usd = resp.estimated_cost_usd
        result.tokens_per_second = resp.tokens_per_second
        result.tps_basis = resp.tps_basis

    except Exception as exc:
        result.status = TaskStatus.FAILED
        result.error_class = type(exc).__name__
        result.error_message = sanitize_error(exc)
        logger.warning(
            "Task %s failed: %s — %s",
            task.name,
            result.error_class,
            result.error_message,
        )

    result.completed_at = datetime.now(UTC)
    return result
