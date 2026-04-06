"""Task runner — executes a single task against a provider and records the result."""

from __future__ import annotations

import logging
from datetime import UTC, datetime

from atomics.models import TaskDefinition, TaskResult, TaskStatus
from atomics.providers.base import BaseProvider

logger = logging.getLogger("atomics.runner")


async def execute_task(
    task: TaskDefinition,
    topic: str,
    *,
    provider: BaseProvider,
    run_id: str,
    model: str | None = None,
) -> TaskResult:
    """Run a single benchmark task and return the result (never raises)."""
    if "{prompt}" in task.prompt_template:
        prompt = topic  # topic is already the fully-built prompt
    else:
        prompt = task.prompt_template.format(topic=topic)
    result = TaskResult(
        run_id=run_id,
        category=task.category,
        task_name=task.name,
        provider=provider.name,
        model=model or "default",
        prompt=prompt,
    )

    try:
        result.status = TaskStatus.RUNNING
        resp = await provider.generate(
            prompt,
            system="You are a knowledgeable technical assistant.",
            model=model,
            max_tokens=task.max_output_tokens,
        )
        result.status = TaskStatus.SUCCESS
        result.response = resp.text
        result.input_tokens = resp.input_tokens
        result.output_tokens = resp.output_tokens
        result.total_tokens = resp.total_tokens
        result.model = resp.model
        result.latency_ms = resp.latency_ms
        result.estimated_cost_usd = resp.estimated_cost_usd

    except Exception as exc:
        result.status = TaskStatus.FAILED
        result.error_class = type(exc).__name__
        result.error_message = str(exc)[:500]
        logger.warning(
            "Task %s failed: %s — %s",
            task.name,
            result.error_class,
            result.error_message,
        )

    result.completed_at = datetime.now(UTC)
    return result
