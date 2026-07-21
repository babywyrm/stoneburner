"""Execute distributed task assignments via the local benchmarking runner."""

from __future__ import annotations

import logging
from typing import Any

from atomics.commands.common import _make_provider
from atomics.config import AtomicsSettings, load_settings
from atomics.core.runner import execute_task
from atomics.distributed.models import TaskAssignment
from atomics.models import TaskCategory, TaskComplexity, TaskDefinition
from atomics.tasks.catalog import TASK_CATALOG

logger = logging.getLogger("atomics.distributed.worker_runner")


def _resolve_task_definition(task_spec: dict[str, Any]) -> tuple[TaskDefinition, str]:
    """Build a TaskDefinition and topic/prompt from an assignment task_spec.

    Expected keys (Phase 1):
      - task_name: either "quick_question" or "general_qa/quick_question"
      - prompt: the topic or prebuilt prompt text
    Optional overrides: max_output_tokens, complexity, category
    """
    prompt = str(task_spec.get("prompt", ""))
    raw_name = str(task_spec.get("task_name", "unknown"))
    category_hint: str | None = None
    task_name = raw_name
    if "/" in raw_name:
        category_hint, task_name = raw_name.split("/", 1)

    catalog_task = next((t for t in TASK_CATALOG if t.name == task_name), None)
    if catalog_task is not None:
        if category_hint and catalog_task.category.value != category_hint:
            logger.warning(
                "task_spec category %r does not match catalog task %s (%s)",
                category_hint,
                catalog_task.name,
                catalog_task.category.value,
            )
        return catalog_task, prompt

    category_value = category_hint or str(task_spec.get("category", TaskCategory.GENERAL_QA.value))
    try:
        category = TaskCategory(category_value)
    except ValueError:
        category = TaskCategory.GENERAL_QA

    complexity_value = str(task_spec.get("complexity", TaskComplexity.MODERATE.value))
    try:
        complexity = TaskComplexity(complexity_value)
    except ValueError:
        complexity = TaskComplexity.MODERATE

    task = TaskDefinition(
        category=category,
        name=task_name,
        prompt_template="{prompt}",
        complexity=complexity,
        max_output_tokens=int(task_spec.get("max_output_tokens", 1024)),
    )
    return task, prompt


def _serialize_result(result: Any) -> dict[str, Any]:
    """Return a JSON-serializable dict from a TaskResult or similar object."""
    if hasattr(result, "model_dump"):
        return result.model_dump(mode="json")
    if hasattr(result, "to_dict"):
        return result.to_dict()
    if isinstance(result, dict):
        return result
    raise TypeError(f"Cannot serialize result of type {type(result)!r}")


async def execute_assignment(
    assignment: TaskAssignment,
    *,
    provider_name: str = "ollama",
    model: str | None = None,
    settings: AtomicsSettings | None = None,
    host: str | None = None,
) -> dict[str, Any]:
    """Execute a single distributed assignment and return a serializable result."""
    settings = settings or load_settings()
    provider = _make_provider(provider_name, model, host, settings)
    task, prompt = _resolve_task_definition(assignment.task_spec)
    logger.info(
        "Executing assignment %s job=%s task=%s",
        assignment.assignment_id,
        assignment.job_id,
        task.name,
    )
    result = await execute_task(
        task,
        prompt,
        provider=provider,
        run_id=assignment.job_id,
        model=model,
    )
    return _serialize_result(result)
