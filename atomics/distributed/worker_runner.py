"""Execute distributed task assignments via the local benchmarking runner."""

from __future__ import annotations

import logging
import tempfile
import uuid
from pathlib import Path
from typing import Any

from atomics.config import AtomicsSettings, load_settings
from atomics.core.engine import LoopEngine
from atomics.core.runner import execute_task
from atomics.distributed.models import TaskAssignment
from atomics.models import BurnTier, TaskCategory, TaskComplexity, TaskDefinition
from atomics.providers.factory import make_provider
from atomics.storage.repository import MetricsRepository
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


def _pinned_execution(task_spec: dict[str, Any]) -> tuple[str | None, str | None]:
    """Return the provider/model this assignment pinned, if any.

    A pinned provider overrides the worker's own default so the submitter's
    routing choice is honored. If the worker cannot build that provider,
    `make_provider` raises and the assignment fails loudly rather than
    silently running somewhere else.
    """
    provider = task_spec.get("provider")
    model = task_spec.get("model")
    return (
        provider if isinstance(provider, str) and provider else None,
        model if isinstance(model, str) and model else None,
    )


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
    pinned_provider, pinned_model = _pinned_execution(assignment.task_spec)
    provider_name = pinned_provider or provider_name
    model = pinned_model or model
    provider = make_provider(
        provider_name,
        model,
        host,
        settings,
        # The vllm branch reads its own `vllm_host` rather than the shared
        # `host`, because CLI commands accept --ollama-host and --vllm-host
        # independently and one positional cannot serve both. A worker has a
        # single --host and exactly one provider, so it routes that host to
        # whichever parameter the chosen provider actually reads.
        vllm_host=host if provider_name == "vllm" else None,
    )
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


def _parse_run_request_host(run_request: dict[str, Any], provider_name: str) -> str | None:
    """Return the host override from the run request for the named provider."""
    if provider_name == "ollama":
        return run_request.get("ollama_host")
    if provider_name == "vllm":
        return run_request.get("vllm_host")
    if provider_name == "brain-gateway":
        return run_request.get("gateway_url")
    return None


async def execute_full_run(
    assignment: TaskAssignment,
    *,
    provider_name: str = "ollama",
    model: str | None = None,
    host: str | None = None,
    settings: AtomicsSettings | None = None,
) -> dict[str, Any]:
    """Execute a full benchmark run locally and return a serializable result."""
    settings = settings or load_settings()
    run_request = assignment.task_spec.get("run_request", {})
    if not isinstance(run_request, dict):
        raise TypeError("run_request must be a dict")

    effective_provider = run_request.get("provider") or provider_name
    effective_model = run_request.get("model") or model
    effective_host = _parse_run_request_host(run_request, effective_provider) or host

    provider = make_provider(
        effective_provider,
        effective_model,
        effective_host,
        settings,
        vllm_host=effective_host if effective_provider == "vllm" else None,
    )

    tier_value = run_request.get("tier", "baseline")
    try:
        tier = BurnTier(str(tier_value))
    except ValueError:
        tier = BurnTier.BASELINE

    iterations = int(run_request.get("iterations", 1))
    budget = run_request.get("budget")
    interval = run_request.get("interval")
    thinking = run_request.get("thinking")
    thinking_budget = run_request.get("thinking_budget")

    db_path = Path(tempfile.gettempdir()) / f"atomics-full-{uuid.uuid4().hex}.db"
    repo = MetricsRepository(db_path)
    try:
        engine = LoopEngine(
            provider,
            repo,
            settings,
            tier=tier,
            budget_override=budget if isinstance(budget, float | int) else None,
            interval_override=interval if isinstance(interval, int) else None,
            model_override=effective_model,
            trigger="distributed",
            thinking=thinking if isinstance(thinking, bool) else None,
            thinking_budget=thinking_budget if isinstance(thinking_budget, int) else None,
        )
        summary = await engine.run(max_iterations=iterations)
        if summary is None:
            raise RuntimeError("Full run produced no summary")

        cursor = repo._conn.execute(
            "SELECT task_id, run_id, category, task_name, provider, model, status, "
            "suite, prompt, response, input_tokens, output_tokens, total_tokens, "
            "thinking_tokens, cache_read_tokens, cache_write_tokens, latency_ms, "
            "estimated_cost_usd, tokens_per_second, tps_basis, error_class, error_message "
            "FROM task_results WHERE run_id = ?",
            (summary.run_id,),
        )
        columns = [desc[0] for desc in cursor.description]
        task_results = [dict(zip(columns, row)) for row in cursor.fetchall()]

        return {
            "summary": summary.model_dump(mode="json"),
            "task_results": task_results,
        }
    finally:
        repo.close()
        try:
            db_path.unlink()
        except FileNotFoundError:
            pass
