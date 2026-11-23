"""Execute distributed task assignments via the local benchmarking runner."""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import Any

from atomics.config import AtomicsSettings, load_settings
from atomics.core.engine import LoopEngine
from atomics.core.runner import execute_task
from atomics.distributed.models import TaskAssignment
from atomics.models import BurnTier, TaskCategory, TaskComplexity, TaskDefinition
from atomics.providers.base import aclose_providers
from atomics.providers.factory import make_provider, resolve_provider_endpoint
from atomics.storage.repository import MetricsRepository
from atomics.tasks.catalog import TASK_CATALOG
from atomics.validation import (
    AllowedEndpoint,
    is_endpoint_allowed,
    is_metadata_or_link_local_endpoint,
    parse_allowed_endpoint,
)

logger = logging.getLogger("atomics.distributed.worker_runner")


def _compile_worker_allowed_hosts(
    allowed_hosts: set[str] | None,
) -> tuple[AllowedEndpoint, ...] | None:
    if allowed_hosts is None:
        return None
    if not allowed_hosts:
        raise ValueError("allowed_hosts must not be empty when set")
    return tuple(
        parse_allowed_endpoint(item, label="allowed_hosts") for item in sorted(allowed_hosts)
    )


def _check_worker_outbound(
    url: str | None,
    *,
    label: str,
    compiled_allowed: tuple[AllowedEndpoint, ...] | None,
) -> None:
    if url is None:
        return
    if compiled_allowed is not None:
        if not is_endpoint_allowed(url, compiled_allowed, label=label):
            raise PermissionError(
                f"{label} is not permitted by the worker's outbound host allowlist"
            )
        return
    if is_metadata_or_link_local_endpoint(url, label=label):
        raise PermissionError(f"{label} targets a blocked link-local or cloud-metadata address")


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
        dumped = result.model_dump(mode="json")
        if isinstance(dumped, dict):
            return dumped
    if hasattr(result, "to_dict"):
        dumped = result.to_dict()
        if isinstance(dumped, dict):
            return dumped
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
    lock_host: bool = False,
    allowed_hosts: set[str] | None = None,
) -> dict[str, Any]:
    """Execute a single distributed assignment and return a serializable result."""
    compiled_allowed = _compile_worker_allowed_hosts(allowed_hosts)
    spec_host = assignment.task_spec.get("host")
    if lock_host and spec_host:
        raise PermissionError("worker --lock-host forbids caller host overrides")
    effective_host = spec_host if isinstance(spec_host, str) and spec_host else host
    _check_worker_outbound(effective_host, label="host", compiled_allowed=compiled_allowed)
    settings = settings or load_settings()
    pinned_provider, pinned_model = _pinned_execution(assignment.task_spec)
    provider_name = pinned_provider or provider_name
    model = pinned_model or model
    if compiled_allowed is not None:
        eff_resolved = resolve_provider_endpoint(
            provider_name, effective_host, settings, host_label="host"
        )
        _check_worker_outbound(eff_resolved, label="host", compiled_allowed=compiled_allowed)
    provider = make_provider(
        provider_name,
        model,
        effective_host,
        settings,
        # The vllm branch reads its own `vllm_host` rather than the shared
        # `host`, because CLI commands accept --ollama-host and --vllm-host
        # independently and one positional cannot serve both. A worker has a
        # single --host and exactly one provider, so it routes that host to
        # whichever parameter the chosen provider actually reads.
        vllm_host=effective_host if provider_name == "vllm" else None,
    )
    task, prompt = _resolve_task_definition(assignment.task_spec)
    logger.info(
        "Executing assignment %s job=%s task=%s",
        assignment.assignment_id,
        assignment.job_id,
        task.name,
    )
    try:
        result = await execute_task(
            task,
            prompt,
            provider=provider,
            run_id=assignment.job_id,
            model=model,
        )
    finally:
        await aclose_providers(provider)
    return _serialize_result(result)


def _parse_run_request_host(run_request: dict[str, Any], provider_name: str) -> str | None:
    """Return the host override from the run request for the named provider."""
    generic = run_request.get("host")
    if isinstance(generic, str) and generic:
        return generic
    if provider_name == "ollama":
        val = run_request.get("ollama_host")
        return val if isinstance(val, str) and val else None
    if provider_name == "vllm":
        val = run_request.get("vllm_host")
        return val if isinstance(val, str) and val else None
    if provider_name == "brain-gateway":
        val = run_request.get("gateway_url")
        return val if isinstance(val, str) and val else None
    return None


async def execute_full_run(
    assignment: TaskAssignment,
    *,
    provider_name: str = "ollama",
    model: str | None = None,
    host: str | None = None,
    settings: AtomicsSettings | None = None,
    allow_codegen: bool = False,
    lock_host: bool = False,
    allowed_hosts: set[str] | None = None,
) -> dict[str, Any]:
    """Execute a full benchmark or eval run locally and return a serializable result."""
    compiled_allowed = _compile_worker_allowed_hosts(allowed_hosts)
    raw_eval = assignment.task_spec.get("eval_request")
    if raw_eval is not None:
        if not isinstance(raw_eval, dict):
            raise TypeError("eval_request must be a dict")
        if lock_host and (raw_eval.get("host") or raw_eval.get("judge_host")):
            raise PermissionError("worker --lock-host forbids caller host overrides")
        from atomics.api._runners import run_eval_suite
        from atomics.api.models import EvalRequest

        eval_payload = dict(raw_eval)
        eval_payload["provider"] = eval_payload.get("provider") or provider_name
        if not eval_payload.get("model") and model is not None:
            eval_payload["model"] = model
        if not eval_payload.get("host") and host is not None:
            eval_payload["host"] = host
        eval_req = EvalRequest(**eval_payload)
        if eval_req.suite == "codegen" and not allow_codegen:
            raise PermissionError(
                "codegen suite is disabled on this worker; start with "
                "--allow-codegen to enable remote code execution"
            )
        _check_worker_outbound(eval_req.host, label="host", compiled_allowed=compiled_allowed)
        _check_worker_outbound(
            eval_req.judge_host, label="judge_host", compiled_allowed=compiled_allowed
        )
        if compiled_allowed is not None:
            app_settings = settings or load_settings()
            eff_eval_host = resolve_provider_endpoint(
                eval_req.provider, eval_req.host, app_settings, host_label="host"
            )
            _check_worker_outbound(eff_eval_host, label="host", compiled_allowed=compiled_allowed)
            if eval_req.suite != "codegen":
                eff_judge_in = (
                    eval_req.judge_host
                    if eval_req.judge_host is not None
                    else (eval_req.host if eval_req.provider == "ollama" else None)
                )
                eff_judge = resolve_provider_endpoint(
                    "ollama", eff_judge_in, app_settings, host_label="judge_host"
                )
                _check_worker_outbound(
                    eff_judge, label="judge_host", compiled_allowed=compiled_allowed
                )
        eval_out = await run_eval_suite(eval_req)
        eval_out.setdefault("suite", eval_req.suite)
        return eval_out

    settings = settings or load_settings()
    run_request = assignment.task_spec.get("run_request", {})
    if not isinstance(run_request, dict):
        raise TypeError("run_request must be a dict")
    if lock_host and any(
        run_request.get(k) for k in ("host", "ollama_host", "vllm_host", "gateway_url")
    ):
        raise PermissionError("worker --lock-host forbids caller host overrides")

    effective_provider = run_request.get("provider") or provider_name
    effective_model = run_request.get("model") or model
    effective_host = _parse_run_request_host(run_request, effective_provider) or host
    _check_worker_outbound(effective_host, label="host", compiled_allowed=compiled_allowed)
    if compiled_allowed is not None:
        eff_run_host = resolve_provider_endpoint(
            effective_provider, effective_host, settings, host_label="host"
        )
        _check_worker_outbound(eff_run_host, label="host", compiled_allowed=compiled_allowed)

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

    with tempfile.TemporaryDirectory(prefix="atomics-full-") as scratch:
        db_path = Path(scratch) / "run.db"
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

            return {
                "summary": summary.model_dump(mode="json"),
                "task_results": repo.get_run_task_results(summary.run_id),
            }
        finally:
            await aclose_providers(provider)
            repo.close()
