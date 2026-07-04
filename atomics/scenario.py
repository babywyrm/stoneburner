"""Mixed-workload scenario runner.

Orchestrates multiple concurrent workload profiles against a shared Ollama
host, measures per-workload latency and SLA compliance, and computes
cross-workload interference scores.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from typing import TYPE_CHECKING

import httpx

from atomics.scenario_models import (
    BASELINE_DURATION_SECONDS,
    ScenarioResult,
    WorkloadResult,
    WorkloadSpec,
)
from atomics.scenario_prompts import resolve_prompts
from atomics.stress import _single_request

if TYPE_CHECKING:
    from atomics.profiles import TargetProfile


async def _run_workload(
    client: httpx.AsyncClient,
    host: str,
    spec: WorkloadSpec,
    duration_seconds: float,
    loaded_profile: TargetProfile | None = None,
    ramp_seconds: float = 0.0,
) -> WorkloadResult:
    """Run a single workload at its specified concurrency for a fixed duration.

    When ``ramp_seconds`` > 0, workers are staggered across the ramp window
    so load builds gradually rather than all starting simultaneously.
    """
    result = WorkloadResult(spec=spec)
    prompts = spec.prompts
    start = time.monotonic()
    prompt_idx = 0

    async def _worker(delay: float) -> None:
        nonlocal prompt_idx
        if delay > 0:
            await asyncio.sleep(delay)
        while time.monotonic() - start < duration_seconds:
            prompt = prompts[prompt_idx % len(prompts)]
            prompt_idx += 1
            try:
                if loaded_profile is not None:
                    from atomics.profiles import _single_request_profile
                    _text, lat, _cls = await _single_request_profile(
                        client, loaded_profile, prompt,
                    )
                    result.requests += 1
                    result.latencies.append(lat)
                else:
                    out, inp, lat, tps = await _single_request(
                        client, host, spec.model, prompt, spec.num_predict,
                    )
                    result.requests += 1
                    result.total_output_tokens += out
                    result.latencies.append(lat)
                    result.per_request_tps.append(tps)
            except Exception:
                result.failed += 1

    concurrency = spec.concurrency
    if ramp_seconds > 0 and concurrency > 1:
        step = ramp_seconds / concurrency
        delays = [i * step for i in range(concurrency)]
    else:
        delays = [0.0] * concurrency

    workers = [asyncio.create_task(_worker(delays[i])) for i in range(concurrency)]
    await asyncio.gather(*workers)
    return result


async def _run_baseline(
    client: httpx.AsyncClient,
    host: str,
    spec: WorkloadSpec,
    loaded_profile: TargetProfile | None = None,
) -> float:
    """Run a workload solo for a short period, return its P50 latency."""
    result = await _run_workload(client, host, spec, BASELINE_DURATION_SECONDS, loaded_profile)
    return result.p50_ms


async def run_scenario(
    host: str,
    specs: list[WorkloadSpec],
    duration_seconds: float = 60.0,
    ramp_seconds: float = 0.0,
    skip_baseline: bool = False,
    on_baseline_done: Callable[[str, float], None] | None = None,
    on_workload_done: Callable[[WorkloadResult], None] | None = None,
) -> ScenarioResult:
    """Run a mixed-workload scenario against a shared Ollama host.

    1. Resolve prompts for each workload
    2. (Optional) Run solo baselines for interference scoring
    3. Run all workloads concurrently
    4. Compute interference factors
    """
    loaded_profiles: dict[str, object] = {}
    for spec in specs:
        if spec.profile:
            from atomics.profiles import load_profile
            tp = load_profile(spec.profile)
            loaded_profiles[spec.name] = tp
            if not spec.prompts and tp.prompts:
                spec.prompts = tp.prompts
            if not spec.model and tp.model:
                spec.model = tp.model
        if not spec.prompts:
            spec.prompts = resolve_prompts(spec.type, spec.prompts_file)

    scenario = ScenarioResult()

    async with httpx.AsyncClient() as client:
        if not skip_baseline:
            for spec in specs:
                lp = loaded_profiles.get(spec.name)
                baseline_p50 = await _run_baseline(client, host, spec, lp)
                scenario.baselines[spec.name] = baseline_p50
                if on_baseline_done:
                    on_baseline_done(spec.name, baseline_p50)

        tasks = [
            _run_workload(client, host, spec, duration_seconds,
                          loaded_profiles.get(spec.name), ramp_seconds)
            for spec in specs
        ]
        results = await asyncio.gather(*tasks)

    for wr in results:
        scenario.workloads.append(wr)
        scenario.total_requests += wr.requests
        scenario.total_failed += wr.failed

        if wr.spec.name in scenario.baselines:
            solo_p50 = scenario.baselines[wr.spec.name]
            mixed_p50 = wr.p50_ms
            if solo_p50 > 0:
                scenario.interference[wr.spec.name] = mixed_p50 / solo_p50
            else:
                scenario.interference[wr.spec.name] = 1.0

        if on_workload_done:
            on_workload_done(wr)

    scenario.duration_seconds = duration_seconds
    scenario.ramp_seconds = ramp_seconds
    return scenario
