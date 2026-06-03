"""Multi-model VRAM contention testing.

Runs two or more models simultaneously against an Ollama host and measures
how each degrades under shared GPU memory pressure.

Usage:
    atomics stress --models qwen2.5:3b,qwen2.5:7b --ollama-host http://gpu:11434
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field

import httpx


@dataclass
class ContentionModelResult:
    model: str
    requests: int = 0
    failed: int = 0
    total_output_tokens: int = 0
    latencies: list[float] = field(default_factory=list)
    per_request_tps: list[float] = field(default_factory=list)

    @property
    def avg_tps(self) -> float:
        if not self.per_request_tps:
            return 0.0
        return sum(self.per_request_tps) / len(self.per_request_tps)

    @property
    def p95_ms(self) -> float:
        return _percentile(sorted(self.latencies), 95)

    @property
    def avg_latency_ms(self) -> float:
        if not self.latencies:
            return 0.0
        return sum(self.latencies) / len(self.latencies)

    @property
    def error_rate(self) -> float:
        total = self.requests + self.failed
        return self.failed / total if total > 0 else 0.0


@dataclass
class ContentionResult:
    host: str
    models: list[str]
    phase_seconds: float
    solo_tps: dict[str, float] = field(default_factory=dict)
    contention_results: list[ContentionModelResult] = field(default_factory=list)
    duration_seconds: float = 0.0

    def contention_factor(self, model: str) -> float | None:
        """Ratio of mixed/solo avg_tps. <1.0 means degradation."""
        solo = self.solo_tps.get(model)
        if not solo:
            return None
        mixed = next((r.avg_tps for r in self.contention_results if r.model == model), None)
        if mixed is None or solo == 0:
            return None
        return round(mixed / solo, 3)


def _percentile(sorted_values: list[float], pct: float) -> float:
    if not sorted_values:
        return 0.0
    k = (len(sorted_values) - 1) * (pct / 100)
    f = int(k)
    c = f + 1
    if c >= len(sorted_values):
        return sorted_values[f]
    return sorted_values[f] + (k - f) * (sorted_values[c] - sorted_values[f])


async def _run_model_phase(
    client: httpx.AsyncClient,
    host: str,
    model: str,
    concurrency: int,
    duration_seconds: float,
    num_predict: int,
) -> ContentionModelResult:
    """Run one model at fixed concurrency for a fixed duration."""
    from atomics.stress import STRESS_PROMPTS, _single_request

    result = ContentionModelResult(model=model)
    start = time.monotonic()
    prompt_idx = 0

    async def _worker() -> None:
        nonlocal prompt_idx
        while time.monotonic() - start < duration_seconds:
            prompt = STRESS_PROMPTS[prompt_idx % len(STRESS_PROMPTS)]
            prompt_idx += 1
            try:
                out, _inp, lat, tps = await _single_request(
                    client, host, model, prompt, num_predict
                )
                result.requests += 1
                result.total_output_tokens += out
                result.latencies.append(lat)
                result.per_request_tps.append(tps)
            except Exception:
                result.failed += 1

    workers = [asyncio.create_task(_worker()) for _ in range(concurrency)]
    await asyncio.gather(*workers)
    return result


async def run_contention(
    host: str,
    models: list[str],
    concurrency: int = 1,
    phase_seconds: float = 20.0,
    num_predict: int = 512,
) -> ContentionResult:
    """Run all models concurrently and measure VRAM contention.

    Phase 1: Each model runs solo to establish a baseline TPS.
    Phase 2: All models run simultaneously to measure contention degradation.
    """
    result = ContentionResult(host=host, models=list(models), phase_seconds=phase_seconds)
    t0 = time.monotonic()

    async with httpx.AsyncClient(timeout=httpx.Timeout(300.0)) as client:
        # Phase 1: solo baselines
        for model in models:
            solo = await _run_model_phase(
                client, host, model, concurrency, phase_seconds, num_predict
            )
            result.solo_tps[model] = round(solo.avg_tps, 2)

        # Phase 2: all models simultaneously
        tasks = [
            _run_model_phase(client, host, model, concurrency, phase_seconds, num_predict)
            for model in models
        ]
        contention_results = await asyncio.gather(*tasks)
        result.contention_results = list(contention_results)

    result.duration_seconds = time.monotonic() - t0
    return result
