"""Long-duration stability test — hold fixed concurrency and track degradation."""

from __future__ import annotations

import asyncio
import re
import time
from collections.abc import Callable
from dataclasses import dataclass, field

from atomics.stats import percentile as _percentile


def parse_duration(s: str) -> float:
    """Parse a human-friendly duration string into seconds.

    Accepts: '30s', '30m', '2h', '1h30m', '1h30m20s', '90' (bare number = minutes).
    """
    s = s.strip().lower()
    if not s:
        raise ValueError("Empty duration string")

    # Bare integer → minutes
    m = re.fullmatch(r"(\d+)", s)
    if m:
        return int(m.group(1)) * 60

    # Full pattern: optional h, optional m, optional s
    pattern = re.fullmatch(r"(?:(\d+)h)?(?:(\d+)m)?(?:(\d+)s)?", s)
    if not pattern or all(pattern.group(i) is None for i in (1, 2, 3)):
        raise ValueError(f"Invalid duration: {s!r}. Use e.g. '30s', '30m', '2h', '1h30m'.")

    hours = int(pattern.group(1) or 0)
    minutes = int(pattern.group(2) or 0)
    seconds = int(pattern.group(3) or 0)
    total = hours * 3600 + minutes * 60 + seconds
    if total == 0:
        raise ValueError(f"Duration must be > 0: {s!r}")
    return float(total)


def _linear_slope(xs: list[float], ys: list[float]) -> float:
    """Least-squares slope. Pure Python, no numpy."""
    n = len(xs)
    if n < 2:
        return 0.0
    x_mean = sum(xs) / n
    y_mean = sum(ys) / n
    num = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys))
    den = sum((x - x_mean) ** 2 for x in xs)
    return num / den if den != 0 else 0.0


def _drift_pct(values: list[float]) -> float:
    """Compute drift percentage over a time series using linear regression."""
    if len(values) < 2:
        return 0.0
    xs = [float(i) for i in range(len(values))]
    slope = _linear_slope(xs, values)
    first = values[0]
    if first == 0:
        return 0.0
    projected_change = slope * (len(values) - 1)
    return (projected_change / first) * 100


def _compute_verdict(
    throughput_drift_pct: float,
    latency_drift_pct: float,
    error_rate: float,
) -> str:
    """Classify soak result as STABLE, DEGRADED, or UNSTABLE."""
    if (
        throughput_drift_pct <= -15
        or latency_drift_pct >= 25
        or error_rate >= 0.05
    ):
        return "UNSTABLE"
    if (
        throughput_drift_pct <= -5
        or latency_drift_pct >= 10
        or error_rate >= 0.005
    ):
        return "DEGRADED"
    return "STABLE"


@dataclass
class SoakSample:
    """One metric snapshot taken every sample_interval seconds."""

    elapsed_seconds: float = 0.0
    requests: int = 0
    failed: int = 0
    total_output_tokens: int = 0
    aggregate_tps: float = 0.0
    avg_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    vram_used_mb: float | None = None


@dataclass
class SoakResult:
    """Complete result of a soak test run."""

    model: str = ""
    host: str = ""
    provider: str = "ollama"
    concurrency: int = 4
    duration_seconds: float = 0.0
    actual_duration_seconds: float = 0.0
    sample_interval: int = 30
    total_requests: int = 0
    total_failed: int = 0
    total_tokens: int = 0
    samples: list[SoakSample] = field(default_factory=list)

    throughput_drift_pct: float = 0.0
    latency_drift_pct: float = 0.0
    vram_start_mb: float | None = None
    vram_end_mb: float | None = None
    vram_drift_mb: float | None = None

    avg_tps: float = 0.0
    peak_tps: float = 0.0
    min_tps: float = 0.0
    avg_p95_ms: float = 0.0
    error_rate: float = 0.0
    verdict: str = "STABLE"
    total_cost_usd: float = 0.0




async def run_soak(
    host: str,
    model: str,
    concurrency: int = 4,
    duration_seconds: float = 1800,
    sample_interval: int = 30,
    num_predict: int = 2048,
    think_time_seconds: float = 0.0,
    on_sample: Callable[[SoakSample], None] | None = None,
) -> SoakResult:
    """Run a soak test against an Ollama endpoint."""
    import httpx

    from atomics.stress import STRESS_PROMPTS, _get_vram_used_mb, _single_request

    result = SoakResult(
        model=model,
        host=host,
        provider="ollama",
        concurrency=concurrency,
        duration_seconds=duration_seconds,
        sample_interval=sample_interval,
    )

    window_latencies: list[float] = []
    window_tokens: int = 0
    window_requests: int = 0
    window_failed: int = 0
    window_lock = asyncio.Lock()

    stop_event = asyncio.Event()
    t0 = time.monotonic()

    result.vram_start_mb = _get_vram_used_mb()

    async def _worker(client: httpx.AsyncClient, worker_id: int) -> None:
        nonlocal window_latencies, window_tokens, window_requests, window_failed
        prompt_idx = worker_id
        while not stop_event.is_set():
            prompt = STRESS_PROMPTS[prompt_idx % len(STRESS_PROMPTS)]
            prompt_idx += concurrency
            try:
                out_tok, _in_tok, lat_ms, _tps = await _single_request(
                    client, host, model, prompt, num_predict
                )
                async with window_lock:
                    window_latencies.append(lat_ms)
                    window_tokens += out_tok
                    window_requests += 1
                if think_time_seconds > 0 and not stop_event.is_set():
                    await asyncio.sleep(think_time_seconds)
            except Exception:
                async with window_lock:
                    window_failed += 1
                    window_requests += 1

    async def _sampler() -> None:
        nonlocal window_latencies, window_tokens, window_requests, window_failed
        while not stop_event.is_set():
            await asyncio.sleep(sample_interval)
            if stop_event.is_set():
                break

            elapsed = time.monotonic() - t0
            async with window_lock:
                lats = window_latencies[:]
                tokens = window_tokens
                reqs = window_requests
                fails = window_failed
                window_latencies = []
                window_tokens = 0
                window_requests = 0
                window_failed = 0

            tps = tokens / sample_interval if sample_interval > 0 else 0.0
            vram = _get_vram_used_mb()

            sample = SoakSample(
                elapsed_seconds=round(elapsed, 1),
                requests=reqs,
                failed=fails,
                total_output_tokens=tokens,
                aggregate_tps=round(tps, 2),
                avg_latency_ms=round(sum(lats) / len(lats), 2) if lats else 0.0,
                p95_latency_ms=round(_percentile(lats, 95), 2),
                vram_used_mb=vram,
            )
            result.samples.append(sample)
            result.total_requests += reqs
            result.total_failed += fails
            result.total_tokens += tokens

            if on_sample:
                on_sample(sample)

    async with httpx.AsyncClient(timeout=httpx.Timeout(120.0)) as client:
        workers = [asyncio.create_task(_worker(client, i)) for i in range(concurrency)]
        sampler_task = asyncio.create_task(_sampler())

        await asyncio.sleep(duration_seconds)
        stop_event.set()

        for w in workers:
            w.cancel()
        sampler_task.cancel()

        await asyncio.gather(*workers, sampler_task, return_exceptions=True)

    result.actual_duration_seconds = round(time.monotonic() - t0, 2)
    result.vram_end_mb = _get_vram_used_mb()

    if result.vram_start_mb is not None and result.vram_end_mb is not None:
        result.vram_drift_mb = round(result.vram_end_mb - result.vram_start_mb, 1)

    if result.samples:
        tps_values = [s.aggregate_tps for s in result.samples]
        p95_values = [s.p95_latency_ms for s in result.samples]
        result.throughput_drift_pct = round(_drift_pct(tps_values), 2)
        result.latency_drift_pct = round(_drift_pct(p95_values), 2)
        result.avg_tps = round(sum(tps_values) / len(tps_values), 2)
        result.peak_tps = round(max(tps_values), 2)
        result.min_tps = round(min(tps_values), 2)
        result.avg_p95_ms = round(sum(p95_values) / len(p95_values), 2)

    result.error_rate = (
        result.total_failed / result.total_requests
        if result.total_requests > 0
        else 0.0
    )
    result.verdict = _compute_verdict(
        result.throughput_drift_pct,
        result.latency_drift_pct,
        result.error_rate,
    )

    return result


async def run_soak_provider(
    provider: object,
    model: str = "",
    concurrency: int = 4,
    duration_seconds: float = 1800,
    sample_interval: int = 30,
    num_predict: int = 2048,
    think_time_seconds: float = 0.0,
    on_sample: Callable[[SoakSample], None] | None = None,
) -> SoakResult:
    """Run a soak test against any provider (cloud or local)."""
    from atomics.stress import STRESS_PROMPTS, _single_request_provider

    result = SoakResult(
        model=model,
        host=getattr(provider, "host", "") or getattr(provider, "url", ""),
        provider=getattr(provider, "name", "unknown"),
        concurrency=concurrency,
        duration_seconds=duration_seconds,
        sample_interval=sample_interval,
    )

    window_latencies: list[float] = []
    window_tokens: int = 0
    window_requests: int = 0
    window_failed: int = 0
    window_cost: float = 0.0
    window_lock = asyncio.Lock()

    stop_event = asyncio.Event()
    t0 = time.monotonic()

    async def _worker(worker_id: int) -> None:
        nonlocal window_latencies, window_tokens, window_requests, window_failed, window_cost
        prompt_idx = worker_id
        while not stop_event.is_set():
            prompt = STRESS_PROMPTS[prompt_idx % len(STRESS_PROMPTS)]
            prompt_idx += concurrency
            try:
                out_tok, _in_tok, lat_ms, _tps, cost = await _single_request_provider(
                    provider, prompt, num_predict
                )
                async with window_lock:
                    window_latencies.append(lat_ms)
                    window_tokens += out_tok
                    window_requests += 1
                    window_cost += cost
                if think_time_seconds > 0 and not stop_event.is_set():
                    await asyncio.sleep(think_time_seconds)
            except Exception:
                async with window_lock:
                    window_failed += 1
                    window_requests += 1

    async def _sampler() -> None:
        nonlocal window_latencies, window_tokens, window_requests, window_failed, window_cost
        while not stop_event.is_set():
            await asyncio.sleep(sample_interval)
            if stop_event.is_set():
                break

            elapsed = time.monotonic() - t0
            async with window_lock:
                lats = window_latencies[:]
                tokens = window_tokens
                reqs = window_requests
                fails = window_failed
                cost = window_cost
                window_latencies = []
                window_tokens = 0
                window_requests = 0
                window_failed = 0
                window_cost = 0.0

            tps = tokens / sample_interval if sample_interval > 0 else 0.0

            sample = SoakSample(
                elapsed_seconds=round(elapsed, 1),
                requests=reqs,
                failed=fails,
                total_output_tokens=tokens,
                aggregate_tps=round(tps, 2),
                avg_latency_ms=round(sum(lats) / len(lats), 2) if lats else 0.0,
                p95_latency_ms=round(_percentile(lats, 95), 2),
            )
            result.samples.append(sample)
            result.total_requests += reqs
            result.total_failed += fails
            result.total_tokens += tokens
            result.total_cost_usd += cost

            if on_sample:
                on_sample(sample)

    workers = [asyncio.create_task(_worker(i)) for i in range(concurrency)]
    sampler_task = asyncio.create_task(_sampler())

    await asyncio.sleep(duration_seconds)
    stop_event.set()

    for w in workers:
        w.cancel()
    sampler_task.cancel()

    await asyncio.gather(*workers, sampler_task, return_exceptions=True)

    result.actual_duration_seconds = round(time.monotonic() - t0, 2)

    if result.samples:
        tps_values = [s.aggregate_tps for s in result.samples]
        p95_values = [s.p95_latency_ms for s in result.samples]
        result.throughput_drift_pct = round(_drift_pct(tps_values), 2)
        result.latency_drift_pct = round(_drift_pct(p95_values), 2)
        result.avg_tps = round(sum(tps_values) / len(tps_values), 2)
        result.peak_tps = round(max(tps_values), 2)
        result.min_tps = round(min(tps_values), 2)
        result.avg_p95_ms = round(sum(p95_values) / len(p95_values), 2)

    result.error_rate = (
        result.total_failed / result.total_requests
        if result.total_requests > 0
        else 0.0
    )
    result.verdict = _compute_verdict(
        result.throughput_drift_pct,
        result.latency_drift_pct,
        result.error_rate,
    )

    return result


async def run_soak_profile(
    profile: object,
    concurrency: int = 4,
    duration_seconds: float = 1800,
    sample_interval: int = 30,
    think_time_seconds: float = 0.0,
    on_sample: Callable[[SoakSample], None] | None = None,
) -> SoakResult:
    """Run a soak test against a custom target profile (ollama or http)."""
    import httpx

    from atomics.profiles import TargetProfile, _single_request_profile

    tp: TargetProfile = profile  # type: ignore[assignment]

    prompts = tp.prompts
    if not prompts:
        from atomics.stress import STRESS_PROMPTS
        prompts = list(STRESS_PROMPTS)

    host = tp.ollama_host if tp.type == "ollama" else tp.http_url
    result = SoakResult(
        model=tp.model,
        host=host,
        provider=f"profile:{tp.type}",
        concurrency=concurrency,
        duration_seconds=duration_seconds,
        sample_interval=sample_interval,
    )

    window_latencies: list[float] = []
    window_requests: int = 0
    window_failed: int = 0
    window_lock = asyncio.Lock()

    stop_event = asyncio.Event()
    t0 = time.monotonic()

    async def _worker(client: httpx.AsyncClient, worker_id: int) -> None:
        nonlocal window_latencies, window_requests, window_failed
        prompt_idx = worker_id
        while not stop_event.is_set():
            prompt = prompts[prompt_idx % len(prompts)]
            prompt_idx += concurrency
            try:
                _text, lat_ms, _cls = await _single_request_profile(
                    client, tp, prompt
                )
                async with window_lock:
                    window_latencies.append(lat_ms)
                    window_requests += 1
                if think_time_seconds > 0 and not stop_event.is_set():
                    await asyncio.sleep(think_time_seconds)
            except Exception:
                async with window_lock:
                    window_failed += 1
                    window_requests += 1

    async def _sampler() -> None:
        nonlocal window_latencies, window_requests, window_failed
        while not stop_event.is_set():
            await asyncio.sleep(sample_interval)
            if stop_event.is_set():
                break

            elapsed = time.monotonic() - t0
            async with window_lock:
                lats = window_latencies[:]
                reqs = window_requests
                fails = window_failed
                window_latencies = []
                window_requests = 0
                window_failed = 0

            rps = reqs / sample_interval if sample_interval > 0 else 0.0

            sample = SoakSample(
                elapsed_seconds=round(elapsed, 1),
                requests=reqs,
                failed=fails,
                total_output_tokens=0,
                aggregate_tps=round(rps, 2),
                avg_latency_ms=round(sum(lats) / len(lats), 2) if lats else 0.0,
                p95_latency_ms=round(_percentile(lats, 95), 2),
            )
            result.samples.append(sample)
            result.total_requests += reqs
            result.total_failed += fails

            if on_sample:
                on_sample(sample)

    timeout = max(float(tp.http_timeout), 120.0) if tp.type == "http" else 300.0
    async with httpx.AsyncClient(timeout=httpx.Timeout(timeout)) as client:
        workers_list = [asyncio.create_task(_worker(client, i)) for i in range(concurrency)]
        sampler_task = asyncio.create_task(_sampler())

        await asyncio.sleep(duration_seconds)
        stop_event.set()

        for w in workers_list:
            w.cancel()
        sampler_task.cancel()

        await asyncio.gather(*workers_list, sampler_task, return_exceptions=True)

    result.actual_duration_seconds = round(time.monotonic() - t0, 2)

    if result.samples:
        tps_values = [s.aggregate_tps for s in result.samples]
        p95_values = [s.p95_latency_ms for s in result.samples]
        result.throughput_drift_pct = round(_drift_pct(tps_values), 2)
        result.latency_drift_pct = round(_drift_pct(p95_values), 2)
        result.avg_tps = round(sum(tps_values) / len(tps_values), 2)
        result.peak_tps = round(max(tps_values), 2)
        result.min_tps = round(min(tps_values), 2)
        result.avg_p95_ms = round(sum(p95_values) / len(p95_values), 2)

    result.error_rate = (
        result.total_failed / result.total_requests
        if result.total_requests > 0
        else 0.0
    )
    result.verdict = _compute_verdict(
        result.throughput_drift_pct,
        result.latency_drift_pct,
        result.error_rate,
    )

    return result
