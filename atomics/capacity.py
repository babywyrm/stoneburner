"""Capacity projection simulator.

Takes stress test data (local GPU or cloud API latency measurements) and
projects how the system behaves under different user load patterns using
Little's Law and linear interpolation from measured data points.

Answers the question: "Can my gpu-host / cloud endpoint handle N users?"
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class LoadProfile:
    users: int
    think_time_s: float = 300.0
    response_tokens: int = 400
    burst_factor: float = 0.2


@dataclass
class CapacityScenario:
    name: str
    concurrent: float
    p50_latency_ms: float
    p95_latency_ms: float
    queue_depth: float
    verdict: str


@dataclass
class CapacityProjection:
    model: str
    peak_tps: float
    profile: LoadProfile
    scenarios: list[CapacityScenario] = field(default_factory=list)
    recommendation: str = ""


def estimate_concurrency(
    users: int, think_time_s: float, avg_response_s: float,
) -> float:
    """Fraction of users actively waiting for a response at any moment.

    Each user cycles: send request → wait response_time → think for think_time → repeat.
    P(active) = response_time / (response_time + think_time).
    """
    if users == 0 or think_time_s <= 0:
        return 0.0
    cycle_time = avg_response_s + think_time_s
    return users * (avg_response_s / cycle_time)


def interpolate_latency(
    concurrency: float,
    phases: list[dict],
    percentile: str = "p50",
) -> float:
    """Interpolate latency from stress test phase data.

    Uses linear interpolation between measured concurrency levels.
    Beyond the max measured point, models latency as scaling roughly
    linearly with concurrency (queuing delay dominates).
    """
    key = "avg_latency_ms" if percentile == "p50" else "p95_latency_ms"
    sorted_phases = sorted(phases, key=lambda p: p["concurrency"])

    if not sorted_phases:
        return 0.0

    if concurrency <= sorted_phases[0]["concurrency"]:
        return sorted_phases[0][key]

    for i in range(len(sorted_phases) - 1):
        lo = sorted_phases[i]
        hi = sorted_phases[i + 1]
        if lo["concurrency"] <= concurrency <= hi["concurrency"]:
            t = (concurrency - lo["concurrency"]) / (hi["concurrency"] - lo["concurrency"])
            return lo[key] + t * (hi[key] - lo[key])

    last = sorted_phases[-1]
    max_conc = last["concurrency"]
    base_lat = last[key]
    ratio = concurrency / max_conc
    return base_lat * ratio


def _solve_steady_state(
    users: float, think_time_s: float, phases: list[dict],
) -> float:
    """Find steady-state concurrency via fixed-point iteration.

    At equilibrium: c = users * lat(c) / (lat(c) + think_time_ms)
    This is bounded [0, users] and monotonically convergent.
    """
    if users <= 0:
        return 0.0
    think_ms = think_time_s * 1000.0

    lo, hi = 0.0, float(users)
    for _ in range(50):
        mid = (lo + hi) / 2.0
        lat = interpolate_latency(max(mid, 0.5), phases)
        expected_c = users * lat / (lat + think_ms)
        if expected_c > mid:
            lo = mid
        else:
            hi = mid
        if hi - lo < 0.01:
            break
    return round((lo + hi) / 2.0, 1)


def _verdict(p50_ms: float, peak_tps: float, concurrent: float) -> str:
    """Classify load scenario severity."""
    if p50_ms < 30_000:
        return "OK"
    if p50_ms < 60_000:
        return "CAUTION"
    if p50_ms < 120_000:
        return "SLOW"
    return "OVERLOAD"


def _estimate_response_time_s(
    response_tokens: int, peak_tps: float, phases: list[dict], concurrent: float,
) -> float:
    """Estimate per-request response time at a given concurrency level."""
    if not phases:
        return response_tokens / max(peak_tps, 1) if peak_tps else 10.0
    base_lat_ms = interpolate_latency(concurrent, phases)
    return base_lat_ms / 1000.0


def project_capacity(
    *,
    profile: LoadProfile,
    phases: list[dict],
    peak_tps: float,
    model: str = "",
) -> CapacityProjection:
    """Project capacity across multiple load scenarios.

    Uses iterative estimation: concurrency depends on response time, which
    depends on concurrency. Converges in a few iterations.
    """
    projection = CapacityProjection(
        model=model, peak_tps=peak_tps, profile=profile,
    )

    scenarios_spec = [
        ("Normal", 1.0),
        ("Peak hour (1.5x)", 1.5),
        (f"Burst ({profile.burst_factor:.0%} spike)", 1.0 + profile.burst_factor * profile.users / max(profile.users, 1) * 3),
        ("Monday morning (2x)", 2.0),
    ]

    sorted_phases = sorted(phases, key=lambda p: p["concurrency"])
    max_measured_conc = sorted_phases[-1]["concurrency"] if sorted_phases else 1

    for name, load_mult in scenarios_spec:
        effective_users = profile.users * load_mult

        concurrent = _solve_steady_state(
            effective_users, profile.think_time_s, phases,
        )

        p50 = interpolate_latency(max(concurrent, 0.5), phases)
        p95 = interpolate_latency(max(concurrent, 0.5), phases, percentile="p95")

        queue_depth = max(0, concurrent - max_measured_conc)

        verdict = _verdict(p50, peak_tps, concurrent)

        projection.scenarios.append(CapacityScenario(
            name=name,
            concurrent=round(concurrent, 1),
            p50_latency_ms=round(p50, 0),
            p95_latency_ms=round(p95, 0),
            queue_depth=round(queue_depth, 1),
            verdict=verdict,
        ))

    worst = max(projection.scenarios, key=lambda s: s.p50_latency_ms)
    best_normal = next((s for s in projection.scenarios if s.name == "Normal"), projection.scenarios[0])

    if best_normal.verdict == "OK":
        projection.recommendation = (
            f"{profile.users} users at {profile.think_time_s:.0f}s cadence is within capacity. "
            f"Normal P50 latency: {best_normal.p50_latency_ms / 1000:.0f}s."
        )
    elif best_normal.verdict == "CAUTION":
        projection.recommendation = (
            f"{profile.users} users is feasible but tight. Consider a smaller/faster model "
            f"or adding a second GPU to keep P50 under 30s."
        )
    elif best_normal.verdict == "SLOW":
        projection.recommendation = (
            f"{profile.users} users will experience slow responses (P50 ~{best_normal.p50_latency_ms / 1000:.0f}s). "
            f"Add a second GPU or switch to a smaller model."
        )
    else:
        projection.recommendation = (
            f"{profile.users} users will overload this setup. "
            f"Need multiple GPUs or a cloud tier with higher throughput."
        )

    if worst.verdict == "OVERLOAD" and best_normal.verdict != "OVERLOAD":
        projection.recommendation += (
            f" Burst scenarios ({worst.name}) will cause significant queuing."
        )

    return projection
