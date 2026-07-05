"""LabCompare — compare two+ inference hosts on throughput and quality parity.

Additive module: imports existing providers/runners/judge as libraries and
never modifies them. Persists only to the labcompare_results table.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger("atomics.labcompare")


@dataclass(frozen=True)
class HostSpec:
    name: str
    url: str


def parse_host_specs(raw: list[str]) -> list[HostSpec]:
    """Parse ``NAME=URL`` strings into HostSpec objects.

    Raises ValueError on malformed input so the CLI can surface a clear error.
    """
    specs: list[HostSpec] = []
    for item in raw:
        if "=" not in item:
            raise ValueError(f"bad --host '{item}': expected NAME=URL")
        name, url = item.split("=", 1)
        name = name.strip()
        url = url.strip()
        if not name:
            raise ValueError(f"bad --host '{item}': empty host name")
        if not url:
            raise ValueError(f"bad --host '{item}': empty url")
        specs.append(HostSpec(name=name, url=url))
    return specs


def vram_fit_from_ps(
    ps_payload: dict, model: str
) -> tuple[float | None, str | None]:
    """Compute VRAM fit ratio for a model from an Ollama /api/ps payload.

    Returns (fit_pct, gpu_name). fit_pct is size_vram/size in [0,1]; 1.0 means
    fully in GPU, <1.0 means CPU offload. Returns (None, None) when the model
    is not loaded or size is unavailable.
    """
    for entry in ps_payload.get("models", []):
        if entry.get("name") != model:
            continue
        size = entry.get("size", 0)
        size_vram = entry.get("size_vram", 0)
        if not size:
            return None, None
        gpu = None
        details = entry.get("details") or {}
        if isinstance(details, dict):
            gpu = details.get("family") or None
        return round(size_vram / size, 4), gpu
    return None, None


def speedup_ratio(fast: float | None, baseline: float | None) -> float | None:
    """How many times faster `fast` is than `baseline` (1 dp). None if unknown."""
    if fast is None or baseline is None or baseline == 0:
        return None
    return round(fast / baseline, 1)


def parity_verdict(
    a: float | None, b: float | None, *, tolerance: float = 0.05
) -> tuple[bool | None, float | None]:
    """Quality parity between two scores. Returns (is_parity, abs_delta).

    (None, None) when either score is missing.
    """
    if a is None or b is None:
        return None, None
    delta = round(abs(a - b), 4)
    return (delta <= tolerance), delta


@dataclass
class ThroughputResult:
    tokens_per_second: float | None
    latency_ms: float | None
    prompt_eval_rate: float | None
    vram_fit_pct: float | None
    gpu_name: str | None


async def probe_throughput(
    provider,
    model: str,
    *,
    ps_fetcher,
    prompts: list[str],
) -> ThroughputResult:
    """Run fixed prompts through provider.generate and average the metrics.

    `ps_fetcher` is an async callable returning an Ollama /api/ps payload; on
    failure VRAM fit is reported as None (throughput still returned).
    """
    tps_vals: list[float] = []
    lat_vals: list[float] = []
    peval_vals: list[float] = []
    for prompt in prompts:
        resp = await provider.generate(prompt, model=model, max_tokens=256)
        if resp.tokens_per_second:
            tps_vals.append(resp.tokens_per_second)
        if resp.latency_ms:
            lat_vals.append(resp.latency_ms)
        raw = resp.raw or {}
        pe_count = raw.get("prompt_eval_count", 0)
        pe_dur = raw.get("prompt_eval_duration", 0)
        if pe_count and pe_dur:
            peval_vals.append(pe_count / (pe_dur / 1e9))

    vram_fit, gpu = None, None
    try:
        ps_payload = await ps_fetcher()
        vram_fit, gpu = vram_fit_from_ps(ps_payload, model)
    except Exception as exc:
        logger.info("labcompare: /api/ps unavailable for %s: %s", model, exc)

    def _avg(xs: list[float]) -> float | None:
        return round(sum(xs) / len(xs), 2) if xs else None

    return ThroughputResult(
        tokens_per_second=_avg(tps_vals),
        latency_ms=_avg(lat_vals),
        prompt_eval_rate=_avg(peval_vals),
        vram_fit_pct=vram_fit,
        gpu_name=gpu,
    )
