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
    from atomics.validation import validate_endpoint_url

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
        url = validate_endpoint_url(url, label=f"--host {name}")
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


@dataclass
class CellResult:
    host_name: str
    host_url: str
    model: str
    tokens_per_second: float | None = None
    latency_ms: float | None = None
    prompt_eval_rate: float | None = None
    vram_fit_pct: float | None = None
    gpu_name: str | None = None
    quality_score: float | None = None
    error: str | None = None


def _throughput_prompts(n: int) -> list[str]:
    """Fixed, deterministic prompts so throughput is comparable across hosts."""
    base = [
        "Explain what a reverse proxy does in two sentences.",
        "List three common causes of high latency in a web service.",
        "Write a one-line summary of what TLS provides.",
        "Describe the difference between a process and a thread.",
        "Name two trade-offs of microservices versus a monolith.",
    ]
    if n <= len(base):
        return base[:n]
    return [base[i % len(base)] for i in range(n)]


async def run_labcompare(
    *,
    hosts: list[HostSpec],
    models: list[str],
    dimensions: list[str],
    quality_suite: str,
    judge_host: str | None,
    judge_model: str | None,
    prompts: int,
    provider_factory,
    quality_fn,
    ps_fetcher_factory,
    on_cell=None,
) -> list[CellResult]:
    """Run throughput and/or quality for every host × model.

    Never raises on a single-cell failure — records the error and continues so
    one dead host or missing model does not abort the whole comparison.
    """
    fixed_prompts = _throughput_prompts(prompts)
    cells: list[CellResult] = []
    for host in hosts:
        provider = provider_factory(host.url)
        ps_fetcher = ps_fetcher_factory(host.url)
        for model in models:
            if on_cell:
                on_cell(host.name, model)
            cell = CellResult(host_name=host.name, host_url=host.url, model=model)
            try:
                if "throughput" in dimensions:
                    tp = await probe_throughput(
                        provider, model, ps_fetcher=ps_fetcher, prompts=fixed_prompts
                    )
                    cell.tokens_per_second = tp.tokens_per_second
                    cell.latency_ms = tp.latency_ms
                    cell.prompt_eval_rate = tp.prompt_eval_rate
                    cell.vram_fit_pct = tp.vram_fit_pct
                    cell.gpu_name = tp.gpu_name
                if "quality" in dimensions:
                    cell.quality_score = await quality_fn(
                        provider, judge_host, judge_model, model
                    )
            except Exception as exc:
                cell.error = (str(exc) or repr(exc))[:200]
                logger.warning(
                    "labcompare: cell %s/%s failed: %s", host.name, model, cell.error
                )
            cells.append(cell)
    return cells
