"""GPU stress testing — ramp concurrency to find saturation point."""

from __future__ import annotations

import asyncio
import shutil
import subprocess
import time
from dataclasses import dataclass, field

import httpx

STRESS_PROMPTS = [
    (
        "Write a detailed technical analysis of how GPU memory bandwidth affects "
        "large language model inference performance. Cover VRAM allocation strategies, "
        "KV-cache management, batch scheduling, and thermal throttling. Include "
        "specific numbers and comparisons between consumer and datacenter GPUs."
    ),
    (
        "Explain the complete architecture of a modern transformer-based language model "
        "from input tokenization through attention layers to output decoding. Cover "
        "embedding matrices, multi-head self-attention, feed-forward networks, layer "
        "normalization, positional encoding, and the softmax sampling process."
    ),
    (
        "Write a comprehensive guide to container orchestration security covering "
        "Kubernetes RBAC, network policies, pod security standards, supply chain "
        "attacks on container images, runtime security monitoring, secrets management, "
        "and incident response procedures for compromised workloads."
    ),
    (
        "Describe the history and evolution of cryptographic protocols from DES through "
        "AES, RSA, elliptic curve cryptography, and into the post-quantum era with "
        "lattice-based schemes like CRYSTALS-Kyber and CRYSTALS-Dilithium. Compare "
        "their security guarantees, performance characteristics, and key sizes."
    ),
    (
        "Write a deep technical comparison of CPU vs GPU vs TPU architectures for "
        "machine learning workloads. Cover SIMD vs SIMT execution models, memory "
        "hierarchies, interconnects, compiler toolchains, and real-world throughput "
        "measurements for training and inference at different batch sizes."
    ),
    (
        "Explain the complete lifecycle of an HTTP request from a browser through "
        "DNS resolution, TCP handshake, TLS negotiation, HTTP/2 multiplexing, "
        "load balancer routing, application server processing, database queries, "
        "response serialization, and CDN caching. Include timing breakdowns."
    ),
    (
        "Write a thorough analysis of supply chain attacks in software development "
        "covering dependency confusion, typosquatting, compromised build systems, "
        "malicious maintainers, CI/CD pipeline injection, and artifact signing. "
        "Include real-world case studies and mitigation strategies."
    ),
    (
        "Describe the internal architecture of a modern database engine covering "
        "the query parser, optimizer, execution engine, buffer pool, WAL logging, "
        "MVCC concurrency control, B-tree and LSM-tree storage engines, and "
        "distributed consensus protocols like Raft and Paxos."
    ),
]


@dataclass
class ConcurrencyResult:
    concurrency: int
    requests: int = 0
    failed: int = 0
    total_output_tokens: int = 0
    total_input_tokens: int = 0
    elapsed_seconds: float = 0.0
    aggregate_tps: float = 0.0
    avg_request_tps: float = 0.0
    avg_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    latencies: list[float] = field(default_factory=list)
    per_request_tps: list[float] = field(default_factory=list)


@dataclass
class StressResult:
    model: str
    host: str
    duration_seconds: float = 0.0
    total_tokens: int = 0
    total_requests: int = 0
    total_failed: int = 0
    phases: list[ConcurrencyResult] = field(default_factory=list)
    saturation_concurrency: int = 0
    peak_tps: float = 0.0
    vram_peak_mb: float | None = None
    vram_total_mb: float | None = None
    gpu_name: str = ""


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    k = (len(s) - 1) * (pct / 100)
    f = int(k)
    c = f + 1
    if c >= len(s):
        return s[f]
    return s[f] + (k - f) * (s[c] - s[f])


def _get_gpu_info() -> tuple[str, float | None]:
    """Try to get GPU name and total VRAM via nvidia-smi."""
    if not shutil.which("nvidia-smi"):
        return "", None
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        if out.returncode == 0:
            parts = out.stdout.strip().split(",")
            name = parts[0].strip()
            total = float(parts[1].strip()) if len(parts) > 1 else None
            return name, total
    except Exception:
        pass
    return "", None


def _get_vram_used_mb() -> float | None:
    if not shutil.which("nvidia-smi"):
        return None
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        if out.returncode == 0:
            return float(out.stdout.strip())
    except Exception:
        pass
    return None


async def _single_request(
    client: httpx.AsyncClient,
    host: str,
    model: str,
    prompt: str,
    num_predict: int,
) -> tuple[int, int, float, float]:
    """Fire one request. Returns (output_tokens, input_tokens, latency_ms, tps)."""
    resp = await client.post(
        f"{host}/api/generate",
        json={
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {"num_predict": num_predict},
        },
        timeout=300.0,
    )
    resp.raise_for_status()
    data = resp.json()
    out = data.get("eval_count", 0)
    inp = data.get("prompt_eval_count", 0)
    eval_dur = data.get("eval_duration", 0)
    total_dur = data.get("total_duration", 0)
    tps = out / (eval_dur / 1e9) if eval_dur > 0 and out > 0 else 0.0
    latency = total_dur / 1e6 if total_dur else 0.0
    return out, inp, latency, tps


async def _run_phase(
    client: httpx.AsyncClient,
    host: str,
    model: str,
    concurrency: int,
    duration_seconds: float,
    num_predict: int,
) -> ConcurrencyResult:
    """Run requests at a given concurrency level for a fixed duration."""
    result = ConcurrencyResult(concurrency=concurrency)
    start = time.monotonic()
    prompt_idx = 0

    async def _worker():
        nonlocal prompt_idx
        while time.monotonic() - start < duration_seconds:
            prompt = STRESS_PROMPTS[prompt_idx % len(STRESS_PROMPTS)]
            prompt_idx += 1
            try:
                out, inp, lat, tps = await _single_request(
                    client, host, model, prompt, num_predict
                )
                result.requests += 1
                result.total_output_tokens += out
                result.total_input_tokens += inp
                result.latencies.append(lat)
                result.per_request_tps.append(tps)
            except Exception:
                result.failed += 1

    workers = [asyncio.create_task(_worker()) for _ in range(concurrency)]
    await asyncio.gather(*workers)

    result.elapsed_seconds = time.monotonic() - start
    if result.elapsed_seconds > 0:
        result.aggregate_tps = result.total_output_tokens / result.elapsed_seconds
    if result.per_request_tps:
        result.avg_request_tps = sum(result.per_request_tps) / len(result.per_request_tps)
    if result.latencies:
        result.avg_latency_ms = sum(result.latencies) / len(result.latencies)
        result.p95_latency_ms = _percentile(result.latencies, 95)
    return result


async def run_stress(
    host: str = "http://localhost:11434",
    model: str = "qwen2.5:7b",
    max_concurrency: int = 8,
    phase_seconds: float = 15.0,
    num_predict: int = 2048,
    on_phase: object = None,
) -> StressResult:
    """Ramp concurrency from 1 to max_concurrency, spending phase_seconds at each level."""
    gpu_name, vram_total = _get_gpu_info()
    result = StressResult(
        model=model,
        host=host,
        gpu_name=gpu_name,
        vram_total_mb=vram_total,
    )

    concurrency_levels = []
    c = 1
    while c <= max_concurrency:
        concurrency_levels.append(c)
        c *= 2
    if max_concurrency not in concurrency_levels:
        concurrency_levels.append(max_concurrency)

    t0 = time.monotonic()
    peak_vram: float | None = None

    async with httpx.AsyncClient() as client:
        for conc in concurrency_levels:
            vram = _get_vram_used_mb()
            if vram is not None:
                peak_vram = max(peak_vram or 0, vram)

            phase = await _run_phase(
                client, host, model, conc, phase_seconds, num_predict
            )
            result.phases.append(phase)
            result.total_tokens += phase.total_output_tokens + phase.total_input_tokens
            result.total_requests += phase.requests
            result.total_failed += phase.failed

            if phase.aggregate_tps > result.peak_tps:
                result.peak_tps = phase.aggregate_tps
                result.saturation_concurrency = conc

            if on_phase:
                on_phase(phase)

            vram = _get_vram_used_mb()
            if vram is not None:
                peak_vram = max(peak_vram or 0, vram)

    result.duration_seconds = time.monotonic() - t0
    result.vram_peak_mb = peak_vram
    return result
