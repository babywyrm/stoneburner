#!/usr/bin/env python3
"""Variance sweep — run N iterations of model sweeps and report statistics."""
from __future__ import annotations

import asyncio
import csv
import json
import math
import os
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from atomics.config import load_settings
from atomics.providers.ollama import OllamaProvider
from atomics.sweep import run_model_sweep

RUNS = 25
FIXTURES = ["ev-01", "ev-06", "ev-08", "ev-11", "ev-17", "ev-22"]
HOST = "http://gpu-host:11434"
MODELS = ["qwen2.5:7b", "qwen3:14b", "deepseek-r1:14b"]


def _stats(values: list[float]) -> dict:
    n = len(values)
    if n == 0:
        return {"mean": 0, "stddev": 0, "min": 0, "max": 0, "n": 0}
    mean = sum(values) / n
    var = sum((v - mean) ** 2 for v in values) / n
    return {
        "mean": mean,
        "stddev": math.sqrt(var),
        "min": min(values),
        "max": max(values),
        "n": n,
    }


async def main() -> None:
    settings = load_settings()
    outdir = Path(f"data/variance_sweep_{datetime.now():%Y%m%d_%H%M%S}")
    outdir.mkdir(parents=True, exist_ok=True)

    csv_path = outdir / "results.csv"
    csvfile = open(csv_path, "w", newline="")
    writer = csv.writer(csvfile)
    writer.writerow(["run", "model", "quality_pct", "avg_latency_ms", "total_tokens", "total_cost", "fixtures"])

    judge = OllamaProvider(host=HOST, default_model="qwen2.5:7b")

    print(f"=== Variance Sweep ===")
    print(f"Models: {', '.join(MODELS)}")
    print(f"Fixtures: {', '.join(FIXTURES)}")
    print(f"Runs: {RUNS}")
    print(f"Output: {outdir}")
    print()

    all_results: dict[str, list[float]] = {m: [] for m in MODELS}
    all_latencies: dict[str, list[float]] = {m: [] for m in MODELS}

    t0 = time.monotonic()

    for run_num in range(1, RUNS + 1):
        elapsed = time.monotonic() - t0
        eta = (elapsed / max(run_num - 1, 1)) * (RUNS - run_num + 1) if run_num > 1 else 0
        print(f"{'─' * 50}")
        print(f"RUN {run_num} / {RUNS}  ({datetime.now():%H:%M:%S})  "
              f"elapsed={elapsed/60:.0f}m  eta={eta/60:.0f}m")
        print(f"{'─' * 50}")

        for model in MODELS:
            def provider_factory(model_name: str) -> OllamaProvider:
                return OllamaProvider(host=HOST, default_model=model_name)

            try:
                results = await run_model_sweep(
                    provider_factory=provider_factory,
                    judge_provider=judge,
                    models=[model],
                    fixture_ids=FIXTURES,
                )

                r = results[0]
                quality = (r.overall_quality or 0) * 100
                latency = r.avg_latency_ms
                tokens = r.total_tokens
                cost = r.total_cost_usd

                all_results[model].append(quality)
                all_latencies[model].append(latency)

                writer.writerow([run_num, model, f"{quality:.1f}", f"{latency:.0f}", tokens, f"{cost:.6f}", len(FIXTURES)])
                csvfile.flush()

                print(f"  [{run_num}/{RUNS}] {model:20s} -> quality={quality:.0f}%  latency={latency:.0f}ms  tokens={tokens}")

            except Exception as exc:
                print(f"  [{run_num}/{RUNS}] {model:20s} -> ERROR: {exc}")
                writer.writerow([run_num, model, "ERR", "ERR", 0, 0, 0])
                csvfile.flush()

    csvfile.close()

    total_elapsed = time.monotonic() - t0

    print()
    print(f"{'=' * 50}")
    print(f"VARIANCE SUMMARY  ({total_elapsed/60:.0f} minutes, {RUNS} runs)")
    print(f"{'=' * 50}")

    summary_lines: list[str] = []

    for model in MODELS:
        q = _stats(all_results[model])
        l = _stats(all_latencies[model])
        lines = [
            f"",
            f"--- {model} ({q['n']} runs) ---",
            f"  Quality:  mean={q['mean']:.1f}%  stddev={q['stddev']:.1f}%  min={q['min']:.0f}%  max={q['max']:.0f}%",
            f"  Latency:  mean={l['mean']:.0f}ms  stddev={l['stddev']:.0f}ms  min={l['min']:.0f}ms  max={l['max']:.0f}ms",
        ]
        for line in lines:
            print(line)
            summary_lines.append(line)

    print()
    print(f"Raw CSV: {csv_path}")

    summary_path = outdir / "summary.txt"
    summary_path.write_text("\n".join(summary_lines) + f"\n\nCompleted: {datetime.now():%Y-%m-%d %H:%M:%S}\n")
    print(f"Summary:  {summary_path}")
    print(f"Done at {datetime.now():%H:%M:%S}")


if __name__ == "__main__":
    asyncio.run(main())
