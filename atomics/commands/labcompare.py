"""Inference-host parity comparison CLI command."""

from __future__ import annotations

from typing import TYPE_CHECKING

import click
from rich.console import Console

from atomics.benchmark.labcompare import (
    parity_verdict,
    parse_host_specs,
    run_labcompare,
    speedup_ratio,
)
from atomics.commands.common import _make_provider, run_async
from atomics.config import load_settings

if TYPE_CHECKING:
    from atomics.benchmark.labcompare import CellResult


def _run_labcompare_sync(**kwargs) -> list[CellResult]:
    """Sync wrapper around the async orchestrator (patch target in tests)."""
    return run_async(run_labcompare(**kwargs))


def _render_labcompare(console, cells, hosts, dims) -> None:
    """Print a per-model side-by-side block with speedup and parity verdicts."""
    by_model: dict[str, dict[str, CellResult]] = {}
    for c in cells:
        by_model.setdefault(c.model, {})[c.host_name] = c

    host_names = [h.name for h in hosts]
    for model, host_cells in by_model.items():
        console.print(f"[bold cyan]{model}[/bold cyan]")

        if "throughput" in dims:
            parts = []
            for hn in host_names:
                c = host_cells.get(hn)
                tps = c.tokens_per_second if c else None
                parts.append(f"{hn} {tps if tps is not None else 'n/a'} tok/s")
            line = "  Throughput   " + "   ".join(parts)
            if len(host_names) == 2:
                a = host_cells.get(host_names[0])
                b = host_cells.get(host_names[1])
                if a and b:
                    sr = speedup_ratio(a.tokens_per_second, b.tokens_per_second)
                    if sr:
                        line += f"   -> {host_names[0]} {sr}x faster"
            console.print(line)

            vparts = []
            for hn in host_names:
                c = host_cells.get(hn)
                v = c.vram_fit_pct if c else None
                vparts.append(f"{hn} {int(v * 100)}% GPU" if v is not None else f"{hn} n/a")
            console.print("  VRAM fit     " + "   ".join(vparts))

        if "quality" in dims:
            qparts = []
            for hn in host_names:
                c = host_cells.get(hn)
                q = c.quality_score if c else None
                qparts.append(f"{hn} {int(q * 100)}%" if q is not None else f"{hn} n/a")
            line = "  Quality      " + "   ".join(qparts)
            if len(host_names) == 2:
                a = host_cells.get(host_names[0])
                b = host_cells.get(host_names[1])
                if a and b:
                    ok, delta = parity_verdict(a.quality_score, b.quality_score)
                    if ok is not None:
                        verdict = "parity OK" if ok else "parity DIFF"
                        line += f"   -> {verdict} (delta {int((delta or 0) * 100)})"
            console.print(line)
        console.print()


@click.command("labcompare")
@click.option(
    "--host",
    "hosts_raw",
    multiple=True,
    required=True,
    help="Labeled endpoint NAME=URL (repeat for each host).",
)
@click.option(
    "--models", "models_csv", required=True, help="Comma-separated models to test on every host."
)
@click.option(
    "--quality-suite", type=click.Choice(["eval", "redblue"]), default="eval", show_default=True
)
@click.option(
    "--dimensions",
    default="throughput,quality",
    show_default=True,
    help="Comma-separated: throughput,quality",
)
@click.option("--judge-host", default=None)
@click.option("--judge-model", default=None)
@click.option("--prompts", type=int, default=3, show_default=True)
@click.option("--save/--no-save", "save_results", default=True, show_default=True)
@click.option("-o", "--json-out", "json_out", type=click.Path(), default=None)
def labcompare(
    hosts_raw,
    models_csv,
    quality_suite,
    dimensions,
    judge_host,
    judge_model,
    prompts,
    save_results,
    json_out,
):
    """Compare two+ inference hosts on throughput and quality parity."""
    console = Console()
    settings = load_settings()

    try:
        hosts = parse_host_specs(list(hosts_raw))
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc
    if len(hosts) < 2:
        raise click.ClickException("labcompare needs at least two --host entries.")

    models = [m.strip() for m in models_csv.split(",") if m.strip()]
    dims = [d.strip() for d in dimensions.split(",") if d.strip()]

    from atomics.eval.redblue.runner import run_redblue
    from atomics.eval.runner import run_eval

    def provider_factory(url):
        return _make_provider("ollama", None, url, settings)

    def ps_fetcher_factory(url):
        async def _ps():
            import httpx

            async with httpx.AsyncClient(timeout=10.0) as client:
                r = await client.get(f"{url}/api/ps")
                r.raise_for_status()
                return r.json()

        return _ps

    async def quality_fn(provider, jhost, jmodel, model):
        judge = _make_provider("ollama", jmodel, jhost or provider._host, settings)
        if quality_suite == "redblue":
            summary = await run_redblue(
                provider,
                judge_provider=judge,
                model=model,
                judge_model=jmodel,
            )
            return summary.overall_quality
        summary = await run_eval(
            provider,
            judge_provider=judge,
            model=model,
            judge_model=jmodel,
        )
        return summary.overall_accuracy

    console.print(
        f"\n[bold]LabCompare[/bold] - {len(hosts)} hosts x {len(models)} models "
        f"| dims: {','.join(dims)} | quality: {quality_suite}\n"
    )

    cells = _run_labcompare_sync(
        hosts=hosts,
        models=models,
        dimensions=dims,
        quality_suite=quality_suite,
        judge_host=judge_host,
        judge_model=judge_model,
        prompts=prompts,
        provider_factory=provider_factory,
        quality_fn=quality_fn,
        ps_fetcher_factory=ps_fetcher_factory,
    )

    _render_labcompare(console, cells, hosts, dims)

    if save_results:
        cmp_id = __import__("uuid").uuid4().hex[:12]
        from atomics.storage.repository import MetricsRepository

        repo = MetricsRepository(settings.db_path)
        for c in cells:
            repo.save_labcompare_result(
                comparison_run_id=cmp_id,
                host_name=c.host_name,
                host_url=c.host_url,
                model=c.model,
                tokens_per_second=c.tokens_per_second,
                latency_ms=c.latency_ms,
                prompt_eval_rate=c.prompt_eval_rate,
                vram_fit_pct=c.vram_fit_pct,
                gpu_name=c.gpu_name,
                quality_score=c.quality_score,
                quality_suite=quality_suite if "quality" in dims else None,
                judge_model=judge_model if "quality" in dims else None,
                dimensions=",".join(dims),
            )
        repo.close()
        console.print(f"[dim]Saved comparison {cmp_id} to DB.[/dim]")

    if json_out:
        import json
        from pathlib import Path as _Path

        _Path(json_out).write_text(json.dumps([c.__dict__ for c in cells], indent=2))
        console.print(f"[dim]Wrote {json_out}[/dim]")
