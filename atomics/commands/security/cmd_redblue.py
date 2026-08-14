"""The `redblue` security evaluation command."""

from __future__ import annotations

import asyncio
import uuid
from pathlib import Path

import click
from rich.console import Console
from rich.markup import escape as _rich_escape
from rich.table import Table

from atomics.commands.common import (
    PROVIDER_CHOICES,
    FixtureProgress,
    _make_provider,
    budget_option,
    eval_budget_from,
    write_summary_json,
)
from atomics.commands.suite_run import finalize_task_run, suite_run
from atomics.config import load_settings
from atomics.eval.budget import share_budget


@click.command("redblue")
@click.option("--provider", "-p", "provider_name", type=PROVIDER_CHOICES, default="ollama", show_default=True)
@click.option("--model", "-m", type=str, default=None)
@click.option("--ollama-host", type=str, default=None)
@click.option("--vllm-host", "vllm_host", type=str, default=None, help="vLLM/OpenAI-compatible base URL.")
@click.option("--judge-provider", "judge_provider_name", type=PROVIDER_CHOICES, default="ollama", show_default=True)
@click.option("--judge-model", type=str, default=None)
@click.option("--judge-host", type=str, default=None)
@click.option("--mode", type=click.Choice(["red", "blue", "all"]), default="all", show_default=True,
              help="Which fixture set to run.")
@click.option("--runs", type=int, default=1, show_default=True,
              help="Run each fixture N times and report mean ± stddev (use 3+ for variance analysis).")
@click.option("--thinking/--no-thinking", "thinking_flag", default=None)
@click.option("--thinking-budget", type=int, default=8000, show_default=True)
@click.option("--save/--no-save", "save_results", default=True, show_default=True)
@click.option("--json-out", "json_out", type=click.Path(dir_okay=False, writable=True), default=None,
              help="Write the full run (per-fixture scores, rationales, latency, cost) as JSON to this file.")
@budget_option
def redblue(
    provider_name: str,
    model: str | None,
    ollama_host: str | None,
    vllm_host: str | None,
    judge_provider_name: str,
    judge_model: str | None,
    judge_host: str | None,
    mode: str,
    runs: int,
    thinking_flag: bool | None,
    thinking_budget: int,
    save_results: bool,
    json_out: str | None,
    budget_usd: float | None,
) -> None:
    """Run red/blue team LLM capability eval — offensive and defensive security tasks.

    Use --runs 3 for variance-aware scoring (mean ± stddev across passes).
    """
    from atomics.eval.redblue.fixtures import ALL_FIXTURES, BLUE_FIXTURES, RED_FIXTURES
    from atomics.eval.redblue.runner import run_redblue

    console = Console()
    fixture_count = {"red": len(RED_FIXTURES), "blue": len(BLUE_FIXTURES), "all": len(ALL_FIXTURES)}[mode]
    settings = load_settings()
    provider = _make_provider(provider_name, model, ollama_host, settings, vllm_host=vllm_host)
    judge = _make_provider(judge_provider_name, judge_model, judge_host or ollama_host, settings, vllm_host=vllm_host)
    provider, judge = share_budget(eval_budget_from(budget_usd), provider, judge)

    console.print(
        f"\n[bold]Red/Blue eval[/bold] — model: [cyan]{provider_name}[/cyan] ({model or 'default'})\n"
        f"Judge: [cyan]{judge_provider_name}[/cyan] | Mode: [bold]{mode}[/bold] | "
        f"Fixtures: [bold]{fixture_count}[/bold] | Runs per fixture: [bold]{runs}[/bold]\n"
    )

    run_id = uuid.uuid4().hex[:12]
    ctx = click.get_current_context()
    show_progress = ctx.obj.get("progress", True) if ctx.obj else True
    verbose = ctx.obj.get("verbose", False) if ctx.obj else False
    progress = FixtureProgress(fixture_count, console, label="redblue") if show_progress else None

    with suite_run(
        suite="redblue",
        db_path=settings.db_path,
        save=save_results,
        finalize=finalize_task_run,
        failure_prefix="Red/blue evaluation failed",
    ) as run:
        # Red/blue fixture rows are stored in task_results, which require a
        # parent row in runs. Create it before on_done persists fixture rows.
        run.begin(
            run_id,
            provider=provider_name,
            model=model or "default",
            tier=f"redblue-{mode}",
        )
        repo = run.repository

        def on_start(idx, fixture):
            if progress:
                progress.on_start(idx, fixture.id, fixture.category)

        def on_done(fr):
            j = fr.judge
            if progress:
                progress.on_done(0)
            if j:
                pct = int(j.score * 100)
                color = "green" if pct >= 80 else ("yellow" if pct >= 60 else "red")
                console.print(
                    f"       [{fr.fixture.team.upper()}] [bold]{fr.fixture.id}[/bold] "
                    f"[{color}]{pct}%[/] ({fr.fixture.category}) — {_rich_escape(j.rationale[:80])}"
                )
                if verbose:
                    console.print(f"       [dim]Response ({fr.task_result.output_tokens} tokens, "
                                  f"{fr.task_result.latency_ms:.0f}ms):[/dim]")
                    console.print(f"       [dim]{_rich_escape((fr.task_result.response or '')[:200])}...[/dim]")
            if repo:
                repo.save_task_result(fr.task_result, suite=f"redblue-{fr.fixture.team}")

        summary = asyncio.run(run_redblue(
            provider,
            judge_provider=judge,
            mode=mode,
            model=model,
            judge_model=judge_model,
            runs=runs,
            run_id=run_id,
            thinking=thinking_flag,
            thinking_budget=thinking_budget,
            on_fixture_start=on_start,
            on_fixture_done=on_done,
        ))

        table = Table(title=f"Red/Blue Eval Summary ({mode})")
        table.add_column("Metric")
        table.add_column("Value")
        table.add_row("Provider", provider_name)
        table.add_row("Model", model or "default")
        table.add_row("Judge", f"{judge_provider_name} / {judge_model or 'default'}")
        table.add_row("Mode", mode)
        table.add_row("Runs per fixture", str(summary.runs))
        quality_str = f"{(summary.overall_quality or 0) * 100:.1f}%"
        if summary.quality_stddev is not None:
            quality_str += f"  ±{summary.quality_stddev * 100:.1f}%"
        table.add_row("Overall Quality", quality_str)
        table.add_row("Fixtures Run", str(summary.total_fixtures))
        table.add_row("Avg Latency", f"{summary.avg_latency_ms:.0f}ms")
        table.add_row("Total Cost", f"${summary.total_cost_usd:.6f}")
        for cat, score in sorted(summary.category_scores.items()):
            table.add_row(f"  {cat}", f"{score * 100:.1f}%")
        console.print(table)

        if json_out:
            write_summary_json(summary, Path(json_out))
            console.print(f"\n[dim]Wrote JSON results to {json_out}[/dim]")

# ── atomics probe ─────────────────────────────────────────────────────────────

