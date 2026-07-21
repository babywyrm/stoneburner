"""Eval and advisor CLI commands."""

from __future__ import annotations

import asyncio
import sys

import click
from rich.console import Console
from rich.table import Table

from atomics.commands.common import PROVIDER_CHOICES, _make_provider, setup_logging
from atomics.config import load_settings


@click.command()
@click.option(
    "--provider", "-p", "provider_name",
    type=PROVIDER_CHOICES,
    default="ollama",
    help="Provider to evaluate (model under test)",
)
@click.option("--model", "-m", type=str, default=None, help="Model override for the provider under test")
@click.option("--ollama-host", type=str, default=None, help="Ollama endpoint for model under test")
@click.option("--vllm-host", "vllm_host", type=str, default=None, help="vLLM/OpenAI-compatible base URL (default: ATOMICS_VLLM_HOST)")
@click.option("--region", type=str, default="us-east-1", help="AWS region for Bedrock")
@click.option(
    "--judge-provider", "judge_provider_name",
    type=PROVIDER_CHOICES,
    default="ollama",
    help="Provider to use as judge (default: ollama — $0 cost)",
)
@click.option("--judge-model", type=str, default=None, help="Model override for the judge")
@click.option("--judge-host", type=str, default=None, help="Ollama host for the judge (if different)")
@click.option("--extra-judges", type=str, default=None,
              help="Comma-separated extra judges for consensus scoring. "
                   "Format: provider:model (e.g. claude:claude-sonnet-4-6,ollama:deepseek-r1:14b). "
                   "Reports mean quality and inter-judge stddev.")
@click.option("--fixtures", "fixtures_filter", type=str, default=None,
              help="Comma-separated fixture IDs to run a subset (e.g. ev-19 or "
                   "ev-01,ev-02). Default: all 25 fixtures.")
@click.option("--save/--no-save", "save_results", default=True, help="Persist results to the database")
@click.option("--json-out", "json_out", type=click.Path(dir_okay=False, writable=True), default=None,
              help="Write the full run (per-fixture scores, rationales, latency, cost) as JSON to this file.")
@click.option("--thinking/--no-thinking", "thinking_flag", default=None, help="Enable/disable thinking for capable models")
@click.option("--thinking-budget", type=int, default=None, help="Max thinking tokens")
def eval(
    provider_name: str,
    model: str | None,
    ollama_host: str | None,
    vllm_host: str | None,
    region: str,
    judge_provider_name: str,
    judge_model: str | None,
    judge_host: str | None,
    extra_judges: str | None,
    fixtures_filter: str | None,
    save_results: bool,
    json_out: str | None,
    thinking_flag: bool | None,
    thinking_budget: int | None,
) -> None:
    """Run the fixed eval fixture set and score quality with an LLM judge.

    Produces an accuracy score (0-100%) for each fixture and an overall quality
    rating. Run against multiple providers then use 'atomics compare --narrative'
    to generate the business-case comparison.

    Example:
      atomics eval --provider ollama --model qwen2.5:7b
      atomics eval --provider claude
      atomics eval --provider openai --model gpt-4o
      atomics compare --narrative
    """
    settings = load_settings()
    setup_logging(settings.log_level)
    console = Console()

    def _build_provider(
        name: str,
        mdl: str | None,
        host: str | None,
        context_tokens: int | None = None,
    ):
        return _make_provider(
            name, mdl, host, settings,
            vllm_host=vllm_host, region=region, context_tokens=context_tokens,
        )

    test_provider = _build_provider(provider_name, model, ollama_host)
    # Judge host falls back to: --judge-host → --ollama-host → ATOMICS_OLLAMA_HOST env/.env
    judge_provider = _build_provider(judge_provider_name, judge_model, judge_host or ollama_host or settings.ollama_host)

    # Parse --extra-judges "claude:claude-sonnet-4-6,ollama:deepseek-r1:14b@http://host:11434"
    extra_judge_pairs: list[tuple] = []
    if extra_judges:
        for spec in extra_judges.split(","):
            spec = spec.strip()
            if not spec:
                continue
            ej_host = None
            if "@" in spec:
                spec, ej_host = spec.rsplit("@", 1)
            parts = spec.split(":", 1)
            ej_provider_name = parts[0]
            ej_model = parts[1] if len(parts) > 1 else None
            ej_provider = _build_provider(
                ej_provider_name, ej_model,
                ej_host or judge_host or ollama_host or settings.ollama_host,
            )
            extra_judge_pairs.append((ej_provider, ej_model))

    from atomics.eval.fixtures import EVAL_FIXTURES
    from atomics.eval.multilingual import ALL_MULTILINGUAL_FIXTURES

    all_available = EVAL_FIXTURES + ALL_MULTILINGUAL_FIXTURES
    selected_fixtures = None
    if fixtures_filter:
        wanted = {fid.strip() for fid in fixtures_filter.split(",") if fid.strip()}
        selected_fixtures = [f for f in all_available if f.id in wanted]
        missing = wanted - {f.id for f in selected_fixtures}
        if missing:
            click.echo(f"Error: unknown fixture id(s): {', '.join(sorted(missing))}", err=True)
            sys.exit(1)
    fixture_count = len(selected_fixtures) if selected_fixtures is not None else len(EVAL_FIXTURES)

    judge_label = judge_model or "default"
    if extra_judge_pairs:
        judge_label += f" + {len(extra_judge_pairs)} consensus"
    console.print(
        f"\n[bold]Eval run[/bold] — model under test: [cyan]{provider_name}[/cyan] "
        f"({model or 'default'})\n"
        f"Judge: [cyan]{judge_provider_name}[/cyan] ({judge_label})\n"
        f"Fixtures: [bold]{fixture_count}[/bold] | Results saved: [bold]{'yes' if save_results else 'no'}[/bold]\n"
    )

    repo = None
    if save_results:
        from atomics.storage.repository import MetricsRepository
        repo = MetricsRepository(settings.db_path)

    from atomics.eval.runner import run_eval

    result_table = Table(title="Eval Results", show_lines=True)
    result_table.add_column("ID", style="dim")
    result_table.add_column("Complexity", style="cyan")
    result_table.add_column("Prompt (truncated)", no_wrap=False, max_width=45)
    result_table.add_column("Quality", justify="right", style="green bold")
    result_table.add_column("Latency", justify="right")
    result_table.add_column("Tokens", justify="right")
    result_table.add_column("Cost", justify="right", style="yellow")
    result_table.add_column("Rationale", no_wrap=False, max_width=40, style="dim")

    # Pre-allocate a run_id and create the parent row so FK constraints are satisfied
    import uuid as _uuid
    eval_run_id = _uuid.uuid4().hex[:12]
    if provider_name == "ollama":
        effective_model = model or settings.ollama_model
    elif provider_name == "vllm":
        effective_model = model or settings.vllm_model
    else:
        effective_model = model or settings.default_model
    if repo:
        repo.create_run(
            eval_run_id,
            tier="eval",
            provider=provider_name,
            model=effective_model,
            trigger="eval",
        )

    def on_done(fr) -> None:
        judge = fr.judge
        tr = fr.task_result
        if tr.status.value == "failed":
            quality = "[red]FAIL[/red]"
            rationale = tr.error_message[:80]
        elif judge and not judge.parse_failed:
            quality = f"{judge.score * 100:.0f}%"
            rationale = judge.rationale[:80]
        else:
            quality = "[yellow]?[/yellow]"
            rationale = "judge parse failed"
        result_table.add_row(
            fr.fixture.id,
            fr.fixture.complexity.value,
            fr.fixture.prompt[:60] + "…",
            quality,
            f"{tr.latency_ms:.0f}ms",
            str(tr.total_tokens),
            f"${tr.estimated_cost_usd:.6f}",
            rationale,
        )
        if repo:
            repo.save_task_result(tr)

    # Auto-detect thinking if not explicitly set
    eff_thinking = thinking_flag
    if eff_thinking is None and model:
        from atomics.model_classes import supports_thinking
        if supports_thinking(model):
            eff_thinking = True

    summary = asyncio.run(run_eval(
        test_provider,
        judge_provider=judge_provider,
        model=model,
        judge_model=judge_model,
        run_id=eval_run_id,
        on_fixture_done=on_done,
        thinking=eff_thinking,
        thinking_budget=thinking_budget,
        extra_judges=extra_judge_pairs,
        fixtures=selected_fixtures,
    ))

    console.print(result_table)

    overall = summary.overall_accuracy
    quality_str = f"{overall * 100:.1f}%" if overall is not None else "—"
    value_str = f"{summary.value_score:.1f}" if summary.value_score is not None else "—"

    summary_table = Table(title="Eval Summary", show_lines=True)
    summary_table.add_column("Metric", style="dim")
    summary_table.add_column("Value", style="bold")
    summary_table.add_row("Provider", provider_name)
    summary_table.add_row("Model", model or "default")
    summary_table.add_row("Overall Quality", f"[green]{quality_str}[/green]")
    summary_table.add_row("Value Score", f"[cyan]{value_str}[/cyan]")
    summary_table.add_row("Avg Latency", f"{summary.avg_latency_ms:.0f}ms")
    summary_table.add_row("Total Tokens", f"{summary.total_tokens:,}")
    summary_table.add_row("Total Cost", f"${summary.total_cost_usd:.6f}")
    summary_table.add_row("Fixtures Run", str(len(summary.fixture_results)))
    pf_rate = summary.parse_failure_rate
    pf_style = "green" if pf_rate == 0 else "yellow" if pf_rate < 0.1 else "red"
    summary_table.add_row("Judge Parse Failures", f"[{pf_style}]{pf_rate * 100:.1f}%[/{pf_style}]")
    console.print(summary_table)

    if repo:
        repo.complete_run(eval_run_id)
        repo.close()
        console.print("\n[dim]Results saved to database. Run [bold]atomics compare --narrative[/bold] after evaluating multiple providers.[/dim]")

    if json_out:
        import json as _json
        with open(json_out, "w", encoding="utf-8") as fh:
            _json.dump(summary.to_dict(), fh, indent=2)
        console.print(f"[dim]Wrote JSON results to {json_out}[/dim]")

@click.command("advisor")
@click.option("--min-quality", type=float, default=0.8, show_default=True,
              help="Minimum acceptable quality score (0.0-1.0).")
@click.option("--since-hours", type=float, default=None,
              help="Only analyze data from the last N hours.")
@click.option("--current-model", type=str, default=None,
              help="Treat this model as the baseline to optimize from.")
@click.option("--json-out", "json_out", type=click.Path(dir_okay=False, writable=True),
              default=None, help="Write recommendations as JSON.")
def advisor(
    min_quality: float,
    since_hours: float | None,
    current_model: str | None,
    json_out: str | None,
) -> None:
    """Analyze historical runs and recommend cheaper models meeting quality thresholds."""
    settings = load_settings()
    console = Console()

    from atomics.advisor import analyze_cost_optimization
    from atomics.storage.schema import init_db

    conn = init_db(settings.db_path)
    try:
        summary = analyze_cost_optimization(
            conn,
            min_quality=min_quality,
            since_hours=since_hours,
            current_model=current_model,
        )
    finally:
        conn.close()

    if not summary.recommendations:
        console.print(
            "[dim]No optimization recommendations found. "
            "Run benchmarks with multiple models first, or lower --min-quality.[/dim]"
        )
        return

    table = Table(title=f"Cost Optimization (min quality: {min_quality:.0%})", show_lines=True)
    table.add_column("Category", style="cyan")
    table.add_column("Current Model", style="dim")
    table.add_column("Quality", justify="right")
    table.add_column("$/Task", justify="right", style="yellow")
    table.add_column("\u2192", justify="center")
    table.add_column("Recommended", style="green bold")
    table.add_column("Quality", justify="right")
    table.add_column("$/Task", justify="right", style="green")
    table.add_column("Savings", justify="right", style="yellow bold")

    for r in summary.recommendations:
        table.add_row(
            r.category,
            r.current_model,
            f"{r.current_quality:.0%}",
            f"${r.current_cost_per_task:.6f}",
            "\u2192",
            r.recommended_model,
            f"{r.recommended_quality:.0%}",
            f"${r.recommended_cost_per_task:.6f}",
            f"{r.cost_savings_pct:.0f}%",
        )
    console.print(table)

    summary_table = Table(title="Savings Summary", show_lines=True)
    summary_table.add_column("Metric", style="dim")
    summary_table.add_column("Value", style="bold")
    summary_table.add_row("Models Analyzed", str(summary.models_analyzed))
    summary_table.add_row("Current Total Cost", f"${summary.total_current_cost:.4f}")
    summary_table.add_row("Recommended Total Cost", f"${summary.total_recommended_cost:.4f}")
    savings_style = "green" if summary.overall_savings_pct > 0 else "yellow"
    summary_table.add_row("Overall Savings",
                          f"[{savings_style}]{summary.overall_savings_pct:.1f}%[/{savings_style}]")
    console.print(summary_table)

    if json_out:
        import json as _json
        with open(json_out, "w", encoding="utf-8") as fh:
            _json.dump(summary.to_dict(), fh, indent=2)
        console.print(f"[dim]Wrote recommendations to {json_out}[/dim]")
