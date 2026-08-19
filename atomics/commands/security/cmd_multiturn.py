"""The `multiturn` security evaluation command."""

from __future__ import annotations

import asyncio
import sys
import uuid
from pathlib import Path

import click
from rich.console import Console
from rich.table import Table

from atomics.commands.common import (
    PROVIDER_CHOICES,
    _make_provider,
    budget_option,
    eval_budget_from,
    extra_judges_option,
    parse_extra_judges,
    setup_logging,
    write_summary_json,
)
from atomics.commands.suite_run import finalize_task_run, suite_run
from atomics.config import load_settings
from atomics.eval.budget import share_budget


@click.command("multiturn")
@click.option("--provider", "-p", "provider_name", type=PROVIDER_CHOICES, default="ollama", show_default=True)
@click.option("--model", "-m", type=str, default=None, help="Model override.")
@click.option("--ollama-host", type=str, default=None, help="Ollama base URL.")
@click.option("--vllm-host", "vllm_host", type=str, default=None, help="vLLM base URL.")
@click.option("--region", type=str, default="us-east-1", help="AWS region for Bedrock.")
@click.option("--judge-provider", "judge_provider_name", type=PROVIDER_CHOICES, default="ollama", show_default=True)
@click.option("--judge-model", type=str, default=None, help="Model for the judge.")
@click.option("--judge-host", type=str, default=None, help="Ollama host for the judge.")
@extra_judges_option
@click.option("--fixtures", "fixtures_filter", type=str, default=None,
              help="Comma-separated fixture IDs (e.g. mt-eval-01,mt-eval-05).")
@click.option("--save/--no-save", "save_results", default=True, help="Persist results.")
@click.option("--json-out", "json_out", type=click.Path(dir_okay=False, writable=True), default=None)
@click.option("--thinking/--no-thinking", "thinking_flag", default=None)
@click.option("--thinking-budget", type=int, default=None)
@budget_option
def multiturn(
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
    budget_usd: float | None,
) -> None:
    """Multi-turn conversation evaluation — context retention, coherence, instruction following."""
    settings = load_settings()
    setup_logging(settings.log_level)
    console = Console()

    from atomics.eval.multiturn.fixtures import ALL_MULTITURN_FIXTURES
    from atomics.eval.multiturn.runner import ConversationResult, run_multiturn

    test_provider = _make_provider(
        provider_name, model, ollama_host, settings,
        vllm_host=vllm_host, region=region,
    )
    judge_provider = _make_provider(
        judge_provider_name, judge_model, judge_host or ollama_host, settings,
        vllm_host=vllm_host, region=region,
    )
    extra_judge_pairs = parse_extra_judges(
        extra_judges,
        build=lambda name, mdl, host: _make_provider(
            name, mdl, host, settings, vllm_host=vllm_host, region=region,
        ),
        default_host=judge_host or ollama_host,
    )
    budget = eval_budget_from(budget_usd)
    if budget is not None:
        guarded = share_budget(
            budget, test_provider, judge_provider, *(p for p, _ in extra_judge_pairs)
        )
        test_provider, judge_provider = guarded[0], guarded[1]
        extra_judge_pairs = [
            (guarded[2 + i], mdl) for i, (_, mdl) in enumerate(extra_judge_pairs)
        ]

    selected_fixtures = ALL_MULTITURN_FIXTURES
    if fixtures_filter:
        ids = [f.strip() for f in fixtures_filter.split(",")]
        fixture_map = {f.id: f for f in ALL_MULTITURN_FIXTURES}
        missing = [i for i in ids if i not in fixture_map]
        if missing:
            console.print(f"[red]Unknown fixture IDs: {', '.join(missing)}[/red]")
            sys.exit(1)
        selected_fixtures = [fixture_map[i] for i in ids]

    total_turns = sum(len(f.turns) for f in selected_fixtures)
    console.print(
        f"\n[bold]Multi-Turn Eval[/bold] — provider: [cyan]{provider_name}[/cyan] | "
        f"model: [cyan]{model or 'default'}[/cyan] | "
        f"judge: [cyan]{judge_provider_name}:{judge_model or 'default'}[/cyan]\n"
        f"Conversations: [bold]{len(selected_fixtures)}[/bold] | "
        f"Total turns: [bold]{total_turns}[/bold]\n"
    )

    mt_run_id = uuid.uuid4().hex[:12]
    if provider_name == "ollama":
        effective_model = model or settings.ollama_model
    elif provider_name == "vllm":
        effective_model = model or settings.vllm_model
    else:
        effective_model = model or settings.default_model

    with suite_run(
        suite="multiturn",
        db_path=settings.db_path,
        save=save_results,
        finalize=finalize_task_run,
        failure_prefix="Multi-turn evaluation failed",
    ) as run:
        run.begin(
            mt_run_id,
            provider=provider_name,
            model=effective_model,
            trigger="eval",
        )
        repo = run.repository

        result_table = Table(title="Multi-Turn Eval Results", show_lines=True)
        result_table.add_column("ID", style="dim")
        result_table.add_column("Turns", justify="right")
        result_table.add_column("Turn Avg", justify="right", style="green")
        result_table.add_column("Retain", justify="right")
        result_table.add_column("Consist", justify="right")
        result_table.add_column("Instruct", justify="right")
        result_table.add_column("Overall", justify="right", style="green bold")
        result_table.add_column("Latency", justify="right")
        result_table.add_column("Cost", justify="right", style="yellow")

        def on_done(cr: ConversationResult) -> None:
            turn_scores = [t.judge.score for t in cr.turn_results if t.judge and not t.judge.parse_failed]
            turn_avg = f"{sum(turn_scores)/len(turn_scores)*100:.0f}%" if turn_scores else "—"
            cj = cr.conversation_judge
            if cj and not cj.parse_failed:
                ret = str(cj.retention)
                con = str(cj.consistency)
                ins = str(cj.instruction)
            else:
                ret = con = ins = "—"
            overall = f"{cr.overall_score*100:.0f}%" if cr.overall_score is not None else "—"

            result_table.add_row(
                cr.fixture.id, str(len(cr.turn_results)),
                turn_avg, ret, con, ins, overall,
                f"{cr.task_result.latency_ms:.0f}ms",
                f"${cr.task_result.estimated_cost_usd:.6f}",
            )
            if repo:
                repo.save_task_result(cr.task_result, suite="multiturn")

        eff_thinking = thinking_flag
        if eff_thinking is None and model:
            from atomics.benchmark.model_classes import supports_thinking
            if supports_thinking(model):
                eff_thinking = True

        summary = asyncio.run(run_multiturn(
            test_provider,
            judge_provider=judge_provider,
            model=model,
            judge_model=judge_model,
            extra_judges=extra_judge_pairs,
            run_id=mt_run_id,
            on_conversation_done=on_done,
            thinking=eff_thinking,
            thinking_budget=thinking_budget,
            fixtures=selected_fixtures,
        ))

        console.print(result_table)

        summary_table = Table(title="Multi-Turn Summary", show_lines=True)
        summary_table.add_column("Metric", style="dim")
        summary_table.add_column("Value", style="bold")
        summary_table.add_row("Provider", provider_name)
        summary_table.add_row("Model", model or "default")
        ts = summary.avg_turn_score
        summary_table.add_row("Avg Turn Score", f"[green]{ts*100:.1f}%[/green]" if ts else "—")
        cs = summary.avg_conversation_score
        summary_table.add_row("Avg Conversation Score", f"[green]{cs*100:.1f}%[/green]" if cs else "—")
        ret = summary.avg_retention
        summary_table.add_row("Avg Retention", f"{ret*100:.1f}%" if ret else "—")
        con = summary.avg_consistency
        summary_table.add_row("Avg Consistency", f"{con*100:.1f}%" if con else "—")
        summary_table.add_row("Total Turns", str(summary.total_turns))
        summary_table.add_row("Total Tokens", f"{summary.total_tokens:,}")
        summary_table.add_row("Total Cost", f"${summary.total_cost_usd:.6f}")
        console.print(summary_table)

        if json_out:
            write_summary_json(summary, Path(json_out))
            console.print(f"[dim]Wrote JSON results to {json_out}[/dim]")

# ── atomics rag ───────────────────────────────────────────────────────────────

