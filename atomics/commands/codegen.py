"""Code generation evaluation CLI command."""

from __future__ import annotations

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
    effort_options,
    eval_budget_from,
    integrity_exit_code,
    run_async,
    setup_logging,
    write_summary_json,
)
from atomics.commands.suite_run import finalize_task_run, suite_run
from atomics.config import load_settings
from atomics.eval.budget import share_budget


@click.command("codegen")
@click.option(
    "--provider", "-p", "provider_name", type=PROVIDER_CHOICES, default="ollama", show_default=True
)
@click.option("--model", "-m", type=str, default=None, help="Model override.")
@click.option("--ollama-host", type=str, default=None, help="Ollama base URL.")
@click.option("--vllm-host", "vllm_host", type=str, default=None, help="vLLM base URL.")
@click.option("--region", type=str, default="us-east-1", help="AWS region for Bedrock.")
@click.option(
    "--fixtures",
    "fixtures_filter",
    type=str,
    default=None,
    help="Comma-separated fixture IDs (e.g. cg-01,cg-05).",
)
@click.option("--save/--no-save", "save_results", default=True, help="Persist results.")
@click.option(
    "--json-out", "json_out", type=click.Path(dir_okay=False, writable=True), default=None
)
@click.option("--thinking/--no-thinking", "thinking_flag", default=None)
@click.option("--thinking-budget", type=int, default=None)
@effort_options
@click.option(
    "--allow-partial",
    is_flag=True,
    help="Return success for a partial run while preserving integrity details.",
)
@budget_option
def codegen(
    provider_name: str,
    model: str | None,
    ollama_host: str | None,
    vllm_host: str | None,
    region: str,
    fixtures_filter: str | None,
    save_results: bool,
    json_out: str | None,
    thinking_flag: bool | None,
    thinking_budget: int | None,
    effort: str | None,
    reasoning_mode: str | None,
    allow_partial: bool,
    budget_usd: float | None,
) -> None:
    """Code generation evaluation — functional correctness via test execution."""
    settings = load_settings()
    setup_logging(settings.log_level)
    console = Console()

    from atomics.eval.codegen.fixtures import ALL_CODEGEN_FIXTURES
    from atomics.eval.codegen.runner import CodegenFixtureResult, run_codegen

    test_provider = _make_provider(
        provider_name,
        model,
        ollama_host,
        settings,
        vllm_host=vllm_host,
        region=region,
    )
    # Codegen scores by executing tests, not by judging, so the model under
    # test is the only thing that spends here.
    (test_provider,) = share_budget(eval_budget_from(budget_usd), test_provider)

    selected_fixtures = ALL_CODEGEN_FIXTURES
    if fixtures_filter:
        ids = [f.strip() for f in fixtures_filter.split(",")]
        fixture_map = {f.id: f for f in ALL_CODEGEN_FIXTURES}
        missing = [i for i in ids if i not in fixture_map]
        if missing:
            console.print(f"[red]Unknown fixture IDs: {', '.join(missing)}[/red]")
            sys.exit(1)
        selected_fixtures = [fixture_map[i] for i in ids]

    if provider_name == "ollama":
        effective_model = model or settings.ollama_model
    elif provider_name == "vllm":
        effective_model = model or settings.vllm_model
    else:
        effective_model = model or settings.default_model

    console.print(
        f"\n[bold]Code Generation Eval[/bold] — provider: [cyan]{provider_name}[/cyan] | "
        f"model: [cyan]{effective_model}[/cyan]\n"
        f"Fixtures: [bold]{len(selected_fixtures)}[/bold]\n"
    )

    cg_run_id = uuid.uuid4().hex[:12]

    result_table = Table(title="Code Generation Results", show_lines=True)
    result_table.add_column("ID", style="dim")
    result_table.add_column("Function", style="cyan")
    result_table.add_column("Tests", justify="right")
    result_table.add_column("Pass", justify="right", style="green bold")
    result_table.add_column("Rate", justify="right")
    result_table.add_column("Latency", justify="right")
    result_table.add_column("Tokens", justify="right")
    result_table.add_column("Cost", justify="right", style="yellow")

    with suite_run(
        suite="codegen",
        db_path=settings.db_path,
        save=save_results,
        finalize=finalize_task_run,
        failure_prefix="Code generation eval failed",
    ) as run:
        run.begin(
            cg_run_id,
            provider=provider_name,
            model=effective_model,
            trigger="eval",
        )
        repo = run.repository

        def on_done(fr: CodegenFixtureResult) -> None:
            rate_style = "green" if fr.pass_rate == 1.0 else "yellow" if fr.pass_rate > 0 else "red"
            result_table.add_row(
                fr.fixture.id,
                fr.fixture.function_name,
                str(fr.tests_total),
                str(fr.tests_passed),
                f"[{rate_style}]{fr.pass_rate * 100:.0f}%[/{rate_style}]",
                f"{fr.task_result.latency_ms:.0f}ms",
                str(fr.task_result.total_tokens),
                f"${fr.task_result.estimated_cost_usd:.6f}",
            )
            if repo:
                repo.save_task_result(fr.task_result, suite="codegen")

        eff_thinking = thinking_flag
        if eff_thinking is None and model:
            from atomics.benchmark.model_classes import supports_thinking

            if supports_thinking(model):
                eff_thinking = True

        summary = run_async(
            run_codegen(
                test_provider,
                model=model,
                run_id=cg_run_id,
                on_fixture_done=on_done,
                thinking=eff_thinking,
                thinking_budget=thinking_budget,
                effort=effort,
                reasoning_mode=reasoning_mode,
                fixtures=selected_fixtures,
            ),
            test_provider,
        )

        console.print(result_table)

        summary_table = Table(title="Code Generation Summary", show_lines=True)
        summary_table.add_column("Metric", style="dim")
        summary_table.add_column("Value", style="bold")
        summary_table.add_row("Provider", provider_name)
        summary_table.add_row("Model", effective_model)
        pr = summary.overall_pass_rate
        pr_style = "green" if pr and pr >= 0.8 else "yellow" if pr and pr >= 0.5 else "red"
        summary_table.add_row(
            "Overall Pass Rate",
            f"[{pr_style}]{pr * 100:.1f}%[/{pr_style}]" if pr is not None else "\u2014",
        )
        summary_table.add_row(
            "Fully Correct", f"{summary.fixtures_fully_correct}/{len(summary.fixture_results)}"
        )
        summary_table.add_row("Total Tokens", f"{summary.total_tokens:,}")
        summary_table.add_row("Total Cost", f"${summary.total_cost_usd:.6f}")
        console.print(summary_table)

        if json_out:
            write_summary_json(summary, Path(json_out))
            console.print(f"[dim]Wrote JSON results to {json_out}[/dim]")

        if integrity_exit_code(summary.integrity, allow_partial=allow_partial):
            raise click.exceptions.Exit(1)
