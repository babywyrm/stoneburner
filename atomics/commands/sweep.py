"""Multi-model eval sweep CLI command."""

from __future__ import annotations

from collections.abc import Callable
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
    run_async,
    setup_logging,
)
from atomics.config import load_settings
from atomics.eval.budget import BudgetMeter
from atomics.providers.base import BaseProvider, aclose_providers


def suite_persister(db_path: Path) -> Callable[[str, str, str, object], None]:
    """Save one sweep job's rows the way that suite's own command does."""
    from atomics.commands.common import evaluation_record_from_fixture
    from atomics.commands.suite_run import (
        finalize_evaluation_run,
        finalize_task_run,
        suite_run,
    )
    from atomics.commands.toolcall import _save as save_toolcall
    from atomics.eval.codereview.runner import CodeReviewSummary
    from atomics.eval.redblue.runner import RedBlueSummary
    from atomics.eval.refusal.runner import RefusalSummary
    from atomics.eval.runner import EvalRunSummary
    from atomics.eval.toolcall.runner import ToolCallSummary

    def persist(suite: str, provider_name: str, model: str, summary: object) -> None:
        if isinstance(summary, ToolCallSummary):
            save_toolcall(summary, db_path=db_path)
            return
        judged = isinstance(summary, (RefusalSummary, CodeReviewSummary))
        with suite_run(
            suite=suite,
            db_path=db_path,
            save=True,
            finalize=finalize_evaluation_run if judged else finalize_task_run,
            failure_prefix=f"Sweep {suite} save failed",
        ) as run:
            if isinstance(summary, RedBlueSummary):
                run.begin(
                    summary.run_id,
                    provider=provider_name,
                    model=model,
                    tier=f"redblue-{summary.mode}",
                    pass_count=summary.runs,
                )
                repo = run.require_repository()
                for fr in summary.results:
                    repo.save_task_result(fr.task_result, suite=f"redblue-{fr.fixture.team}")
            elif isinstance(summary, EvalRunSummary):
                run.begin(summary.run_id, provider=provider_name, model=model, trigger="eval")
                repo = run.require_repository()
                for er in summary.fixture_results:
                    repo.save_task_result(er.task_result)
            elif isinstance(summary, (RefusalSummary, CodeReviewSummary)):
                run.begin(summary.run_id, provider=provider_name, model=model)
                repo = run.require_repository()
                for result in summary.results:
                    repo.save_evaluation_result(
                        evaluation_record_from_fixture(
                            run_id=summary.run_id,
                            suite=suite,
                            provider=provider_name,
                            model=model,
                            payload=result.to_dict(),
                        )
                    )
            else:
                raise TypeError(f"no saver for {type(summary).__name__}")

    return persist


@click.command("sweep")
@click.option(
    "--provider",
    "-p",
    "provider_name",
    type=PROVIDER_CHOICES,
    default="ollama",
    help="Provider to evaluate (default: ollama for local models)",
)
@click.option(
    "--models",
    type=str,
    default=None,
    help="Comma-separated list of models to sweep (e.g. qwen2.5:1.5b,mistral:7b)",
)
@click.option(
    "--all-local",
    "all_local",
    is_flag=True,
    default=False,
    help="Discover and sweep all models on the Ollama host (ollama provider only)",
)
@click.option(
    "--ollama-host",
    "ollama_host",
    type=str,
    default=None,
    help="Ollama host URL (default: ATOMICS_OLLAMA_HOST)",
)
@click.option(
    "--host",
    "ollama_host",
    type=str,
    default=None,
    hidden=True,
    help="Deprecated alias for --ollama-host",
)
@click.option(
    "--vllm-host",
    "vllm_host",
    type=str,
    default=None,
    help="vLLM/OpenAI-compatible base URL (default: ATOMICS_VLLM_HOST)",
)
@click.option(
    "--judge-provider",
    "judge_provider_name",
    type=PROVIDER_CHOICES,
    default="ollama",
    help="Provider for quality judge (default: ollama — $0)",
)
@click.option("--judge-model", type=str, default=None, help="Judge model override")
@click.option("--judge-host", type=str, default=None, help="Ollama host for judge")
@click.option(
    "--fixtures", type=str, default=None, help="Comma-separated fixture IDs (default: all)"
)
@click.option("--thinking/--no-thinking", "thinking_flag", default=None)
@click.option("--thinking-budget", type=int, default=None)
@effort_options
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    default=False,
    help="Print each model's full reply alongside scores",
)
@click.option(
    "--save/--no-save",
    "save_results",
    default=False,
    help="Persist results to the database (default: off). With --suites, each "
    "model×suite is saved as its suite's own command would save it.",
)
@click.option(
    "--suites",
    type=str,
    default="eval",
    show_default=True,
    help="Comma-separated suites: eval, redblue, refusal, toolcall, codereview",
)
@click.option(
    "--runs",
    type=click.IntRange(min=1),
    default=1,
    show_default=True,
    help="Passes per fixture for suites that support --runs (redblue, toolcall).",
)
@click.option(
    "--status",
    "status_path",
    type=click.Path(dir_okay=False, writable=True),
    default=None,
    help="Rewrite a JSON status file after each model×suite job.",
)
@click.option(
    "--log",
    "log_path",
    type=click.Path(dir_okay=False, writable=True),
    default=None,
    help="Append a detachable log. Survives a dead chat (SIGPIPE).",
)
@click.option(
    "--models-from",
    "models_from",
    type=click.Choice(["ollama"]),
    default=None,
    help="Discover models from the provider (ollama is --all-local).",
)
@click.option(
    "--resume",
    is_flag=True,
    default=False,
    help="Keep jobs the --status file records as ok. Re-run failed and unfinished ones.",
)
@budget_option
def sweep(
    provider_name: str,
    models: str | None,
    all_local: bool,
    ollama_host: str | None,
    vllm_host: str | None,
    judge_provider_name: str,
    judge_model: str | None,
    judge_host: str | None,
    fixtures: str | None,
    thinking_flag: bool | None,
    thinking_budget: int | None,
    effort: str | None,
    reasoning_mode: str | None,
    verbose: bool,
    save_results: bool,
    suites: str,
    runs: int,
    status_path: str | None,
    log_path: str | None,
    models_from: str | None,
    resume: bool,
    budget_usd: float | None,
) -> None:
    """Sweep eval fixtures across multiple models and compare results.

    Works with any provider — local Ollama, Claude, OpenAI, Bedrock, or brain-gateway.
    Use --all-local with ollama to auto-discover models, or --models for any provider.

    Examples:
      atomics sweep --all-local --ollama-host http://gpu-host:11434
      atomics sweep --models qwen2.5:1.5b,qwen2.5:3b,mistral:7b
      atomics sweep --provider claude --models claude-sonnet-4-6,claude-haiku-4-5-20251001
      atomics sweep --provider openai --models gpt-4o,gpt-4o-mini
      atomics sweep --all-local --fixtures ev-01,ev-02,ev-03
      atomics sweep --all-local --save
      atomics sweep --suites redblue,refusal,toolcall,codereview --runs 3 \\
        --no-thinking --models-from ollama --status sweep.status.json --log sweep.log
    """
    from pathlib import Path

    from atomics.benchmark.sweep import ModelSweepResult, run_model_sweep
    from atomics.eval.gauntlet import (
        format_headline_cell,
        ignore_broken_pipe,
        make_suite_runner,
        parse_suites,
        run_gauntlet,
    )

    if resume and not status_path:
        raise click.UsageError("--resume needs --status: that file is what it resumes from")
    settings = load_settings()
    setup_logging(settings.log_level)
    console = Console()
    effective_host = ollama_host or settings.ollama_host
    try:
        suite_list = parse_suites(suites)
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc
    if models_from == "ollama":
        all_local = True

    if all_local:
        if provider_name != "ollama":
            console.print("[red]--all-local only works with --provider ollama[/red]")
            raise SystemExit(1)
        from atomics.providers.ollama import OllamaProvider

        disc = OllamaProvider(host=effective_host)
        try:
            available = run_async(disc.list_models(), disc)
        except ConnectionError as exc:
            click.echo(str(exc), err=True)
            raise SystemExit(1)
        model_list = [str(m["name"]) for m in available]
        if not model_list:
            console.print("[red]No models found on Ollama host.[/red]")
            raise SystemExit(1)
        console.print(f"[bold]Discovered {len(model_list)} models on {effective_host}[/bold]\n")
    elif models:
        model_list = [m.strip() for m in models.split(",") if m.strip()]
    else:
        console.print("[red]Specify --models or --all-local[/red]")
        raise SystemExit(1)

    fixture_ids = [f.strip() for f in fixtures.split(",") if f.strip()] if fixtures else None

    # A sweep builds one provider per model as it goes, so the ceiling has to
    # outlive any single provider. Metering each model separately would make an
    # N-model sweep cost N times what --budget asked for.
    meter = BudgetMeter(eval_budget_from(budget_usd))

    created: list[BaseProvider] = []

    def provider_factory(model_name: str):
        wrapped = meter.wrap(
            _make_provider(provider_name, model_name, ollama_host, settings, vllm_host=vllm_host)
        )
        created.append(wrapped)
        return wrapped

    async def _with_created(coro):
        try:
            return await coro
        finally:
            await aclose_providers(*created)

    judge_provider = meter.wrap(
        _make_provider(
            judge_provider_name,
            judge_model,
            judge_host or effective_host,
            settings,
            vllm_host=vllm_host,
        )
    )

    if suite_list != ["eval"] or status_path or log_path:
        ignore_broken_pipe()
        suite_results = run_async(
            _with_created(
                run_gauntlet(
                    models=model_list,
                    suites=suite_list,
                    run_suite=make_suite_runner(
                        provider_factory=provider_factory,
                        judge_provider=judge_provider,
                        judge_model=judge_model,
                        runs=runs,
                        thinking=thinking_flag,
                        thinking_budget=thinking_budget,
                        fixture_ids=fixture_ids,
                        effort=effort,
                        reasoning_mode=reasoning_mode,
                        persist=suite_persister(settings.db_path) if save_results else None,
                    ),
                    status_path=Path(status_path) if status_path else None,
                    log_path=Path(log_path) if log_path else None,
                    skip_incapable=False,
                    resume=resume,
                )
            ),
            judge_provider,
        )
        table = Table(title="Suite Sweep Results", show_lines=True)
        table.add_column("Model", style="cyan bold")
        table.add_column("Suite")
        table.add_column("Result", justify="right")
        table.add_column("Headline", justify="right")
        for row in suite_results:
            mark = "[green]OK[/green]" if row.ok else "[red]FAIL[/red]"
            if row.tool_capable is False:
                mark = "[yellow]SKIP[/yellow]" if row.ok else "[red]INCAPABLE[/red]"
            table.add_row(row.model, row.suite, mark, format_headline_cell(row))
            try:
                console.print(f"  [dim]{row.suite}[/dim] {row.model}")
            except BrokenPipeError:
                pass
        try:
            console.print(table)
        except BrokenPipeError:
            pass
        if any(not row.ok for row in suite_results):
            raise SystemExit(1)
        return

    result_table = Table(title="Model Sweep Results", show_lines=True)
    result_table.add_column("Model", style="cyan bold")
    result_table.add_column("Quality", justify="right")
    result_table.add_column("Avg Latency", justify="right")
    result_table.add_column("Tokens", justify="right")
    result_table.add_column("Cost", justify="right", style="yellow")
    result_table.add_column("Fixtures", justify="right", style="dim")

    def on_fixture_done_verbose(fr) -> None:
        if not verbose:
            return
        from atomics.eval.display import format_eval_verbose_block

        console.print()
        console.print(format_eval_verbose_block(fr), markup=False)

    def on_model_done(r: ModelSweepResult) -> None:
        q = (
            f"[green]{r.overall_quality * 100:.0f}%[/green]"
            if r.overall_quality is not None
            else "[red]FAIL[/red]"
        )
        result_table.add_row(
            r.model,
            q,
            f"{r.avg_latency_ms:.0f}ms",
            f"{r.total_tokens:,}",
            f"${r.total_cost_usd:.6f}",
            str(r.fixtures_run),
        )
        console.print(f"\n  [dim]Done:[/dim] {r.model}")

    console.print(
        f"[bold]Sweeping {len(model_list)} models × "
        f"{'all' if fixture_ids is None else len(fixture_ids)} fixtures[/bold]\n"
    )

    results = run_async(
        _with_created(
            run_model_sweep(
                provider_factory=provider_factory,
                judge_provider=judge_provider,
                models=model_list,
                fixture_ids=fixture_ids,
                judge_model=judge_model,
                thinking=thinking_flag,
                thinking_budget=thinking_budget,
                effort=effort,
                reasoning_mode=reasoning_mode,
                on_model_done=on_model_done,
                on_fixture_done=on_fixture_done_verbose,
            )
        ),
        judge_provider,
    )

    console.print(result_table)

    ranked = sorted(
        [r for r in results if r.overall_quality is not None],
        key=lambda r: r.overall_quality or 0,
        reverse=True,
    )
    if ranked:
        best = ranked[0]
        console.print(
            f"\n[bold green]Best local model:[/bold green] {best.model} "
            f"({(best.overall_quality or 0) * 100:.0f}% quality, "
            f"{best.avg_latency_ms:.0f}ms avg latency, "
            f"${best.total_cost_usd:.2f} total cost)"
        )

    failed = [r for r in results if r.overall_quality is None]
    if failed:
        console.print(f"\n[yellow]{len(failed)} model(s) failed:[/yellow]")
        for r in failed:
            reason = f" — [dim]{r.error}[/dim]" if r.error else ""
            console.print(f"[yellow]  • {r.model}[/yellow]{reason}")

    if save_results:
        from atomics.storage.repository import MetricsRepository

        repo = MetricsRepository(settings.db_path)
        for r in results:
            repo.save_sweep_result(r)
        repo.close()
        console.print("\n[dim]Sweep results saved to database.[/dim]")
