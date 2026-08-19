"""The `judge-agreement` study command."""

from __future__ import annotations

import asyncio
import uuid
from pathlib import Path

import click
from rich.console import Console
from rich.markup import escape

from atomics.commands.common import (
    PROVIDER_CHOICES,
    _make_provider,
    budget_option,
    eval_budget_from,
    parse_extra_judges,
    write_summary_json,
)
from atomics.commands.suite_run import SuiteRun, suite_run
from atomics.config import load_settings
from atomics.eval.agreement import STUDY_SUITES, AgreementSummary, run_agreement_study
from atomics.eval.budget import share_budget
from atomics.storage import MetricsRepository
from atomics.validation import sanitize_error


def _no_parent_finalize(_repository: MetricsRepository, _run_id: str) -> None:
    """Study rows have no parent `runs` row to roll up."""


@click.command("judge-agreement")
@click.option(
    "--suite",
    type=click.Choice(list(STUDY_SUITES), case_sensitive=False),
    required=True,
    help=(
        "Suite whose judge headline to study. "
        "rag is included; probe and archreview are not (no fixture catalog)."
    ),
)
@click.option(
    "--judges",
    type=str,
    required=True,
    help="Comma-separated judges, at least two. "
    "Format: provider:model[@host] "
    "(e.g. ollama:qwen2.5:14b,claude:claude-sonnet-4-6).",
)
@click.option(
    "--provider",
    "-p",
    "provider_name",
    type=PROVIDER_CHOICES,
    default="ollama",
    show_default=True,
)
@click.option("--model", "-m", type=str, default=None)
@click.option("--ollama-host", type=str, default=None)
@click.option("--vllm-host", "vllm_host", type=str, default=None)
@click.option(
    "--fixtures",
    "fixtures_filter",
    type=str,
    default=None,
    help="Comma-separated fixture IDs. Default: the suite's full set.",
)
@click.option(
    "-o",
    "--json-out",
    "json_out",
    type=click.Path(path_type=Path),
    default=None,
)
@click.option(
    "--save/--no-save",
    default=False,
    show_default=True,
    help="Write study rows. Default off — this is not a leaderboard run.",
)
@budget_option
def judge_agreement(
    suite: str,
    judges: str,
    provider_name: str,
    model: str | None,
    ollama_host: str | None,
    vllm_host: str | None,
    fixtures_filter: str | None,
    json_out: Path | None,
    save: bool,
    budget_usd: float | None,
) -> None:
    """Generate each fixture once and score it with every judge.

    Reports pairwise agreement and how often the primary judge's headline
    would flip under the panel. Does not write a parent eval run.
    """
    console = Console()
    settings = load_settings()
    try:
        provider = _make_provider(provider_name, model, ollama_host, settings, vllm_host=vllm_host)
        judge_pairs = parse_extra_judges(
            judges,
            build=lambda name, mdl, host: _make_provider(
                name, mdl, host, settings, vllm_host=vllm_host
            ),
            default_host=ollama_host,
        )
        if len(judge_pairs) < 2:
            raise click.BadParameter(
                "need at least two judges (comma-separated provider:model[@host])",
                param_hint="--judges",
            )
        fixture_ids = (
            [item.strip() for item in fixtures_filter.split(",") if item.strip()]
            if fixtures_filter
            else None
        )
        guarded = share_budget(
            eval_budget_from(budget_usd),
            provider,
            *(p for p, _ in judge_pairs),
        )
        provider = guarded[0]
        judge_pairs = [(guarded[1 + i], mdl) for i, (_, mdl) in enumerate(judge_pairs)]
    except click.ClickException:
        raise
    except Exception as exc:
        raise click.ClickException(f"Judge-agreement setup failed: {sanitize_error(exc)}") from exc

    run_id = uuid.uuid4().hex[:12]
    with suite_run(
        suite="judge-agreement",
        db_path=settings.db_path,
        save=save,
        finalize=_no_parent_finalize,
        failure_prefix="Judge-agreement study failed",
    ) as run:
        summary = asyncio.run(
            run_agreement_study(
                suite=suite.lower(),
                provider=provider,
                judges=judge_pairs,
                model=model,
                fixture_ids=fixture_ids,
                run_id=run_id,
            )
        )
        if run.repository is not None:
            _save_rows(run, summary)
        _render_summary(console, summary)
        if json_out is not None:
            write_summary_json(summary, json_out)
            console.print(f"\n[dim]Wrote JSON results to {escape(str(json_out))}[/dim]")


def _save_rows(run: SuiteRun, summary: AgreementSummary) -> None:
    repository = run.require_repository()
    for row in summary.fixtures:
        repository.save_agreement_result(
            run_id=summary.run_id,
            suite=summary.suite,
            fixture_id=row.fixture_id,
            votes=row.to_dict(),
            agreement=row.agreement,
            flipped=row.flipped,
        )


def _render_summary(console: Console, summary: AgreementSummary) -> None:
    n = len(summary.fixtures)
    pair = f"{summary.pairwise_agreement:.2f}" if summary.pairwise_agreement is not None else "—"
    flip = f"{summary.flip_rate:.2f}" if summary.flip_rate is not None else "—"
    stdev = (
        "—      (categorical)"
        if summary.suite in {"refusal", "codereview"}
        else (f"{summary.mean_stdev:.3f}" if summary.mean_stdev is not None else "—")
    )
    console.print(
        f"\n[bold]judge-agreement[/bold]  suite={escape(summary.suite)}  "
        f"fixtures={n}  judges={summary.n_judges}"
    )
    console.print(f"  pairwise agreement   {pair}")
    console.print(
        f"  majority-flip rate   {flip}   ({summary.n_flipped} of {n} would change the headline)"
    )
    console.print(f"  unresolved (ties)    {summary.n_unresolved}")
    console.print(f"  mean stdev           {stdev}")
    console.print(f"  cost                 ${summary.total_cost_usd:.2f}")
