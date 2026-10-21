"""The `judge-calibrate` command."""

from __future__ import annotations

import uuid
from pathlib import Path

import click
import yaml
from rich.console import Console
from rich.markup import escape

from atomics.commands.common import (
    PROVIDER_CHOICES,
    _make_provider,
    budget_option,
    effective_model,
    eval_budget_from,
    integrity_exit_code,
    run_async,
    write_summary_json,
)
from atomics.commands.suite_run import suite_run
from atomics.config import load_settings
from atomics.eval.budget import share_budget
from atomics.eval.calibration import (
    CALIBRATE_SUITE,
    DEFAULT_MAX_ERROR,
    ItemCalibration,
    JudgeCalibration,
    default_set_path,
    load_graded_set,
    run_judge_calibration,
    save_calibration,
)
from atomics.eval.outcomes import RunStatus
from atomics.storage import MetricsRepository
from atomics.validation import sanitize_error


def _no_finalize(_repository: MetricsRepository, _run_id: str) -> None:
    """save_calibration completes its own parent run."""


@click.command("judge-calibrate")
@click.option(
    "--judge-provider",
    "judge_provider_name",
    type=PROVIDER_CHOICES,
    default="ollama",
    show_default=True,
)
@click.option("--judge-model", type=str, default=None)
@click.option("--judge-host", type=str, default=None)
@click.option(
    "--set",
    "set_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="Graded set YAML. Default: the shipped blue red/blue set (6 items).",
)
@click.option(
    "--max-error",
    type=click.FloatRange(0.0, 1.0),
    default=DEFAULT_MAX_ERROR,
    show_default=True,
    help="Pass when the mean error against the human grade is at most this.",
)
@click.option(
    "--save/--no-save",
    default=True,
    show_default=True,
    help="Store the result so judged suites can label this judge.",
)
@click.option(
    "-o",
    "--json-out",
    "json_out",
    type=click.Path(path_type=Path),
    default=None,
)
@budget_option
def judge_calibrate(
    judge_provider_name: str,
    judge_model: str | None,
    judge_host: str | None,
    set_path: Path | None,
    max_error: float,
    save: bool,
    json_out: Path | None,
    budget_usd: float | None,
) -> None:
    """Score a judge against a human-graded set: mean error, bias, pass or fail.

    Judged suites print the stored result next to their judge. The label is
    advisory: it never changes a suite's scores, integrity, or exit code.
    """
    console = Console()
    settings = load_settings()
    try:
        graded = load_graded_set(set_path or default_set_path())
    except (OSError, ValueError, yaml.YAMLError) as exc:
        raise click.BadParameter(sanitize_error(exc), param_hint="--set") from exc
    judge = _make_provider(judge_provider_name, judge_model, judge_host, settings)
    judge = share_budget(eval_budget_from(budget_usd), judge)[0]

    console.print(
        f"\n[bold]Judge calibration[/bold] — judge: [cyan]{escape(judge_provider_name)}[/cyan] "
        f"({escape(effective_model(judge_model, judge))})\n"
        f"Set: [bold]{escape(graded.name)}[/bold] ({graded.digest}) | "
        f"items: [bold]{len(graded.items)}[/bold]\n"
    )
    calibration = run_async(
        run_judge_calibration(
            judge,
            provider_name=judge_provider_name,
            judge_model=judge_model,
            graded=graded,
            max_error=max_error,
            on_item=lambda row: _print_item(console, row),
        ),
        judge,
    )
    complete = calibration.integrity.status is RunStatus.COMPLETE
    with suite_run(
        suite=CALIBRATE_SUITE,
        db_path=settings.db_path,
        save=save and complete,
        finalize=_no_finalize,
        failure_prefix="Judge calibration save failed",
    ) as run:
        if run.repository is not None:
            save_calibration(run.repository, uuid.uuid4().hex[:12], calibration)
    _render_summary(console, calibration, saved=run.repository is not None)
    if json_out is not None:
        write_summary_json(calibration, json_out)
        console.print(f"[dim]Wrote JSON results to {escape(str(json_out))}[/dim]")
    if not complete:
        raise SystemExit(integrity_exit_code(calibration.integrity, allow_partial=False) or 1)
    if not calibration.passed:
        raise SystemExit(1)


def _print_item(console: Console, row: ItemCalibration) -> None:
    if row.judge is None:
        console.print(f"  {escape(row.id)}  human {row.human:.2f}  judge  n/a")
        return
    console.print(
        f"  {escape(row.id)}  human {row.human:.2f}  judge {row.judge:.2f}  "
        f"error {row.judge - row.human:+.2f}"
    )


def _render_summary(console: Console, calibration: JudgeCalibration, *, saved: bool) -> None:
    n = len(calibration.items)
    scored = len(calibration.pairs)
    if scored < n:
        console.print(f"\n  [bold]INCOMPLETE[/bold]  n/a ({scored}/{n} scored); nothing stored")
        return
    order = calibration.pair_order
    console.print(
        f"\n  mean error   {calibration.mae:.3f}   (max {calibration.max_error:.3f}; "
        f"each item is 1/{n} of it)"
    )
    console.print(f"  bias         {calibration.bias:+.3f}   (positive: the judge scores high)")
    console.print(f"  pair order   {'n/a' if order is None else f'{order:.3f}'}")
    verdict = "[green]PASS[/green]" if calibration.passed else "[red]FAIL[/red]"
    console.print(f"  {verdict}  {'stored' if saved else 'not stored (--no-save)'}")
