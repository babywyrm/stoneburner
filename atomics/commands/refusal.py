"""Refusal-calibration CLI command."""

from __future__ import annotations

import asyncio
import uuid
from pathlib import Path
from typing import TYPE_CHECKING

import click
from rich.console import Console
from rich.markup import escape
from rich.table import Table

from atomics.commands.common import (
    PROVIDER_CHOICES,
    FixtureProgress,
    _make_provider,
    effective_model,
    evaluation_record_from_fixture,
    integrity_exit_code,
    write_summary_json,
)
from atomics.config import load_settings
from atomics.storage import MetricsRepository
from atomics.validation import sanitize_error

if TYPE_CHECKING:
    from atomics.eval.refusal.fixtures import RefusalFixture
    from atomics.eval.refusal.runner import RefusalResult, RefusalSummary


@click.command("refusal")
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
    "--judge-provider",
    "judge_provider_name",
    type=PROVIDER_CHOICES,
    default="ollama",
    show_default=True,
)
@click.option("--judge-model", type=str, default=None)
@click.option("--judge-host", type=str, default=None)
@click.option(
    "-o",
    "--json-out",
    "json_out",
    type=click.Path(path_type=Path),
    default=None,
)
@click.option("--save/--no-save", default=True, show_default=True)
@click.option(
    "--allow-partial",
    is_flag=True,
    help="Return success for a partial run while preserving integrity details.",
)
@click.pass_context
def refusal(
    ctx: click.Context,
    provider_name: str,
    model: str | None,
    ollama_host: str | None,
    vllm_host: str | None,
    judge_provider_name: str,
    judge_model: str | None,
    judge_host: str | None,
    json_out: Path | None,
    save: bool,
    allow_partial: bool,
) -> None:
    """Refusal calibration — measure over- and under-refusal."""
    from atomics.eval.refusal import REFUSAL_FIXTURES, run_refusal

    console = Console()
    settings = load_settings()
    provider = _make_provider(
        provider_name,
        model,
        ollama_host,
        settings,
        vllm_host=vllm_host,
    )
    judge = _make_provider(
        judge_provider_name,
        judge_model,
        judge_host or ollama_host,
        settings,
        vllm_host=vllm_host,
    )
    attributed_model = effective_model(model, provider)
    attributed_judge = effective_model(judge_model, judge)
    console.print(
        f"\n[bold]Refusal calibration[/bold] — model: [cyan]"
        f"{escape(provider_name)}[/cyan] ({escape(attributed_model)})\n"
        f"Judge: [cyan]{escape(judge_provider_name)}[/cyan] "
        f"({escape(attributed_judge)})\n"
    )

    show_progress = bool((ctx.obj or {}).get("progress", True))
    progress = (
        FixtureProgress(len(REFUSAL_FIXTURES), console, label="refusal")
        if show_progress
        else None
    )
    current_index = -1
    run_id = uuid.uuid4().hex[:12]
    repository: MetricsRepository | None = None
    parent_created = False

    def on_start(fixture: RefusalFixture) -> None:
        nonlocal current_index
        current_index += 1
        if progress is not None:
            progress.on_start(
                current_index,
                escape(fixture.id),
                escape(fixture.category),
            )

    def on_done(result: RefusalResult) -> None:
        if progress is not None:
            progress.on_done(max(current_index, 0))
        mark = "[green]OK[/green]" if result.correct else "[red]MISS[/red]"
        flag = ""
        if result.over_refusal:
            flag = " [yellow](over-refusal)[/yellow]"
        elif result.under_refusal:
            flag = " [red](under-refusal)[/red]"
        console.print(
            f"  {mark} [bold]{escape(result.fixture.id)}[/bold] "
            f"expected={escape(result.fixture.expected)} "
            f"got={escape(result.classification)}{flag}"
        )
        if repository is not None:
            repository.save_evaluation_result(
                evaluation_record_from_fixture(
                    run_id=run_id,
                    suite="refusal",
                    provider=provider_name,
                    model=attributed_model,
                    payload=result.to_dict(),
                )
            )

    try:
        if save:
            repository = MetricsRepository(settings.db_path)
            repository.create_run(
                run_id,
                tier="refusal",
                provider=provider_name,
                model=attributed_model,
            )
            parent_created = True
        summary = asyncio.run(
            run_refusal(
                provider,
                judge_provider=judge,
                model=model,
                judge_model=judge_model,
                run_id=run_id,
                on_fixture_start=on_start,
                on_fixture_done=on_done,
            )
        )
        _render_summary(
            console,
            summary,
            provider_name=provider_name,
            model=attributed_model,
            judge_provider=judge_provider_name,
            judge_model=attributed_judge,
        )

        if json_out is not None:
            write_summary_json(summary, json_out)
            console.print(
                f"\n[dim]Wrote JSON results to {escape(str(json_out))}[/dim]"
            )

        if integrity_exit_code(summary.integrity, allow_partial=allow_partial):
            raise click.exceptions.Exit(1)
    except (click.ClickException, click.exceptions.Exit):
        raise
    except Exception as exc:
        raise click.ClickException(
            f"Refusal evaluation failed: {sanitize_error(exc)}"
        ) from exc
    finally:
        if repository is not None:
            try:
                if parent_created:
                    repository.complete_evaluation_run(run_id)
            finally:
                repository.close()


def _render_summary(
    console: Console,
    summary: RefusalSummary,
    *,
    provider_name: str,
    model: str,
    judge_provider: str,
    judge_model: str,
) -> None:
    table = Table(title="Refusal Calibration Summary")
    table.add_column("Metric")
    table.add_column("Value")
    table.add_row("Provider", provider_name)
    table.add_row("Model", model)
    table.add_row("Judge", f"{judge_provider} / {judge_model}")
    calibration = summary.calibration_score
    over_refusal = summary.over_refusal_rate
    under_refusal = summary.under_refusal_rate
    table.add_row(
        "Calibration score",
        f"{calibration * 100:.1f}%" if calibration is not None else "n/a",
    )
    table.add_row(
        "Over-refusal rate",
        f"{over_refusal * 100:.1f}%" if over_refusal is not None else "n/a",
    )
    table.add_row(
        "Under-refusal rate",
        f"{under_refusal * 100:.1f}%" if under_refusal is not None else "n/a",
    )
    table.add_row("Fixtures", str(len(summary.results)))
    table.add_row("Integrity", summary.integrity.status.value)
    table.add_row(
        "Fixture coverage",
        f"{summary.integrity.fixture_coverage * 100:.1f}%",
    )
    console.print(table)
