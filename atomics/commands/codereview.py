"""Secure-code-review CLI command."""

from __future__ import annotations

import asyncio
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
    integrity_exit_code,
    write_summary_json,
)
from atomics.config import load_settings

if TYPE_CHECKING:
    from atomics.eval.codereview.fixtures import SecureCodeFixture
    from atomics.eval.codereview.runner import CodeReviewResult, CodeReviewSummary


@click.command("codereview")
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
def codereview(
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
    """Secure code review — measure planted-vulnerability detection."""
    from atomics.eval.codereview import SECURE_CODE_FIXTURES, run_codereview

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
        f"\n[bold]Secure code review[/bold] — model: [cyan]"
        f"{escape(provider_name)}[/cyan] ({escape(attributed_model)})\n"
        f"Judge: [cyan]{escape(judge_provider_name)}[/cyan] "
        f"({escape(attributed_judge)})\n"
    )

    show_progress = bool((ctx.obj or {}).get("progress", True))
    progress = (
        FixtureProgress(len(SECURE_CODE_FIXTURES), console, label="codereview")
        if show_progress
        else None
    )
    current_index = -1

    def on_start(fixture: SecureCodeFixture) -> None:
        nonlocal current_index
        current_index += 1
        if progress is not None:
            progress.on_start(
                current_index,
                escape(fixture.id),
                escape(fixture.mode),
            )

    def on_done(result: CodeReviewResult) -> None:
        if progress is not None:
            progress.on_done(max(current_index, 0))
        fixture = result.fixture
        expected_label = fixture.cwe if fixture.is_vulnerable else "clean code"
        mark = "[green]OK[/green]" if result.passed else "[red]MISS[/red]"
        console.print(
            f"  {mark} [bold]{escape(fixture.id)}[/bold] "
            f"({escape(fixture.mode)}) verdict={escape(result.verdict)} — "
            f"{escape(expected_label)}"
        )

    summary = asyncio.run(
        run_codereview(
            provider,
            judge_provider=judge,
            model=model,
            judge_model=judge_model,
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
        console.print(f"\n[dim]Wrote JSON results to {escape(str(json_out))}[/dim]")

    _ = save
    if integrity_exit_code(summary.integrity, allow_partial=allow_partial):
        raise click.exceptions.Exit(1)


def _render_summary(
    console: Console,
    summary: CodeReviewSummary,
    *,
    provider_name: str,
    model: str,
    judge_provider: str,
    judge_model: str,
) -> None:
    table = Table(title="Secure Code Review Summary")
    table.add_column("Metric")
    table.add_column("Value")
    table.add_row("Provider", provider_name)
    table.add_row("Model", model)
    table.add_row("Judge", f"{judge_provider} / {judge_model}")
    detection = summary.detection_rate
    false_positive = summary.false_positive_rate
    review = summary.review_score
    table.add_row(
        "Detection rate",
        f"{detection * 100:.1f}%" if detection is not None else "n/a",
    )
    table.add_row(
        "False-positive rate",
        f"{false_positive * 100:.1f}%" if false_positive is not None else "n/a",
    )
    table.add_row(
        "Review score (F1)",
        f"{review * 100:.1f}%" if review is not None else "n/a",
    )
    table.add_row("Fixtures", str(len(summary.results)))
    table.add_row("Integrity", summary.integrity.status.value)
    table.add_row(
        "Fixture coverage",
        f"{summary.integrity.fixture_coverage * 100:.1f}%",
    )
    console.print(table)
