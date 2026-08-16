"""The `atomics toolcall` command.

Its own module rather than growing `commands/security.py`, already the largest
command file in the project.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import click
from rich.console import Console
from rich.table import Table

from atomics.commands.common import (
    PROVIDER_CHOICES,
    FixtureProgress,
    _make_provider,
    budget_option,
    effective_model,
    eval_budget_from,
    extra_judges_option,
    parse_extra_judges,
    write_summary_json,
)
from atomics.commands.suite_run import finalize_evaluation_run, suite_run
from atomics.config import load_settings
from atomics.eval.budget import share_budget
from atomics.eval.toolcall.fixtures import (
    ALL_FIXTURES,
    GROUP_ALIASES,
    fixtures_for_category,
)
from atomics.eval.toolcall.runner import ERROR_OUTCOME, run_toolcall_suite
from atomics.eval.toolcall.scorer import ToolOutcome
from atomics.storage.records import EvaluationResultRecord

console = Console()

_SUITE = "toolcall"

# Worst first: the table is read top-down and the dangerous rows are the point.
_OUTCOME_STYLE = {
    ToolOutcome.DANGEROUS_CALL: ("red", "DANGEROUS"),
    ToolOutcome.MALFORMED_CALL: ("yellow", "MALFORMED"),
    ToolOutcome.SAFE_CALL: ("green", "safe call"),
    ToolOutcome.NO_CALL: ("cyan", "no call"),
    ERROR_OUTCOME: ("magenta", "error"),
}

_ALIAS_HELP = ", ".join(sorted(GROUP_ALIASES))


def _valid_categories() -> set[str]:
    return set(GROUP_ALIASES) | {f.category for f in ALL_FIXTURES}


def _rate(rate: float | None, numerator: int, denominator: int) -> str:
    """Render a rate with its denominator, or say plainly that it was not measured.

    A bare percentage is not interpretable here: both divergence measures are
    computed over subsets, and an empty denominator means "no fixture qualified",
    not "no divergence found".
    """
    if rate is None:
        return "[dim]not measured (no qualifying fixtures)[/dim]"
    return f"{rate * 100:.0f}%  ({numerator}/{denominator})"


@click.command("toolcall")
@click.option(
    "--provider", "-p", "provider_name",
    type=PROVIDER_CHOICES, default="ollama", show_default=True,
    help="Provider for the model under test. It must support tool calling.",
)
@click.option("--model", "-m", type=str, default=None,
              help="Model override for the provider under test.")
@click.option("--ollama-host", type=str, default=None,
              help="Ollama base URL for the model under test.")
@click.option("--vllm-host", "vllm_host", type=str, default=None,
              help="vLLM/OpenAI-compatible base URL for the model under test.")
@click.option("--judge-provider", "judge_provider_name",
              type=PROVIDER_CHOICES, default=None,
              help="Judge for the prose channel and the tool-channel text. "
                   "Omit to skip judging: the tool channel is scored "
                   "deterministically and needs no judge, but divergence against "
                   "the prose channel cannot be computed without one.")
@click.option("--judge-model", type=str, default=None, help="Judge model override.")
@click.option("--judge-host", type=str, default=None,
              help="Ollama base URL for the judge.")
@extra_judges_option
@click.option("--runs", type=click.IntRange(min=1), default=1, show_default=True,
              help="Run each fixture N times. Both channels run per pass and pair "
                   "within it; the reported outcome is the modal one.")
@click.option("--category", type=str, default=None,
              help=f"Category or group alias to run (default: all). "
                   f"Group aliases: {_ALIAS_HELP}.")
@click.option("--channel", type=click.Choice(["both", "tools", "prose"]),
              default="both", show_default=True,
              help="Which channels to run. Divergence needs both; a single-channel "
                   "run reports the outcome distribution only.")
@click.option("--skip-incapable/--no-skip-incapable", default=True, show_default=True,
              help="Exit zero when the model cannot emit tool calls. With "
                   "--no-skip-incapable that is an error, which is what you want "
                   "in a sweep where a silently skipped model looks like a pass.")
@click.option("--save/--no-save", "save_results", default=True, show_default=True)
@click.option("--json-out", "json_out",
              type=click.Path(dir_okay=False, writable=True), default=None,
              help="Write the full run as JSON to this file.")
@click.option("--verbose", "-v", is_flag=True,
              help="Show the emitted call arguments and the accompanying text.")
@click.option("--thinking/--no-thinking", "thinking_flag", default=None)
@click.option("--thinking-budget", type=int, default=8000, show_default=True)
@budget_option
def toolcall(
    provider_name: str,
    model: str | None,
    ollama_host: str | None,
    vllm_host: str | None,
    judge_provider_name: str | None,
    judge_model: str | None,
    judge_host: str | None,
    extra_judges: str | None,
    runs: int,
    category: str | None,
    channel: str,
    skip_incapable: bool,
    save_results: bool,
    json_out: str | None,
    verbose: bool,
    thinking_flag: bool | None,
    thinking_budget: int,
    budget_usd: float | None,
) -> None:
    """Measure tool-call divergence: refuses in prose, complies with a function.

    Each fixture is sent twice — once as plain conversation with no tools, once
    with tool schemas attached — and the gap between those two answers is the
    result. The tool channel is scored deterministically: a declared dangerous
    call either was emitted or was not, with no judge involved.

    Tool calls are never executed. The suite records what the model asked to do.
    """
    if category and category not in _valid_categories():
        raise click.BadParameter(
            f"{category!r} is not a known category. "
            f"Valid: {', '.join(sorted(_valid_categories()))}",
            param_hint="--category",
        )
    fixtures = fixtures_for_category(category)

    settings = load_settings()
    provider = _make_provider(
        provider_name, model, ollama_host, settings, vllm_host=vllm_host
    )
    judge = None
    extra_judge_pairs: list = []
    if judge_provider_name:
        judge = _make_provider(
            judge_provider_name, judge_model, judge_host or ollama_host, settings,
            vllm_host=vllm_host,
        )
        extra_judge_pairs = parse_extra_judges(
            extra_judges,
            build=lambda name, mdl, host: _make_provider(
                name, mdl, host, settings, vllm_host=vllm_host
            ),
            default_host=judge_host or ollama_host,
        )

    # The judge is optional here — the tool channel is scored deterministically
    # — so the pair is built conditionally to keep one shared ceiling either way.
    budget = eval_budget_from(budget_usd)
    if judge is None:
        (provider,) = share_budget(budget, provider)
    else:
        guarded = share_budget(
            budget, provider, judge, *(p for p, _ in extra_judge_pairs)
        )
        provider, judge = guarded[0], guarded[1]
        extra_judge_pairs = [
            (guarded[2 + i], mdl) for i, (_, mdl) in enumerate(extra_judge_pairs)
        ]

    resolved_model = effective_model(model, provider)
    console.print(
        f"\n[bold]Tool-call divergence[/bold] — {provider_name}/{resolved_model}"
    )
    console.print(
        f"[dim]{len(fixtures)} fixtures · channel={channel} · runs={runs} · "
        f"judge={judge_provider_name or 'none'} · calls are never executed[/dim]\n"
    )
    if judge is None and channel != "tools":
        console.print(
            "[yellow]No judge: prose stays unjudged and channel divergence "
            "cannot be measured. Pass --judge-provider to score the "
            "refused-in-chat / complied-with-tools gap.[/yellow]"
        )
    if channel != "prose":
        console.print(
            "[dim]Capability probe — checking that the model can emit a tool "
            "call before scoring silence as refusal...[/dim]"
        )

    progress = FixtureProgress(len(fixtures), console, label="toolcall")

    def _called(record: dict) -> str:
        return ", ".join(
            c.get("name", "") for c in record.get("calls") or []
        ) or "none"

    def on_start(index: int, fixture) -> None:
        progress.on_start(index, fixture.id, fixture.category)
        console.print(
            f"  [{index + 1}/{len(fixtures)}] {fixture.id} ({fixture.category}) "
            f"— generating..."
        )

    def on_run(index: int, fixture, run_number: int, run_count: int, record: dict) -> None:
        if run_count <= 1:
            return
        style, label = _OUTCOME_STYLE.get(record.get("tool_outcome"), ("white", "?"))
        console.print(
            f"    {fixture.id} run {run_number + 1}/{run_count} "
            f"— [{style}]{label}[/{style}]  called={_called(record)}"
        )

    def on_done(index: int, fixture, aggregated: dict) -> None:
        progress.on_done(index)
        style, label = _OUTCOME_STYLE.get(aggregated.get("tool_outcome"), ("white", "?"))
        prose = aggregated.get("prose_label") or "—"
        console.print(
            f"  [{index + 1}/{len(fixtures)}] {fixture.id} ({fixture.category}) "
            f"— [{style}]{label}[/{style}]  prose={prose}  called={_called(aggregated)}"
        )

    summary = asyncio.run(
        run_toolcall_suite(
            provider=provider,
            model=model,
            judge_provider=judge,
            judge_model=judge_model,
            extra_judges=extra_judge_pairs,
            fixtures=fixtures,
            runs=runs,
            channel=channel,
            thinking=thinking_flag,
            thinking_budget=thinking_budget,
            on_fixture_start=on_start,
            on_fixture_done=on_done,
            on_run_done=on_run,
        )
    )

    if not summary.tool_capable:
        console.print(
            f"[yellow]{provider_name}/{resolved_model} did not emit a tool call for "
            f"the capability probe.[/yellow]"
        )
        console.print(
            "[dim]No fixtures were run. A model that cannot emit tool calls would "
            "produce no calls on every fixture, which is indistinguishable from "
            "refusing all of them — scoring that as resistance would rank it "
            "above models that genuinely resist.[/dim]\n"
        )
        if json_out:
            write_summary_json(summary, Path(json_out))
        raise SystemExit(0 if skip_incapable else 1)

    _render_results(summary, verbose=verbose)
    _render_summary(summary)

    if json_out:
        write_summary_json(summary, Path(json_out))
        console.print(f"[dim]JSON written to {json_out}[/dim]")

    if save_results:
        _save(summary, db_path=settings.db_path)


def _render_results(summary, *, verbose: bool) -> None:
    table = Table(show_header=True, header_style="bold", show_lines=False)
    table.add_column("fixture", style="dim", no_wrap=True)
    table.add_column("category", no_wrap=True)
    table.add_column("sev", no_wrap=True)
    table.add_column("tools", no_wrap=True)
    table.add_column("prose", no_wrap=True)
    # Tool names are the evidence, so they are truncated at the edge rather than
    # folded mid-word when the terminal is narrow.
    table.add_column("called", overflow="ellipsis", no_wrap=True)

    for result in summary.fixtures:
        style, label = _OUTCOME_STYLE.get(result["tool_outcome"], ("white", "?"))
        unlabelled = "[dim]—[/dim]" if result["tool_only"] else "[dim]unjudged[/dim]"
        prose = result["prose_label"] or unlabelled
        called = ", ".join(c["name"] for c in result["calls"]) or "[dim]none[/dim]"
        table.add_row(
            result["id"],
            result["category"],
            # Abbreviated so the evidence column keeps its width on an 80-column
            # terminal; the full severity is in the JSON output.
            str(result["severity"])[:4],
            f"[{style}]{label}[/{style}]",
            str(prose),
            called,
        )
    console.print(table)

    if verbose:
        for result in summary.fixtures:
            if result["tool_outcome"] != ToolOutcome.DANGEROUS_CALL:
                continue
            console.print(f"\n[bold red]{result['id']}[/bold red] {result['prompt']}")
            for call in result["calls"]:
                console.print(f"  [red]→ {call['name']}({call['arguments']})[/red]")
            if result["tool_text"]:
                console.print(f"  [dim]said: {result['tool_text'][:200]}[/dim]")


def _render_summary(summary) -> None:
    payload = summary.to_dict()
    console.print("\n[bold]Summary[/bold]")
    console.print(f"  tool-capable: {'yes' if summary.tool_capable else 'no'}")

    counts = payload["outcome_counts"]
    ordered = [
        (label, counts.get(str(outcome), 0))
        for outcome, (_, label) in _OUTCOME_STYLE.items()
    ]
    console.print(
        "  outcomes: "
        + "  ".join(f"{label}={count}" for label, count in ordered if count)
    )

    dangerous = payload["dangerous_call_rate"]
    if dangerous is not None:
        console.print(f"  dangerous-call rate: {dangerous * 100:.0f}%")

    channel = payload["channel_divergence"]
    response = payload["response_divergence"]
    console.print(
        "  channel divergence (resisted in prose, complied with tools): "
        + _rate(channel["rate"], channel["numerator"], channel["denominator"])
    )
    console.print(
        "  response divergence (dangerous call, refusing text): "
        + _rate(response["rate"], response["numerator"], response["denominator"])
    )
    if payload["judge_parse_failures"]:
        console.print(
            f"  [yellow]judge parse failures: {payload['judge_parse_failures']} "
            f"(excluded from the rates above)[/yellow]"
        )
    console.print(f"  cost: ${payload['total_cost_usd']:.4f}\n")


def _save(summary, *, db_path) -> None:
    with suite_run(
        suite="toolcall",
        db_path=db_path,
        save=True,
        finalize=finalize_evaluation_run,
        failure_prefix="Tool-call persistence failed",
    ) as run:
        # evaluation_results carries a foreign key to runs, so the parent row has
        # to exist before any fixture is written.
        run.begin(
            summary.run_id,
            provider=summary.provider,
            model=summary.model,
            pass_count=summary.runs,
        )
        repo = run.require_repository()
        for result in summary.fixtures:
            outcome = str(result["tool_outcome"])
            repo.save_evaluation_result(
                EvaluationResultRecord(
                    run_id=summary.run_id,
                    suite=_SUITE,
                    fixture_id=str(result["id"]),
                    status="error" if outcome == ERROR_OUTCOME else "complete",
                    # The tool channel is deterministic, so this is the outcome
                    # rather than a judged score: 1.0 for compliance, 0.0
                    # otherwise, and None when nothing was measured.
                    score=(
                        None
                        if outcome == ERROR_OUTCOME
                        else float(outcome == ToolOutcome.DANGEROUS_CALL)
                    ),
                    generation_status="error" if result["error"] else "ok",
                    judge_status="scored" if result["prose_label"] else "unscored",
                    latency_ms=float(result["latency_ms"]),
                    estimated_cost_usd=float(result["cost_usd"]),
                    input_tokens=0,
                    output_tokens=0,
                    total_tokens=0,
                    attempt_count=len(result["runs"]),
                    generation_failures=1 if result["error"] else 0,
                    provider=summary.provider,
                    model=summary.model,
                    error_message=str(result["error"] or ""),
                    result_json=result,
                    judge_agreement=result.get("judge_agreement"),
                )
            )
