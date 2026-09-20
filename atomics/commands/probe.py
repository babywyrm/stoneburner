"""Live ecosystem health probe CLI command."""

from __future__ import annotations

import uuid

import click
from rich.console import Console
from rich.table import Table

from atomics.commands.common import (
    PROVIDER_CHOICES,
    _make_provider,
    budget_option,
    effort_options,
    eval_budget_from,
    extra_judges_option,
    parse_extra_judges,
    run_async,
    write_summary_json,
)
from atomics.commands.suite_run import finalize_probe_run, suite_run
from atomics.config import load_settings
from atomics.eval.budget import share_budget


@click.command("probe")
@click.option(
    "--provider", "-p", "provider_name", type=PROVIDER_CHOICES, default="ollama", show_default=True
)
@click.option("--model", "-m", type=str, default=None)
@click.option("--ollama-host", type=str, default=None)
@click.option(
    "--vllm-host", "vllm_host", type=str, default=None, help="vLLM/OpenAI-compatible base URL."
)
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
    "--probes-file",
    type=click.Path(exists=True),
    default=None,
    help="Path to probes.yaml config file.",
)
@click.option(
    "--artifact",
    type=click.Choice(
        [
            "json-security-report",
            "inference-api",
            "access-log",
            "k8s-audit-log",
            "config-file",
            "api-response",
        ]
    ),
    default=None,
    help="Artifact type for single-file mode.",
)
@click.option(
    "--file",
    "artifact_file",
    type=click.Path(exists=True),
    default=None,
    help="Artifact file path for single-file mode (use with --artifact).",
)
@click.option("--thinking/--no-thinking", "thinking_flag", default=None)
@click.option("--thinking-budget", type=int, default=8000, show_default=True)
@effort_options
@click.option(
    "--alert-on-regression/--no-alert-on-regression",
    default=False,
    help="Warn if any check score drops >10% from last run.",
)
@click.option("--save/--no-save", "save_results", default=True, show_default=True)
@click.option(
    "--json-out",
    "json_out",
    type=click.Path(dir_okay=False, writable=True),
    default=None,
    help="Write the full run (per-target scores, rationales, regressions) as JSON to this file.",
)
@extra_judges_option
@budget_option
def probe(
    provider_name: str,
    model: str | None,
    ollama_host: str | None,
    vllm_host: str | None,
    judge_provider_name: str,
    judge_model: str | None,
    judge_host: str | None,
    probes_file: str | None,
    artifact: str | None,
    artifact_file: str | None,
    thinking_flag: bool | None,
    thinking_budget: int,
    effort: str | None,
    reasoning_mode: str | None,
    alert_on_regression: bool,
    save_results: bool,
    json_out: str | None,
    extra_judges: str | None,
    budget_usd: float | None,
) -> None:
    """Run LLM-evaluated live ecosystem health probes against configured artifact targets."""
    from pathlib import Path

    from atomics.probe.config import ProbeTarget, load_probe_config
    from atomics.probe.runner import run_probe

    console = Console()
    settings = load_settings()
    provider = _make_provider(provider_name, model, ollama_host, settings, vllm_host=vllm_host)
    judge = _make_provider(
        judge_provider_name, judge_model, judge_host or ollama_host, settings, vllm_host=vllm_host
    )
    extra_judge_pairs = parse_extra_judges(
        extra_judges,
        build=lambda name, mdl, host: _make_provider(
            name,
            mdl,
            host,
            settings,
            vllm_host=vllm_host,
        ),
        default_host=judge_host or ollama_host,
    )
    budget = eval_budget_from(budget_usd)
    if budget is not None:
        guarded = share_budget(budget, provider, judge, *(p for p, _ in extra_judge_pairs))
        provider, judge = guarded[0], guarded[1]
        extra_judge_pairs = [(guarded[2 + i], mdl) for i, (_, mdl) in enumerate(extra_judge_pairs)]

    targets = []
    if probes_file:
        targets = load_probe_config(Path(probes_file))
    elif artifact and artifact_file:
        targets = [
            ProbeTarget(
                name=Path(artifact_file).name,
                artifact_type=artifact,
                source="file",
                path=artifact_file,
            )
        ]
    else:
        console.print("[red]Provide --probes-file or both --artifact and --file.[/red]")
        raise SystemExit(2)

    console.print(
        f"\n[bold]Ecosystem probe[/bold] — model: [cyan]{provider_name}[/cyan] "
        f"({model or 'default'})\n"
        f"Judge: [cyan]{judge_provider_name}[/cyan] | Targets: [bold]{len(targets)}[/bold]\n"
    )

    run_id = uuid.uuid4().hex[:12]

    with suite_run(
        suite="probe",
        db_path=settings.db_path,
        save=save_results,
        finalize=finalize_probe_run,
        failure_prefix="Probe run failed",
    ) as run:
        # Parent run row so probe runs are listable/queryable like other suites.
        run.begin(
            run_id,
            provider=provider_name,
            model=model or "default",
        )
        repo = run.repository

        def on_result(r):
            color = (
                "green" if (r.score or 0) >= 0.8 else ("yellow" if (r.score or 0) >= 0.6 else "red")
            )
            reg_tag = " [bold red][REGRESSION][/bold red]" if r.regressed else ""
            console.print(
                f" [bold]{r.target_name}[/bold] ({r.artifact_type}) "
                f"[{color}]{(r.score or 0) * 100:.1f}%[/]{reg_tag} — {r.judge_rationale[:80]}"
            )
            if repo:
                repo.save_probe_result(run_id, r)

        summary = run_async(
            run_probe(
                provider,
                judge_provider=judge,
                targets=targets,
                model=model,
                judge_model=judge_model,
                extra_judges=extra_judge_pairs,
                thinking=thinking_flag,
                thinking_budget=thinking_budget,
                effort=effort,
                reasoning_mode=reasoning_mode,
                regression_threshold=0.10,
                on_result=on_result,
            ),
            provider,
            judge,
            *(p for p, _ in extra_judge_pairs),
        )

        table = Table(title="Probe Summary")
        table.add_column("Metric")
        table.add_column("Value")
        table.add_row("Provider", provider_name)
        table.add_row("Targets", str(len(summary.results)))
        table.add_row("Overall Score", f"{(summary.overall_score or 0) * 100:.1f}%")
        if summary.regressions:
            table.add_row("[red]Regressions[/red]", str(len(summary.regressions)))
        console.print(table)

        if json_out:
            write_summary_json(summary, Path(json_out))
            console.print(f"[dim]Wrote JSON results to {json_out}[/dim]")

    if alert_on_regression and summary.regressions:
        console.print(
            f"\n[bold red]⚠ {len(summary.regressions)} probe(s) "
            f"regressed >10% from last run[/bold red]"
        )
        for r in summary.regressions:
            console.print(
                f"  • {r.target_name}: "
                f"{(r.prev_score or 0) * 100:.1f}% → {(r.score or 0) * 100:.1f}%"
            )
