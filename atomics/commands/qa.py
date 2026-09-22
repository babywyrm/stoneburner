"""QA fixture validation CLI command."""

from __future__ import annotations

import click
from rich.console import Console
from rich.table import Table

from atomics.commands.common import effort_options


@click.command("qa")
@click.option(
    "--file",
    "-f",
    "qa_file",
    type=click.Path(exists=True),
    required=True,
    help="QA fixture YAML file (prompts + pass/fail patterns — no secrets).",
)
@click.option(
    "--profile",
    "-p",
    "profile_path",
    type=click.Path(exists=True),
    default=None,
    help="Target profile YAML for app-level gates (gitignored, replaces --model/--ollama-host).",
)
@click.option(
    "--model",
    "-m",
    type=str,
    default=None,
    help="Override model from fixture file (raw Ollama mode).",
)
@click.option(
    "--ollama-host",
    type=str,
    default=None,
    help="Override Ollama host from fixture file (raw Ollama mode).",
)
@click.option(
    "--num-predict",
    type=int,
    default=1024,
    show_default=True,
    help="Max output tokens per fixture prompt (raw Ollama mode only).",
)
@click.option(
    "--fail-fast", is_flag=True, default=False, help="Stop after the first FAIL or ERROR."
)
@click.option(
    "--thinking/--no-thinking",
    "thinking_flag",
    default=None,
    help="Force thinking on or off for raw Ollama QA (default: auto-detect). "
    "Ignored with --profile.",
)
@click.option(
    "--thinking-budget",
    type=int,
    default=None,
    help="Added to --num-predict when thinking is on. Raw Ollama only.",
)
@effort_options
def qa(
    qa_file: str,
    profile_path: str | None,
    model: str | None,
    ollama_host: str | None,
    num_predict: int,
    fail_fast: bool,
    thinking_flag: bool | None,
    thinking_budget: int | None,
    effort: str | None,
    reasoning_mode: str | None,
) -> None:
    """QA validation — fire fixture prompts and check pass/fail patterns.

    Two modes:

    \b
    RAW OLLAMA (default): talks directly to an Ollama model.
      atomics qa --file qa/examples/ctf-solvability.yaml --model gemma4:26b

    \b
    PROFILE MODE: routes requests through an app-level HTTP target.
    The profile lives in profiles/local/ (gitignored — keeps your real
    box IPs and credentials out of the repo). The fixture file is safe
    to commit; it only contains prompts and patterns.
      atomics qa --file qa/examples/app-gate-guardrails.yaml \\
                 --profile profiles/local/my-gate.yaml

    \b
    Other examples:
      atomics qa --file qa/examples/app-gate-guardrails.yaml --model qwen3.8:27b --no-thinking
      atomics qa --file qa/examples/ai-gate-regression.yaml --fail-fast
      atomics qa --file qa/examples/app-gate-guardrails.yaml \\
                 --profile profiles/local/my-policy.yaml
    """
    import asyncio as _asyncio
    import logging as _logging

    from atomics.benchmark.qa_runner import QAResult, format_qa_tokens, load_qa_suite, run_qa_suite

    _ = reasoning_mode

    _logging.getLogger("httpx").setLevel(_logging.WARNING)
    _logging.getLogger("httpcore").setLevel(_logging.WARNING)

    console = Console()

    file_model, file_host, fixtures = load_qa_suite(qa_file)

    # Set in both modes. --fail-fast raises KeyboardInterrupt from the
    # result callback, and that handler records the partial suite. Profile
    # mode never assigns these, so the handler used to raise UnboundLocalError
    # instead of reporting the stop.
    effective_model = ""
    effective_host = ""

    # Load profile if given — it handles all transport details
    loaded_profile = None
    target_label: str
    if profile_path:
        from atomics.load.profiles import load_profile

        loaded_profile = load_profile(profile_path)
        target_label = (
            f"profile:[bold cyan]{loaded_profile.name}[/bold cyan] ({loaded_profile.type})"
        )
    else:
        effective_model = model or file_model
        effective_host = ollama_host or file_host
        if not effective_model:
            console.print("[red]No model specified. Set 'model' in the YAML or use --model.[/red]")
            raise SystemExit(1)
        target_label = f"[cyan]{effective_model}[/cyan]  Host: {effective_host}"

    console.print(f"[bold]QA Suite[/bold] — {len(fixtures)} fixture(s)\nTarget: {target_label}\n")

    stopped_early = False
    results: list[QAResult] = []

    def _on_result(r: QAResult) -> None:
        icon = {
            "PASS": "[green]✓[/green]",
            "FAIL": "[red]✗[/red]",
            "ERROR": "[yellow]![/yellow]",
        }.get(r.status, "?")
        tokens = format_qa_tokens(r)
        suffix = f"  {tokens}" if tokens else ""
        console.print(
            f"  {icon} [{r.status}] {r.fixture.id}  ({r.latency_ms / 1000:.1f}s){suffix}"
        )
        results.append(r)
        if fail_fast and r.status in ("FAIL", "ERROR"):
            raise KeyboardInterrupt("fail-fast triggered")

    try:
        suite = _asyncio.run(
            run_qa_suite(
                model=effective_model if not loaded_profile else "",
                host=effective_host if not loaded_profile else "",
                fixtures=fixtures,
                num_predict=num_predict,
                on_result=_on_result,
                profile=loaded_profile,
                thinking=thinking_flag,
                thinking_budget=thinking_budget,
                effort=effort,
            )
        )
    except KeyboardInterrupt:
        stopped_early = True
        from atomics.benchmark.qa_runner import QASuiteResult

        suite = QASuiteResult(model=effective_model, host=effective_host, results=results)

    console.print()
    rtable = Table(title="QA Results", show_lines=True)
    rtable.add_column("ID", style="cyan")
    rtable.add_column("Status", justify="center")
    rtable.add_column("Matched pass patterns")
    rtable.add_column("Matched fail patterns")
    rtable.add_column("Latency", justify="right")
    rtable.add_column("Tokens", justify="right")

    status_style_map = {
        "PASS": "[green]PASS[/green]",
        "FAIL": "[red]FAIL[/red]",
        "ERROR": "[yellow]ERROR[/yellow]",
    }
    for r in suite.results:
        rtable.add_row(
            r.fixture.id,
            status_style_map.get(r.status, r.status),
            ", ".join(r.matched_pass) or "-",
            ", ".join(r.matched_fail) or "-",
            f"{r.latency_ms / 1000:.1f}s" if r.latency_ms else "-",
            format_qa_tokens(r) or "-",
        )

    console.print(rtable)

    pass_color = (
        "green" if suite.pass_rate == 1.0 else ("yellow" if suite.pass_rate >= 0.5 else "red")
    )
    console.print(
        f"\n[bold]Pass rate:[/bold] [{pass_color}]{suite.passed}/{suite.total}[/{pass_color}]"
        + (" [dim](stopped early)[/dim]" if stopped_early else "")
    )
    if suite.failed or suite.errors or stopped_early:
        raise SystemExit(1)
