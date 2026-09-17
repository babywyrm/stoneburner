"""Named security batteries — list, show, and run."""

from __future__ import annotations

import click
from rich.console import Console
from rich.table import Table

from atomics.commands.common import PROVIDER_CHOICES
from atomics.eval.batteries import BATTERIES, get_battery, step_args, visible_steps


def _format_command(args: list[str]) -> str:
    return "atomics " + " ".join(args)


def _thinking_argv(thinking: bool) -> str:
    return "--thinking" if thinking else "--no-thinking"


@click.group("battery")
def battery() -> None:
    """Named security jobs that compose existing suites.

    Use these instead of the full 72-fixture adversarial suite unless you
    mean to. `list` the packs. `show NAME` prints purpose, labels, and
    copy-pasteable commands. `run NAME` executes those steps in order.
    """


@battery.command("list")
def battery_list() -> None:
    """Table of named batteries."""
    console = Console()
    table = Table(title="Security batteries")
    table.add_column("id")
    table.add_column("cost")
    table.add_column("judge")
    table.add_column("purpose")
    for item in BATTERIES:
        table.add_row(
            item.id,
            item.cost_band,
            "yes" if item.needs_judge else "no",
            item.purpose,
        )
    console.print(table)


@battery.command("show")
@click.argument("name")
@click.option(
    "-m",
    "--model",
    default=None,
    help="Omit to let inference.env fill when -p matches.",
)
@click.option(
    "-p",
    "--provider",
    "provider",
    type=PROVIDER_CHOICES,
    default="ollama",
    show_default=True,
)
@click.option("--ollama-host", default=None)
@click.option("--vllm-host", default=None, help="vLLM / OpenAI-compatible lab URL.")
@click.option("--judge-provider", type=PROVIDER_CHOICES, default=None)
@click.option("--judge-model", default=None)
@click.option("--judge-host", default=None)
@click.option("--budget", default=None, help="Forwarded onto judged / toolcall steps.")
@click.option(
    "--profile",
    default=None,
    help="App-gate profile; enables qa on non-Ollama providers.",
)
@click.option(
    "--thinking/--no-thinking",
    default=False,
    help="Default --no-thinking so short fixtures stay visible. "
    "Pass --thinking for tags that 500 or skip the tool probe with think off.",
)
def battery_show(
    name: str,
    model: str | None,
    provider: str,
    ollama_host: str | None,
    vllm_host: str | None,
    judge_provider: str | None,
    judge_model: str | None,
    judge_host: str | None,
    budget: str | None,
    profile: str | None,
    thinking: bool,
) -> None:
    """Print purpose, what a pass is not, and commands for NAME."""
    console = Console()
    try:
        item = get_battery(name)
    except KeyError as exc:
        click.echo(str(exc), err=True)
        raise SystemExit(2) from exc
    console.print(f"[bold]{item.id}[/bold] — {item.title}")
    console.print(item.purpose)
    console.print(f"When: {item.when}")
    console.print(f"Label: {item.label_hint}")
    console.print(f"Not a pass: {item.not_a_pass}")
    if item.needs_judge:
        console.print("Needs a judge. Pass --judge-provider / --judge-model.")
    console.print()
    for step in visible_steps(item, provider=provider, profile=profile):
        args = step_args(
            step,
            model=model,
            provider=provider,
            thinking_flag=_thinking_argv(thinking),
            ollama_host=ollama_host,
            vllm_host=vllm_host,
            judge_provider=judge_provider,
            judge_model=judge_model,
            judge_host=judge_host,
            budget=budget,
            profile=profile,
        )
        console.print(f"# {step.purpose}")
        console.print(_format_command(args))
    if item.optional_steps:
        console.print("\n# optional")
        for step in item.optional_steps:
            args = step_args(
                step,
                model=model,
                provider=provider,
                thinking_flag=_thinking_argv(thinking),
                ollama_host=ollama_host,
                vllm_host=vllm_host,
                judge_provider=judge_provider,
                judge_model=judge_model,
                judge_host=judge_host,
                budget=budget,
                profile=profile,
            )
            console.print(f"# {step.purpose}")
            console.print("# " + _format_command(args))


def invoke_atomics(args: list[str]) -> int:
    """Run one suite command in-process. Tests monkeypatch this.

    `standalone_mode=False` so Click does not `sys.exit`. Usage errors
    (`BadParameter`, `ctx.exit`) must still become an integer code or
    `battery run` tracebacks on the first bad step.
    """
    from atomics.cli import cli

    try:
        cli.main(args=["--no-progress", *args], standalone_mode=False)
    except SystemExit as exc:
        code = exc.code
        if code in (None, False):
            return 0
        if isinstance(code, int):
            return code
        return 1
    except click.exceptions.Exit as exc:
        return int(exc.exit_code)
    except click.Abort:
        return 1
    except click.ClickException as exc:
        exc.show()
        return int(exc.exit_code)
    return 0


_PAID = frozenset({"openai", "claude", "bedrock", "groq", "together", "gemini"})
_RUNS_SUITES = frozenset(
    {"adversarial", "redblue", "refusal", "codereview", "toolcall", "archreview"}
)


def _positive_budget(budget: str | None) -> bool:
    if not budget:
        return False
    try:
        return float(budget) > 0
    except ValueError:
        return False


@battery.command("run")
@click.argument("name")
@click.option(
    "-m",
    "--model",
    default=None,
    help="Omit to let inference.env fill when -p matches.",
)
@click.option(
    "-p",
    "--provider",
    "provider",
    type=PROVIDER_CHOICES,
    default="ollama",
    show_default=True,
)
@click.option("--ollama-host", default=None)
@click.option("--vllm-host", default=None, help="vLLM / OpenAI-compatible lab URL.")
@click.option("--judge-provider", type=PROVIDER_CHOICES, default=None)
@click.option("--judge-model", default=None)
@click.option("--judge-host", default=None)
@click.option("--budget", default=None, help="Forwarded onto judged / toolcall steps.")
@click.option(
    "--profile",
    default=None,
    help="App-gate profile; enables qa on non-Ollama providers.",
)
@click.option(
    "--runs",
    type=int,
    default=None,
    help="Forwarded onto suite steps that accept it.",
)
@click.option(
    "--keep-going",
    is_flag=True,
    default=False,
    help="Do not stop on the first failure.",
)
@click.option(
    "--thinking/--no-thinking",
    default=False,
    help="Default --no-thinking so short fixtures stay visible. "
    "Pass --thinking for tags that 500 or skip the tool probe with think off.",
)
def battery_run(
    name: str,
    model: str | None,
    provider: str,
    ollama_host: str | None,
    vllm_host: str | None,
    judge_provider: str | None,
    judge_model: str | None,
    judge_host: str | None,
    budget: str | None,
    profile: str | None,
    runs: int | None,
    keep_going: bool,
    thinking: bool,
) -> None:
    """Execute the named battery. Stops on the first nonzero step unless --keep-going.

    Paid providers (`openai`, `claude`, `bedrock`, `groq`, `together`,
    `gemini`) as `-p` or `--judge-provider` require a positive `--budget`.
    """
    console = Console()
    try:
        item = get_battery(name)
    except KeyError as exc:
        click.echo(str(exc), err=True)
        raise SystemExit(2) from exc
    if item.needs_judge and not judge_model:
        click.echo(
            f"{item.id} needs a judge. Pass --judge-model (and --judge-provider).",
            err=True,
        )
        raise SystemExit(2)
    if budget is not None and not _positive_budget(budget):
        click.echo("Pass a positive --budget.", err=True)
        raise SystemExit(2)
    paid = provider in _PAID or (judge_provider in _PAID)
    if paid and not _positive_budget(budget):
        click.echo(
            "Paid provider or judge. Pass --budget.",
            err=True,
        )
        raise SystemExit(2)
    console.print(f"[bold]{item.id}[/bold] — {item.title}")
    console.print(f"Not a pass: {item.not_a_pass}")
    failed = 0
    for step in visible_steps(item, provider=provider, profile=profile):
        args = step_args(
            step,
            model=model,
            provider=provider,
            thinking_flag=_thinking_argv(thinking),
            ollama_host=ollama_host,
            vllm_host=vllm_host,
            judge_provider=judge_provider,
            judge_model=judge_model,
            judge_host=judge_host,
            budget=budget,
            profile=profile,
        )
        if runs is not None and step.suite in _RUNS_SUITES:
            args.extend(["--runs", str(runs)])
        console.print(f"# {step.purpose}")
        console.print(_format_command(args))
        code = invoke_atomics(args)
        if code != 0:
            failed = code if failed == 0 else failed
            if not keep_going:
                raise SystemExit(code)
    if failed:
        raise SystemExit(failed)
