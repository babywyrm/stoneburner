"""CLI entry point — run, report, schedule, provider-test."""

from __future__ import annotations

import asyncio
import logging
import sys

import click
from rich.console import Console
from rich.logging import RichHandler

from atomics.config import load_settings


def _setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(rich_tracebacks=True, markup=True)],
    )


@click.group()
@click.version_option(package_name="atomics")
def cli() -> None:
    """Atomics — Agentic token usage benchmarking platform."""


@cli.command()
@click.option("--max-iterations", "-n", type=int, default=None, help="Stop after N tasks (omit for continuous)")
@click.option("--model", "-m", type=str, default=None, help="Override default model")
@click.option("--budget", "-b", type=float, default=None, help="Override budget limit (USD)")
@click.option("--interval", "-i", type=int, default=None, help="Override loop interval (seconds)")
def run(max_iterations: int | None, model: str | None, budget: float | None, interval: int | None) -> None:
    """Start the benchmarking loop."""
    settings = load_settings()
    _setup_logging(settings.log_level)

    if model:
        settings.default_model = model
    if budget is not None:
        settings.budget_limit_usd = budget
    if interval is not None:
        settings.loop_interval_seconds = interval

    if not settings.anthropic_api_key:
        Console().print("[red]ANTHROPIC_API_KEY not set. Export it or add to .env[/red]")
        sys.exit(1)

    from atomics.core.engine import LoopEngine
    from atomics.providers.claude import ClaudeProvider
    from atomics.storage.repository import MetricsRepository

    provider = ClaudeProvider(api_key=settings.anthropic_api_key, default_model=settings.default_model)
    repo = MetricsRepository(settings.db_path)
    engine = LoopEngine(provider=provider, repo=repo, settings=settings)

    try:
        asyncio.run(engine.run(max_iterations=max_iterations))
    except KeyboardInterrupt:
        Console().print("\n[yellow]Interrupted — finalizing run...[/yellow]")
    finally:
        repo.close()


@cli.command()
@click.option("--hours", "-h", type=int, default=24, help="Hours of history to show")
@click.option("--runs", "-r", type=int, default=10, help="Number of recent runs to show")
def report(hours: int, runs: int) -> None:
    """Show usage reports and trends."""
    settings = load_settings()
    from atomics.reporting import print_category_breakdown, print_hourly_usage, print_recent_runs
    from atomics.storage.repository import MetricsRepository

    repo = MetricsRepository(settings.db_path)
    try:
        print_recent_runs(repo, limit=runs)
        print_hourly_usage(repo, hours=hours)
        print_category_breakdown(repo)
    finally:
        repo.close()


@cli.command("provider-test")
@click.option("--model", "-m", type=str, default=None, help="Model to test")
def provider_test(model: str | None) -> None:
    """Quick health check against the configured provider."""
    settings = load_settings()
    _setup_logging(settings.log_level)

    if not settings.anthropic_api_key:
        Console().print("[red]ANTHROPIC_API_KEY not set.[/red]")
        sys.exit(1)

    from atomics.providers.claude import ClaudeProvider

    console = Console()
    provider = ClaudeProvider(
        api_key=settings.anthropic_api_key,
        default_model=model or settings.default_model,
    )

    async def _test() -> None:
        console.print(f"Testing provider [cyan]{provider.name}[/cyan] with model [cyan]{model or settings.default_model}[/cyan]...")
        ok = await provider.health_check()
        if ok:
            console.print("[green]Provider health check passed.[/green]")
        else:
            console.print("[red]Provider health check failed.[/red]")
            sys.exit(1)

        resp = await provider.generate(
            "What is 2+2? Reply with just the number.",
            model=model,
            max_tokens=32,
        )
        console.print(f"Response: {resp.text.strip()}")
        console.print(f"Tokens: in={resp.input_tokens} out={resp.output_tokens} total={resp.total_tokens}")
        console.print(f"Latency: {resp.latency_ms:.0f}ms")
        console.print(f"Cost: ${resp.estimated_cost_usd:.6f}")

    asyncio.run(_test())


@cli.command()
@click.option("--interval", "-i", type=int, default=30, help="Minutes between runs")
@click.option("--max-iterations", "-n", type=int, default=10, help="Tasks per scheduled run")
@click.option("--format", "fmt", type=click.Choice(["crontab", "systemd", "launchd"]), default="crontab")
def schedule(interval: int, max_iterations: int, fmt: str) -> None:
    """Generate scheduler config (crontab, systemd timer, or launchd plist)."""
    from atomics.scheduler.cron import (
        generate_crontab_entry,
        generate_launchd_plist,
        generate_systemd_timer,
    )

    console = Console()

    if fmt == "crontab":
        entry = generate_crontab_entry(interval, max_iterations)
        console.print("[bold]Add this to your crontab (crontab -e):[/bold]\n")
        console.print(entry)
    elif fmt == "systemd":
        service, timer = generate_systemd_timer(interval, max_iterations)
        console.print("[bold]atomics.service:[/bold]")
        console.print(service)
        console.print("[bold]atomics.timer:[/bold]")
        console.print(timer)
    elif fmt == "launchd":
        plist = generate_launchd_plist(interval, max_iterations)
        console.print("[bold]Save to ~/Library/LaunchAgents/com.babywyrm.atomics.plist:[/bold]\n")
        console.print(plist)
