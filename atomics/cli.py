"""CLI entry point — run, report, schedule, provider-test."""

from __future__ import annotations

import asyncio
import logging
import sys

import click
from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table

from atomics.config import load_settings
from atomics.models import BurnTier


def _setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(rich_tracebacks=True, markup=True)],
    )


TIER_CHOICES = click.Choice([t.value for t in BurnTier], case_sensitive=False)


@click.group()
@click.version_option(package_name="atomics")
def cli() -> None:
    """Atomics — Agentic token usage benchmarking platform."""


PROVIDER_CHOICES = click.Choice(["claude", "bedrock"], case_sensitive=False)


@cli.command()
@click.option(
    "--tier",
    "-t",
    type=TIER_CHOICES,
    default="baseline",
    help="Burn tier (ez/baseline/mega)",
)
@click.option(
    "--provider",
    "-p",
    "provider_name",
    type=PROVIDER_CHOICES,
    default="claude",
    help="LLM provider",
)
@click.option(
    "--max-iterations",
    "-n",
    type=int,
    default=None,
    help="Stop after N tasks (omit for continuous)",
)
@click.option("--model", "-m", type=str, default=None, help="Override default model")
@click.option("--budget", "-b", type=float, default=None, help="Override budget limit (USD)")
@click.option("--interval", "-i", type=int, default=None, help="Override loop interval (seconds)")
@click.option("--region", type=str, default="us-east-1", help="AWS region for Bedrock")
def run(
    tier: str,
    provider_name: str,
    max_iterations: int | None,
    model: str | None,
    budget: float | None,
    interval: int | None,
    region: str,
) -> None:
    """Start the benchmarking loop."""
    settings = load_settings()
    _setup_logging(settings.log_level)
    console = Console()

    burn_tier = BurnTier(tier)

    from atomics.core.engine import LoopEngine
    from atomics.storage.repository import MetricsRepository
    from atomics.tiers import get_tier_profile

    profile = get_tier_profile(burn_tier)

    if provider_name == "claude":
        if not settings.anthropic_api_key:
            console.print("[red]ANTHROPIC_API_KEY not set. Export it or add to .env[/red]")
            sys.exit(1)
        from atomics.providers.claude import ClaudeProvider

        effective_model = model or profile.preferred_model or settings.default_model
        provider = ClaudeProvider(api_key=settings.anthropic_api_key, default_model=effective_model)
    elif provider_name == "bedrock":
        from atomics.providers.bedrock import BedrockProvider

        bedrock_model = model or "anthropic.claude-3-5-sonnet-20241022-v2:0"
        provider = BedrockProvider(region=region, model_id=bedrock_model)
    else:
        console.print(f"[red]Unknown provider: {provider_name}[/red]")
        sys.exit(1)

    repo = MetricsRepository(settings.db_path)
    engine = LoopEngine(
        provider=provider,
        repo=repo,
        settings=settings,
        tier=burn_tier,
        interval_override=interval,
        budget_override=budget,
    )

    try:
        asyncio.run(engine.run(max_iterations=max_iterations))
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted — finalizing run...[/yellow]")
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
        model_label = model or settings.default_model
        console.print(
            f"Testing provider [cyan]{provider.name}[/cyan] with model "
            f"[cyan]{model_label}[/cyan]..."
        )
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
        console.print(
            f"Tokens: in={resp.input_tokens} out={resp.output_tokens} total={resp.total_tokens}"
        )
        console.print(f"Latency: {resp.latency_ms:.0f}ms")
        console.print(f"Cost: ${resp.estimated_cost_usd:.6f}")

    asyncio.run(_test())


@cli.command()
@click.option("--tier", "-t", type=TIER_CHOICES, default="baseline", help="Burn tier")
@click.option("--interval", "-i", type=int, default=30, help="Minutes between runs")
@click.option("--max-iterations", "-n", type=int, default=10, help="Tasks per scheduled run")
@click.option(
    "--format",
    "fmt",
    type=click.Choice(["crontab", "systemd", "launchd", "auto"]),
    default="auto",
)
@click.option("--install", is_flag=True, help="Install the schedule on this system")
@click.option("--uninstall", is_flag=True, help="Remove installed atomics schedule")
def schedule(
    tier: str,
    interval: int,
    max_iterations: int,
    fmt: str,
    install: bool,
    uninstall: bool,
) -> None:
    """Generate or install scheduler config (crontab, systemd timer, or launchd plist)."""
    from atomics.scheduler.cron import (
        detect_best_scheduler,
        generate_crontab_entry,
        generate_launchd_plist,
        generate_systemd_timer,
        install_crontab,
        install_launchd,
        install_systemd,
        uninstall_crontab,
        uninstall_launchd,
        uninstall_systemd,
    )

    console = Console()

    if fmt == "auto":
        fmt = detect_best_scheduler()
        console.print(f"[dim]Auto-detected scheduler: {fmt}[/dim]")

    if uninstall:
        if fmt == "crontab":
            msg = uninstall_crontab()
        elif fmt == "systemd":
            msg = uninstall_systemd(tier=tier)
        elif fmt == "launchd":
            msg = uninstall_launchd(tier=tier)
        else:
            msg = "Unknown format"
        console.print(f"[green]{msg}[/green]")
        return

    if fmt == "crontab":
        entry = generate_crontab_entry(interval, max_iterations, tier=tier)
        if install:
            msg = install_crontab(entry)
            console.print(f"[green]{msg}[/green]")
        else:
            console.print("[bold]Add this to your crontab (crontab -e):[/bold]\n")
            console.print(entry)
    elif fmt == "systemd":
        service, timer = generate_systemd_timer(interval, max_iterations, tier=tier)
        if install:
            msg = install_systemd(service, timer, tier=tier)
            console.print(f"[green]{msg}[/green]")
        else:
            console.print("[bold]atomics.service:[/bold]")
            console.print(service)
            console.print("[bold]atomics.timer:[/bold]")
            console.print(timer)
    elif fmt == "launchd":
        plist = generate_launchd_plist(interval, max_iterations, tier=tier)
        if install:
            msg = install_launchd(plist, tier=tier)
            console.print(f"[green]{msg}[/green]")
        else:
            console.print(
                "[bold]Save to ~/Library/LaunchAgents/com.babywyrm.atomics.plist:[/bold]\n"
            )
            console.print(plist)


@cli.command("tiers")
def show_tiers() -> None:
    """Show available burn tiers and their profiles."""
    from atomics.tiers import TIER_PROFILES

    console = Console()
    table = Table(title="Burn Tiers", show_lines=True)
    table.add_column("Tier", style="cyan bold")
    table.add_column("Description")
    table.add_column("Interval", justify="right")
    table.add_column("Tokens/hr", justify="right")
    table.add_column("Req/min", justify="right")
    table.add_column("Budget", justify="right", style="yellow")
    table.add_column("Model", style="dim")

    for profile in TIER_PROFILES.values():
        table.add_row(
            profile.tier.value.upper(),
            profile.description,
            f"{profile.loop_interval_seconds}s",
            f"{profile.max_tokens_per_hour:,}",
            str(profile.max_requests_per_minute),
            f"${profile.budget_limit_usd:.2f}",
            profile.preferred_model or "(default)",
        )
    console.print(table)
