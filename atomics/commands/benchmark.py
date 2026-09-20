"""Benchmark CLI commands."""

from __future__ import annotations

import sys

import click
from rich.console import Console
from rich.table import Table

from atomics.commands.common import (
    PROVIDER_CHOICES,
    _make_provider,
    effort_options,
    run_async,
    setup_logging,
)
from atomics.commands.common import effective_model as resolve_effective_model
from atomics.config import load_settings
from atomics.models import BurnTier
from atomics.providers.base import BaseProvider

TIER_CHOICES = click.Choice([t.value for t in BurnTier], case_sensitive=False)


@click.command()
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
@click.option(
    "--ollama-host",
    type=str,
    default=None,
    help="Ollama endpoint (default: ATOMICS_OLLAMA_HOST or http://localhost:11434)",
)
@click.option(
    "--vllm-host",
    type=str,
    default=None,
    help="vLLM/OpenAI-compatible base URL (default: ATOMICS_VLLM_HOST or http://localhost:8000/v1)",
)
@click.option(
    "--gateway-url",
    type=str,
    default=None,
    help="Brain-gateway endpoint (default: ATOMICS_BRAIN_GATEWAY_URL or http://localhost:8080)",
)
@click.option(
    "--hook",
    "hook_cmd",
    type=str,
    default=None,
    help="Shell command after a finished run (overrides ATOMICS_POST_RUN_HOOK)",
)
@click.option(
    "--notify/--no-notify",
    "notify_flag",
    default=None,
    help="Desktop notification when the run completes (default: ATOMICS_NOTIFY)",
)
@click.option(
    "--trigger",
    type=click.Choice(["manual", "scheduled", "test"], case_sensitive=False),
    default="manual",
    help="How this run was triggered (set automatically by scheduled runs)",
)
@click.option(
    "--thinking/--no-thinking",
    "thinking_flag",
    default=None,
    help="Enable/disable extended thinking for capable models (auto-detects if omitted)",
)
@click.option(
    "--thinking-budget",
    type=int,
    default=None,
    help="Max thinking tokens to allocate (provider-specific defaults if omitted)",
)
@effort_options
def run(
    tier: str,
    provider_name: str,
    max_iterations: int | None,
    model: str | None,
    budget: float | None,
    interval: int | None,
    region: str,
    ollama_host: str | None,
    vllm_host: str | None,
    gateway_url: str | None,
    hook_cmd: str | None,
    notify_flag: bool | None,
    trigger: str,
    thinking_flag: bool | None,
    thinking_budget: int | None,
    effort: str | None,
    reasoning_mode: str | None,
) -> None:
    """Start the benchmarking loop."""
    settings = load_settings()
    setup_logging(settings.log_level)
    console = Console()

    burn_tier = BurnTier(tier)

    from atomics.benchmark.tiers import get_tier_profile
    from atomics.core.engine import LoopEngine
    from atomics.storage.repository import MetricsRepository

    profile = get_tier_profile(burn_tier)

    provider: BaseProvider
    if provider_name == "openai" and not settings.openai_api_key:
        # The keyless OpenAI path auto-detects a local login (e.g. Codex) and
        # prints which credential it found. That reach into local auth and the
        # console is a CLI affordance the shared factory deliberately lacks, so
        # the API server and distributed workers stay headless. This branch stays.
        from atomics.auth import auto_detect_auth
        from atomics.providers.openai import OpenAIProvider

        try:
            auth = auto_detect_auth()
        except RuntimeError as exc:
            console.print(f"[red]{exc}[/red]")
            sys.exit(1)
        console.print(f"[dim]Auth: {auth.description}[/dim]")
        provider = OpenAIProvider(default_model=model or "gpt-4o", auth=auth)
    else:
        # claude and brain-gateway fall back to the tier's preferred model before
        # the account default; the factory only knows the account default, so
        # pre-resolve that one step and let the factory own the rest.
        requested_model = model
        if provider_name in ("claude", "brain-gateway"):
            requested_model = model or profile.preferred_model or settings.default_model
        host = gateway_url if provider_name == "brain-gateway" else ollama_host
        provider = _make_provider(
            provider_name, requested_model, host, settings, vllm_host=vllm_host, region=region
        )

    effective_model = resolve_effective_model(model, provider)

    # Auto-detect thinking capability when not explicitly set
    if thinking_flag is None and effective_model:
        from atomics.benchmark.model_classes import supports_thinking

        if supports_thinking(effective_model):
            thinking_flag = True

    repo = MetricsRepository(settings.db_path)
    engine = LoopEngine(
        provider=provider,
        repo=repo,
        settings=settings,
        tier=burn_tier,
        interval_override=interval,
        model_override=effective_model,
        budget_override=budget,
        trigger=trigger,
        thinking=thinking_flag,
        thinking_budget=thinking_budget,
        effort=effort,
        reasoning_mode=reasoning_mode,
    )

    from atomics.reporting.hooks import hook_env, notify_run_complete, run_post_hook

    summary = None
    try:
        summary = run_async(engine.run(max_iterations=max_iterations), provider)
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted — finalizing run...[/yellow]")
    finally:
        if summary is not None:
            do_notify = settings.notify_on_finish if notify_flag is None else notify_flag
            if do_notify:
                notify_run_complete(summary)
            eff_hook = (hook_cmd or "").strip() or (settings.post_run_hook.strip() or None)
            if eff_hook:
                env = hook_env(summary, tier=burn_tier.value, provider=provider_name)
                rc = run_post_hook(eff_hook, env)
                if rc != 0:
                    console.print(f"[yellow]Post-run hook exited with code {rc}[/yellow]")
            if settings.webhook_url:
                from atomics.reporting.webhooks import send_webhook

                send_webhook(
                    settings.webhook_url,
                    summary,
                    tier=burn_tier.value,
                    provider=provider_name,
                )
        repo.close()


@click.command()
@click.option("--hours", "-h", type=int, default=24, help="Hours of history to show")
@click.option("--runs", "-r", type=int, default=10, help="Number of recent runs to show")
def report(hours: int, runs: int) -> None:
    """Show usage reports and trends."""
    settings = load_settings()
    from atomics.reporting import (
        print_category_breakdown,
        print_hourly_usage,
        print_provider_summary,
        print_recent_runs,
    )
    from atomics.storage.repository import MetricsRepository

    repo = MetricsRepository(settings.db_path)
    try:
        print_recent_runs(repo, limit=runs)
        print_provider_summary(repo, since_hours=hours)
        print_hourly_usage(repo, hours=hours)
        print_category_breakdown(repo)
    finally:
        repo.close()


@click.command("tiers")
def tiers() -> None:
    """Show available burn tiers and their profiles."""
    from atomics.benchmark.tiers import TIER_PROFILES

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
