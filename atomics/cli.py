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


PROVIDER_CHOICES = click.Choice(["claude", "bedrock", "openai"], case_sensitive=False)


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
def run(
    tier: str,
    provider_name: str,
    max_iterations: int | None,
    model: str | None,
    budget: float | None,
    interval: int | None,
    region: str,
    hook_cmd: str | None,
    notify_flag: bool | None,
    trigger: str,
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

        effective_model = model or "us.anthropic.claude-sonnet-4-6"
        provider = BedrockProvider(region=region, model_id=effective_model)
    elif provider_name == "openai":
        from atomics.providers.openai import OpenAIProvider

        effective_model = model or "gpt-4o"
        if settings.openai_api_key:
            provider = OpenAIProvider(
                api_key=settings.openai_api_key, default_model=effective_model
            )
        else:
            try:
                from atomics.auth import auto_detect_auth

                auth = auto_detect_auth()
                console.print(f"[dim]Auth: {auth.description}[/dim]")
                provider = OpenAIProvider(default_model=effective_model, auth=auth)
            except RuntimeError as exc:
                console.print(f"[red]{exc}[/red]")
                sys.exit(1)
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
        model_override=effective_model,
        budget_override=budget,
        trigger=trigger,
    )

    from atomics.hooks import hook_env, notify_run_complete, run_post_hook

    summary = None
    try:
        summary = asyncio.run(engine.run(max_iterations=max_iterations))
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
        repo.close()


@cli.command()
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


@cli.command("provider-test")
@click.option(
    "--provider",
    "-p",
    "provider_name",
    type=PROVIDER_CHOICES,
    default="claude",
    help="LLM provider to test",
)
@click.option("--model", "-m", type=str, default=None, help="Model to test")
@click.option("--region", type=str, default="us-east-1", help="AWS region for Bedrock")
def provider_test(provider_name: str, model: str | None, region: str) -> None:
    """Quick health check against the configured provider."""
    settings = load_settings()
    _setup_logging(settings.log_level)
    console = Console()

    if provider_name == "claude":
        if not settings.anthropic_api_key:
            console.print("[red]ANTHROPIC_API_KEY not set.[/red]")
            sys.exit(1)
        from atomics.providers.claude import ClaudeProvider

        prov = ClaudeProvider(
            api_key=settings.anthropic_api_key,
            default_model=model or settings.default_model,
        )
        model_label = model or settings.default_model
    elif provider_name == "bedrock":
        from atomics.providers.bedrock import BedrockProvider

        bedrock_model = model or "us.anthropic.claude-sonnet-4-6"
        prov = BedrockProvider(region=region, model_id=bedrock_model)
        model_label = bedrock_model
    elif provider_name == "openai":
        from atomics.providers.openai import OpenAIProvider

        openai_model = model or "gpt-4o"
        model_label = openai_model
        if settings.openai_api_key:
            prov = OpenAIProvider(api_key=settings.openai_api_key, default_model=openai_model)
        else:
            try:
                from atomics.auth import auto_detect_auth

                auth = auto_detect_auth()
                console.print(f"[dim]Auth: {auth.description}[/dim]")
                prov = OpenAIProvider(default_model=openai_model, auth=auth)
            except RuntimeError as exc:
                console.print(f"[red]{exc}[/red]")
                sys.exit(1)
    else:
        console.print(f"[red]Unknown provider: {provider_name}[/red]")
        sys.exit(1)

    async def _test() -> None:
        console.print(
            f"Testing provider [cyan]{prov.name}[/cyan] with model "
            f"[cyan]{model_label}[/cyan]..."
        )
        ok = await prov.health_check()
        if ok:
            console.print("[green]Provider health check passed.[/green]")
        else:
            console.print("[red]Provider health check failed.[/red]")
            sys.exit(1)

        try:
            resp = await prov.generate(
                "What is 2+2? Reply with just the number.",
                model=model,
                max_tokens=32,
            )
        except Exception as exc:
            console.print(f"[red]Generate failed:[/red] {exc}")
            sys.exit(1)
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
    "--provider",
    "-p",
    "provider_name",
    type=PROVIDER_CHOICES,
    default="claude",
    help="LLM provider for scheduled runs",
)
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
    provider_name: str,
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
    from atomics.storage.repository import MetricsRepository

    settings = load_settings()
    console = Console()

    if fmt == "auto":
        fmt = detect_best_scheduler()
        console.print(f"[dim]Auto-detected scheduler: {fmt}[/dim]")

    schedule_id = f"{fmt}.{tier}.{provider_name}"

    if uninstall:
        if fmt == "crontab":
            msg = uninstall_crontab()
        elif fmt == "systemd":
            msg = uninstall_systemd(tier=tier)
        elif fmt == "launchd":
            msg = uninstall_launchd(tier=tier)
        else:
            msg = "Unknown format"
        repo = MetricsRepository(settings.db_path)
        repo.remove_schedule(schedule_id)
        repo.close()
        console.print(f"[green]{msg}[/green]")
        return

    if fmt == "crontab":
        entry = generate_crontab_entry(interval, max_iterations, tier=tier, provider=provider_name)
        if install:
            msg = install_crontab(entry)
            repo = MetricsRepository(settings.db_path)
            repo.save_schedule(
                schedule_id=schedule_id, format=fmt, tier=tier,
                provider=provider_name, model=None,
                interval_minutes=interval, max_iterations=max_iterations,
            )
            repo.close()
            console.print(f"[green]{msg}[/green]")
        else:
            console.print("[bold]Add this to your crontab (crontab -e):[/bold]\n")
            console.print(entry)
    elif fmt == "systemd":
        service, timer = generate_systemd_timer(
            interval, max_iterations, tier=tier, provider=provider_name
        )
        if install:
            msg = install_systemd(service, timer, tier=tier)
            repo = MetricsRepository(settings.db_path)
            repo.save_schedule(
                schedule_id=schedule_id, format=fmt, tier=tier,
                provider=provider_name, model=None,
                interval_minutes=interval, max_iterations=max_iterations,
            )
            repo.close()
            console.print(f"[green]{msg}[/green]")
        else:
            console.print("[bold]atomics.service:[/bold]")
            console.print(service)
            console.print("[bold]atomics.timer:[/bold]")
            console.print(timer)
    elif fmt == "launchd":
        plist = generate_launchd_plist(
            interval, max_iterations, tier=tier, provider=provider_name
        )
        if install:
            msg = install_launchd(plist, tier=tier)
            repo = MetricsRepository(settings.db_path)
            repo.save_schedule(
                schedule_id=schedule_id, format=fmt, tier=tier,
                provider=provider_name, model=None,
                interval_minutes=interval, max_iterations=max_iterations,
            )
            repo.close()
            console.print(f"[green]{msg}[/green]")
        else:
            console.print(
                "[bold]Save to ~/Library/LaunchAgents/com.babywyrm.atomics.plist:[/bold]\n"
            )
            console.print(plist)


@cli.command("schedule-status")
def schedule_status() -> None:
    """Show installed schedules and their health."""
    from atomics.scheduler.cron import check_schedule_health
    from atomics.storage.repository import MetricsRepository

    settings = load_settings()
    console = Console()
    repo = MetricsRepository(settings.db_path)
    try:
        schedules = repo.get_schedules()
        if not schedules:
            console.print("[dim]No schedules installed. Use [bold]atomics schedule --install[/bold].[/dim]")
            return

        table = Table(title="Installed Schedules", show_lines=True)
        table.add_column("ID", style="cyan")
        table.add_column("Format")
        table.add_column("Tier")
        table.add_column("Provider", style="magenta")
        table.add_column("Interval", justify="right")
        table.add_column("Tasks/Run", justify="right")
        table.add_column("Installed", style="green")
        table.add_column("Last Run")
        table.add_column("Status")
        table.add_column("OS Health")

        for s in schedules:
            health = check_schedule_health(
                s["format"], s["tier"],
            )
            status_style = (
                "[green]success[/green]" if s.get("last_status") == "success"
                else "[red]failed[/red]" if s.get("last_status") == "failed"
                else "[dim]—[/dim]"
            )
            health_style = (
                "[green]alive[/green]" if health
                else "[red]missing[/red]"
            )
            table.add_row(
                s["schedule_id"],
                s["format"],
                s["tier"],
                s["provider"],
                f"{s['interval_minutes']}m",
                str(s["max_iterations"]),
                s["installed_at"][:19] if s.get("installed_at") else "—",
                s["last_run_at"][:19] if s.get("last_run_at") else "—",
                status_style,
                health_style,
            )
        console.print(table)
    finally:
        repo.close()


@cli.command()
@click.option(
    "--by",
    type=click.Choice(["provider", "model"], case_sensitive=False),
    default="provider",
    help="Group comparison by provider or model",
)
@click.option("--since-hours", type=float, default=None, help="Only include recent data")
@click.option("--tier", "-t", type=TIER_CHOICES, default=None, help="Filter by tier")
@click.option("--category", type=str, default=None, help="Filter by task category")
def compare(by: str, since_hours: float | None, tier: str | None, category: str | None) -> None:
    """Compare providers or models side-by-side."""
    settings = load_settings()
    from atomics.storage.repository import MetricsRepository

    console = Console()
    repo = MetricsRepository(settings.db_path)
    try:
        rows = repo.compare_providers(
            since_hours=since_hours,
            tier=tier,
            category=category,
            group_by=by,
        )
        if not rows:
            console.print("[dim]No data to compare. Run benchmarks with multiple providers first.[/dim]")
            return

        label = "Provider" if by == "provider" else "Model"
        detail_label = "Model(s)" if by == "provider" else "Provider"
        table = Table(title=f"Comparison by {label}", show_lines=True)
        table.add_column(label, style="magenta bold")
        table.add_column(detail_label, style="dim")
        table.add_column("Tasks", justify="right")
        table.add_column("Success %", justify="right", style="green")
        table.add_column("Avg Tokens", justify="right")
        table.add_column("P50 Lat.", justify="right")
        table.add_column("P95 Lat.", justify="right")
        table.add_column("$/1K tok", justify="right", style="yellow")
        table.add_column("Avg $/Task", justify="right", style="yellow")
        table.add_column("Total $", justify="right", style="yellow bold")

        for r in rows:
            success_pct = (
                f"{r['successes'] / r['task_count'] * 100:.0f}%"
                if r["task_count"] > 0 else "—"
            )
            table.add_row(
                r["group_key"],
                r.get("models_used", "—") or "—",
                str(r["task_count"]),
                success_pct,
                f"{r['avg_tokens']:.0f}",
                f"{r['p50_latency_ms']:.0f}ms",
                f"{r['p95_latency_ms']:.0f}ms",
                f"${r['cost_per_1k_tokens']:.4f}",
                f"${r['avg_cost_per_task']:.6f}",
                f"${r['total_cost']:.4f}",
            )
        console.print(table)
    finally:
        repo.close()


@cli.command()
def doctor() -> None:
    """Check Python, database, API keys, optional deps, and scheduler tooling."""
    from atomics.doctor import run_doctor

    sys.exit(run_doctor())


@cli.command("export")
@click.option(
    "--since-hours",
    type=float,
    default=None,
    help="Only include tasks started in the last N hours",
)
@click.option("--limit", "-n", type=int, default=None, help="Maximum rows to export")
@click.option(
    "--format",
    "fmt",
    type=click.Choice(["jsonl", "csv"]),
    default="jsonl",
    help="Output format",
)
@click.option(
    "--output",
    "-o",
    "out_file",
    type=click.File("w", encoding="utf-8"),
    default="-",
    help="Output file (default: stdout)",
)
def export_tasks(
    since_hours: float | None,
    limit: int | None,
    fmt: str,
    out_file,
) -> None:
    """Export stored task metrics as JSON lines or CSV."""
    settings = load_settings()
    from atomics.exporters import write_tasks_export
    from atomics.storage.repository import MetricsRepository

    repo = MetricsRepository(settings.db_path)
    try:
        rows = repo.query_task_results(since_hours=since_hours, limit=limit)
        write_tasks_export(rows, fmt, out_file)
    finally:
        repo.close()


@cli.command()
@click.argument("shell", type=click.Choice(["bash", "zsh", "fish"]))
def completion(shell: str) -> None:
    """Print shell tab-completion script. Example: eval \"$(atomics completion zsh)\"."""
    from click.shell_completion import get_completion_class

    cls = get_completion_class(shell)
    comp = cls(cli, {}, "atomics", "_ATOMICS_COMPLETE")
    click.echo(comp.source())


AUTH_MODE_CHOICES = click.Choice(["auto", "apikey", "oauth", "codex"], case_sensitive=False)


@cli.command()
@click.option(
    "--profile",
    "oidc_profile",
    type=str,
    default="openai",
    help="Built-in OIDC profile name",
)
@click.option("--issuer", type=str, default=None, help="Custom OIDC issuer URL")
@click.option("--client-id", type=str, default=None, help="Custom OIDC client ID")
@click.option("--scopes", type=str, default=None, help="Space-separated OIDC scopes")
@click.option("--headless", is_flag=True, help="Use device code flow (no browser)")
def login(
    oidc_profile: str,
    issuer: str | None,
    client_id: str | None,
    scopes: str | None,
    headless: bool,
) -> None:
    """Log in via OAuth/OIDC (opens browser or prints device code)."""
    from atomics.auth.oauth import OAuthPKCEAuth
    from atomics.auth.profiles import OIDCProfile, get_profile
    from atomics.auth.store import TokenStore

    console = Console()

    if issuer and client_id:
        profile = OIDCProfile(
            name="custom",
            issuer=issuer,
            authorization_endpoint=f"{issuer.rstrip('/')}/authorize",
            token_endpoint=f"{issuer.rstrip('/')}/oauth/token",
            device_authorization_endpoint=f"{issuer.rstrip('/')}/oauth/device/code",
            client_id=client_id,
            scopes=scopes.split() if scopes else ["openid", "profile", "email"],
        )
    else:
        profile = get_profile(oidc_profile)

    store = TokenStore()
    auth = OAuthPKCEAuth(profile=profile, store=store)

    async def _login():
        tokens = await auth.login(headless=headless)
        console.print(f"[green]Logged in via {profile.name}[/green]")
        console.print(f"[dim]Tokens cached at {store.path}[/dim]")
        return tokens

    asyncio.run(_login())


@cli.command()
def logout() -> None:
    """Clear cached OAuth tokens."""
    from atomics.auth.store import TokenStore

    console = Console()
    store = TokenStore()
    store.clear()
    console.print("[green]Logged out — cached tokens cleared.[/green]")


@cli.command()
def whoami() -> None:
    """Show current auth mode and identity."""
    from atomics.auth.store import TokenStore

    console = Console()
    settings = load_settings()

    if settings.openai_api_key:
        masked = settings.openai_api_key[:8] + "..."
        console.print(f"[cyan]Auth mode:[/cyan] API key ({masked})")
        return

    from atomics.auth.codex import CodexTokenAuth

    codex = CodexTokenAuth()
    if codex.tokens_available():
        console.print("[cyan]Auth mode:[/cyan] Codex CLI API key (~/.codex/auth.json)")
        return
    if codex.codex_installed():
        console.print(
            "[yellow]Codex CLI detected[/yellow] but its ChatGPT tokens can't access "
            "the OpenAI API. Create a key at https://platform.openai.com/api-keys"
        )

    store = TokenStore()
    tokens = store.load()
    if tokens.access_token and not tokens.expired:
        console.print(f"[cyan]Auth mode:[/cyan] OAuth ({tokens.profile_name or 'unknown'})")
        # Decode identity from id_token if available
        if tokens.id_token:
            try:
                import base64
                import json

                parts = tokens.id_token.split(".")
                payload = parts[1] + "=" * (4 - len(parts[1]) % 4)
                claims = json.loads(base64.urlsafe_b64decode(payload))
                email = claims.get("email", "")
                name = claims.get("name", "")
                if name or email:
                    console.print(f"[dim]Identity:[/dim] {name} ({email})" if name else f"[dim]Identity:[/dim] {email}")
            except Exception:
                pass
        return

    console.print("[yellow]Not authenticated.[/yellow] Run [bold]atomics login[/bold] or set OPENAI_API_KEY.")


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
