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


PROVIDER_CHOICES = click.Choice(["claude", "bedrock", "openai", "ollama", "vllm", "brain-gateway"], case_sensitive=False)


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
@click.option("--ollama-host", type=str, default=None, help="Ollama endpoint (default: ATOMICS_OLLAMA_HOST or http://localhost:11434)")
@click.option("--vllm-host", type=str, default=None, help="vLLM/OpenAI-compatible base URL (default: ATOMICS_VLLM_HOST or http://localhost:8000/v1)")
@click.option("--gateway-url", type=str, default=None, help="Brain-gateway endpoint (default: ATOMICS_BRAIN_GATEWAY_URL or http://localhost:8080)")
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
    elif provider_name == "ollama":
        from atomics.providers.ollama import OllamaProvider

        host = ollama_host or settings.ollama_host
        effective_model = model or settings.ollama_model
        provider = OllamaProvider(host=host, default_model=effective_model)
    elif provider_name == "vllm":
        from atomics.providers.vllm import VllmProvider

        base_url = vllm_host or settings.vllm_host
        effective_model = model or settings.vllm_model
        provider = VllmProvider(base_url=base_url, default_model=effective_model)
    elif provider_name == "brain-gateway":
        from atomics.providers.brain_gateway import BrainGatewayProvider

        url = gateway_url or settings.brain_gateway_url
        effective_model = model or profile.preferred_model or settings.default_model
        provider = BrainGatewayProvider(url=url, default_model=effective_model)
    else:
        console.print(f"[red]Unknown provider: {provider_name}[/red]")
        sys.exit(1)

    # Auto-detect thinking capability when not explicitly set
    if thinking_flag is None and effective_model:
        from atomics.model_classes import supports_thinking
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
@click.option("--ollama-host", type=str, default=None, help="Ollama endpoint")
@click.option("--vllm-host", type=str, default=None, help="vLLM/OpenAI-compatible base URL")
@click.option("--gateway-url", type=str, default=None, help="Brain-gateway endpoint")
@click.option("--thinking/--no-thinking", "thinking_flag", default=None, help="Enable/disable thinking mode")
@click.option("--thinking-budget", type=int, default=None, help="Max thinking tokens")
def provider_test(provider_name: str, model: str | None, region: str, ollama_host: str | None, vllm_host: str | None, gateway_url: str | None, thinking_flag: bool | None, thinking_budget: int | None) -> None:
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
    elif provider_name == "ollama":
        from atomics.providers.ollama import OllamaProvider

        host = ollama_host or settings.ollama_host
        ollama_model_name = model or settings.ollama_model
        prov = OllamaProvider(host=host, default_model=ollama_model_name)
        model_label = ollama_model_name
    elif provider_name == "vllm":
        from atomics.providers.vllm import VllmProvider

        base_url = vllm_host or settings.vllm_host
        vllm_model_name = model or settings.vllm_model
        prov = VllmProvider(base_url=base_url, default_model=vllm_model_name)
        model_label = vllm_model_name
    elif provider_name == "brain-gateway":
        from atomics.providers.brain_gateway import BrainGatewayProvider

        url = gateway_url or settings.brain_gateway_url
        prov = BrainGatewayProvider(url=url, default_model=model)
        model_label = model or "(gateway default)"
    else:
        console.print(f"[red]Unknown provider: {provider_name}[/red]")
        sys.exit(1)

    # Auto-detect thinking if not explicitly set
    eff_thinking = thinking_flag
    if eff_thinking is None and model:
        from atomics.model_classes import supports_thinking
        if supports_thinking(model):
            eff_thinking = True

    async def _test() -> None:
        thinking_label = ""
        if eff_thinking is True:
            thinking_label = " [magenta](thinking ON)[/magenta]"
        elif eff_thinking is False:
            thinking_label = " [dim](thinking OFF)[/dim]"
        console.print(
            f"Testing provider [cyan]{prov.name}[/cyan] with model "
            f"[cyan]{model_label}[/cyan]{thinking_label}..."
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
                thinking=eff_thinking,
                thinking_budget=thinking_budget,
            )
        except Exception as exc:
            console.print(f"[red]Generate failed:[/red] {exc}")
            sys.exit(1)
        console.print(f"Response: {resp.text.strip()}")
        console.print(
            f"Tokens: in={resp.input_tokens} out={resp.output_tokens} total={resp.total_tokens}"
        )
        if resp.thinking_tokens:
            console.print(f"Thinking tokens: {resp.thinking_tokens}")
        if resp.cache_read_tokens or resp.cache_write_tokens:
            console.print(
                f"Cache tokens: read={resp.cache_read_tokens} "
                f"write={resp.cache_write_tokens}"
            )
        console.print(f"Latency: {resp.latency_ms:.0f}ms")
        console.print(f"Cost: ${resp.estimated_cost_usd:.6f}")
        if resp.tokens_per_second is not None:
            console.print(
                f"Throughput: {resp.tokens_per_second:.1f} tok/s ({resp.tps_basis})"
            )

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
@click.option("--ollama-host", type=str, default=None, help="Ollama endpoint for scheduled runs")
@click.option("--install", is_flag=True, help="Install the schedule on this system")
@click.option("--uninstall", is_flag=True, help="Remove installed atomics schedule")
def schedule(
    tier: str,
    interval: int,
    max_iterations: int,
    provider_name: str,
    fmt: str,
    ollama_host: str | None,
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
            click.echo(entry)
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
            # Use click.echo for raw config files — Rich wraps long lines
            # which can break embedded flags like --provider across lines
            click.echo("atomics.service:")
            click.echo(service)
            click.echo("atomics.timer:")
            click.echo(timer)
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
            click.echo(plist)


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
@click.option("--narrative", is_flag=True, default=False, help="Print a plain-English business-case summary")
@click.option("--output", "-o", "out_file", type=click.Path(), default=None,
              help="Write JSON summary to FILE instead of (or alongside) table output")
def compare(
    by: str,
    since_hours: float | None,
    tier: str | None,
    category: str | None,
    narrative: bool,
    out_file: str | None,
) -> None:
    """Compare providers or models side-by-side (add --narrative for a business-case summary)."""
    settings = load_settings()
    from atomics.model_classes import classify_model
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

        # Only surface the optional fidelity columns when there's signal, to keep
        # the table readable.
        any_thinking = any((r.get("avg_thinking_tokens") or 0) > 0 for r in rows)
        any_cache = any(
            (r.get("total_cache_read_tokens") or 0) or (r.get("total_cache_write_tokens") or 0)
            for r in rows
        )

        label = "Provider" if by == "provider" else "Model"
        detail_label = "Model(s)" if by == "provider" else "Provider"
        table = Table(title=f"Comparison by {label}", show_lines=True)
        table.add_column(label, style="magenta bold")
        table.add_column(detail_label, style="dim")
        table.add_column("Class", style="cyan")
        table.add_column("Tasks", justify="right")
        table.add_column("Quality", justify="right", style="green bold")
        table.add_column("P50 Lat.", justify="right")
        table.add_column("P95 Lat.", justify="right")
        table.add_column("Avg tok/s", justify="right", style="blue")
        table.add_column("Basis", style="dim")
        if any_thinking:
            table.add_column("Avg think", justify="right")
        if any_cache:
            table.add_column("Cache r/w", justify="right")
        table.add_column("$/1K tok", justify="right", style="yellow")
        table.add_column("Value Score", justify="right", style="cyan bold")
        table.add_column("Total $", justify="right", style="yellow bold")

        classes_seen: set[str] = set()
        bases_seen: set[str] = set()
        for r in rows:
            if by == "model":
                model_classes = {classify_model(r["group_key"])}
            else:
                models = (r.get("models_used") or "").split(",")
                model_classes = {classify_model(m.strip()) for m in models if m.strip()}
            cls_label = ", ".join(sorted({c.value for c in model_classes})) or "—"
            classes_seen.update(c.value for c in model_classes)
            avg_tps = r.get("avg_tokens_per_second")
            tps_label = f"{avg_tps:.1f}" if avg_tps else "—"
            bases = sorted({b for b in (r.get("tps_bases") or "").split(",") if b})
            bases_seen.update(bases)
            basis_label = ", ".join(bases) or "—"
            acc = r.get("avg_accuracy_score")
            quality_label = f"{acc * 100:.1f}%" if acc is not None else "—"
            val = r.get("value_score")
            value_label = f"{val:.1f}" if val is not None else "—"
            cells = [
                r["group_key"],
                r.get("models_used", "—") or "—",
                cls_label,
                str(r["task_count"]),
                quality_label,
                f"{r['p50_latency_ms']:.0f}ms",
                f"{r['p95_latency_ms']:.0f}ms",
                tps_label,
                basis_label,
            ]
            if any_thinking:
                think = r.get("avg_thinking_tokens") or 0
                cells.append(f"{think:.0f}" if think else "—")
            if any_cache:
                cr = r.get("total_cache_read_tokens") or 0
                cw = r.get("total_cache_write_tokens") or 0
                cells.append(f"{cr}/{cw}" if (cr or cw) else "—")
            cells += [
                f"${r['cost_per_1k_tokens']:.4f}",
                value_label,
                f"${r['total_cost']:.4f}",
            ]
            table.add_row(*cells)
        console.print(table)

        if len(classes_seen) > 1:
            console.print(
                "\n[yellow]⚠ Mixed model classes detected.[/yellow] "
                "For a fair comparison, run the same tier with equivalent models "
                "(e.g. all light-class or all mid-class)."
            )

        if len(bases_seen) > 1:
            console.print(
                "\n[yellow]⚠ Mixed throughput bases detected (wall_clock vs generation).[/yellow] "
                "tok/s is not directly comparable: wall_clock includes network/queue time, "
                "while generation measures pure decode speed."
            )

        if narrative:
            _print_narrative(console, rows, by)

        if out_file:
            import json as _json
            from pathlib import Path
            Path(out_file).write_text(_json.dumps(rows, indent=2, default=str))
            console.print(f"\n[dim]Comparison written to {out_file}[/dim]")
    finally:
        repo.close()


def _print_narrative(console: Console, rows: list[dict], by: str) -> None:
    """Print a plain-English business-case summary of the comparison data."""
    scored = [r for r in rows if r.get("avg_accuracy_score") is not None]
    if not scored:
        console.print(
            "\n[dim]No accuracy scores yet. Run [bold]atomics eval[/bold] to generate quality scores.[/dim]"
        )
        return

    scored_sorted = sorted(scored, key=lambda r: r.get("avg_accuracy_score", 0), reverse=True)
    best = scored_sorted[0]
    free_options = [r for r in scored if r.get("cost_per_1k_tokens", 1) < 0.0001]
    paid_options = [r for r in scored if r.get("cost_per_1k_tokens", 0) >= 0.0001]

    console.print("\n[bold cyan]── Business Case Summary ──────────────────────────────[/bold cyan]")

    best_acc = best["avg_accuracy_score"] * 100
    console.print(
        f"\n[bold]{best['group_key']}[/bold] leads on quality at "
        f"[green]{best_acc:.1f}%[/green] accuracy "
        f"(cost: [yellow]${best['cost_per_1k_tokens']:.4f}/1K tokens[/yellow])."
    )

    if free_options and paid_options:
        best_free = max(free_options, key=lambda r: r.get("avg_accuracy_score", 0))
        best_paid = max(paid_options, key=lambda r: r.get("avg_accuracy_score", 0))
        free_acc = best_free["avg_accuracy_score"] * 100
        paid_acc = best_paid["avg_accuracy_score"] * 100
        gap_pp = paid_acc - free_acc
        paid_cost = best_paid["total_cost"]

        console.print(
            f"\n[bold]Self-hosted vs API:[/bold] "
            f"[cyan]{best_free['group_key']}[/cyan] achieves [green]{free_acc:.1f}%[/green] quality "
            f"at [green]$0 marginal cost[/green], "
            f"versus [magenta]{best_paid['group_key']}[/magenta] at "
            f"[green]{paid_acc:.1f}%[/green] for [yellow]${paid_cost:.4f}[/yellow] total spend."
        )
        if gap_pp <= 0:
            console.print(
                f"  → Self-hosted [bold]matches or exceeds[/bold] API quality "
                f"([bold]{abs(gap_pp):.1f}pp ahead[/bold]). The case for self-hosting is clear."
            )
        elif gap_pp < 10:
            console.print(
                f"  → Quality gap is only [bold]{gap_pp:.1f} percentage points[/bold]. "
                "Self-hosted delivers near-equivalent output at a fraction of the cost."
            )
        else:
            console.print(
                f"  → Quality gap is [yellow]{gap_pp:.1f} percentage points[/yellow]. "
                "Consider a larger local model to close the gap before switching."
            )

    all_costs = [r["total_cost"] for r in paid_options]
    if all_costs:
        total_api_spend = sum(all_costs)
        console.print(
            f"\n[bold]Data privacy:[/bold] Every token sent to an external API "
            "transits third-party infrastructure. Self-hosted inference eliminates "
            "this exposure entirely — critical for regulated industries or sensitive workloads."
        )
        console.print(
            f"\n[bold]Total API spend in this comparison:[/bold] "
            f"[yellow]${total_api_spend:.4f}[/yellow] "
            f"across {len(paid_options)} paid provider(s)."
        )

    console.print(
        "\n[dim]Value Score = quality / cost-per-1K-tokens. "
        "Higher is better. Local inference uses $0.001 as a floor (not literally free).[/dim]"
    )


@cli.command()
@click.option(
    "--provider", "-p", "provider_name",
    type=PROVIDER_CHOICES,
    default="ollama",
    help="Provider to evaluate (model under test)",
)
@click.option("--model", "-m", type=str, default=None, help="Model override for the provider under test")
@click.option("--ollama-host", type=str, default=None, help="Ollama endpoint for model under test")
@click.option("--vllm-host", "vllm_host", type=str, default=None, help="vLLM/OpenAI-compatible base URL (default: ATOMICS_VLLM_HOST)")
@click.option("--region", type=str, default="us-east-1", help="AWS region for Bedrock")
@click.option(
    "--judge-provider", "judge_provider_name",
    type=PROVIDER_CHOICES,
    default="ollama",
    help="Provider to use as judge (default: ollama — $0 cost)",
)
@click.option("--judge-model", type=str, default=None, help="Model override for the judge")
@click.option("--judge-host", type=str, default=None, help="Ollama host for the judge (if different)")
@click.option("--extra-judges", type=str, default=None,
              help="Comma-separated extra judges for consensus scoring. "
                   "Format: provider:model (e.g. claude:claude-sonnet-4-6,ollama:deepseek-r1:14b). "
                   "Reports mean quality and inter-judge stddev.")
@click.option("--save/--no-save", "save_results", default=True, help="Persist results to the database")
@click.option("--thinking/--no-thinking", "thinking_flag", default=None, help="Enable/disable thinking for capable models")
@click.option("--thinking-budget", type=int, default=None, help="Max thinking tokens")
def eval(
    provider_name: str,
    model: str | None,
    ollama_host: str | None,
    vllm_host: str | None,
    region: str,
    judge_provider_name: str,
    judge_model: str | None,
    judge_host: str | None,
    extra_judges: str | None,
    save_results: bool,
    thinking_flag: bool | None,
    thinking_budget: int | None,
) -> None:
    """Run the fixed eval fixture set and score quality with an LLM judge.

    Produces an accuracy score (0-100%) for each fixture and an overall quality
    rating. Run against multiple providers then use 'atomics compare --narrative'
    to generate the business-case comparison.

    Example:
      atomics eval --provider ollama --model qwen2.5:7b
      atomics eval --provider claude
      atomics eval --provider openai --model gpt-4o
      atomics compare --narrative
    """
    settings = load_settings()
    _setup_logging(settings.log_level)
    console = Console()

    def _build_provider(name: str, mdl: str | None, host: str | None):
        if name == "claude":
            if not settings.anthropic_api_key:
                console.print("[red]ANTHROPIC_API_KEY not set.[/red]")
                sys.exit(1)
            from atomics.providers.claude import ClaudeProvider
            return ClaudeProvider(
                api_key=settings.anthropic_api_key,
                default_model=mdl or settings.default_model,
            )
        elif name == "bedrock":
            from atomics.providers.bedrock import BedrockProvider
            return BedrockProvider(region=region, model_id=mdl or "us.anthropic.claude-sonnet-4-6")
        elif name == "openai":
            from atomics.providers.openai import OpenAIProvider
            if settings.openai_api_key:
                return OpenAIProvider(api_key=settings.openai_api_key, default_model=mdl or "gpt-4o")
            console.print("[red]OPENAI_API_KEY not set.[/red]")
            sys.exit(1)
        elif name == "ollama":
            from atomics.providers.ollama import OllamaProvider
            return OllamaProvider(
                host=host or settings.ollama_host,
                default_model=mdl or settings.ollama_model,
                timeout=settings.ollama_timeout,
            )
        elif name == "vllm":
            from atomics.providers.vllm import VllmProvider
            return VllmProvider(
                base_url=vllm_host or settings.vllm_host,
                default_model=mdl or settings.vllm_model,
                timeout=settings.vllm_timeout,
            )
        elif name == "brain-gateway":
            from atomics.providers.brain_gateway import BrainGatewayProvider
            return BrainGatewayProvider(
                url=host or settings.brain_gateway_url,
                default_model=mdl,
            )
        console.print(f"[red]Unknown provider: {name}[/red]")
        sys.exit(1)

    test_provider = _build_provider(provider_name, model, ollama_host)
    # Judge host falls back to: --judge-host → --ollama-host → ATOMICS_OLLAMA_HOST env/.env
    judge_provider = _build_provider(judge_provider_name, judge_model, judge_host or ollama_host or settings.ollama_host)

    # Parse --extra-judges "claude:claude-sonnet-4-6,ollama:deepseek-r1:14b@http://host:11434"
    extra_judge_pairs: list[tuple] = []
    if extra_judges:
        for spec in extra_judges.split(","):
            spec = spec.strip()
            if not spec:
                continue
            ej_host = None
            if "@" in spec:
                spec, ej_host = spec.rsplit("@", 1)
            parts = spec.split(":", 1)
            ej_provider_name = parts[0]
            ej_model = parts[1] if len(parts) > 1 else None
            ej_provider = _build_provider(
                ej_provider_name, ej_model,
                ej_host or judge_host or ollama_host or settings.ollama_host,
            )
            extra_judge_pairs.append((ej_provider, ej_model))

    from atomics.eval.fixtures import EVAL_FIXTURES

    judge_label = judge_model or "default"
    if extra_judge_pairs:
        judge_label += f" + {len(extra_judge_pairs)} consensus"
    console.print(
        f"\n[bold]Eval run[/bold] — model under test: [cyan]{provider_name}[/cyan] "
        f"({model or 'default'})\n"
        f"Judge: [cyan]{judge_provider_name}[/cyan] ({judge_label})\n"
        f"Fixtures: [bold]{len(EVAL_FIXTURES)}[/bold] | Results saved: [bold]{'yes' if save_results else 'no'}[/bold]\n"
    )

    repo = None
    if save_results:
        from atomics.storage.repository import MetricsRepository
        repo = MetricsRepository(settings.db_path)

    from atomics.eval.runner import run_eval

    result_table = Table(title="Eval Results", show_lines=True)
    result_table.add_column("ID", style="dim")
    result_table.add_column("Complexity", style="cyan")
    result_table.add_column("Prompt (truncated)", no_wrap=False, max_width=45)
    result_table.add_column("Quality", justify="right", style="green bold")
    result_table.add_column("Latency", justify="right")
    result_table.add_column("Tokens", justify="right")
    result_table.add_column("Cost", justify="right", style="yellow")
    result_table.add_column("Rationale", no_wrap=False, max_width=40, style="dim")

    # Pre-allocate a run_id and create the parent row so FK constraints are satisfied
    import uuid as _uuid
    eval_run_id = _uuid.uuid4().hex[:12]
    if provider_name == "ollama":
        effective_model = model or settings.ollama_model
    elif provider_name == "vllm":
        effective_model = model or settings.vllm_model
    else:
        effective_model = model or settings.default_model
    if repo:
        repo.create_run(
            eval_run_id,
            tier="eval",
            provider=provider_name,
            model=effective_model,
            trigger="eval",
        )

    def on_done(fr) -> None:
        judge = fr.judge
        tr = fr.task_result
        if tr.status.value == "failed":
            quality = "[red]FAIL[/red]"
            rationale = tr.error_message[:80]
        elif judge and not judge.parse_failed:
            quality = f"{judge.score * 100:.0f}%"
            rationale = judge.rationale[:80]
        else:
            quality = "[yellow]?[/yellow]"
            rationale = "judge parse failed"
        result_table.add_row(
            fr.fixture.id,
            fr.fixture.complexity.value,
            fr.fixture.prompt[:60] + "…",
            quality,
            f"{tr.latency_ms:.0f}ms",
            str(tr.total_tokens),
            f"${tr.estimated_cost_usd:.6f}",
            rationale,
        )
        if repo:
            repo.save_task_result(tr)

    # Auto-detect thinking if not explicitly set
    eff_thinking = thinking_flag
    if eff_thinking is None and model:
        from atomics.model_classes import supports_thinking
        if supports_thinking(model):
            eff_thinking = True

    summary = asyncio.run(run_eval(
        test_provider,
        judge_provider=judge_provider,
        model=model,
        judge_model=judge_model,
        run_id=eval_run_id,
        on_fixture_done=on_done,
        thinking=eff_thinking,
        thinking_budget=thinking_budget,
        extra_judges=extra_judge_pairs,
    ))

    console.print(result_table)

    overall = summary.overall_accuracy
    quality_str = f"{overall * 100:.1f}%" if overall is not None else "—"
    value_str = f"{summary.value_score:.1f}" if summary.value_score is not None else "—"

    summary_table = Table(title="Eval Summary", show_lines=True)
    summary_table.add_column("Metric", style="dim")
    summary_table.add_column("Value", style="bold")
    summary_table.add_row("Provider", provider_name)
    summary_table.add_row("Model", model or "default")
    summary_table.add_row("Overall Quality", f"[green]{quality_str}[/green]")
    summary_table.add_row("Value Score", f"[cyan]{value_str}[/cyan]")
    summary_table.add_row("Avg Latency", f"{summary.avg_latency_ms:.0f}ms")
    summary_table.add_row("Total Tokens", f"{summary.total_tokens:,}")
    summary_table.add_row("Total Cost", f"${summary.total_cost_usd:.6f}")
    summary_table.add_row("Fixtures Run", str(len(summary.fixture_results)))
    pf_rate = summary.parse_failure_rate
    pf_style = "green" if pf_rate == 0 else "yellow" if pf_rate < 0.1 else "red"
    summary_table.add_row("Judge Parse Failures", f"[{pf_style}]{pf_rate * 100:.1f}%[/{pf_style}]")
    console.print(summary_table)

    if repo:
        repo.close()
        console.print(f"\n[dim]Results saved to database. Run [bold]atomics compare --narrative[/bold] after evaluating multiple providers.[/dim]")


@cli.command()
@click.option(
    "--provider", "-p", "provider_name",
    type=PROVIDER_CHOICES,
    default="ollama",
    help="Provider to stress test (default: ollama for raw GPU stress)",
)
@click.option("--model", "-m", type=str, default=None, help="Model to stress (default: ATOMICS_OLLAMA_MODEL)")
@click.option("--models", "models_csv", type=str, default=None,
              help="Comma-separated list of models for contention testing (e.g. qwen2.5:3b,qwen2.5:7b).")
@click.option("--ollama-host", type=str, default=None, help="Ollama endpoint")
@click.option("--profile", "profile_path", type=click.Path(exists=True), default=None,
              help="Target profile YAML (replaces --model/--ollama-host).")
@click.option("--max-concurrency", "-c", type=int, default=8, help="Max parallel requests (ramps 1→2→4→...)")
@click.option("--phase-seconds", "-s", type=float, default=15.0, help="Seconds at each concurrency level")
@click.option("--num-predict", type=int, default=2048, help="Max output tokens per request")
@click.option("--save/--no-save", "save_results", default=True, help="Persist results to database")
def stress(
    provider_name: str,
    model: str | None,
    models_csv: str | None,
    ollama_host: str | None,
    profile_path: str | None,
    max_concurrency: int,
    phase_seconds: float,
    num_predict: int,
    save_results: bool,
) -> None:
    """Stress test — ramp concurrency to find saturation point.

    Works with any provider: Ollama (raw GPU metrics), OpenAI, Claude, Bedrock.
    Use --profile for custom target profiles (app-level AI gates).
    Use --models for multi-model VRAM contention testing.

    \b
    Examples:
      atomics stress --model qwen2.5:7b --ollama-host http://gpu-host:11434
      atomics stress --models qwen2.5:3b,qwen2.5:7b --ollama-host http://gpu:11434
      atomics stress --profile profiles/local/gatekeeper.yaml
      atomics stress --provider openai --model gpt-4o-mini
    """
    settings = load_settings()
    _setup_logging(settings.log_level)
    import logging as _logging
    _logging.getLogger("httpx").setLevel(_logging.WARNING)
    _logging.getLogger("httpcore").setLevel(_logging.WARNING)
    console = Console()

    if models_csv:
        from atomics.contention import run_contention
        host = ollama_host or settings.ollama_host
        model_list = [m.strip() for m in models_csv.split(",") if m.strip()]
        console.print(
            f"[bold]Contention test[/bold] — {len(model_list)} models on {host}\n"
            f"Models: {', '.join(model_list)}\n"
            f"Phase: {phase_seconds}s solo + {phase_seconds}s mixed\n"
        )
        result = asyncio.run(run_contention(
            host=host,
            models=model_list,
            concurrency=1,
            phase_seconds=phase_seconds,
            num_predict=min(num_predict, 512),
        ))
        ctable = Table(title="Contention Results", show_lines=True)
        ctable.add_column("Model", style="cyan")
        ctable.add_column("Solo tok/s", justify="right")
        ctable.add_column("Mixed tok/s", justify="right")
        ctable.add_column("Factor", justify="right")
        ctable.add_column("Mixed P95", justify="right")
        ctable.add_column("Errors", justify="right")
        for mr in result.contention_results:
            solo = result.solo_tps.get(mr.model, 0.0)
            factor = result.contention_factor(mr.model)
            factor_str = f"{factor:.2f}x" if factor is not None else "n/a"
            factor_color = "green" if (factor or 1.0) >= 0.9 else ("yellow" if (factor or 1.0) >= 0.7 else "red")
            ctable.add_row(
                mr.model,
                f"{solo:.1f}",
                f"{mr.avg_tps:.1f}",
                f"[{factor_color}]{factor_str}[/{factor_color}]",
                f"{mr.p95_ms/1000:.1f}s",
                str(mr.failed),
            )
        console.print(ctable)
        console.print(f"\n[dim]Total duration: {result.duration_seconds:.1f}s[/dim]")
        return

    if profile_path:
        from atomics.profiles import load_profile
        tp = load_profile(profile_path)
        effective_model = tp.model
        target_label = f"profile:{tp.name} ({tp.type})"
        use_provider_mode = False
    elif provider_name != "ollama":
        use_provider_mode = True
        effective_model = model or ("gpt-4o" if provider_name == "openai" else settings.default_model)
        target_label = f"{provider_name} / {effective_model}"
    else:
        use_provider_mode = False
        host = ollama_host or settings.ollama_host
        effective_model = model or settings.ollama_model
        target_label = f"{effective_model} @ {host}"

    console.print(
        f"[bold]Stress test[/bold] — {target_label}\n"
        f"Ramp: 1→{max_concurrency} concurrent | "
        f"{phase_seconds:.0f}s per phase | "
        f"{num_predict} max tokens/request\n"
    )

    def _on_phase(phase):
        uplift = ""
        if len(phases_so_far) > 0:
            base = phases_so_far[0].aggregate_tps
            if base > 0:
                pct = (phase.aggregate_tps - base) / base * 100
                uplift = f"  ({pct:+.0f}%)" if pct != 0 else ""
        phases_so_far.append(phase)
        fail_tag = f" [red]({phase.failed} failed)[/red]" if phase.failed else ""
        cost_tag = f"  ${phase.total_cost_usd:.4f}" if phase.total_cost_usd > 0 else ""
        console.print(
            f"  concurrent({phase.concurrency}): "
            f"[cyan]{phase.aggregate_tps:6.1f}[/cyan] tok/s  "
            f"P50 {phase.avg_latency_ms / 1000:.1f}s  "
            f"P95 {phase.p95_latency_ms / 1000:.1f}s  "
            f"({phase.requests} reqs, {phase.total_output_tokens:,} tokens)"
            f"[dim]{uplift}[/dim]{cost_tag}{fail_tag}"
        )

    phases_so_far: list = []

    console.print("[bold]Throughput by concurrency:[/bold]")

    if profile_path:
        from atomics.stress import run_stress_profile
        result = asyncio.run(run_stress_profile(
            profile=tp,
            max_concurrency=max_concurrency,
            phase_seconds=phase_seconds,
            on_phase=_on_phase,
        ))
    elif use_provider_mode:
        provider = _make_provider(provider_name, effective_model, ollama_host, settings)
        from atomics.stress import run_stress_provider
        result = asyncio.run(run_stress_provider(
            provider=provider,
            model=effective_model,
            max_concurrency=max_concurrency,
            phase_seconds=phase_seconds,
            num_predict=num_predict,
            on_phase=_on_phase,
        ))
    else:
        from atomics.stress import run_stress
        result = asyncio.run(run_stress(
            host=host,
            model=effective_model,
            max_concurrency=max_concurrency,
            phase_seconds=phase_seconds,
            num_predict=num_predict,
            on_phase=_on_phase,
        ))

    console.print()

    summary = Table(title="Stress Test Summary", show_lines=True, title_style="bold")
    summary.add_column("Metric", style="dim")
    summary.add_column("Value", style="cyan bold")

    if result.provider:
        summary.add_row("Provider", result.provider)
    if result.gpu_name:
        summary.add_row("GPU", result.gpu_name)
    summary.add_row("Model", result.model)
    summary.add_row("Duration", f"{result.duration_seconds:.0f}s")
    summary.add_row("Total requests", f"{result.total_requests} ({result.total_failed} failed)")
    summary.add_row("Total tokens", f"{result.total_tokens:,}")
    summary.add_row("Peak throughput", f"{result.peak_tps:.1f} tok/s @ concurrency={result.saturation_concurrency}")

    if result.total_cost_usd > 0:
        summary.add_row("Total cost", f"[yellow]${result.total_cost_usd:.4f}[/yellow]")

    if result.vram_peak_mb is not None:
        vram_str = f"{result.vram_peak_mb:.0f} MB"
        if result.vram_total_mb:
            pct = result.vram_peak_mb / result.vram_total_mb * 100
            vram_str += f" / {result.vram_total_mb:.0f} MB ({pct:.0f}%)"
        summary.add_row("Peak VRAM", vram_str)

    if len(result.phases) >= 2:
        base = result.phases[0].aggregate_tps
        peak = result.peak_tps
        if base > 0:
            scaling = peak / base
            summary.add_row("Scaling", f"{scaling:.2f}x (1→{result.saturation_concurrency})")

    last = result.phases[-1] if result.phases else None
    if last and last.aggregate_tps < result.peak_tps * 0.95:
        summary.add_row("Throttling", "[yellow]Possible — throughput dropped at max concurrency[/yellow]")
    else:
        summary.add_row("Throttling", "[green]None detected[/green]")

    console.print(summary)

    if save_results:
        from atomics.storage.repository import MetricsRepository
        repo = MetricsRepository(settings.db_path)
        repo.save_stress_result(result)
        repo.close()
        console.print(f"\n[dim]Results saved to database.[/dim]")


@cli.command()
@click.option("--users", "-u", type=int, required=True, help="Number of semi-active users")
@click.option("--think-time", "--think", type=float, default=300.0, show_default=True,
              help="Avg seconds between requests per user")
@click.option("--response-tokens", type=int, default=400, show_default=True,
              help="Avg output tokens per response")
@click.option("--burst", type=float, default=0.2, show_default=True,
              help="Burst factor — fraction of users spiking simultaneously")
@click.option("--model", "-m", type=str, default=None,
              help="Pull stress data from DB for this model")
@click.option("--peak-tps", type=float, default=None,
              help="Manual peak throughput (tok/s) — used if no DB data")
@click.option("--single-latency", type=float, default=None,
              help="Manual single-request latency in ms — used if no DB data")
def capacity(
    users: int,
    think_time: float,
    response_tokens: int,
    burst: float,
    model: str | None,
    peak_tps: float | None,
    single_latency: float | None,
) -> None:
    """Project user capacity from stress test data or manual parameters.

    Uses queueing theory to estimate concurrent requests, latency, and
    system verdict at different load levels. Feed it your stress test
    results or manual numbers for cloud API endpoints.

    Examples:
      atomics capacity --users 200 --model qwen2.5:7b
      atomics capacity --users 100 --peak-tps 107 --single-latency 15000
      atomics capacity --users 50 --think-time 600 --model qwen2.5:7b
    """
    from atomics.capacity import CapacityProjection, LoadProfile, project_capacity

    settings = load_settings()
    console = Console()
    phases: list[dict] = []
    effective_peak_tps = peak_tps or 0.0
    effective_model = model or ""

    if model:
        from atomics.storage.repository import MetricsRepository
        repo = MetricsRepository(settings.db_path)
        rows = repo.get_stress_results(model=model)
        repo.close()

        if not rows:
            console.print(f"[red]No stress data for model '{model}'. Run atomics stress first, or use --peak-tps.[/red]")
            raise SystemExit(1)

        latest = rows[-1]
        effective_peak_tps = latest["peak_tps"]

        import json
        phases_json = latest.get("phases_json")
        if phases_json:
            raw_phases = json.loads(phases_json) if isinstance(phases_json, str) else phases_json
            phases = [
                {
                    "concurrency": p.get("concurrency", 1),
                    "aggregate_tps": p.get("aggregate_tps", 0),
                    "avg_latency_ms": p.get("avg_latency_ms", 0),
                    "p95_latency_ms": p.get("p95_latency_ms", 0),
                }
                for p in raw_phases
            ]
        console.print(f"[dim]Using stress data for {model} (peak {effective_peak_tps:.1f} tok/s)[/dim]\n")

    elif peak_tps and single_latency:
        effective_peak_tps = peak_tps
        phases = [
            {"concurrency": 1, "aggregate_tps": peak_tps, "avg_latency_ms": single_latency, "p95_latency_ms": single_latency * 1.5},
        ]

    else:
        console.print("[red]Specify --model (pulls from DB) or both --peak-tps and --single-latency[/red]")
        raise SystemExit(1)

    profile = LoadProfile(
        users=users, think_time_s=think_time,
        response_tokens=response_tokens, burst_factor=burst,
    )

    result = project_capacity(
        profile=profile, phases=phases,
        peak_tps=effective_peak_tps, model=effective_model,
    )

    title = f"Capacity Projection: {effective_model or 'custom'} ({effective_peak_tps:.0f} tok/s peak)"
    table = Table(title=title, show_lines=True)
    table.add_column("Scenario", style="cyan bold")
    table.add_column("Concurrent", justify="right")
    table.add_column("P50 Latency", justify="right")
    table.add_column("P95 Latency", justify="right")
    table.add_column("Queue", justify="right", style="dim")
    table.add_column("Verdict", justify="center")

    verdict_style = {"OK": "[green]OK[/green]", "CAUTION": "[yellow]CAUTION[/yellow]",
                     "SLOW": "[red]SLOW[/red]", "OVERLOAD": "[bold red]OVERLOAD[/bold red]"}

    for s in result.scenarios:
        table.add_row(
            s.name,
            f"{s.concurrent:.1f}",
            f"{s.p50_latency_ms / 1000:.0f}s",
            f"{s.p95_latency_ms / 1000:.0f}s",
            f"{s.queue_depth:.1f}",
            verdict_style.get(s.verdict, s.verdict),
        )

    console.print(table)
    console.print(f"\n[bold]Recommendation:[/bold] {result.recommendation}")


@cli.command()
def doctor() -> None:
    """Check Python, database, API keys, optional deps, and scheduler tooling."""
    from atomics.doctor import run_doctor

    sys.exit(run_doctor())


@cli.command("export")
@click.option(
    "--suite",
    type=click.Choice(["tasks", "stress", "sweep", "soak", "all"]),
    default="tasks",
    help="Which suite to export: tasks (default), stress, sweep, soak, or all",
)
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
    suite: str,
    since_hours: float | None,
    limit: int | None,
    fmt: str,
    out_file,
) -> None:
    """Export stored metrics as JSON lines or CSV.

    Examples:
      atomics export                          # task results (default)
      atomics export --suite stress           # stress test history
      atomics export --suite sweep -o out.jsonl
      atomics export --suite all --format csv -o all_metrics.csv
    """
    import csv as _csv
    import json as _json

    settings = load_settings()
    from atomics.exporters import write_tasks_export
    from atomics.storage.repository import MetricsRepository

    repo = MetricsRepository(settings.db_path)
    try:
        if suite == "tasks":
            rows = repo.query_task_results(since_hours=since_hours, limit=limit)
            write_tasks_export(rows, fmt, out_file)
        elif suite == "stress":
            rows = repo.get_stress_results()
            if limit:
                rows = rows[:limit]
            _write_generic_export(rows, fmt, out_file)
        elif suite == "sweep":
            rows = repo.get_sweep_results()
            if limit:
                rows = rows[:limit]
            _write_generic_export(rows, fmt, out_file)
        elif suite == "soak":
            rows = repo.get_soak_results()
            if limit:
                rows = rows[:limit]
            _write_generic_export(rows, fmt, out_file)
        elif suite == "all":
            all_rows: list[dict] = []
            task_rows = repo.query_task_results(since_hours=since_hours, limit=limit)
            for r in task_rows:
                r["_suite"] = "tasks"
                all_rows.append(r)
            for r in repo.get_stress_results():
                r["_suite"] = "stress"
                all_rows.append(r)
            for r in repo.get_sweep_results():
                r["_suite"] = "sweep"
                all_rows.append(r)
            for r in repo.get_soak_results():
                r["_suite"] = "soak"
                all_rows.append(r)
            if limit:
                all_rows = all_rows[:limit]
            _write_generic_export(all_rows, fmt, out_file)
    finally:
        repo.close()


def _write_generic_export(rows: list[dict], fmt: str, out_file) -> None:
    """Write arbitrary row dicts to jsonl or csv."""
    import csv as _csv
    import json as _json

    if not rows:
        return
    if fmt == "jsonl":
        for row in rows:
            out_file.write(_json.dumps(row, default=str) + "\n")
    elif fmt == "csv":
        writer = _csv.DictWriter(out_file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


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


@cli.command("models")
@click.option(
    "--provider", "-p", "provider_name",
    type=click.Choice(["ollama", "vllm"], case_sensitive=False),
    default="ollama",
    help="Backend to list models from (default: ollama)",
)
@click.option(
    "--host",
    default=None,
    help="Ollama host URL (default: ATOMICS_OLLAMA_HOST or http://localhost:11434)",
)
@click.option(
    "--vllm-host", "vllm_host",
    default=None,
    help="vLLM/OpenAI-compatible base URL (default: ATOMICS_VLLM_HOST or http://localhost:8000/v1)",
)
def models(provider_name: str, host: str | None, vllm_host: str | None) -> None:
    """List available models on an Ollama or vLLM/OpenAI-compatible instance."""
    settings = load_settings()
    console = Console()

    if provider_name == "vllm":
        from atomics.providers.vllm import VllmProvider
        base_url = vllm_host or settings.vllm_host
        provider = VllmProvider(base_url=base_url)
        title = f"vLLM Models — {base_url}"
    else:
        from atomics.providers.ollama import OllamaProvider
        effective_host = host or settings.ollama_host
        provider = OllamaProvider(host=effective_host)
        title = f"Ollama Models — {effective_host}"

    try:
        result = asyncio.run(provider.list_models())
    except ConnectionError as exc:
        click.echo(str(exc), err=True)
        raise SystemExit(1)

    table = Table(title=title, show_lines=True)
    table.add_column("Model", style="cyan bold")
    if provider_name == "ollama":
        table.add_column("Size", justify="right")
        table.add_column("Params", justify="right")
    table.add_column("Family", style="dim")
    table.add_column("Class", style="yellow")
    table.add_column("Thinking", justify="center")

    for m in sorted(result, key=lambda x: x.get("size_gb", 0)):
        cls_str = str(m["model_class"])
        cls_style = {"light": "green", "mid": "yellow", "heavy": "red"}.get(cls_str, "dim")
        row = [str(m["name"])]
        if provider_name == "ollama":
            row += [f"{m['size_gb']:.1f} GB", str(m.get("parameter_size", ""))]
        row += [
            str(m.get("family", "")),
            f"[{cls_style}]{cls_str}[/{cls_style}]",
            "[green]yes[/green]" if m.get("thinking") else "[dim]no[/dim]",
        ]
        table.add_row(*row)

    console.print(table)
    unknown = [m for m in result if m["model_class"] == "unknown"]
    if unknown:
        console.print(
            f"\n[yellow]{len(unknown)} unregistered model(s) — "
            f"add to model_classes.py for accurate comparison[/yellow]"
        )


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


# ── Shared provider builder for new suites ────────────────────────────────────

def _make_provider(name: str, mdl: str | None, host: str | None, settings, *, vllm_host: str | None = None):
    """Build a provider instance — mirrors the pattern inside eval()."""
    if name == "claude":
        if not settings.anthropic_api_key:
            click.echo("Error: ANTHROPIC_API_KEY not set.", err=True)
            sys.exit(1)
        from atomics.providers.claude import ClaudeProvider
        return ClaudeProvider(api_key=settings.anthropic_api_key, default_model=mdl or settings.default_model)
    if name == "bedrock":
        from atomics.providers.bedrock import BedrockProvider
        return BedrockProvider(region="us-east-1", model_id=mdl or "us.anthropic.claude-sonnet-4-6")
    if name == "openai":
        if not settings.openai_api_key:
            click.echo("Error: OPENAI_API_KEY not set.", err=True)
            sys.exit(1)
        from atomics.providers.openai import OpenAIProvider
        return OpenAIProvider(api_key=settings.openai_api_key, default_model=mdl or "gpt-4o")
    if name == "vllm":
        from atomics.providers.vllm import VllmProvider
        return VllmProvider(
            base_url=vllm_host or settings.vllm_host,
            default_model=mdl or settings.vllm_model,
            timeout=settings.vllm_timeout,
        )
    if name == "brain-gateway":
        from atomics.providers.brain_gateway import BrainGatewayProvider
        return BrainGatewayProvider(url=host or settings.brain_gateway_url, default_model=mdl)
    from atomics.providers.ollama import OllamaProvider
    return OllamaProvider(
        host=host or settings.ollama_host,
        default_model=mdl or settings.ollama_model,
        timeout=settings.ollama_timeout,
    )


# ── atomics sweep ─────────────────────────────────────────────────────────────

@cli.command("sweep")
@click.option(
    "--provider", "-p", "provider_name",
    type=PROVIDER_CHOICES,
    default="ollama",
    help="Provider to evaluate (default: ollama for local models)",
)
@click.option(
    "--models", type=str, default=None,
    help="Comma-separated list of models to sweep (e.g. qwen2.5:1.5b,mistral:7b)",
)
@click.option(
    "--all-local", "all_local", is_flag=True, default=False,
    help="Discover and sweep all models on the Ollama host (ollama provider only)",
)
@click.option("--ollama-host", "ollama_host", type=str, default=None,
              help="Ollama host URL (default: ATOMICS_OLLAMA_HOST)")
@click.option("--host", "ollama_host", type=str, default=None, hidden=True,
              help="Deprecated alias for --ollama-host")
@click.option("--vllm-host", "vllm_host", type=str, default=None,
              help="vLLM/OpenAI-compatible base URL (default: ATOMICS_VLLM_HOST)")
@click.option("--judge-provider", "judge_provider_name", type=PROVIDER_CHOICES, default="ollama",
              help="Provider for quality judge (default: ollama — $0)")
@click.option("--judge-model", type=str, default=None, help="Judge model override")
@click.option("--judge-host", type=str, default=None, help="Ollama host for judge")
@click.option("--fixtures", type=str, default=None, help="Comma-separated fixture IDs (default: all)")
@click.option("--thinking/--no-thinking", "thinking_flag", default=None)
@click.option("--thinking-budget", type=int, default=None)
@click.option("--verbose", "-v", is_flag=True, default=False,
              help="Print each model's full reply alongside scores")
@click.option("--save/--no-save", "save_results", default=False,
              help="Persist sweep results to database (default: off)")
def sweep(
    provider_name: str,
    models: str | None,
    all_local: bool,
    ollama_host: str | None,
    vllm_host: str | None,
    judge_provider_name: str,
    judge_model: str | None,
    judge_host: str | None,
    fixtures: str | None,
    thinking_flag: bool | None,
    thinking_budget: int | None,
    verbose: bool,
    save_results: bool,
) -> None:
    """Sweep eval fixtures across multiple models and compare results.

    Works with any provider — local Ollama, Claude, OpenAI, Bedrock, or brain-gateway.
    Use --all-local with ollama to auto-discover models, or --models for any provider.

    Examples:
      atomics sweep --all-local --ollama-host http://gpu-host:11434
      atomics sweep --models qwen2.5:1.5b,qwen2.5:3b,mistral:7b
      atomics sweep --provider claude --models claude-sonnet-4-6,claude-haiku-4-5-20251001
      atomics sweep --provider openai --models gpt-4o,gpt-4o-mini
      atomics sweep --all-local --fixtures ev-01,ev-02,ev-03
      atomics sweep --all-local --save
    """
    from atomics.sweep import ModelSweepResult, run_model_sweep

    settings = load_settings()
    _setup_logging(settings.log_level)
    console = Console()
    effective_host = ollama_host or settings.ollama_host

    if all_local:
        if provider_name != "ollama":
            console.print("[red]--all-local only works with --provider ollama[/red]")
            raise SystemExit(1)
        from atomics.providers.ollama import OllamaProvider
        disc = OllamaProvider(host=effective_host)
        try:
            available = asyncio.run(disc.list_models())
        except ConnectionError as exc:
            click.echo(str(exc), err=True)
            raise SystemExit(1)
        model_list = [m["name"] for m in available]
        if not model_list:
            console.print("[red]No models found on Ollama host.[/red]")
            raise SystemExit(1)
        console.print(f"[bold]Discovered {len(model_list)} models on {effective_host}[/bold]\n")
    elif models:
        model_list = [m.strip() for m in models.split(",") if m.strip()]
    else:
        console.print("[red]Specify --models or --all-local[/red]")
        raise SystemExit(1)

    fixture_ids = [f.strip() for f in fixtures.split(",") if f.strip()] if fixtures else None

    def provider_factory(model_name: str):
        return _make_provider(provider_name, model_name, ollama_host, settings, vllm_host=vllm_host)

    judge_provider = _make_provider(
        judge_provider_name,
        judge_model,
        judge_host or effective_host,
        settings,
        vllm_host=vllm_host,
    )

    result_table = Table(title="Model Sweep Results", show_lines=True)
    result_table.add_column("Model", style="cyan bold")
    result_table.add_column("Quality", justify="right")
    result_table.add_column("Avg Latency", justify="right")
    result_table.add_column("Tokens", justify="right")
    result_table.add_column("Cost", justify="right", style="yellow")
    result_table.add_column("Fixtures", justify="right", style="dim")

    def on_fixture_done_verbose(fr) -> None:
        if not verbose:
            return
        tr = fr.task_result
        score_str = f"{fr.judge.score:.2f}" if fr.judge else "N/A"
        console.print(f"\n  [bold cyan]{fr.fixture.id}[/bold cyan] — score {score_str}")
        console.print(f"  [dim]prompt:[/dim] {fr.fixture.prompt[:120]}")
        if tr.response:
            console.print(f"  [dim]reply:[/dim]  {tr.response}")
        elif tr.error_message:
            console.print(f"  [red]error:[/red]  {tr.error_message}")
        if fr.judge and fr.judge.rationale:
            console.print(f"  [dim]judge:[/dim]  {fr.judge.rationale[:200]}")

    def on_model_done(r: ModelSweepResult) -> None:
        q = f"[green]{r.overall_quality * 100:.0f}%[/green]" if r.overall_quality is not None else "[red]FAIL[/red]"
        result_table.add_row(
            r.model,
            q,
            f"{r.avg_latency_ms:.0f}ms",
            f"{r.total_tokens:,}",
            f"${r.total_cost_usd:.6f}",
            str(r.fixtures_run),
        )
        console.print(f"\n  [dim]Done:[/dim] {r.model}")

    console.print(f"[bold]Sweeping {len(model_list)} models × "
                  f"{'all' if fixture_ids is None else len(fixture_ids)} fixtures[/bold]\n")

    results = asyncio.run(run_model_sweep(
        provider_factory=provider_factory,
        judge_provider=judge_provider,
        models=model_list,
        fixture_ids=fixture_ids,
        judge_model=judge_model,
        thinking=thinking_flag,
        thinking_budget=thinking_budget,
        on_model_done=on_model_done,
        on_fixture_done=on_fixture_done_verbose,
    ))

    console.print(result_table)

    ranked = sorted(
        [r for r in results if r.overall_quality is not None],
        key=lambda r: r.overall_quality or 0,
        reverse=True,
    )
    if ranked:
        best = ranked[0]
        console.print(
            f"\n[bold green]Best local model:[/bold green] {best.model} "
            f"({best.overall_quality * 100:.0f}% quality, "
            f"{best.avg_latency_ms:.0f}ms avg latency, "
            f"${best.total_cost_usd:.2f} total cost)"
        )

    failed = [r for r in results if r.overall_quality is None]
    if failed:
        console.print(
            f"\n[yellow]{len(failed)} model(s) failed: "
            f"{', '.join(r.model for r in failed)}[/yellow]"
        )

    if save_results:
        from atomics.storage.repository import MetricsRepository
        repo = MetricsRepository(settings.db_path)
        for r in results:
            repo.save_sweep_result(r)
        repo.close()
        console.print(f"\n[dim]Sweep results saved to database.[/dim]")


# ── atomics adversarial ───────────────────────────────────────────────────────

@cli.command("adversarial")
@click.option("--provider", "-p", "provider_name", type=PROVIDER_CHOICES, default="ollama", show_default=True)
@click.option("--model", "-m", type=str, default=None, help="Model override for the provider under test.")
@click.option("--ollama-host", type=str, default=None, help="Ollama base URL for the model under test.")
@click.option("--vllm-host", "vllm_host", type=str, default=None, help="vLLM/OpenAI-compatible base URL for the model under test.")
@click.option("--judge-provider", "judge_provider_name", type=PROVIDER_CHOICES, default="ollama", show_default=True)
@click.option("--judge-model", type=str, default=None, help="Primary judge model override.")
@click.option("--judge-host", type=str, default=None, help="Ollama base URL for the primary judge.")
@click.option("--extra-judges", type=str, default=None,
              help="Comma-separated extra judges for consensus scoring. "
                   "Format: provider:model or provider:model@host. "
                   "Example: ollama:deepseek-r1:14b,claude:claude-sonnet-4-20250514")
@click.option("--runs", type=int, default=1, show_default=True,
              help="Run each fixture N times and report mean ± stddev (use 3+ for variance analysis).")
@click.option("--category", type=str, default=None,
              help="Comma-separated categories to run (default: all). "
                   "Options: prompt_injection,role_confusion,context_escape,"
                   "instruction_override,social_engineering,data_exfil_attempt")
@click.option("--thinking/--no-thinking", "thinking_flag", default=None,
              help="Force thinking mode on or off (default: auto-detect).")
@click.option("--thinking-budget", type=int, default=8000, show_default=True)
@click.option("--save/--no-save", "save_results", default=True, show_default=True)
def adversarial(
    provider_name: str,
    model: str | None,
    ollama_host: str | None,
    vllm_host: str | None,
    judge_provider_name: str,
    judge_model: str | None,
    judge_host: str | None,
    extra_judges: str | None,
    runs: int,
    category: str | None,
    thinking_flag: bool | None,
    thinking_budget: int,
    save_results: bool,
) -> None:
    """Run adversarial LLM resilience eval — measures resistance to manipulation.

    Use --runs 3 for variance-aware scoring. Use --extra-judges for consensus.

    \b
    Examples:
      atomics adversarial --provider ollama -m qwen3:14b --runs 3
      atomics adversarial --judge-model deepseek-r1:14b --extra-judges "claude:claude-sonnet-4-20250514"
      atomics adversarial --runs 3 --extra-judges "ollama:deepseek-r1:14b@http://ollama-host:11434"
    """
    from atomics.eval.adversarial.runner import run_adversarial
    from atomics.eval.adversarial.fixtures import ADVERSARIAL_FIXTURES

    console = Console()
    settings = load_settings()
    provider = _make_provider(provider_name, model, ollama_host, settings, vllm_host=vllm_host)
    judge = _make_provider(judge_provider_name, judge_model, judge_host or ollama_host, settings, vllm_host=vllm_host)
    categories = [c.strip() for c in category.split(",")] if category else None

    # Parse --extra-judges "ollama:deepseek-r1:14b@http://host:port,claude:model"
    extra_judge_pairs: list[tuple] = []
    if extra_judges:
        for spec in extra_judges.split(","):
            spec = spec.strip()
            host_override = None
            if "@" in spec:
                spec, host_override = spec.rsplit("@", 1)
            parts = spec.split(":", 1)
            ej_provider_name = parts[0]
            ej_model = parts[1] if len(parts) > 1 else None
            ej_provider = _make_provider(ej_provider_name, ej_model, host_override or judge_host or ollama_host, settings, vllm_host=vllm_host)
            extra_judge_pairs.append((ej_provider, ej_model))

    judge_label = judge_model or "default"
    if extra_judge_pairs:
        judge_label += f" + {len(extra_judge_pairs)} extra"

    console.print(
        f"\n[bold]Adversarial eval[/bold] — model under test: [cyan]{provider_name}[/cyan] "
        f"({model or 'default'})\n"
        f"Judge: [cyan]{judge_provider_name}[/cyan] ({judge_label}) | "
        f"Runs per fixture: [bold]{runs}[/bold]\n"
        f"Fixtures: [bold]{len(ADVERSARIAL_FIXTURES)}[/bold] | "
        f"Categories: [bold]{category or 'all'}[/bold]\n"
    )

    repo = None
    if save_results:
        from atomics.storage.repository import MetricsRepository
        repo = MetricsRepository(settings.db_path)

    run_id = __import__("uuid").uuid4().hex[:12]

    def on_done(fr):
        res = fr.resistance
        if res:
            color = "green" if res.label == "resisted" else ("yellow" if res.label == "partial" else "red")
            icon = "✓" if res.label == "resisted" else ("~" if res.label == "partial" else "✗")
            run_tag = f" ×{runs}" if runs > 1 else ""
            judge_tag = f" [{len(fr.run_scores)} scores]" if fr.run_scores else ""
            console.print(
                f" [{icon}] [bold]{fr.fixture.id}[/bold]{run_tag} "
                f"[{color}]{res.label}[/] ({res.score:.2f}){judge_tag} — {res.rationale[:70]}"
            )
        if repo and res:
            repo.save_adversarial_result(run_id, fr, thinking_enabled=thinking_flag is True)

    summary = asyncio.run(run_adversarial(
        provider,
        judge_provider=judge,
        model=model,
        judge_model=judge_model,
        extra_judges=extra_judge_pairs,
        categories=categories,
        runs=runs,
        run_id=run_id,
        thinking=thinking_flag,
        thinking_budget=thinking_budget,
        on_fixture_done=on_done,
    ))

    title = f"Adversarial Resilience Summary (runs={summary.runs}, judges={len(summary.judges)})"
    table = Table(title=title)
    table.add_column("Metric")
    table.add_column("Value")
    table.add_row("Provider", provider_name)
    table.add_row("Model", model or "default")
    resilience_str = f"{summary.overall_resilience * 100:.1f}%"
    if summary.resilience_stddev is not None:
        resilience_str += f"  ±{summary.resilience_stddev * 100:.1f}%"
    table.add_row("Overall Resilience", resilience_str)
    table.add_row("Runs per fixture", str(summary.runs))
    table.add_row("Judges", ", ".join(summary.judges))
    table.add_row("Fixtures Run", str(summary.total_fixtures))
    table.add_row("Critical Failures", str(len(summary.critical_failures)))
    for cat, score in sorted(summary.category_scores.items()):
        table.add_row(f"  {cat}", f"{score * 100:.1f}%")
    console.print(table)

    if summary.critical_failures:
        console.print(
            f"\n[bold red]⚠ {len(summary.critical_failures)} CRITICAL/HIGH fixture(s) where model complied:[/bold red]"
        )
        for fr in summary.critical_failures:
            console.print(f"  • {fr.fixture.id} [{fr.fixture.severity}] {fr.fixture.category}")


# ── atomics redblue ───────────────────────────────────────────────────────────

@cli.command("redblue")
@click.option("--provider", "-p", "provider_name", type=PROVIDER_CHOICES, default="ollama", show_default=True)
@click.option("--model", "-m", type=str, default=None)
@click.option("--ollama-host", type=str, default=None)
@click.option("--vllm-host", "vllm_host", type=str, default=None, help="vLLM/OpenAI-compatible base URL.")
@click.option("--judge-provider", "judge_provider_name", type=PROVIDER_CHOICES, default="ollama", show_default=True)
@click.option("--judge-model", type=str, default=None)
@click.option("--judge-host", type=str, default=None)
@click.option("--mode", type=click.Choice(["red", "blue", "all"]), default="all", show_default=True,
              help="Which fixture set to run.")
@click.option("--thinking/--no-thinking", "thinking_flag", default=None)
@click.option("--thinking-budget", type=int, default=8000, show_default=True)
@click.option("--save/--no-save", "save_results", default=True, show_default=True)
def redblue(
    provider_name: str,
    model: str | None,
    ollama_host: str | None,
    vllm_host: str | None,
    judge_provider_name: str,
    judge_model: str | None,
    judge_host: str | None,
    mode: str,
    thinking_flag: bool | None,
    thinking_budget: int,
    save_results: bool,
) -> None:
    """Run red/blue team LLM capability eval — offensive and defensive security tasks."""
    from atomics.eval.redblue.runner import run_redblue
    from atomics.eval.redblue.fixtures import RED_FIXTURES, BLUE_FIXTURES, ALL_FIXTURES

    console = Console()
    fixture_count = {"red": len(RED_FIXTURES), "blue": len(BLUE_FIXTURES), "all": len(ALL_FIXTURES)}[mode]
    settings = load_settings()
    provider = _make_provider(provider_name, model, ollama_host, settings, vllm_host=vllm_host)
    judge = _make_provider(judge_provider_name, judge_model, judge_host or ollama_host, settings, vllm_host=vllm_host)

    console.print(
        f"\n[bold]Red/Blue eval[/bold] — model: [cyan]{provider_name}[/cyan] ({model or 'default'})\n"
        f"Judge: [cyan]{judge_provider_name}[/cyan] | Mode: [bold]{mode}[/bold] | "
        f"Fixtures: [bold]{fixture_count}[/bold]\n"
    )

    repo = None
    if save_results:
        from atomics.storage.repository import MetricsRepository
        repo = MetricsRepository(settings.db_path)

    def on_done(fr):
        j = fr.judge
        if j:
            pct = int(j.score * 100)
            color = "green" if pct >= 80 else ("yellow" if pct >= 60 else "red")
            console.print(
                f" [{fr.fixture.team.upper()}] [bold]{fr.fixture.id}[/bold] "
                f"[{color}]{pct}%[/] ({fr.fixture.category}) — {j.rationale[:80]}"
            )
        if repo:
            repo.save_task_result(fr.task_result, suite=f"redblue-{fr.fixture.team}")

    summary = asyncio.run(run_redblue(
        provider,
        judge_provider=judge,
        mode=mode,
        model=model,
        judge_model=judge_model,
        thinking=thinking_flag,
        thinking_budget=thinking_budget,
        on_fixture_done=on_done,
    ))

    table = Table(title=f"Red/Blue Eval Summary ({mode})")
    table.add_column("Metric")
    table.add_column("Value")
    table.add_row("Provider", provider_name)
    table.add_row("Model", model or "default")
    table.add_row("Mode", mode)
    table.add_row("Overall Quality", f"{(summary.overall_quality or 0) * 100:.1f}%")
    table.add_row("Fixtures Run", str(summary.total_fixtures))
    table.add_row("Avg Latency", f"{summary.avg_latency_ms:.0f}ms")
    table.add_row("Total Cost", f"${summary.total_cost_usd:.6f}")
    for cat, score in sorted(summary.category_scores.items()):
        table.add_row(f"  {cat}", f"{score * 100:.1f}%")
    console.print(table)


# ── atomics probe ─────────────────────────────────────────────────────────────

@cli.command("probe")
@click.option("--provider", "-p", "provider_name", type=PROVIDER_CHOICES, default="ollama", show_default=True)
@click.option("--model", "-m", type=str, default=None)
@click.option("--ollama-host", type=str, default=None)
@click.option("--vllm-host", "vllm_host", type=str, default=None, help="vLLM/OpenAI-compatible base URL.")
@click.option("--judge-provider", "judge_provider_name", type=PROVIDER_CHOICES, default="ollama", show_default=True)
@click.option("--judge-model", type=str, default=None)
@click.option("--judge-host", type=str, default=None)
@click.option("--probes-file", type=click.Path(exists=True), default=None,
              help="Path to probes.yaml config file.")
@click.option("--artifact", type=click.Choice([
    "json-security-report", "inference-api", "access-log",
    "k8s-audit-log", "config-file", "api-response",
]), default=None, help="Artifact type for single-file mode.")
@click.option("--file", "artifact_file", type=click.Path(exists=True), default=None,
              help="Artifact file path for single-file mode (use with --artifact).")
@click.option("--thinking/--no-thinking", "thinking_flag", default=None)
@click.option("--thinking-budget", type=int, default=8000, show_default=True)
@click.option("--alert-on-regression/--no-alert-on-regression", default=False,
              help="Warn if any check score drops >10% from last run.")
@click.option("--save/--no-save", "save_results", default=True, show_default=True)
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
    alert_on_regression: bool,
    save_results: bool,
) -> None:
    """Run LLM-evaluated live ecosystem health probes against configured artifact targets."""
    from pathlib import Path
    from atomics.probe.config import load_probe_config, ProbeTarget
    from atomics.probe.runner import run_probe

    console = Console()
    settings = load_settings()
    provider = _make_provider(provider_name, model, ollama_host, settings, vllm_host=vllm_host)
    judge = _make_provider(judge_provider_name, judge_model, judge_host or ollama_host, settings, vllm_host=vllm_host)

    targets = []
    if probes_file:
        targets = load_probe_config(Path(probes_file))
    elif artifact and artifact_file:
        targets = [ProbeTarget(
            name=Path(artifact_file).name,
            artifact_type=artifact,
            source="file",
            path=artifact_file,
        )]
    else:
        console.print("[red]Provide --probes-file or both --artifact and --file.[/red]")
        raise SystemExit(2)

    console.print(
        f"\n[bold]Ecosystem probe[/bold] — model: [cyan]{provider_name}[/cyan] ({model or 'default'})\n"
        f"Judge: [cyan]{judge_provider_name}[/cyan] | Targets: [bold]{len(targets)}[/bold]\n"
    )

    repo = None
    if save_results:
        from atomics.storage.repository import MetricsRepository
        repo = MetricsRepository(settings.db_path)

    run_id = __import__("uuid").uuid4().hex[:12]

    def on_result(r):
        color = "green" if (r.score or 0) >= 0.8 else ("yellow" if (r.score or 0) >= 0.6 else "red")
        reg_tag = " [bold red][REGRESSION][/bold red]" if r.regressed else ""
        console.print(
            f" [bold]{r.target_name}[/bold] ({r.artifact_type}) "
            f"[{color}]{(r.score or 0) * 100:.1f}%[/]{reg_tag} — {r.judge_rationale[:80]}"
        )
        if repo:
            repo.save_probe_result(run_id, r)

    summary = asyncio.run(run_probe(
        provider,
        judge_provider=judge,
        targets=targets,
        model=model,
        judge_model=judge_model,
        thinking=thinking_flag,
        thinking_budget=thinking_budget,
        regression_threshold=0.10,
        on_result=on_result,
    ))

    table = Table(title="Probe Summary")
    table.add_column("Metric")
    table.add_column("Value")
    table.add_row("Provider", provider_name)
    table.add_row("Targets", str(len(summary.results)))
    table.add_row("Overall Score", f"{(summary.overall_score or 0) * 100:.1f}%")
    if summary.regressions:
        table.add_row("[red]Regressions[/red]", str(len(summary.regressions)))
    console.print(table)

    if alert_on_regression and summary.regressions:
        console.print(
            f"\n[bold red]⚠ {len(summary.regressions)} probe(s) regressed >10% from last run[/bold red]"
        )
        for r in summary.regressions:
            console.print(f"  • {r.target_name}: {(r.prev_score or 0)*100:.1f}% → {(r.score or 0)*100:.1f}%")


# ── atomics soak ──────────────────────────────────────────────────────────────

@cli.command("soak")
@click.option("--model", "-m", type=str, default=None, help="Model to soak test.")
@click.option(
    "--provider", "-p", "provider_name",
    type=PROVIDER_CHOICES,
    default="ollama",
    help="Provider to use (default: ollama for raw GPU soak).",
)
@click.option("--ollama-host", type=str, default=None,
              help="Ollama endpoint (default: ATOMICS_OLLAMA_HOST or http://localhost:11434)")
@click.option("--profile", "profile_path", type=click.Path(exists=True), default=None,
              help="Target profile YAML (replaces --model/--ollama-host).")
@click.option("--duration", "-d", type=str, default="30m", show_default=True,
              help="Test duration: e.g. '30m', '2h', '1h30m', or bare minutes like '90'.")
@click.option("--concurrency", "-c", type=int, default=4, show_default=True,
              help="Fixed concurrent request count.")
@click.option("--sample-interval", "-s", type=int, default=30, show_default=True,
              help="Seconds between metric snapshots.")
@click.option("--num-predict", type=int, default=2048, show_default=True,
              help="Max output tokens per request.")
@click.option("--save/--no-save", "save_results", default=True, show_default=True,
              help="Persist results to the database.")
@click.option("--verbose", "-v", is_flag=True, default=False,
              help="Show every HTTP request (httpx debug output).")
@click.option("--think-time", type=float, default=0.0, show_default=True,
              help="Seconds to wait between requests per worker (simulates user think time).")
@click.option("--save-baseline", "save_baseline_name", type=str, default=None,
              help="Save this run as a named baseline for future regression checks.")
@click.option("--compare-baseline", "compare_baseline_name", type=str, default=None,
              help="Compare this run against a previously saved baseline.")
def soak(
    model: str | None,
    provider_name: str,
    ollama_host: str | None,
    profile_path: str | None,
    duration: str,
    concurrency: int,
    sample_interval: int,
    num_predict: int,
    save_results: bool,
    verbose: bool,
    think_time: float,
    save_baseline_name: str | None,
    compare_baseline_name: str | None,
) -> None:
    """Soak test — hold fixed concurrency and track degradation over time.

    Measures throughput drift, latency drift, VRAM drift, and error rate.
    Classifies the result as STABLE, DEGRADED, or UNSTABLE.

    \b
    Examples:
      atomics soak --model qwen2.5:7b --duration 30m
      atomics soak --model qwen2.5:7b -d 2h -c 8 --ollama-host http://gpu:11434
      atomics soak --profile profiles/local/gatekeeper.yaml -d 30m
      atomics soak --provider openai --model gpt-4o-mini -d 15m -c 2
      atomics soak --model qwen2.5:3b -d 5m --verbose
    """
    from atomics.soak import parse_duration, run_soak, run_soak_profile, run_soak_provider

    settings = load_settings()
    _setup_logging(settings.log_level)
    console = Console()

    if not model and not profile_path:
        console.print("[red]Specify --model or --profile.[/red]")
        sys.exit(1)

    try:
        duration_seconds = parse_duration(duration)
    except ValueError as exc:
        console.print(f"[red]{exc}[/red]")
        sys.exit(1)

    def _dur_label(secs: float) -> str:
        secs = int(secs)
        if secs < 60:
            return f"{secs}s"
        m = secs // 60
        if m < 60:
            extra_s = secs % 60
            return f"{m}m" if extra_s == 0 else f"{m}m{extra_s}s"
        h = m // 60
        rem_m = m % 60
        return f"{h}h{rem_m:02d}m"

    dur_label = _dur_label(duration_seconds)

    if profile_path:
        from atomics.profiles import load_profile
        tp = load_profile(profile_path)
        target_label = f"profile:{tp.name} ({tp.type})"
        console.print(
            f"[bold]Soak test[/bold] — {target_label}\n"
            f"Duration: [bold]{dur_label}[/bold] | "
            f"Concurrency: [bold]{concurrency}[/bold] | "
            f"Sample interval: [bold]{sample_interval}s[/bold]\n"
        )
    else:
        use_provider_mode = provider_name != "ollama"
        host = ollama_host or settings.ollama_host
        target_label = f"{provider_name} / {model}" if use_provider_mode else f"{model} @ {host}"
        console.print(
            f"[bold]Soak test[/bold] — {target_label}\n"
            f"Duration: [bold]{dur_label}[/bold] | "
            f"Concurrency: [bold]{concurrency}[/bold] | "
            f"Sample interval: [bold]{sample_interval}s[/bold] | "
            f"Max tokens: {num_predict}\n"
        )

    import logging as _logging
    if not verbose:
        _logging.getLogger("httpx").setLevel(_logging.WARNING)
        _logging.getLogger("httpcore").setLevel(_logging.WARNING)

    if think_time > 0:
        console.print(
            f"Think time: [bold]{think_time}s[/bold] per worker — "
            f"simulates ~{concurrency} users with natural pauses\n"
        )

    tps_label = "req/s" if profile_path else "tok/s"
    sample_count = 0

    def _on_sample(s) -> None:
        nonlocal sample_count
        sample_count += 1
        elapsed_m = int(s.elapsed_seconds // 60)
        elapsed_s = int(s.elapsed_seconds % 60)
        fail_tag = f"  [red]({s.failed} err)[/red]" if s.failed else ""
        vram_tag = f"  VRAM {s.vram_used_mb:.0f}MB" if s.vram_used_mb else ""
        tokens_tag = f"  {s.total_output_tokens:,} tok" if s.total_output_tokens else ""
        console.print(
            f"  [{elapsed_m:02d}:{elapsed_s:02d}] "
            f"[cyan]{s.aggregate_tps:6.1f}[/cyan] {tps_label}  "
            f"P95 {s.p95_latency_ms / 1000:.1f}s  "
            f"({s.requests} reqs{tokens_tag})"
            f"{vram_tag}{fail_tag}"
        )

    console.print("[bold]Live samples:[/bold]")

    if profile_path:
        result = asyncio.run(run_soak_profile(
            profile=tp,
            concurrency=concurrency,
            duration_seconds=duration_seconds,
            sample_interval=sample_interval,
            think_time_seconds=think_time,
            on_sample=_on_sample,
        ))
    elif provider_name != "ollama":
        provider = _make_provider(provider_name, model, ollama_host, settings)
        result = asyncio.run(run_soak_provider(
            provider=provider,
            model=model or "",
            concurrency=concurrency,
            duration_seconds=duration_seconds,
            sample_interval=sample_interval,
            num_predict=num_predict,
            think_time_seconds=think_time,
            on_sample=_on_sample,
        ))
    else:
        host = ollama_host or settings.ollama_host
        result = asyncio.run(run_soak(
            host=host,
            model=model or settings.ollama_model,
            concurrency=concurrency,
            duration_seconds=duration_seconds,
            sample_interval=sample_interval,
            num_predict=num_predict,
            think_time_seconds=think_time,
            on_sample=_on_sample,
        ))

    console.print()

    verdict_style = {
        "STABLE": "[bold green]STABLE[/bold green]",
        "DEGRADED": "[bold yellow]DEGRADED[/bold yellow]",
        "UNSTABLE": "[bold red]UNSTABLE[/bold red]",
    }

    summary = Table(title="Soak Test Summary", show_lines=True, title_style="bold")
    summary.add_column("Metric", style="dim")
    summary.add_column("Value", style="cyan bold")

    summary.add_row("Model", result.model)
    if result.provider and result.provider != "ollama":
        summary.add_row("Provider", result.provider)
    summary.add_row("Duration", f"{_dur_label(result.actual_duration_seconds)} (target: {dur_label})")
    summary.add_row("Concurrency", str(result.concurrency))
    summary.add_row("Samples", str(len(result.samples)))
    summary.add_row("Total requests", f"{result.total_requests} ({result.total_failed} failed)")
    summary.add_row("Total tokens", f"{result.total_tokens:,}")

    summary.add_row("Avg throughput", f"{result.avg_tps:.1f} tok/s")
    summary.add_row("Peak throughput", f"{result.peak_tps:.1f} tok/s")
    summary.add_row("Min throughput", f"{result.min_tps:.1f} tok/s")
    summary.add_row("Avg P95 latency", f"{result.avg_p95_ms / 1000:.1f}s")

    drift_color = "green" if abs(result.throughput_drift_pct) < 5 else ("yellow" if abs(result.throughput_drift_pct) < 15 else "red")
    summary.add_row("Throughput drift", f"[{drift_color}]{result.throughput_drift_pct:+.1f}%[/{drift_color}]")

    lat_color = "green" if result.latency_drift_pct < 10 else ("yellow" if result.latency_drift_pct < 25 else "red")
    summary.add_row("Latency drift", f"[{lat_color}]{result.latency_drift_pct:+.1f}%[/{lat_color}]")

    err_color = "green" if result.error_rate < 0.005 else ("yellow" if result.error_rate < 0.05 else "red")
    summary.add_row("Error rate", f"[{err_color}]{result.error_rate * 100:.2f}%[/{err_color}]")

    if result.vram_drift_mb is not None:
        vram_color = "green" if abs(result.vram_drift_mb) < 100 else "yellow"
        summary.add_row("VRAM drift", f"[{vram_color}]{result.vram_drift_mb:+.0f} MB[/{vram_color}]")

    if result.total_cost_usd > 0:
        summary.add_row("Total cost", f"[yellow]${result.total_cost_usd:.4f}[/yellow]")

    summary.add_row("Verdict", verdict_style.get(result.verdict, result.verdict))

    console.print(summary)

    if save_results:
        from atomics.storage.repository import MetricsRepository
        repo = MetricsRepository(settings.db_path)
        repo.save_soak_result(result)
        repo.close()
        console.print(f"\n[dim]Results saved to database.[/dim]")

    if save_baseline_name or compare_baseline_name:
        from atomics.storage.schema import init_db
        from atomics.regression import save_baseline, load_baseline, compute_regression
        conn = init_db(settings.db_path)

    if save_baseline_name:
        from atomics.regression import save_baseline
        from atomics.storage.schema import init_db
        conn = init_db(settings.db_path)
        save_baseline(
            conn, name=save_baseline_name, suite="soak",
            model=result.model, host=result.host,
            avg_tps=result.avg_tps, peak_tps=result.peak_tps,
            avg_p95_ms=result.avg_p95_ms, error_rate=result.error_rate,
            verdict=result.verdict, concurrency=result.concurrency,
        )
        conn.close()
        console.print(f"\n[green]Baseline '[bold]{save_baseline_name}[/bold]' saved.[/green]")

    if compare_baseline_name:
        from atomics.regression import load_baseline, compute_regression
        from atomics.storage.schema import init_db
        conn = init_db(settings.db_path)
        bl = load_baseline(conn, compare_baseline_name, "soak")
        conn.close()
        if bl is None:
            console.print(f"\n[red]Baseline '[bold]{compare_baseline_name}[/bold]' not found. "
                          f"Run with --save-baseline first.[/red]")
        else:
            report = compute_regression(
                bl,
                current_avg_tps=result.avg_tps,
                current_peak_tps=result.peak_tps,
                current_avg_p95_ms=result.avg_p95_ms,
                current_error_rate=result.error_rate,
                current_verdict=result.verdict,
            )
            status_style = {
                "IMPROVED": "[bold green]IMPROVED[/bold green]",
                "STABLE": "[bold cyan]STABLE[/bold cyan]",
                "REGRESSED": "[bold red]REGRESSED[/bold red]",
            }
            rtable = Table(
                title=f"Regression vs baseline '{compare_baseline_name}'",
                show_lines=True, title_style="bold",
            )
            rtable.add_column("Metric", style="dim")
            rtable.add_column("Baseline", justify="right")
            rtable.add_column("Current", justify="right")
            rtable.add_column("Delta", justify="right")

            def _delta_style(v: float, invert: bool = False) -> str:
                bad = v > 0 if invert else v < 0
                tag = "red" if bad else ("green" if abs(v) >= 1.0 else "dim")
                sign = "+" if v >= 0 else ""
                return f"[{tag}]{sign}{v:.1f}%[/{tag}]"

            rtable.add_row("Avg tok/s",
                f"{bl.avg_tps:.1f}", f"{result.avg_tps:.1f}",
                _delta_style(report.avg_tps_delta_pct))
            rtable.add_row("Peak tok/s",
                f"{bl.peak_tps:.1f}", f"{result.peak_tps:.1f}",
                _delta_style(report.peak_tps_delta_pct))
            rtable.add_row("Avg P95 latency",
                f"{bl.avg_p95_ms/1000:.1f}s", f"{result.avg_p95_ms/1000:.1f}s",
                _delta_style(report.p95_delta_pct, invert=True))
            rtable.add_row("Error rate",
                f"{bl.error_rate*100:.2f}%", f"{result.error_rate*100:.2f}%",
                f"[{'red' if report.error_rate_delta > 0 else 'green'}]"
                f"{report.error_rate_delta:+.4f}[/{'red' if report.error_rate_delta > 0 else 'green'}]")
            rtable.add_row("Verdict", bl.verdict, result.verdict,
                "[yellow]changed[/yellow]" if report.verdict_changed else "[dim]same[/dim]")

            console.print()
            console.print(rtable)
            console.print(
                f"\nRegression status: {status_style.get(report.status, report.status)}"
            )


# ── atomics qa ────────────────────────────────────────────────────────────────

@cli.command("qa")
@click.option("--file", "-f", "qa_file", type=click.Path(exists=True), required=True,
              help="QA fixture YAML file (prompts + pass/fail patterns — no secrets).")
@click.option("--profile", "-p", "profile_path", type=click.Path(exists=True), default=None,
              help="Target profile YAML for app-level gates (gitignored, replaces --model/--ollama-host).")
@click.option("--model", "-m", type=str, default=None,
              help="Override model from fixture file (raw Ollama mode).")
@click.option("--ollama-host", type=str, default=None,
              help="Override Ollama host from fixture file (raw Ollama mode).")
@click.option("--num-predict", type=int, default=1024, show_default=True,
              help="Max output tokens per fixture prompt (raw Ollama mode only).")
@click.option("--fail-fast", is_flag=True, default=False,
              help="Stop after the first FAIL or ERROR.")
def qa(
    qa_file: str,
    profile_path: str | None,
    model: str | None,
    ollama_host: str | None,
    num_predict: int,
    fail_fast: bool,
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
      atomics qa --file qa/examples/ai-gate-regression.yaml --fail-fast
      atomics qa --file qa/examples/app-gate-guardrails.yaml \\
                 --profile profiles/local/my-policy.yaml
    """
    import asyncio as _asyncio
    import logging as _logging
    from atomics.qa_runner import load_qa_suite, run_qa_suite, QAResult

    _logging.getLogger("httpx").setLevel(_logging.WARNING)
    _logging.getLogger("httpcore").setLevel(_logging.WARNING)

    console = Console()

    file_model, file_host, fixtures = load_qa_suite(qa_file)

    # Load profile if given — it handles all transport details
    loaded_profile = None
    target_label: str
    if profile_path:
        from atomics.profiles import load_profile
        loaded_profile = load_profile(profile_path)
        target_label = f"profile:[bold cyan]{loaded_profile.name}[/bold cyan] ({loaded_profile.type})"
    else:
        effective_model = model or file_model
        effective_host = ollama_host or file_host
        if not effective_model:
            console.print("[red]No model specified. Set 'model' in the YAML or use --model.[/red]")
            raise SystemExit(1)
        target_label = f"[cyan]{effective_model}[/cyan]  Host: {effective_host}"

    console.print(
        f"[bold]QA Suite[/bold] — {len(fixtures)} fixture(s)\n"
        f"Target: {target_label}\n"
    )

    stopped_early = False
    results: list[QAResult] = []

    def _on_result(r: QAResult) -> None:
        icon = {"PASS": "[green]✓[/green]", "FAIL": "[red]✗[/red]", "ERROR": "[yellow]![/yellow]"}.get(r.status, "?")
        console.print(f"  {icon} [{r.status}] {r.fixture.id}  ({r.latency_ms/1000:.1f}s)")
        results.append(r)
        if fail_fast and r.status in ("FAIL", "ERROR"):
            raise KeyboardInterrupt("fail-fast triggered")

    try:
        suite = _asyncio.run(run_qa_suite(
            model=effective_model if not loaded_profile else "",
            host=effective_host if not loaded_profile else "",
            fixtures=fixtures,
            num_predict=num_predict,
            on_result=_on_result,
            profile=loaded_profile,
        ))
    except KeyboardInterrupt:
        stopped_early = True
        from atomics.qa_runner import QASuiteResult
        suite = QASuiteResult(model=effective_model, host=effective_host, results=results)

    console.print()
    rtable = Table(title="QA Results", show_lines=True)
    rtable.add_column("ID", style="cyan")
    rtable.add_column("Status", justify="center")
    rtable.add_column("Matched pass patterns")
    rtable.add_column("Matched fail patterns")
    rtable.add_column("Latency", justify="right")

    status_style_map = {"PASS": "[green]PASS[/green]", "FAIL": "[red]FAIL[/red]", "ERROR": "[yellow]ERROR[/yellow]"}
    for r in suite.results:
        rtable.add_row(
            r.fixture.id,
            status_style_map.get(r.status, r.status),
            ", ".join(r.matched_pass) or "-",
            ", ".join(r.matched_fail) or "-",
            f"{r.latency_ms/1000:.1f}s" if r.latency_ms else "-",
        )

    console.print(rtable)

    pass_color = "green" if suite.pass_rate == 1.0 else ("yellow" if suite.pass_rate >= 0.5 else "red")
    console.print(
        f"\n[bold]Pass rate:[/bold] [{pass_color}]{suite.passed}/{suite.total}[/{pass_color}]"
        + (" [dim](stopped early)[/dim]" if stopped_early else "")
    )


# ── atomics baselines ─────────────────────────────────────────────────────────

@cli.command("baselines")
def baselines_cmd() -> None:
    """List all saved baselines."""
    from atomics.regression import list_baselines
    from atomics.storage.schema import init_db
    settings = load_settings()
    console = Console()
    conn = init_db(settings.db_path)
    records = list_baselines(conn)
    conn.close()

    if not records:
        console.print("[dim]No baselines saved yet. Use --save-baseline on a soak run.[/dim]")
        return

    table = Table(title="Saved Baselines", show_lines=True)
    table.add_column("Name", style="bold")
    table.add_column("Suite")
    table.add_column("Model")
    table.add_column("Avg tok/s", justify="right")
    table.add_column("P95 lat", justify="right")
    table.add_column("Verdict")
    table.add_column("Saved", style="dim")

    for r in records:
        table.add_row(
            r.name, r.suite, r.model,
            f"{r.avg_tps:.1f}",
            f"{r.avg_p95_ms/1000:.1f}s",
            r.verdict,
            r.timestamp[:10],
        )
    console.print(table)


# ── atomics scenario ──────────────────────────────────────────────────────────

@cli.command("scenario")
@click.option("--file", "-f", "scenario_file", type=click.Path(exists=True), default=None,
              help="YAML scenario file defining workloads.")
@click.option("--workload", "-w", "workload_flags", type=str, multiple=True,
              help="Repeatable CLI shorthand: type:model:concurrency[:sla_ms]. "
                   "Example: gate:qwen2.5:3b:2:5000")
@click.option("--ollama-host", type=str, default=None,
              help="Ollama endpoint (default: ATOMICS_OLLAMA_HOST or http://localhost:11434)")
@click.option("--duration", "-d", type=float, default=60.0, show_default=True,
              help="Test duration in seconds for the mixed phase.")
@click.option("--ramp", "ramp_seconds", type=float, default=0.0, show_default=True,
              help="Seconds over which to gradually start workers (ramp-up period).")
@click.option("--skip-baseline", is_flag=True, default=False,
              help="Skip solo baseline phase (faster, but no interference score).")
@click.option("--save/--no-save", "save_results", default=True, show_default=True,
              help="Persist results to the database.")
def scenario(
    scenario_file: str | None,
    workload_flags: tuple[str, ...],
    ollama_host: str | None,
    duration: float,
    ramp_seconds: float,
    skip_baseline: bool,
    save_results: bool,
) -> None:
    """Run mixed-workload scenario — simulate multiple agentic services competing for one GPU.

    Measures per-workload latency, SLA compliance, and cross-workload interference.

    \b
    Examples:
      atomics scenario -w "gate:qwen2.5:3b:2:5000" -w "eval:qwen2.5:7b:1:15000" -d 60
      atomics scenario --file scenario.yaml --ollama-host http://gpu-host:11434
      atomics scenario -w "gate:qwen2.5:3b:3" -d 30
    """
    from atomics.scenario import run_scenario
    from atomics.scenario_models import load_scenario_yaml, parse_workload_flag

    settings = load_settings()
    _setup_logging(settings.log_level)
    console = Console()
    host = ollama_host or settings.ollama_host

    if scenario_file and workload_flags:
        console.print("[red]Cannot use both --file and --workload. Pick one.[/red]")
        sys.exit(1)

    if scenario_file:
        specs = load_scenario_yaml(scenario_file)
    elif workload_flags:
        specs = [parse_workload_flag(f) for f in workload_flags]
    else:
        console.print("[red]Specify --file or at least one --workload.[/red]")
        sys.exit(1)

    total_conc = sum(s.concurrency for s in specs)
    console.print(
        f"\n[bold]Scenario[/bold] — {len(specs)} workload(s), "
        f"{total_conc} total concurrent workers\n"
        f"Target: [cyan]{host}[/cyan] | Duration: [bold]{duration:.0f}s[/bold] | "
        f"Baseline: [bold]{'skip' if skip_baseline else 'auto'}[/bold]\n"
    )

    for s in specs:
        sla_tag = f" SLA {s.sla_ms:.0f}ms" if s.sla_ms else ""
        console.print(f"  • {s.name} [{s.type}] {s.model} ×{s.concurrency}{sla_tag}")
    console.print()

    def on_baseline(name: str, p50: float) -> None:
        console.print(f"  [dim]baseline[/dim] {name}: P50 = {p50 / 1000:.2f}s")

    def on_workload(wr) -> None:
        pass

    if not skip_baseline:
        console.print("[bold]Solo baselines:[/bold]")

    result = asyncio.run(run_scenario(
        host=host,
        specs=specs,
        duration_seconds=duration,
        ramp_seconds=ramp_seconds,
        skip_baseline=skip_baseline,
        on_baseline_done=on_baseline,
        on_workload_done=on_workload,
    ))

    workload_table = Table(title="Scenario Results", show_lines=True)
    workload_table.add_column("Workload", style="cyan bold")
    workload_table.add_column("Type", style="dim")
    workload_table.add_column("Model")
    workload_table.add_column("Conc.", justify="right")
    workload_table.add_column("Reqs", justify="right")
    workload_table.add_column("Failed", justify="right")
    workload_table.add_column("P50", justify="right")
    workload_table.add_column("P95", justify="right")
    workload_table.add_column("tok/s", justify="right", style="blue")
    workload_table.add_column("SLA", justify="right")
    workload_table.add_column("Compliance", justify="right")

    for wr in result.workloads:
        sla_str = f"{wr.spec.sla_ms:.0f}ms" if wr.spec.sla_ms else "—"
        comp_pct = wr.sla_compliance_pct
        if wr.spec.sla_ms is not None:
            comp_color = "green" if comp_pct >= 95 else ("yellow" if comp_pct >= 80 else "red")
            comp_str = f"[{comp_color}]{comp_pct:.1f}%[/{comp_color}]"
        else:
            comp_str = "—"
        fail_str = str(wr.failed) if wr.failed == 0 else f"[red]{wr.failed}[/red]"

        workload_table.add_row(
            wr.spec.name,
            wr.spec.type,
            wr.spec.model,
            str(wr.spec.concurrency),
            str(wr.requests),
            fail_str,
            f"{wr.p50_ms / 1000:.1f}s",
            f"{wr.p95_ms / 1000:.1f}s",
            f"{wr.avg_tps:.1f}",
            sla_str,
            comp_str,
        )

    console.print(workload_table)

    if result.interference:
        intf_table = Table(title="Interference Analysis", show_lines=True)
        intf_table.add_column("Workload", style="cyan bold")
        intf_table.add_column("Solo P50", justify="right")
        intf_table.add_column("Mixed P50", justify="right")
        intf_table.add_column("Factor", justify="right")

        for wr in result.workloads:
            name = wr.spec.name
            if name in result.interference:
                solo = result.baselines[name]
                mixed = wr.p50_ms
                factor = result.interference[name]
                factor_color = "green" if factor < 1.5 else ("yellow" if factor < 2.5 else "red")
                intf_table.add_row(
                    name,
                    f"{solo / 1000:.2f}s",
                    f"{mixed / 1000:.2f}s",
                    f"[{factor_color}]{factor:.2f}x[/{factor_color}]",
                )

        console.print(intf_table)

    summary = Table(title="Summary", show_lines=True)
    summary.add_column("Metric", style="dim")
    summary.add_column("Value", style="bold")
    summary.add_row("Duration", f"{result.duration_seconds:.0f}s")
    summary.add_row("Total requests", f"{result.total_requests} ({result.total_failed} failed)")
    summary.add_row("Workloads", str(len(result.workloads)))

    sla_workloads = [wr for wr in result.workloads if wr.spec.sla_ms is not None]
    if sla_workloads:
        all_compliant = all(wr.sla_compliance_pct >= 95 for wr in sla_workloads)
        verdict = "[green]ALL PASS[/green]" if all_compliant else "[red]SLA BREACH[/red]"
        summary.add_row("SLA Verdict", verdict)

    if result.interference:
        max_intf = max(result.interference.values())
        intf_color = "green" if max_intf < 1.5 else ("yellow" if max_intf < 2.5 else "red")
        summary.add_row("Max Interference", f"[{intf_color}]{max_intf:.2f}x[/{intf_color}]")

    console.print(summary)

    if save_results:
        from atomics.storage.repository import MetricsRepository
        repo = MetricsRepository(settings.db_path)
        repo.save_scenario_result(result)
        repo.close()
        console.print(f"\n[dim]Results saved to database.[/dim]")
