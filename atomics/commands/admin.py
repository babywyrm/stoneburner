"""Admin and utility CLI commands."""

from __future__ import annotations

import asyncio
import sys

import click
from rich.console import Console
from rich.markup import escape as _rich_escape
from rich.table import Table

from atomics.commands.common import PROVIDER_CHOICES, setup_logging
from atomics.config import load_settings
from atomics.models import BurnTier
from atomics.providers.base import BaseProvider

TIER_CHOICES = click.Choice([t.value for t in BurnTier], case_sensitive=False)

@click.command()
def doctor() -> None:
    """Check Python, database, API keys, optional deps, and scheduler tooling."""
    from atomics.doctor import run_doctor

    sys.exit(run_doctor())

@click.command()
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

@click.command("schedule-status")
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

@click.command("export")
@click.option(
    "--suite",
    type=click.Choice(["tasks", "eval", "redblue", "stress", "sweep", "soak", "adversarial", "all"]),
    default="tasks",
    help="Which suite to export: tasks (all task_results), eval, redblue, stress, "
         "sweep, soak, adversarial, or all",
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
def export(
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

    settings = load_settings()
    from atomics.exporters import write_tasks_export
    from atomics.storage.repository import MetricsRepository

    repo = MetricsRepository(settings.db_path)
    try:
        if suite == "tasks":
            rows = repo.query_task_results(since_hours=since_hours, limit=limit)
            write_tasks_export(rows, fmt, out_file)
        elif suite == "eval":
            rows = repo.query_task_results(since_hours=since_hours, limit=limit, suite="eval")
            write_tasks_export(rows, fmt, out_file)
        elif suite == "redblue":
            rows = repo.query_task_results(since_hours=since_hours, limit=limit, suite_prefix="redblue-")
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
        elif suite == "adversarial":
            rows = repo.get_adversarial_results(limit=limit)
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
            for r in repo.get_adversarial_results():
                r["_suite"] = "adversarial"
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

@click.command("models")
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

    provider: BaseProvider
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

@click.command("provider-test")
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
    setup_logging(settings.log_level)
    console = Console()

    prov: BaseProvider
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
    elif provider_name == "llamacpp":
        from atomics.providers.llamacpp import LlamaCppProvider

        llamacpp_model = model or "local"
        prov = LlamaCppProvider(
            base_url=ollama_host or settings.llamacpp_host,
            default_model=llamacpp_model,
        )
        model_label = llamacpp_model
    elif provider_name == "groq":
        if not settings.groq_api_key:
            console.print("[red]GROQ_API_KEY not set.[/red]")
            sys.exit(1)
        from atomics.providers.groq import GroqProvider

        groq_model = model or "llama-3.3-70b-versatile"
        prov = GroqProvider(api_key=settings.groq_api_key, default_model=groq_model)
        model_label = groq_model
    elif provider_name == "together":
        if not settings.together_api_key:
            console.print("[red]TOGETHER_API_KEY not set.[/red]")
            sys.exit(1)
        from atomics.providers.together import TogetherProvider

        together_model = model or "meta-llama/Llama-3.3-70B-Instruct-Turbo"
        prov = TogetherProvider(api_key=settings.together_api_key, default_model=together_model)
        model_label = together_model
    elif provider_name == "gemini":
        if not settings.gemini_api_key:
            console.print("[red]GEMINI_API_KEY not set.[/red]")
            sys.exit(1)
        from atomics.providers.gemini import GeminiProvider

        gemini_model = model or "gemini-2.5-flash"
        prov = GeminiProvider(api_key=settings.gemini_api_key, default_model=gemini_model)
        model_label = gemini_model
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
        console.print(f"Response: {_rich_escape(resp.text.strip())}")
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

@click.command()
@click.argument("shell", type=click.Choice(["bash", "zsh", "fish"]))
def completion(shell: str) -> None:
    """Print shell tab-completion script. Example: eval \"$(atomics completion zsh)\"."""
    from click.shell_completion import get_completion_class

    cls = get_completion_class(shell)
    if cls is None:
        console = Console()
        console.print(f"[red]No completion support for shell: {shell}[/red]")
        sys.exit(1)
    root = click.get_current_context().find_root().command
    comp = cls(root, {}, "atomics", "_ATOMICS_COMPLETE")
    click.echo(comp.source())
