"""Benchmark CLI commands."""

from __future__ import annotations

import asyncio
import sys
from typing import TYPE_CHECKING

import click
from rich.console import Console
from rich.markup import escape as _rich_escape
from rich.table import Table

from atomics.commands.common import PROVIDER_CHOICES, _make_provider, setup_logging
from atomics.config import load_settings
from atomics.labcompare import (
    parity_verdict,
    parse_host_specs,
    run_labcompare,
    speedup_ratio,
)
from atomics.models import BurnTier
from atomics.providers.base import BaseProvider

if TYPE_CHECKING:
    from atomics.labcompare import CellResult

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
    setup_logging(settings.log_level)
    console = Console()

    burn_tier = BurnTier(tier)

    from atomics.core.engine import LoopEngine
    from atomics.storage.repository import MetricsRepository
    from atomics.tiers import get_tier_profile

    profile = get_tier_profile(burn_tier)

    provider: BaseProvider
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
    elif provider_name == "llamacpp":
        from atomics.providers.llamacpp import LlamaCppProvider

        effective_model = model or "local"
        provider = LlamaCppProvider(
            base_url=ollama_host or settings.llamacpp_host,
            default_model=effective_model,
        )
    elif provider_name == "groq":
        if not settings.groq_api_key:
            console.print("[red]GROQ_API_KEY not set. Get one at https://console.groq.com/keys[/red]")
            sys.exit(1)
        from atomics.providers.groq import GroqProvider

        effective_model = model or "llama-3.3-70b-versatile"
        provider = GroqProvider(api_key=settings.groq_api_key, default_model=effective_model)
    elif provider_name == "together":
        if not settings.together_api_key:
            console.print("[red]TOGETHER_API_KEY not set.[/red]")
            sys.exit(1)
        from atomics.providers.together import TogetherProvider

        effective_model = model or "meta-llama/Llama-3.3-70B-Instruct-Turbo"
        provider = TogetherProvider(api_key=settings.together_api_key, default_model=effective_model)
    elif provider_name == "gemini":
        if not settings.gemini_api_key:
            console.print("[red]GEMINI_API_KEY not set. Get one at https://aistudio.google.com/apikey[/red]")
            sys.exit(1)
        from atomics.providers.gemini import GeminiProvider

        effective_model = model or "gemini-2.5-flash"
        provider = GeminiProvider(api_key=settings.gemini_api_key, default_model=effective_model)
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
            if settings.webhook_url:
                from atomics.webhooks import send_webhook
                send_webhook(
                    settings.webhook_url, summary,
                    tier=burn_tier.value, provider=provider_name,
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

@click.command()
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
            "\n[bold]Data privacy:[/bold] Every token sent to an external API "
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

@click.command("tiers")
def tiers() -> None:
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

@click.command("sweep")
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
    setup_logging(settings.log_level)
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
        model_list = [str(m["name"]) for m in available]
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
            console.print(f"  [dim]reply:[/dim]  {_rich_escape(tr.response or '')}")
        elif tr.error_message:
            console.print(f"  [red]error:[/red]  {_rich_escape(tr.error_message or '')}")
        if fr.judge and fr.judge.rationale:
            console.print(f"  [dim]judge:[/dim]  {_rich_escape(fr.judge.rationale[:200])}")

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
            f"({(best.overall_quality or 0) * 100:.0f}% quality, "
            f"{best.avg_latency_ms:.0f}ms avg latency, "
            f"${best.total_cost_usd:.2f} total cost)"
        )

    failed = [r for r in results if r.overall_quality is None]
    if failed:
        console.print(f"\n[yellow]{len(failed)} model(s) failed:[/yellow]")
        for r in failed:
            reason = f" — [dim]{r.error}[/dim]" if r.error else ""
            console.print(f"[yellow]  • {r.model}[/yellow]{reason}")

    if save_results:
        from atomics.storage.repository import MetricsRepository
        repo = MetricsRepository(settings.db_path)
        for r in results:
            repo.save_sweep_result(r)
        repo.close()
        console.print("\n[dim]Sweep results saved to database.[/dim]")

def _run_labcompare_sync(**kwargs) -> list[CellResult]:
    """Sync wrapper around the async orchestrator (patch target in tests)."""
    return asyncio.run(run_labcompare(**kwargs))

def _render_labcompare(console, cells, hosts, dims) -> None:
    """Print a per-model side-by-side block with speedup and parity verdicts."""
    by_model: dict[str, dict[str, CellResult]] = {}
    for c in cells:
        by_model.setdefault(c.model, {})[c.host_name] = c

    host_names = [h.name for h in hosts]
    for model, host_cells in by_model.items():
        console.print(f"[bold cyan]{model}[/bold cyan]")

        if "throughput" in dims:
            parts = []
            for hn in host_names:
                c = host_cells.get(hn)
                tps = c.tokens_per_second if c else None
                parts.append(f"{hn} {tps if tps is not None else 'n/a'} tok/s")
            line = "  Throughput   " + "   ".join(parts)
            if len(host_names) == 2:
                a = host_cells.get(host_names[0])
                b = host_cells.get(host_names[1])
                if a and b:
                    sr = speedup_ratio(a.tokens_per_second, b.tokens_per_second)
                    if sr:
                        line += f"   -> {host_names[0]} {sr}x faster"
            console.print(line)

            vparts = []
            for hn in host_names:
                c = host_cells.get(hn)
                v = c.vram_fit_pct if c else None
                vparts.append(f"{hn} {int(v * 100)}% GPU" if v is not None else f"{hn} n/a")
            console.print("  VRAM fit     " + "   ".join(vparts))

        if "quality" in dims:
            qparts = []
            for hn in host_names:
                c = host_cells.get(hn)
                q = c.quality_score if c else None
                qparts.append(f"{hn} {int(q * 100)}%" if q is not None else f"{hn} n/a")
            line = "  Quality      " + "   ".join(qparts)
            if len(host_names) == 2:
                a = host_cells.get(host_names[0])
                b = host_cells.get(host_names[1])
                if a and b:
                    ok, delta = parity_verdict(a.quality_score, b.quality_score)
                    if ok is not None:
                        verdict = "parity OK" if ok else "parity DIFF"
                        line += f"   -> {verdict} (delta {int((delta or 0) * 100)})"
            console.print(line)
        console.print()

@click.command("labcompare")
@click.option("--host", "hosts_raw", multiple=True, required=True,
              help="Labeled endpoint NAME=URL (repeat for each host).")
@click.option("--models", "models_csv", required=True,
              help="Comma-separated models to test on every host.")
@click.option("--quality-suite", type=click.Choice(["eval", "redblue"]),
              default="eval", show_default=True)
@click.option("--dimensions", default="throughput,quality", show_default=True,
              help="Comma-separated: throughput,quality")
@click.option("--judge-host", default=None)
@click.option("--judge-model", default=None)
@click.option("--prompts", type=int, default=3, show_default=True)
@click.option("--save/--no-save", "save_results", default=True, show_default=True)
@click.option("-o", "--json-out", "json_out", type=click.Path(), default=None)
def labcompare(hosts_raw, models_csv, quality_suite, dimensions, judge_host,
               judge_model, prompts, save_results, json_out):
    """Compare two+ inference hosts on throughput and quality parity."""
    console = Console()
    settings = load_settings()

    try:
        hosts = parse_host_specs(list(hosts_raw))
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc
    if len(hosts) < 2:
        raise click.ClickException("labcompare needs at least two --host entries.")

    models = [m.strip() for m in models_csv.split(",") if m.strip()]
    dims = [d.strip() for d in dimensions.split(",") if d.strip()]

    from atomics.eval.redblue.runner import run_redblue
    from atomics.eval.runner import run_eval

    def provider_factory(url):
        return _make_provider("ollama", None, url, settings)

    def ps_fetcher_factory(url):
        async def _ps():
            import httpx
            async with httpx.AsyncClient(timeout=10.0) as client:
                r = await client.get(f"{url}/api/ps")
                r.raise_for_status()
                return r.json()
        return _ps

    async def quality_fn(provider, jhost, jmodel, model):
        judge = _make_provider("ollama", jmodel, jhost or provider._host, settings)
        if quality_suite == "redblue":
            summary = await run_redblue(
                provider, judge_provider=judge, model=model, judge_model=jmodel,
            )
            return summary.overall_quality
        summary = await run_eval(
            provider, judge_provider=judge, model=model, judge_model=jmodel,
        )
        return summary.overall_accuracy

    console.print(
        f"\n[bold]LabCompare[/bold] - {len(hosts)} hosts x {len(models)} models "
        f"| dims: {','.join(dims)} | quality: {quality_suite}\n"
    )

    cells = _run_labcompare_sync(
        hosts=hosts, models=models, dimensions=dims, quality_suite=quality_suite,
        judge_host=judge_host, judge_model=judge_model, prompts=prompts,
        provider_factory=provider_factory, quality_fn=quality_fn,
        ps_fetcher_factory=ps_fetcher_factory,
    )

    _render_labcompare(console, cells, hosts, dims)

    if save_results:
        cmp_id = __import__("uuid").uuid4().hex[:12]
        from atomics.storage.repository import MetricsRepository
        repo = MetricsRepository(settings.db_path)
        for c in cells:
            repo.save_labcompare_result(
                comparison_run_id=cmp_id, host_name=c.host_name,
                host_url=c.host_url, model=c.model,
                tokens_per_second=c.tokens_per_second, latency_ms=c.latency_ms,
                prompt_eval_rate=c.prompt_eval_rate, vram_fit_pct=c.vram_fit_pct,
                gpu_name=c.gpu_name, quality_score=c.quality_score,
                quality_suite=quality_suite if "quality" in dims else None,
                judge_model=judge_model if "quality" in dims else None,
                dimensions=",".join(dims),
            )
        repo.close()
        console.print(f"[dim]Saved comparison {cmp_id} to DB.[/dim]")

    if json_out:
        import json
        from pathlib import Path as _Path
        _Path(json_out).write_text(json.dumps([c.__dict__ for c in cells], indent=2))
        console.print(f"[dim]Wrote {json_out}[/dim]")
