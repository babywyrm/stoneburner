"""Load-testing CLI commands."""

from __future__ import annotations

import asyncio

import click
from rich.console import Console
from rich.table import Table

from atomics.commands.common import PROVIDER_CHOICES, _make_provider, run_async, setup_logging
from atomics.config import load_settings


@click.command()
@click.option(
    "--provider",
    "-p",
    "provider_name",
    type=PROVIDER_CHOICES,
    default="ollama",
    help="Provider to stress test (default: ollama for raw GPU stress)",
)
@click.option(
    "--model", "-m", type=str, default=None, help="Model to stress (default: ATOMICS_OLLAMA_MODEL)"
)
@click.option(
    "--models",
    "models_csv",
    type=str,
    default=None,
    help="Comma-separated list of models for contention testing (e.g. qwen2.5:3b,qwen2.5:7b).",
)
@click.option("--ollama-host", type=str, default=None, help="Ollama endpoint")
@click.option(
    "--profile",
    "profile_path",
    type=click.Path(exists=True),
    default=None,
    help="Target profile YAML (replaces --model/--ollama-host).",
)
@click.option(
    "--max-concurrency", "-c", type=int, default=8, help="Max parallel requests (ramps 1→2→4→...)"
)
@click.option(
    "--phase-seconds", "-s", type=float, default=15.0, help="Seconds at each concurrency level"
)
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
    setup_logging(settings.log_level)
    import logging as _logging

    _logging.getLogger("httpx").setLevel(_logging.WARNING)
    _logging.getLogger("httpcore").setLevel(_logging.WARNING)
    console = Console()

    if models_csv:
        from atomics.load.contention import run_contention

        host = ollama_host or settings.ollama_host
        model_list = [m.strip() for m in models_csv.split(",") if m.strip()]
        console.print(
            f"[bold]Contention test[/bold] — {len(model_list)} models on {host}\n"
            f"Models: {', '.join(model_list)}\n"
            f"Phase: {phase_seconds}s solo + {phase_seconds}s mixed\n"
        )
        contention = asyncio.run(
            run_contention(
                host=host,
                models=model_list,
                concurrency=1,
                phase_seconds=phase_seconds,
                num_predict=min(num_predict, 512),
            )
        )
        ctable = Table(title="Contention Results", show_lines=True)
        ctable.add_column("Model", style="cyan")
        ctable.add_column("Solo tok/s", justify="right")
        ctable.add_column("Mixed tok/s", justify="right")
        ctable.add_column("Factor", justify="right")
        ctable.add_column("Mixed P95", justify="right")
        ctable.add_column("Errors", justify="right")
        for mr in contention.contention_results:
            solo = contention.solo_tps.get(mr.model, 0.0)
            factor = contention.contention_factor(mr.model)
            factor_str = f"{factor:.2f}x" if factor is not None else "n/a"
            factor_color = (
                "green"
                if (factor or 1.0) >= 0.9
                else ("yellow" if (factor or 1.0) >= 0.7 else "red")
            )
            ctable.add_row(
                mr.model,
                f"{solo:.1f}",
                f"{mr.avg_tps:.1f}",
                f"[{factor_color}]{factor_str}[/{factor_color}]",
                f"{mr.p95_ms / 1000:.1f}s",
                str(mr.failed),
            )
        console.print(ctable)
        console.print(f"\n[dim]Total duration: {contention.duration_seconds:.1f}s[/dim]")
        return

    if profile_path:
        from atomics.load.profiles import load_profile

        tp = load_profile(profile_path)
        effective_model = tp.model
        target_label = f"profile:{tp.name} ({tp.type})"
        use_provider_mode = False
    elif provider_name != "ollama":
        use_provider_mode = True
        effective_model = model or (
            "gpt-4o" if provider_name == "openai" else settings.default_model
        )
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
        from atomics.load.stress import run_stress_profile

        result = asyncio.run(
            run_stress_profile(
                profile=tp,
                max_concurrency=max_concurrency,
                phase_seconds=phase_seconds,
                on_phase=_on_phase,
            )
        )
    elif use_provider_mode:
        provider = _make_provider(provider_name, effective_model, ollama_host, settings)
        from atomics.load.stress import run_stress_provider

        result = run_async(
            run_stress_provider(
                provider=provider,
                model=effective_model,
                max_concurrency=max_concurrency,
                phase_seconds=phase_seconds,
                num_predict=num_predict,
                on_phase=_on_phase,
            ),
            provider,
        )
    else:
        from atomics.load.stress import run_stress

        result = asyncio.run(
            run_stress(
                host=host,
                model=effective_model,
                max_concurrency=max_concurrency,
                phase_seconds=phase_seconds,
                num_predict=num_predict,
                on_phase=_on_phase,
            )
        )

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
    summary.add_row(
        "Peak throughput",
        f"{result.peak_tps:.1f} tok/s @ concurrency={result.saturation_concurrency}",
    )

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
        summary.add_row(
            "Throttling", "[yellow]Possible — throughput dropped at max concurrency[/yellow]"
        )
    else:
        summary.add_row("Throttling", "[green]None detected[/green]")

    console.print(summary)

    if save_results:
        from atomics.storage.repository import MetricsRepository

        repo = MetricsRepository(settings.db_path)
        repo.save_stress_result(result)
        repo.close()
        console.print("\n[dim]Results saved to database.[/dim]")


@click.command()
@click.option("--users", "-u", type=int, required=True, help="Number of semi-active users")
@click.option(
    "--think-time",
    "--think",
    type=float,
    default=300.0,
    show_default=True,
    help="Avg seconds between requests per user",
)
@click.option(
    "--response-tokens",
    type=int,
    default=400,
    show_default=True,
    help="Avg output tokens per response",
)
@click.option(
    "--burst",
    type=float,
    default=0.2,
    show_default=True,
    help="Burst factor — fraction of users spiking simultaneously",
)
@click.option(
    "--model", "-m", type=str, default=None, help="Pull stress data from DB for this model"
)
@click.option(
    "--peak-tps",
    type=float,
    default=None,
    help="Manual peak throughput (tok/s) — used if no DB data",
)
@click.option(
    "--single-latency",
    type=float,
    default=None,
    help="Manual single-request latency in ms — used if no DB data",
)
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
    from atomics.load.capacity import LoadProfile, project_capacity

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
            console.print(
                f"[red]No stress data for model '{model}'. "
                f"Run atomics stress first, or use --peak-tps.[/red]"
            )
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
        console.print(
            f"[dim]Using stress data for {model} (peak {effective_peak_tps:.1f} tok/s)[/dim]\n"
        )

    elif peak_tps and single_latency:
        effective_peak_tps = peak_tps
        phases = [
            {
                "concurrency": 1,
                "aggregate_tps": peak_tps,
                "avg_latency_ms": single_latency,
                "p95_latency_ms": single_latency * 1.5,
            },
        ]

    else:
        console.print(
            "[red]Specify --model (pulls from DB) or both --peak-tps and --single-latency[/red]"
        )
        raise SystemExit(1)

    profile = LoadProfile(
        users=users,
        think_time_s=think_time,
        response_tokens=response_tokens,
        burst_factor=burst,
    )

    result = project_capacity(
        profile=profile,
        phases=phases,
        peak_tps=effective_peak_tps,
        model=effective_model,
    )

    title = (
        f"Capacity Projection: {effective_model or 'custom'} ({effective_peak_tps:.0f} tok/s peak)"
    )
    table = Table(title=title, show_lines=True)
    table.add_column("Scenario", style="cyan bold")
    table.add_column("Concurrent", justify="right")
    table.add_column("P50 Latency", justify="right")
    table.add_column("P95 Latency", justify="right")
    table.add_column("Queue", justify="right", style="dim")
    table.add_column("Verdict", justify="center")

    verdict_style = {
        "OK": "[green]OK[/green]",
        "CAUTION": "[yellow]CAUTION[/yellow]",
        "SLOW": "[red]SLOW[/red]",
        "OVERLOAD": "[bold red]OVERLOAD[/bold red]",
    }

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


@click.command("baselines")
def baselines() -> None:
    """List all saved baselines."""
    from atomics.load.regression import list_baselines
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
            r.name,
            r.suite,
            r.model,
            f"{r.avg_tps:.1f}",
            f"{r.avg_p95_ms / 1000:.1f}s",
            r.verdict,
            r.timestamp[:10],
        )
    console.print(table)
