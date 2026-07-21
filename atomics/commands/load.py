"""Load-testing CLI commands."""

from __future__ import annotations

import asyncio
import sys

import click
from rich.console import Console
from rich.table import Table

from atomics.commands.common import PROVIDER_CHOICES, _make_provider, setup_logging
from atomics.config import load_settings


@click.command()
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
    setup_logging(settings.log_level)
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
        contention = asyncio.run(run_contention(
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
        for mr in contention.contention_results:
            solo = contention.solo_tps.get(mr.model, 0.0)
            factor = contention.contention_factor(mr.model)
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
        console.print(f"\n[dim]Total duration: {contention.duration_seconds:.1f}s[/dim]")
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
        console.print("\n[dim]Results saved to database.[/dim]")

@click.command()
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
    from atomics.capacity import LoadProfile, project_capacity

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

@click.command("soak")
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
    setup_logging(settings.log_level)
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
        console.print("\n[dim]Results saved to database.[/dim]")

    if save_baseline_name or compare_baseline_name:
        from atomics.regression import compute_regression, load_baseline, save_baseline
        from atomics.storage.schema import init_db
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
        from atomics.regression import compute_regression, load_baseline
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

@click.command("baselines")
def baselines() -> None:
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

@click.command("scenario")
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
    setup_logging(settings.log_level)
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
        console.print("\n[dim]Results saved to database.[/dim]")

