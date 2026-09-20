"""Soak-test CLI command."""

from __future__ import annotations

import asyncio
import sys

import click
from rich.console import Console
from rich.table import Table

from atomics.commands.common import PROVIDER_CHOICES, _make_provider, run_async, setup_logging
from atomics.config import load_settings


@click.command("soak")
@click.option("--model", "-m", type=str, default=None, help="Model to soak test.")
@click.option(
    "--provider",
    "-p",
    "provider_name",
    type=PROVIDER_CHOICES,
    default="ollama",
    help="Provider to use (default: ollama for raw GPU soak).",
)
@click.option(
    "--ollama-host",
    type=str,
    default=None,
    help="Ollama endpoint (default: ATOMICS_OLLAMA_HOST or http://localhost:11434)",
)
@click.option(
    "--profile",
    "profile_path",
    type=click.Path(exists=True),
    default=None,
    help="Target profile YAML (replaces --model/--ollama-host).",
)
@click.option(
    "--duration",
    "-d",
    type=str,
    default="30m",
    show_default=True,
    help="Test duration: e.g. '30m', '2h', '1h30m', or bare minutes like '90'.",
)
@click.option(
    "--concurrency",
    "-c",
    type=int,
    default=4,
    show_default=True,
    help="Fixed concurrent request count.",
)
@click.option(
    "--sample-interval",
    "-s",
    type=int,
    default=30,
    show_default=True,
    help="Seconds between metric snapshots.",
)
@click.option(
    "--num-predict",
    type=int,
    default=2048,
    show_default=True,
    help="Max output tokens per request.",
)
@click.option(
    "--save/--no-save",
    "save_results",
    default=True,
    show_default=True,
    help="Persist results to the database.",
)
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    default=False,
    help="Show every HTTP request (httpx debug output).",
)
@click.option(
    "--think-time",
    type=float,
    default=0.0,
    show_default=True,
    help="Seconds to wait between requests per worker (simulates user think time).",
)
@click.option(
    "--save-baseline",
    "save_baseline_name",
    type=str,
    default=None,
    help="Save this run as a named baseline for future regression checks.",
)
@click.option(
    "--compare-baseline",
    "compare_baseline_name",
    type=str,
    default=None,
    help="Compare this run against a previously saved baseline.",
)
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
    from atomics.load.soak import parse_duration, run_soak, run_soak_profile, run_soak_provider

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
        from atomics.load.profiles import load_profile

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
        result = asyncio.run(
            run_soak_profile(
                profile=tp,
                concurrency=concurrency,
                duration_seconds=duration_seconds,
                sample_interval=sample_interval,
                think_time_seconds=think_time,
                on_sample=_on_sample,
            )
        )
    elif provider_name != "ollama":
        provider = _make_provider(provider_name, model, ollama_host, settings)
        result = run_async(
            run_soak_provider(
                provider=provider,
                model=model or "",
                concurrency=concurrency,
                duration_seconds=duration_seconds,
                sample_interval=sample_interval,
                num_predict=num_predict,
                think_time_seconds=think_time,
                on_sample=_on_sample,
            ),
            provider,
        )
    else:
        host = ollama_host or settings.ollama_host
        result = asyncio.run(
            run_soak(
                host=host,
                model=model or settings.ollama_model,
                concurrency=concurrency,
                duration_seconds=duration_seconds,
                sample_interval=sample_interval,
                num_predict=num_predict,
                think_time_seconds=think_time,
                on_sample=_on_sample,
            )
        )

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
    summary.add_row(
        "Duration", f"{_dur_label(result.actual_duration_seconds)} (target: {dur_label})"
    )
    summary.add_row("Concurrency", str(result.concurrency))
    summary.add_row("Samples", str(len(result.samples)))
    summary.add_row("Total requests", f"{result.total_requests} ({result.total_failed} failed)")
    summary.add_row("Total tokens", f"{result.total_tokens:,}")

    summary.add_row("Avg throughput", f"{result.avg_tps:.1f} tok/s")
    summary.add_row("Peak throughput", f"{result.peak_tps:.1f} tok/s")
    summary.add_row("Min throughput", f"{result.min_tps:.1f} tok/s")
    summary.add_row("Avg P95 latency", f"{result.avg_p95_ms / 1000:.1f}s")

    drift_color = (
        "green"
        if abs(result.throughput_drift_pct) < 5
        else ("yellow" if abs(result.throughput_drift_pct) < 15 else "red")
    )
    summary.add_row(
        "Throughput drift", f"[{drift_color}]{result.throughput_drift_pct:+.1f}%[/{drift_color}]"
    )

    lat_color = (
        "green"
        if result.latency_drift_pct < 10
        else ("yellow" if result.latency_drift_pct < 25 else "red")
    )
    summary.add_row("Latency drift", f"[{lat_color}]{result.latency_drift_pct:+.1f}%[/{lat_color}]")

    err_color = (
        "green" if result.error_rate < 0.005 else ("yellow" if result.error_rate < 0.05 else "red")
    )
    summary.add_row("Error rate", f"[{err_color}]{result.error_rate * 100:.2f}%[/{err_color}]")

    if result.vram_drift_mb is not None:
        vram_color = "green" if abs(result.vram_drift_mb) < 100 else "yellow"
        summary.add_row(
            "VRAM drift", f"[{vram_color}]{result.vram_drift_mb:+.0f} MB[/{vram_color}]"
        )

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
        from atomics.load.regression import compute_regression, load_baseline, save_baseline
        from atomics.storage.schema import init_db

        conn = init_db(settings.db_path)

    if save_baseline_name:
        from atomics.load.regression import save_baseline
        from atomics.storage.schema import init_db

        conn = init_db(settings.db_path)
        save_baseline(
            conn,
            name=save_baseline_name,
            suite="soak",
            model=result.model,
            host=result.host,
            avg_tps=result.avg_tps,
            peak_tps=result.peak_tps,
            avg_p95_ms=result.avg_p95_ms,
            error_rate=result.error_rate,
            verdict=result.verdict,
            concurrency=result.concurrency,
        )
        conn.close()
        console.print(f"\n[green]Baseline '[bold]{save_baseline_name}[/bold]' saved.[/green]")

    if compare_baseline_name:
        from atomics.load.regression import compute_regression, load_baseline
        from atomics.storage.schema import init_db

        conn = init_db(settings.db_path)
        bl = load_baseline(conn, compare_baseline_name, "soak")
        conn.close()
        if bl is None:
            console.print(
                f"\n[red]Baseline '[bold]{compare_baseline_name}[/bold]' not found. "
                f"Run with --save-baseline first.[/red]"
            )
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
                show_lines=True,
                title_style="bold",
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

            rtable.add_row(
                "Avg tok/s",
                f"{bl.avg_tps:.1f}",
                f"{result.avg_tps:.1f}",
                _delta_style(report.avg_tps_delta_pct),
            )
            rtable.add_row(
                "Peak tok/s",
                f"{bl.peak_tps:.1f}",
                f"{result.peak_tps:.1f}",
                _delta_style(report.peak_tps_delta_pct),
            )
            rtable.add_row(
                "Avg P95 latency",
                f"{bl.avg_p95_ms / 1000:.1f}s",
                f"{result.avg_p95_ms / 1000:.1f}s",
                _delta_style(report.p95_delta_pct, invert=True),
            )
            rtable.add_row(
                "Error rate",
                f"{bl.error_rate * 100:.2f}%",
                f"{result.error_rate * 100:.2f}%",
                (
                    f"[{'red' if report.error_rate_delta > 0 else 'green'}]"
                    f"{report.error_rate_delta:+.4f}"
                    f"[/{'red' if report.error_rate_delta > 0 else 'green'}]"
                ),
            )
            rtable.add_row(
                "Verdict",
                bl.verdict,
                result.verdict,
                "[yellow]changed[/yellow]" if report.verdict_changed else "[dim]same[/dim]",
            )

            console.print()
            console.print(rtable)
            console.print(f"\nRegression status: {status_style.get(report.status, report.status)}")
