"""Mixed-workload scenario CLI command."""

from __future__ import annotations

import asyncio
import sys

import click
from rich.console import Console
from rich.table import Table

from atomics.commands.common import setup_logging
from atomics.config import load_settings


@click.command("scenario")
@click.option(
    "--file",
    "-f",
    "scenario_file",
    type=click.Path(exists=True),
    default=None,
    help="YAML scenario file defining workloads.",
)
@click.option(
    "--workload",
    "-w",
    "workload_flags",
    type=str,
    multiple=True,
    help="Repeatable CLI shorthand: type:model:concurrency[:sla_ms]. "
    "Example: gate:YOUR_MODEL:2:5000",
)
@click.option(
    "--ollama-host",
    type=str,
    default=None,
    help="Ollama endpoint (default: ATOMICS_OLLAMA_HOST or http://localhost:11434)",
)
@click.option(
    "--duration",
    "-d",
    type=float,
    default=60.0,
    show_default=True,
    help="Test duration in seconds for the mixed phase.",
)
@click.option(
    "--ramp",
    "ramp_seconds",
    type=float,
    default=0.0,
    show_default=True,
    help="Seconds over which to gradually start workers (ramp-up period).",
)
@click.option(
    "--skip-baseline",
    is_flag=True,
    default=False,
    help="Skip solo baseline phase (faster, but no interference score).",
)
@click.option(
    "--save/--no-save",
    "save_results",
    default=True,
    show_default=True,
    help="Persist results to the database.",
)
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
      atomics scenario --file profiles/examples/scenario-gate-and-eval.yaml -d 60
      atomics scenario -w "gate:YOUR_MODEL:2:5000" -w "eval:YOUR_MODEL:1:15000" -d 60
      atomics scenario -w "gate:YOUR_MODEL:3" -d 30
    """
    from atomics.load.scenario import run_scenario
    from atomics.load.scenario_models import load_scenario_yaml, parse_workload_flag

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

    result = asyncio.run(
        run_scenario(
            host=host,
            specs=specs,
            duration_seconds=duration,
            ramp_seconds=ramp_seconds,
            skip_baseline=skip_baseline,
            on_baseline_done=on_baseline,
            on_workload_done=on_workload,
        )
    )

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
