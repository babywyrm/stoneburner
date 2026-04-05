"""Trend reporting — renders usage data from the metrics store."""

from __future__ import annotations

from rich.console import Console
from rich.table import Table

from atomics.storage.repository import MetricsRepository


def print_recent_runs(repo: MetricsRepository, limit: int = 10) -> None:
    console = Console()
    runs = repo.get_recent_runs(limit)
    if not runs:
        console.print("[dim]No runs recorded yet.[/dim]")
        return

    table = Table(title="Recent Runs", show_lines=True)
    table.add_column("Run ID", style="cyan")
    table.add_column("Started", style="green")
    table.add_column("Tasks", justify="right")
    table.add_column("OK", justify="right", style="green")
    table.add_column("Fail", justify="right", style="red")
    table.add_column("Tokens", justify="right")
    table.add_column("Cost", justify="right", style="yellow")
    table.add_column("Avg Latency", justify="right")

    for r in runs:
        table.add_row(
            r["run_id"],
            r["started_at"][:19] if r["started_at"] else "—",
            str(r["total_tasks"]),
            str(r["successful_tasks"]),
            str(r["failed_tasks"]),
            f"{r['total_tokens']:,}",
            f"${r['total_cost_usd']:.4f}",
            f"{r['avg_latency_ms']:.0f}ms",
        )
    console.print(table)


def print_hourly_usage(repo: MetricsRepository, hours: int = 24) -> None:
    console = Console()
    rows = repo.get_token_usage_by_hour(hours)
    if not rows:
        console.print("[dim]No hourly data yet.[/dim]")
        return

    table = Table(title=f"Token Usage by Hour (last {hours}h)", show_lines=True)
    table.add_column("Hour", style="cyan")
    table.add_column("Tasks", justify="right")
    table.add_column("Input Tokens", justify="right")
    table.add_column("Output Tokens", justify="right")
    table.add_column("Total Tokens", justify="right", style="bold")
    table.add_column("Cost", justify="right", style="yellow")

    for r in rows:
        table.add_row(
            r["hour"],
            str(r["task_count"]),
            f"{r['input_tokens']:,}",
            f"{r['output_tokens']:,}",
            f"{r['total_tokens']:,}",
            f"${r['cost']:.4f}",
        )
    console.print(table)


def print_category_breakdown(repo: MetricsRepository) -> None:
    console = Console()
    rows = repo.get_usage_by_category()
    if not rows:
        console.print("[dim]No category data yet.[/dim]")
        return

    table = Table(title="Usage by Task Category", show_lines=True)
    table.add_column("Category", style="cyan")
    table.add_column("Tasks", justify="right")
    table.add_column("Total Tokens", justify="right", style="bold")
    table.add_column("Total Cost", justify="right", style="yellow")
    table.add_column("Avg Latency", justify="right")

    for r in rows:
        table.add_row(
            r["category"],
            str(r["task_count"]),
            f"{r['total_tokens']:,}",
            f"${r['total_cost']:.4f}",
            f"{r['avg_latency']:.0f}ms",
        )
    console.print(table)
