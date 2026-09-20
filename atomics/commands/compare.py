"""Provider and model comparison CLI command."""

from __future__ import annotations

import click
from rich.console import Console
from rich.table import Table

from atomics.commands.benchmark import TIER_CHOICES
from atomics.config import load_settings


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
@click.option(
    "--narrative", is_flag=True, default=False, help="Print a plain-English business-case summary"
)
@click.option(
    "--output",
    "-o",
    "out_file",
    type=click.Path(),
    default=None,
    help="Write JSON summary to FILE instead of (or alongside) table output",
)
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
    from atomics.benchmark.model_classes import classify_model
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
            console.print(
                "[dim]No data to compare. Run benchmarks with multiple providers first.[/dim]"
            )
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
            "\n[dim]No accuracy scores yet. "
            "Run [bold]atomics eval[/bold] to generate quality scores.[/dim]"
        )
        return

    scored_sorted = sorted(scored, key=lambda r: r.get("avg_accuracy_score", 0), reverse=True)
    best = scored_sorted[0]
    free_options = [r for r in scored if r.get("cost_per_1k_tokens", 1) < 0.0001]
    paid_options = [r for r in scored if r.get("cost_per_1k_tokens", 0) >= 0.0001]

    console.print(
        "\n[bold cyan]── Business Case Summary ──────────────────────────────[/bold cyan]"
    )

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
            f"[cyan]{best_free['group_key']}[/cyan] achieves "
            f"[green]{free_acc:.1f}%[/green] quality "
            f"at [green]$0 marginal cost[/green], "
            f"versus [magenta]{best_paid['group_key']}[/magenta] at "
            f"[green]{paid_acc:.1f}%[/green] for "
            f"[yellow]${paid_cost:.4f}[/yellow] total spend."
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
