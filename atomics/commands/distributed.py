"""Distributed run CLI commands."""

from __future__ import annotations

import json
from pathlib import Path

import click
import httpx
from rich.console import Console
from rich.table import Table

from atomics.commands.common import PROVIDER_CHOICES


@click.group()
def distributed() -> None:
    """Distributed benchmark runs across multiple workers."""


@distributed.command()
@click.option("--coordinator", default="http://127.0.0.1:8000", show_default=True)
@click.option("--api-key", envvar="ATOMICS_API_KEY", help="Client API key")
@click.option(
    "--mode",
    default="split",
    show_default=True,
    type=click.Choice(["split", "fleet"]),
    help="split: divide the tasks across workers. fleet: run every task on every "
    "matching worker, for comparing hosts.",
)
@click.option(
    "--provider",
    "-p",
    type=PROVIDER_CHOICES,
    help="Pin every task to this provider. Default: each worker's own provider.",
)
@click.option("--tier", "-t", default="baseline", show_default=True)
@click.option("--model", "-m", help="Model override for the executing provider")
@click.option("-n", "iterations", default=1, show_default=True, help="Number of tasks")
@click.option(
    "--label",
    "labels",
    multiple=True,
    help="Worker selector key=value, repeatable. Fleet mode only; a worker must "
    "match every pair. Omit to broadcast to all online workers.",
)
def run(
    coordinator: str,
    api_key: str,
    mode: str,
    provider: str | None,
    tier: str,
    model: str | None,
    iterations: int,
    labels: tuple[str, ...],
) -> None:
    """Submit a distributed run to the coordinator."""
    if not api_key:
        raise click.UsageError("--api-key is required (or set ATOMICS_API_KEY)")
    if labels and mode == "split":
        raise click.UsageError(
            "--label only applies to fleet mode: split assigns each task to the "
            "next available worker, so a selector would be silently ignored. "
            "Use --mode fleet to target workers by label."
        )
    selector: dict[str, str] = {}
    for label in labels:
        if "=" not in label:
            raise click.BadParameter(f"Label must be key=value: {label!r}")
        key, value = label.split("=", 1)
        selector[key] = value
    run_request: dict[str, object] = {"tier": tier, "iterations": iterations}
    if provider:
        run_request["provider"] = provider
    if model:
        run_request["model"] = model
    payload: dict[str, object] = {"mode": mode, "run_request": run_request}
    # Omitted rather than sent empty: an absent selector means every online worker.
    if selector:
        payload["worker_selector"] = selector
    headers = {"X-API-Key": api_key}
    resp = httpx.post(f"{coordinator}/api/v1/distributed/runs", json=payload, headers=headers)
    resp.raise_for_status()
    data = resp.json()
    Console().print(f"Submitted distributed run: {data['job_id']}")


def _render_fleet_table(job: dict, summary: dict) -> None:
    """Print one row per host so the comparison is readable at a glance.

    Nine columns do not fit an 80-column terminal: Rich shrinks them until a
    worker id reads `5b5fc…` and a label reads `box=a…`, which cannot answer
    which host was faster — the only question fleet mode exists to answer. So
    identifiers are allowed to wrap instead of truncate, and the model moves to
    the caption when every host ran the same one, which is the usual case for a
    comparison and buys a whole column back.
    """
    workers = summary.get("workers", [])
    # A host that completed nothing has no model to report, and must not be the
    # reason the column reappears: the run where one host died is exactly when
    # the remaining numbers most need the room.
    reported_models = sorted({str(w.get("model")) for w in workers if w.get("model")})
    show_model_column = len(reported_models) > 1
    uniform_model = reported_models[0] if len(reported_models) == 1 else None

    table = Table(
        title=f"Fleet run {job.get('job_id', '')} — {job.get('status', '')}",
        caption=f"model: {uniform_model}" if uniform_model else None,
    )
    # A 12-hex worker id has to arrive in one piece; wrapped after 11 characters
    # it is no more usable than the truncation this replaced.
    table.add_column("Worker", overflow="fold", min_width=12)
    table.add_column("Labels", overflow="fold")
    if show_model_column:
        table.add_column("Model", overflow="fold")
    table.add_column("Done", justify="right")
    table.add_column("Fail", justify="right")
    table.add_column("Mean ms", justify="right")
    table.add_column("p95 ms", justify="right")
    table.add_column("Tok/s", justify="right")
    table.add_column("Cost", justify="right")

    for worker in workers:
        labels = " ".join(f"{k}={v}" for k, v in (worker.get("labels") or {}).items())
        failed = worker.get("failed", 0)
        row = [
            str(worker.get("worker_id", "")),
            labels or "-",
        ]
        if show_model_column:
            row.append(str(worker.get("model") or "-"))
        row += [
            str(worker.get("completed", 0)),
            f"[red]{failed}[/red]" if failed else "0",
            f"{worker.get('mean_latency_ms', 0.0):.1f}",
            f"{worker.get('p95_latency_ms', 0.0):.1f}",
            # No decimals: the fractional part of a tokens-per-second figure is
            # noise, and the two characters it costs pushed the table past 80
            # columns, which clipped the cost column off the right edge.
            f"{worker.get('mean_tokens_per_second', 0.0):.0f}",
            f"{worker.get('estimated_cost_usd', 0.0):.4f}",
        ]
        table.add_row(*row)

    console = Console()
    console.print(table)
    console.print(
        f"Total: {summary.get('completed', 0)} completed, "
        f"{summary.get('failed', 0)} failed, "
        f"{summary.get('total_output_tokens', 0)} output tokens, "
        f"${summary.get('estimated_cost_usd', 0.0)}"
    )


@distributed.command()
@click.option("--coordinator", default="http://127.0.0.1:8000", show_default=True)
@click.option("--api-key", envvar="ATOMICS_API_KEY", help="Client API key")
@click.option(
    "--json-out",
    type=click.Path(dir_okay=False, writable=True, path_type=Path),
    help="Write the job and its per-worker rollup to this file.",
)
@click.argument("job_id")
def status(
    coordinator: str, api_key: str, job_id: str, json_out: Path | None
) -> None:
    """Check status of a distributed run."""
    if not api_key:
        raise click.UsageError("--api-key is required (or set ATOMICS_API_KEY)")
    headers = {"X-API-Key": api_key}
    resp = httpx.get(f"{coordinator}/api/v1/distributed/runs/{job_id}", headers=headers)
    resp.raise_for_status()
    job = resp.json()

    summary = {}
    raw_summary = job.get("summary_json")
    if raw_summary:
        try:
            summary = json.loads(raw_summary)
        except json.JSONDecodeError:
            summary = {}

    if json_out:
        json_out.write_text(json.dumps({**job, "summary": summary}, indent=2))
        Console().print(f"Wrote {json_out}")
        return

    # A fleet run's whole purpose is the per-host comparison, which does not read
    # well as nested JSON. Split runs keep the machine-readable output they had.
    if job.get("mode") == "fleet" and summary.get("workers"):
        _render_fleet_table(job, summary)
        return
    click.echo(json.dumps(job, indent=2))
