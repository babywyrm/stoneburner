"""Distributed run CLI commands."""

from __future__ import annotations

import json

import click
import httpx
from rich.console import Console

from atomics.commands.common import PROVIDER_CHOICES


@click.group()
def distributed() -> None:
    """Distributed benchmark runs across multiple workers."""


@distributed.command()
@click.option("--coordinator", default="http://127.0.0.1:8000", show_default=True)
@click.option("--api-key", envvar="ATOMICS_API_KEY", help="Client API key")
@click.option("--mode", default="split", show_default=True, type=click.Choice(["split"]))
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
    help="Worker selector key=value. Not supported yet — see fleet mode (phase 2).",
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
    if labels:
        raise click.UsageError(
            "--label is not supported yet: split mode assigns each task to the "
            "next available worker, so a selector would be silently ignored. "
            "Label-based targeting arrives with fleet mode."
        )
    run_request: dict[str, object] = {"tier": tier, "iterations": iterations}
    if provider:
        run_request["provider"] = provider
    if model:
        run_request["model"] = model
    payload = {"mode": mode, "run_request": run_request}
    headers = {"X-API-Key": api_key}
    resp = httpx.post(f"{coordinator}/api/v1/distributed/runs", json=payload, headers=headers)
    resp.raise_for_status()
    data = resp.json()
    Console().print(f"Submitted distributed run: {data['job_id']}")


@distributed.command()
@click.option("--coordinator", default="http://127.0.0.1:8000", show_default=True)
@click.option("--api-key", envvar="ATOMICS_API_KEY", help="Client API key")
@click.argument("job_id")
def status(coordinator: str, api_key: str, job_id: str) -> None:
    """Check status of a distributed run."""
    if not api_key:
        raise click.UsageError("--api-key is required (or set ATOMICS_API_KEY)")
    headers = {"X-API-Key": api_key}
    resp = httpx.get(f"{coordinator}/api/v1/distributed/runs/{job_id}", headers=headers)
    resp.raise_for_status()
    click.echo(json.dumps(resp.json(), indent=2))
