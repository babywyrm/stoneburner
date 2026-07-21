"""Distributed run CLI commands."""

from __future__ import annotations

import click
import httpx
from rich.console import Console
from rich.json import JSON


@click.group()
def distributed() -> None:
    """Distributed benchmark runs across multiple workers."""


@distributed.command()
@click.option("--coordinator", default="http://127.0.0.1:8000", show_default=True)
@click.option("--api-key", envvar="ATOMICS_API_KEY", help="Client API key")
@click.option("--mode", default="split", show_default=True, type=click.Choice(["split"]))
@click.option("--provider", "-p", default="claude", show_default=True)
@click.option("--tier", "-t", default="baseline", show_default=True)
@click.option("--model", "-m", help="Model override")
@click.option("-n", "iterations", default=1, show_default=True, help="Number of tasks")
@click.option("--label", "labels", multiple=True, help="Worker selector key=value")
def run(coordinator: str, api_key: str, mode: str, provider: str, tier: str, model: str | None, iterations: int, labels: tuple[str, ...]) -> None:
    """Submit a distributed run to the coordinator."""
    if not api_key:
        raise click.UsageError("--api-key is required (or set ATOMICS_API_KEY)")
    label_dict = {}
    for label in labels:
        if "=" not in label:
            raise click.BadParameter(f"Label must be key=value: {label}")
        k, v = label.split("=", 1)
        label_dict[k] = v
    run_request = {"provider": provider, "tier": tier, "iterations": iterations}
    if model:
        run_request["model"] = model
    payload = {"mode": mode, "run_request": run_request}
    if label_dict:
        payload["worker_selector"] = label_dict
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
    Console().print(JSON(resp.text))
