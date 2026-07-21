"""Distributed worker CLI command."""

from __future__ import annotations

import asyncio

import click

from atomics.commands.common import setup_logging
from atomics.distributed.worker_client import WorkerClient


@click.command()
@click.option("--coordinator", default="http://127.0.0.1:8000", show_default=True, help="Coordinator URL")
@click.option("--api-key", envvar="ATOMICS_WORKER_API_KEY", help="Worker API key")
@click.option("--label", "labels", multiple=True, help="Worker label as key=value")
@click.option("--endpoint", help="Push endpoint URL for this worker")
@click.option("--heartbeat-interval", default=30, show_default=True, help="Heartbeat interval in seconds")
def worker(coordinator: str, api_key: str, labels: tuple[str, ...], endpoint: str | None, heartbeat_interval: int) -> None:
    """Start a distributed worker process."""
    if not api_key:
        raise click.UsageError("--api-key is required (or set ATOMICS_WORKER_API_KEY)")
    label_dict = {}
    for label in labels:
        if "=" not in label:
            raise click.BadParameter(f"Label must be key=value: {label}")
        k, v = label.split("=", 1)
        label_dict[k] = v
    setup_logging("INFO")
    client = WorkerClient(
        coordinator,
        api_key,
        labels=label_dict,
        endpoint=endpoint,
        heartbeat_interval=heartbeat_interval,
    )
    try:
        asyncio.run(client.run())
    except KeyboardInterrupt:
        client.shutdown()
