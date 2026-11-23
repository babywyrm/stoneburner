"""Distributed worker CLI command."""

from __future__ import annotations

import asyncio

import click

from atomics.commands.common import setup_logging
from atomics.distributed.worker_client import WorkerClient


@click.command()
@click.option(
    "--coordinator", default="http://127.0.0.1:8000", show_default=True, help="Coordinator URL"
)
@click.option("--api-key", envvar="ATOMICS_WORKER_API_KEY", help="Worker API key")
@click.option("--label", "labels", multiple=True, help="Worker label as key=value")
@click.option("--endpoint", help="Push endpoint URL for this worker")
@click.option(
    "--heartbeat-interval", default=30, show_default=True, help="Heartbeat interval in seconds"
)
@click.option(
    "--provider",
    "-p",
    default="ollama",
    show_default=True,
    help="Provider used by this worker (ollama, brain-gateway, vllm, ...)",
)
@click.option("--model", "-m", help="Model override for this worker")
@click.option("--host", "-h", help="Provider host/URL override (e.g. http://gpu-host:30080)")
@click.option(
    "--lock-host",
    is_flag=True,
    default=False,
    help="Reject any caller-supplied host/judge_host overrides in dispatched assignments.",
)
@click.option(
    "--allowed-host",
    "--allowed-hosts",
    "allowed_hosts",
    multiple=True,
    help=(
        "Allowlisted outbound inference/judge origin or URL prefix (repeatable or comma-separated)."
    ),
)
@click.option(
    "--allow-codegen",
    is_flag=True,
    default=False,
    help="Allow distributed execution of the codegen suite (executes model-generated Python).",
)
def worker(
    coordinator: str,
    api_key: str,
    labels: tuple[str, ...],
    endpoint: str | None,
    heartbeat_interval: int,
    provider: str,
    model: str | None,
    host: str | None,
    lock_host: bool,
    allowed_hosts: tuple[str, ...],
    allow_codegen: bool,
) -> None:
    """Start a distributed worker process."""
    if not api_key:
        raise click.UsageError("--api-key is required (or set ATOMICS_WORKER_API_KEY)")
    label_dict = {}
    for label in labels:
        if "=" not in label:
            raise click.BadParameter(f"Label must be key=value: {label}")
        k, v = label.split("=", 1)
        label_dict[k] = v
    allowed_set: set[str] | None = (
        {part.strip() for raw in allowed_hosts for part in raw.split(",") if part.strip()}
        if allowed_hosts
        else None
    )
    setup_logging("INFO")
    try:
        client = WorkerClient(
            coordinator,
            api_key,
            labels=label_dict,
            endpoint=endpoint,
            heartbeat_interval=heartbeat_interval,
            provider_name=provider,
            model=model,
            host=host,
            allow_codegen=allow_codegen,
            lock_host=lock_host,
            allowed_hosts=allowed_set,
        )
    except ValueError as exc:
        raise click.BadParameter(str(exc)) from exc
    try:
        asyncio.run(client.run())
    except KeyboardInterrupt:
        client.shutdown()
