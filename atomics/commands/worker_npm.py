"""CLI command to start a Node.js worker bridge for distributed runs."""

from __future__ import annotations

import asyncio
import logging
import os
import shutil
import sys
from pathlib import Path

import click

from atomics.commands.common import setup_logging

logger = logging.getLogger("atomics.worker_npm")


NPM_DIR = Path(__file__).resolve().parent.parent / "workers" / "npm"


@click.command()
@click.option("--coordinator", default="http://127.0.0.1:8000", show_default=True)
@click.option("--api-key", envvar="ATOMICS_WORKER_API_KEY", help="Worker API key")
@click.option("--label", "labels", multiple=True, help="Worker label key=value, repeatable")
@click.option(
    "--capability",
    "capabilities",
    multiple=True,
    default=["node"],
    show_default=True,
    help="Worker capability, repeatable",
)
@click.option("--endpoint", help="Optional push endpoint URL for this worker")
@click.option("--worker-cmd", default="node task-runner.js", show_default=True, help="Command the npm worker uses to execute each task")
@click.option("--heartbeat-interval", default=30, show_default=True, help="Heartbeat interval in seconds")
@click.option("--npm-dir", type=click.Path(exists=True, file_okay=False, path_type=Path), default=str(NPM_DIR), help="Path to the npm worker package")
def worker_npm(
    coordinator: str,
    api_key: str,
    labels: tuple[str, ...],
    capabilities: tuple[str, ...],
    endpoint: str | None,
    worker_cmd: str,
    heartbeat_interval: int,
    npm_dir: Path,
) -> None:
    """Start a Node.js worker that joins the Atomics distributed pool.

    The Node.js worker registers with the coordinator, heartbeats, polls for
    task assignments, and executes them via the bridge command specified by
    --worker-cmd. The default command is the bundled task-runner.js, which is
    useful for testing; real deployments should point it at their own runner.
    """
    if not api_key:
        raise click.UsageError("--api-key is required (or set ATOMICS_WORKER_API_KEY)")
    if not shutil.which("node"):
        raise click.UsageError("node is required to run the npm worker bridge")

    setup_logging("INFO")

    for label in labels:
        if "=" not in label:
            raise click.BadParameter(f"Label must be key=value: {label!r}")

    worker_script = npm_dir / "worker.js"
    if not worker_script.exists():
        raise click.UsageError(f"npm worker script not found: {worker_script}")

    env = os.environ.copy()
    env["ATOMICS_COORDINATOR_URL"] = coordinator
    env["ATOMICS_WORKER_API_KEY"] = api_key
    env["ATOMICS_WORKER_LABELS"] = ",".join(labels)
    env["ATOMICS_WORKER_CAPABILITIES"] = ",".join(capabilities)
    env["ATOMICS_WORKER_CMD"] = worker_cmd
    if endpoint:
        env["ATOMICS_WORKER_ENDPOINT"] = endpoint
    env["ATOMICS_WORKER_HEARTBEAT_INTERVAL"] = str(heartbeat_interval)

    logger.info("Starting npm worker from %s", npm_dir)
    try:
        asyncio.run(_run_node_worker(str(worker_script), str(npm_dir), env))
    except KeyboardInterrupt:
        logger.info("npm worker stopped")


async def _run_node_worker(script: str, cwd: str, env: dict[str, str]) -> None:
    proc = await asyncio.create_subprocess_exec(
        "node", script,
        cwd=cwd,
        env=env,
    )
    await proc.wait()
    if proc.returncode != 0:
        sys.exit(proc.returncode)
