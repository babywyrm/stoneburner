"""API server CLI command."""

from __future__ import annotations

from pathlib import Path

import click
from rich.console import Console


@click.command()
@click.option("--host", default="127.0.0.1", help="Bind host")
@click.option("--port", type=int, default=8000, help="Bind port")
@click.option(
    "--api-key",
    "api_keys",
    multiple=True,
    help="API key(s) allowed (can be repeated). If none, --no-auth is required.",
)
@click.option(
    "--worker-api-key",
    "worker_api_keys",
    multiple=True,
    help="Key(s) accepted on worker endpoints only (can be repeated). Without "
    "this, workers share --api-key and a worker credential can also submit evals.",
)
@click.option(
    "--no-auth",
    is_flag=True,
    default=False,
    help="Disable API key authentication (loopback binds only)",
)
@click.option("--log-level", default="info", help="Uvicorn log level")
@click.option(
    "--worker-absent-after",
    default=120.0,
    show_default=True,
    type=float,
    help="Seconds without a heartbeat before a distributed worker is marked "
    "offline and its pinned fleet work fails. Raise it above roughly four times "
    "your workers' --heartbeat-interval.",
)
@click.option(
    "--with-dashboard",
    is_flag=True,
    default=False,
    help="Serve an optional web dashboard at /dashboard (default: off)",
)
@click.option(
    "--db-path",
    type=click.Path(path_type=Path),
    help="SQLite database path (default: atomics default state directory)",
)
@click.option(
    "--max-active-jobs",
    default=16,
    show_default=True,
    type=int,
    help="Maximum concurrent in-memory API jobs before returning 429.",
)
@click.option(
    "--max-retained-jobs",
    default=256,
    show_default=True,
    type=int,
    help="Maximum finished jobs retained for polling before evicting the oldest.",
)
@click.option(
    "--max-active-jobs-per-caller",
    default=4,
    show_default=True,
    type=int,
    help="Maximum concurrent active jobs allowed for a single API key.",
)
@click.option(
    "--max-caller-spend-usd",
    default=None,
    type=float,
    help="Optional cumulative USD spend ceiling per API key across runs.",
)
@click.option(
    "--max-request-bytes",
    default=4 * 1024 * 1024,
    show_default=True,
    type=int,
    help="Maximum HTTP request body size in bytes before returning 413.",
)
@click.option(
    "--max-caller-rpm",
    default=None,
    type=int,
    help="Optional sliding-window requests-per-minute ceiling per API key.",
)
@click.option(
    "--allowed-host",
    "--allowed-hosts",
    "allowed_hosts",
    multiple=True,
    help="Allowlisted outbound inference/judge origin or URL prefix "
    "(repeatable or comma-separated). Rejects non-matching caller hosts with 403.",
)
def server(
    host: str,
    port: int,
    api_keys: tuple[str, ...],
    worker_api_keys: tuple[str, ...],
    no_auth: bool,
    log_level: str,
    worker_absent_after: float,
    with_dashboard: bool,
    db_path: Path | None,
    max_active_jobs: int,
    max_retained_jobs: int,
    max_active_jobs_per_caller: int,
    max_caller_spend_usd: float | None,
    max_request_bytes: int,
    max_caller_rpm: int | None,
    allowed_hosts: tuple[str, ...],
) -> None:
    """Run the atomics API server."""
    console = Console()
    try:
        import uvicorn
    except ImportError as exc:
        console.print("[red]API server requires the [api] extra:[/red] uv sync --extra api")
        raise SystemExit(1) from exc

    from atomics.api.config import ServerSettings
    from atomics.api.server import create_app
    from atomics.commands.common import setup_logging

    if not api_keys and not no_auth:
        console.print("[red]Error: supply --api-key or --no-auth[/red]")
        raise SystemExit(1)

    allowed_set: set[str] | None = (
        {part.strip() for raw in allowed_hosts for part in raw.split(",") if part.strip()}
        if allowed_hosts
        else None
    )

    try:
        settings = ServerSettings(
            host=host,
            port=port,
            api_keys=set(api_keys),
            worker_api_keys=set(worker_api_keys),
            no_auth=no_auth,
            log_level=log_level,
            worker_absent_after_seconds=worker_absent_after,
            with_dashboard=with_dashboard,
            db_path=db_path or ServerSettings().db_path,
            max_active_jobs=max_active_jobs,
            max_retained_jobs=max_retained_jobs,
            max_active_jobs_per_caller=max_active_jobs_per_caller,
            max_spend_usd_per_caller=max_caller_spend_usd,
            max_request_body_bytes=max_request_bytes,
            max_requests_per_minute_per_caller=max_caller_rpm,
            allowed_hosts=allowed_set,
        )
    except ValueError as exc:
        console.print(f"[red]Error:[/red] {exc}")
        raise SystemExit(1) from exc
    app = create_app(settings)
    # Without this the atomics loggers never reach a handler at INFO, so the
    # access log and every job_submitted/job_finished line silently vanish and
    # a correlation ID correlates nothing.
    setup_logging(log_level, plain=True)
    # Uvicorn's own access log writes the raw request line, query string
    # included, which defeats the middleware's deliberate omission of it: a key
    # passed as ?api_key= would land in the log anyway. Ours replaces it and
    # carries the correlation ID and caller besides.
    uvicorn.run(app, host=host, port=port, log_level=log_level, access_log=False)
