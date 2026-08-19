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
