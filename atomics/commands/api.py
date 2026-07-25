"""API server CLI command."""

from __future__ import annotations

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
    "--no-auth",
    is_flag=True,
    default=False,
    help="Disable API key authentication (local dev only)",
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
def server(
    host: str,
    port: int,
    api_keys: tuple[str, ...],
    no_auth: bool,
    log_level: str,
    worker_absent_after: float,
) -> None:
    """Run the atomics API server."""
    console = Console()
    try:
        import uvicorn
    except ImportError as exc:
        console.print(
            "[red]API server requires the [api] extra:[/red] uv sync --extra api"
        )
        raise SystemExit(1) from exc

    from atomics.api.config import ServerSettings
    from atomics.api.server import create_app

    if not api_keys and not no_auth:
        console.print("[red]Error: supply --api-key or --no-auth[/red]")
        raise SystemExit(1)

    settings = ServerSettings(
        host=host,
        port=port,
        api_keys=set(api_keys),
        no_auth=no_auth,
        log_level=log_level,
        worker_absent_after_seconds=worker_absent_after,
    )
    app = create_app(settings)
    uvicorn.run(app, host=host, port=port, log_level=log_level)
