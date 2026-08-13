"""MCP server CLI command."""

from __future__ import annotations

import os

import click
from rich.console import Console

from atomics.mcp.client import API_KEY_ENV, API_URL_ENV, DEFAULT_API_URL


@click.command(name="mcp")
@click.option(
    "--api-url",
    default=None,
    help=f"Base URL of a running atomics API server (or ${API_URL_ENV}; "
    f"default {DEFAULT_API_URL}).",
)
@click.option(
    "--api-key",
    default=None,
    help=f"Client API key, sent as X-API-Key (or ${API_KEY_ENV}).",
)
def mcp(api_url: str | None, api_key: str | None) -> None:
    """Serve atomics over MCP, proxying a running API server.

    This holds no provider, storage, or budget logic: every tool call is one
    authenticated request to `atomics server`, so an agent inherits the same
    authentication and spend ceilings any remote HTTP caller has.

    Serves on stdio only, so the client that spawns it is the only thing that
    can drive it. To reach atomics from another host, expose the API server —
    which authenticates — and run this against it locally.

    Start the API first, then point an MCP client at this command:

    \b
        atomics server --api-key "$ATOMICS_API_KEY"
        atomics mcp
    """
    console = Console(stderr=True)
    try:
        from atomics.mcp.server import build_server
    except ImportError as exc:
        console.print("[red]The MCP server requires the [mcp] extra:[/red] uv sync --extra mcp")
        raise SystemExit(1) from exc

    from atomics.commands.common import setup_logging
    from atomics.mcp.client import AtomicsApiClient

    # On stdio, stdout *is* the JSON-RPC channel. The CLI group installed a Rich
    # log handler that writes there, so a single warning would interleave with
    # protocol frames and break the session. Plain logging uses a StreamHandler,
    # which writes to stderr; the console above is pinned to stderr for the same
    # reason. Nothing this command prints may go to stdout.
    setup_logging("INFO", plain=True)

    base_url = api_url or os.environ.get(API_URL_ENV) or DEFAULT_API_URL
    key = api_key or os.environ.get(API_KEY_ENV) or None
    if key is None:
        console.print(
            f"[yellow]No API key set.[/yellow] Requests will be unauthenticated, which "
            f"only works against a server started with --no-auth. Set ${API_KEY_ENV} "
            f"or pass --api-key."
        )

    console.print(f"[dim]atomics MCP server → {base_url} (stdio)[/dim]")
    server = build_server(AtomicsApiClient(base_url, key))
    # stdio only, deliberately. The HTTP transports would open a port that no
    # MCP-layer credential guards while this process holds an API key with spend
    # authority — anyone who could reach it could run up a provider bill through
    # the very guardrails this proxy exists to inherit.
    server.run()
