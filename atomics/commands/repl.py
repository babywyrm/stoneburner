"""Interactive REPL — human client of a running atomics API server."""

from __future__ import annotations

import os

import click
from rich.console import Console

from atomics.mcp.client import API_KEY_ENV, API_URL_ENV, DEFAULT_API_URL, AtomicsApiClient
from atomics.repl.loop import run_repl


@click.command(name="repl")
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
def repl(api_url: str | None, api_key: str | None) -> None:
    """Interactive prompt over a running atomics API server.

    Same trust model as `atomics mcp`: this process holds no provider or
    budget logic. Start the API first. If nothing is listening, this exits.

    \b
        atomics server --api-key "$ATOMICS_API_KEY"
        atomics repl
    """
    console = Console(stderr=True)
    base_url = api_url or os.environ.get(API_URL_ENV) or DEFAULT_API_URL
    key = api_key or os.environ.get(API_KEY_ENV) or None
    if key is None:
        console.print(
            f"[yellow]No API key set.[/yellow] Requests will be unauthenticated, which "
            f"only works against a server started with --no-auth. Set ${API_KEY_ENV} "
            f"or pass --api-key."
        )
    raise SystemExit(run_repl(AtomicsApiClient(base_url, key)))
