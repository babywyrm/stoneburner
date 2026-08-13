"""MCP server for atomics, as a proxy over a running API server.

`client` deliberately imports nothing from the `mcp` package, so the proxy logic
is importable — and testable — without the optional `[mcp]` extra installed.
Only `atomics.mcp.server` needs the SDK, and it is imported lazily by the
`atomics mcp` command so a missing extra produces an install hint rather than a
traceback.
"""

from __future__ import annotations

from atomics.mcp.client import AtomicsApiClient, AtomicsApiError

__all__ = ["AtomicsApiClient", "AtomicsApiError"]
