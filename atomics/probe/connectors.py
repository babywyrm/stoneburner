"""Probe connectors — fetch raw artifact content from file or HTTP."""

from __future__ import annotations

from pathlib import Path

from atomics.probe.config import ProbeTarget


class ProbeConnectorError(OSError):
    """Raised when an artifact cannot be fetched."""


async def fetch_artifact(target: ProbeTarget, *, max_bytes: int = 64_000) -> str:
    """Fetch artifact content for a probe target.

    Returns the content as a string, truncated to max_bytes if necessary.
    """
    if target.source == "file":
        return await _fetch_file(target, max_bytes=max_bytes)
    if target.source == "http":
        return await _fetch_http(target, max_bytes=max_bytes)
    raise ProbeConnectorError(
        f"Unsupported source '{target.source}' for target '{target.name}'. "
        "Valid sources: file, http"
    )


async def _fetch_file(target: ProbeTarget, *, max_bytes: int) -> str:
    path = Path(target.path or "")
    if not path.exists():
        raise ProbeConnectorError(
            f"Artifact file not found for target '{target.name}': {path}"
        )
    raw = path.read_bytes()
    if len(raw) > max_bytes:
        raw = raw[:max_bytes]
    try:
        return raw.decode("utf-8", errors="replace")
    except Exception as exc:
        raise ProbeConnectorError(
            f"Failed to decode artifact '{target.name}' ({path}): {exc}"
        ) from exc


async def _fetch_http(target: ProbeTarget, *, max_bytes: int) -> str:
    import httpx

    url = target.url or ""
    headers = target.headers or {}

    try:
        async with httpx.AsyncClient(headers=headers, timeout=15.0) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            content = resp.content
            if len(content) > max_bytes:
                content = content[:max_bytes]
            return content.decode("utf-8", errors="replace")
    except httpx.HTTPError as exc:
        raise ProbeConnectorError(
            f"HTTP fetch failed for target '{target.name}' ({url}): {exc}"
        ) from exc
