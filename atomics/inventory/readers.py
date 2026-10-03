"""Read what a model host says about its models."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from typing import Any
from urllib.parse import urlsplit

import httpx

from atomics.inventory import HostRecord, Inventory, ModelRecord

_TIMEOUT = 10.0

Reader = Callable[..., Awaitable[tuple[HostRecord, list[ModelRecord]]]]


def new_client() -> httpx.AsyncClient:
    return httpx.AsyncClient(timeout=_TIMEOUT)


def host_label(url: str) -> str:
    return urlsplit(url).hostname or url


async def _fetch(client: httpx.AsyncClient, url: str, body: dict[str, str] | None = None) -> Any:
    """GET, or POST when there is a body. Any HTTP failure is a ConnectionError."""
    try:
        if body is None:
            response = await client.get(url)
        else:
            response = await client.post(url, json=body)
        response.raise_for_status()
    except httpx.HTTPError as exc:
        raise ConnectionError(f"Cannot reach {url}: {type(exc).__name__}") from exc
    return response.json()


async def read_ollama(
    client: httpx.AsyncClient, url: str, *, label: str
) -> tuple[HostRecord, list[ModelRecord]]:
    base = url.rstrip("/")
    tags = await _fetch(client, f"{base}/api/tags")
    try:
        version = (await _fetch(client, f"{base}/api/version")).get("version")
    except ConnectionError:
        version = None
    models: list[ModelRecord] = []
    for entry in tags.get("models", []):
        details = entry.get("details") or {}
        caps = entry.get("capabilities")
        models.append(
            ModelRecord(
                host=label,
                name=entry.get("name", ""),
                digest=entry.get("digest"),
                size_bytes=entry.get("size"),
                parameter_size=details.get("parameter_size") or None,
                quantization=details.get("quantization_level") or None,
                declared=frozenset(caps) if isinstance(caps, list) else None,
            )
        )
    return HostRecord(label=label, provider="ollama", version=version), models


async def read_openai(
    client: httpx.AsyncClient, url: str, *, label: str
) -> tuple[HostRecord, list[ModelRecord]]:
    base = url.rstrip("/")
    listing = await _fetch(client, f"{base}/models")
    models = [ModelRecord(host=label, name=e.get("id", "")) for e in listing.get("data", [])]
    return HostRecord(label=label, provider="vllm"), models


_READERS: dict[str, Reader] = {"ollama": read_ollama, "vllm": read_openai}


async def take_inventory(provider: str, url: str) -> Inventory:
    async with new_client() as client:
        host, models = await _READERS[provider](client, url, label=host_label(url))
    taken_at = datetime.now(UTC).isoformat(timespec="seconds")
    return Inventory(taken_at=taken_at, hosts=[host], models=models)
