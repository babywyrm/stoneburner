"""Read what a model host says about its models."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Sequence
from datetime import UTC, datetime
from fnmatch import fnmatch
from typing import Any
from urllib.parse import urlsplit

import httpx

from atomics.inventory import HostRecord, Inventory, ModelRecord, mark_digest_mismatches
from atomics.inventory.probe import probe_model
from atomics.providers.base import BaseProvider
from atomics.providers.llamacpp import LlamaCppProvider
from atomics.providers.ollama import DEFAULT_NUM_CTX, OllamaProvider
from atomics.providers.vllm import VllmProvider

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


def _show_context(show: Any) -> int | None:
    info = show.get("model_info") or {}
    # Only the architecture's own key is the window; rope keys also end in context_length.
    value = info.get(f"{info.get('general.architecture')}.context_length")
    return value if isinstance(value, int) else None


async def _loaded(client: httpx.AsyncClient, base: str) -> dict[str, int | None]:
    try:
        ps = await _fetch(client, f"{base}/api/ps")
    except ConnectionError:
        return {}
    return {m.get("name", ""): m.get("context_length") for m in ps.get("models", [])}


async def read_ollama(
    client: httpx.AsyncClient, url: str, *, label: str, context_tokens: int | None
) -> tuple[HostRecord, list[ModelRecord]]:
    base = url.rstrip("/")
    tags = await _fetch(client, f"{base}/api/tags")
    try:
        version = (await _fetch(client, f"{base}/api/version")).get("version")
    except ConnectionError:
        version = None
    loaded = await _loaded(client, base)
    models: list[ModelRecord] = []
    for entry in tags.get("models", []):
        details = entry.get("details") or {}
        caps = entry.get("capabilities")
        record = ModelRecord(
            host=label,
            name=entry.get("name", ""),
            digest=entry.get("digest"),
            size_bytes=entry.get("size"),
            parameter_size=details.get("parameter_size") or None,
            quantization=details.get("quantization_level") or None,
            declared=frozenset(caps) if isinstance(caps, list) else None,
            requested_context=context_tokens or DEFAULT_NUM_CTX,
        )
        record.loaded_context = loaded.get(record.name)
        try:
            show = await _fetch(client, f"{base}/api/show", body={"model": record.name})
        except ConnectionError as exc:
            record.errors.append(f"show: {exc}")
        else:
            record.declared_context = _show_context(show)
            if record.declared is None and isinstance(show.get("capabilities"), list):
                record.declared = frozenset(show["capabilities"])
        models.append(record)
    host = HostRecord(label=label, provider="ollama", version=version, loaded=list(loaded.items()))
    return host, models


async def read_openai(
    client: httpx.AsyncClient, url: str, *, label: str, context_tokens: int | None
) -> tuple[HostRecord, list[ModelRecord]]:
    base = url.rstrip("/")
    listing = await _fetch(client, f"{base}/models")
    models = []
    for entry in listing.get("data", []):
        window = entry.get("max_model_len")
        models.append(
            ModelRecord(
                host=label,
                name=entry.get("id", ""),
                declared_context=window if isinstance(window, int) else None,
                requested_context=context_tokens,
            )
        )
    return HostRecord(label=label, provider="vllm"), models


async def read_llamacpp(
    client: httpx.AsyncClient, url: str, *, label: str, context_tokens: int | None
) -> tuple[HostRecord, list[ModelRecord]]:
    base = url.rstrip("/")
    listing = await _fetch(client, f"{base}/v1/models")
    try:
        props = await _fetch(client, f"{base}/props")
    except ConnectionError:
        props = {}
    settings = props.get("default_generation_settings") or {}
    n_ctx = settings.get("n_ctx", props.get("n_ctx"))
    models = [
        ModelRecord(
            host=label,
            name=entry.get("id", ""),
            declared_context=n_ctx if isinstance(n_ctx, int) else None,
            requested_context=context_tokens,
        )
        for entry in listing.get("data", [])
    ]
    return HostRecord(label=label, provider="llamacpp"), models


_READERS: dict[str, Reader] = {
    "ollama": read_ollama,
    "vllm": read_openai,
    "llamacpp": read_llamacpp,
}


def probe_provider(
    provider: str, url: str, client: httpx.AsyncClient, context_tokens: int | None
) -> BaseProvider:
    """Build the generating provider. Each request sets its own long timeout."""
    if provider == "vllm":
        return VllmProvider(base_url=url, client=client)
    if provider == "llamacpp":
        return LlamaCppProvider(base_url=url, client=client)
    return OllamaProvider(host=url, client=client, context_tokens=context_tokens)


def unique_labels(urls: Sequence[str]) -> list[str]:
    names = [host_label(u) for u in urls]
    return [urlsplit(u).netloc if names.count(n) > 1 else n for u, n in zip(urls, names)]


async def _take_host(
    client: httpx.AsyncClient,
    provider: str,
    url: str,
    label: str,
    *,
    context_tokens: int | None,
    patterns: Sequence[str],
    probe: bool,
    on_model: Callable[[ModelRecord], None] | None,
) -> tuple[HostRecord, list[ModelRecord]]:
    try:
        host, models = await _READERS[provider](
            client, url, label=label, context_tokens=context_tokens
        )
    except ConnectionError as exc:
        return HostRecord(label=label, provider=provider, error=str(exc)), []
    if patterns:
        models = [m for m in models if any(fnmatch(m.name, p) for p in patterns)]
    if probe:
        target = probe_provider(provider, url, client, context_tokens)
        for record in models:
            if record.evaluable:
                await probe_model(target, record)
            if on_model is not None:
                on_model(record)
    return host, models


async def take_inventory(
    provider: str,
    urls: Sequence[str],
    *,
    context_tokens: int | None = None,
    patterns: Sequence[str] = (),
    probe: bool = False,
    on_model: Callable[[ModelRecord], None] | None = None,
) -> Inventory:
    """Read every host at once. Within a host, probes run one model at a time:
    a host serves one generate at once, so parallel probes would only queue."""
    async with new_client() as client:
        taken = await asyncio.gather(
            *(
                _take_host(
                    client,
                    provider,
                    url,
                    label,
                    context_tokens=context_tokens,
                    patterns=patterns,
                    probe=probe,
                    on_model=on_model,
                )
                for url, label in zip(urls, unique_labels(urls))
            )
        )
    hosts = [h for h, _ in taken]
    if all(h.error for h in hosts):
        raise ConnectionError("; ".join(f"{h.label}: {h.error}" for h in hosts))
    models = [m for _, ms in taken for m in ms]
    mark_digest_mismatches(models)
    taken_at = datetime.now(UTC).isoformat(timespec="seconds")
    return Inventory(taken_at=taken_at, hosts=hosts, models=models)
