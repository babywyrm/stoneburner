"""List models and probe a provider for the API — not jobs.

`models` and `provider-test` are seconds, not hours. They share `make_provider`
with the rest of the server so host validation and credentials stay in one place.
The generate prompt is fixed: a caller-supplied prompt would be a spend amp.
"""

from __future__ import annotations

from typing import Any

from fastapi import HTTPException, status

from atomics.api.models import ProviderTestRequest
from atomics.config import load_settings
from atomics.providers.factory import ProviderConfigError, make_provider
from atomics.validation import sanitize_error

LISTABLE_PROVIDERS = frozenset({"ollama", "vllm"})
PROVIDER_TEST_PROMPT = "What is 2+2? Reply with just the number."
PROVIDER_TEST_MAX_TOKENS = 32


def _provider(name: str, model: str | None, host: str | None):
    settings = load_settings()
    try:
        if name == "vllm":
            return make_provider(name, model, None, settings, vllm_host=host)
        return make_provider(name, model, host, settings)
    except ProviderConfigError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)
        ) from exc


async def list_models(provider: str, host: str | None) -> dict[str, Any]:
    name = provider.lower()
    if name not in LISTABLE_PROVIDERS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="provider must be 'ollama' or 'vllm'",
        )
    adapter = _provider(name, None, host)
    try:
        models = await adapter.list_models()
    except ConnectionError as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY, detail=sanitize_error(exc)
        ) from exc
    return {"provider": name, "models": models}


async def run_provider_test(payload: ProviderTestRequest) -> dict[str, Any]:
    adapter = _provider(payload.provider, payload.model, payload.host)
    health = await adapter.health_check()
    if not health:
        return {
            "ok": False,
            "health": False,
            "provider": adapter.name,
            "model": payload.model,
            "response": None,
            "error": "Provider health check failed.",
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "thinking_tokens": 0,
            "latency_ms": 0.0,
            "cost_usd": 0.0,
        }
    try:
        resp = await adapter.generate(
            PROVIDER_TEST_PROMPT,
            model=payload.model,
            max_tokens=PROVIDER_TEST_MAX_TOKENS,
            thinking=payload.thinking,
        )
    except Exception as exc:
        return {
            "ok": False,
            "health": True,
            "provider": adapter.name,
            "model": payload.model,
            "response": None,
            "error": sanitize_error(exc),
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "thinking_tokens": 0,
            "latency_ms": 0.0,
            "cost_usd": 0.0,
        }
    return {
        "ok": True,
        "health": True,
        "provider": adapter.name,
        "model": payload.model,
        "response": (resp.text or "").strip(),
        "error": None,
        "input_tokens": resp.input_tokens,
        "output_tokens": resp.output_tokens,
        "total_tokens": resp.total_tokens,
        "thinking_tokens": resp.thinking_tokens,
        "latency_ms": resp.latency_ms,
        "cost_usd": resp.estimated_cost_usd,
    }
