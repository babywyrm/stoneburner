"""OpenAI / Codex provider adapter.

Requires: pip install atomics[openai]  (adds openai)
Auth: OPENAI_API_KEY, OAuth/OIDC, or Codex CLI tokens.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from atomics.providers.base import BaseProvider, ProviderResponse

if TYPE_CHECKING:
    from atomics.auth import AuthStrategy

MODEL_PRICING: dict[str, tuple[float, float]] = {
    "gpt-4o": (2.50, 10.0),
    "gpt-4o-2024-11-20": (2.50, 10.0),
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o-mini-2024-07-18": (0.15, 0.60),
    "gpt-4.1": (2.00, 8.00),
    "gpt-4.1-mini": (0.40, 1.60),
    "gpt-4.1-nano": (0.10, 0.40),
    "gpt-5": (15.0, 60.0),
    "gpt-5-turbo": (5.00, 20.0),
    "gpt-5.3": (10.0, 40.0),
    "gpt-5.5": (15.0, 60.0),
    "o3": (2.00, 8.00),
    "o3-mini": (1.10, 4.40),
    "o3-pro": (20.0, 80.0),
    "o4-mini": (1.10, 4.40),
    "codex-mini-latest": (1.50, 6.00),
}

# Models that require max_completion_tokens instead of max_tokens
_MAX_COMPLETION_TOKENS_MODELS = {
    "gpt-5", "gpt-5-turbo", "gpt-5.3", "gpt-5.5", "o3", "o3-pro", "o3-mini", "o4-mini",
}

DEFAULT_PRICING = (2.50, 10.0)


def _estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    inp_price, out_price = MODEL_PRICING.get(model, DEFAULT_PRICING)
    return (input_tokens * inp_price + output_tokens * out_price) / 1_000_000


class OpenAIProvider(BaseProvider):
    """OpenAI / Codex adapter. Uses Chat Completions with API keys,
    Responses API with OAuth tokens (required by ChatGPT OAuth scopes)."""

    def __init__(
        self,
        api_key: str = "",
        default_model: str = "gpt-4o",
        *,
        client: object | None = None,
        auth: AuthStrategy | None = None,
    ) -> None:
        self._auth = auth
        self._use_responses_api = auth is not None
        if client is not None:
            self._client = client
        else:
            try:
                from openai import AsyncOpenAI
            except ImportError as exc:
                raise ImportError(
                    "openai is required for the OpenAI provider. "
                    "Install with: uv sync --extra openai"
                ) from exc
            if auth is not None:
                self._client = AsyncOpenAI(api_key="oauth-managed")
            else:
                self._client = AsyncOpenAI(api_key=api_key)
        self._default_model = default_model

    @property
    def name(self) -> str:
        return "openai"

    async def generate(
        self,
        prompt: str,
        *,
        system: str = "",
        model: str | None = None,
        max_tokens: int = 1024,
    ) -> ProviderResponse:
        model = model or self._default_model

        if self._auth is not None:
            headers = await self._auth.get_headers()
            token = headers.get("Authorization", "").removeprefix("Bearer ")
            if token:
                self._client.api_key = token

        if self._use_responses_api:
            return await self._generate_responses(prompt, system=system, model=model, max_tokens=max_tokens)
        return await self._generate_completions(prompt, system=system, model=model, max_tokens=max_tokens)

    async def _generate_completions(
        self, prompt: str, *, system: str, model: str, max_tokens: int
    ) -> ProviderResponse:
        """Chat Completions API — used with static API keys."""
        messages: list[dict] = []
        if system:
            messages.append({"role": "system", "content": system})
        else:
            messages.append({"role": "system", "content": "You are a helpful assistant."})
        messages.append({"role": "user", "content": prompt})

        # Newer models (o3, gpt-5 family) require max_completion_tokens
        token_param = (
            {"max_completion_tokens": max_tokens}
            if model in _MAX_COMPLETION_TOKENS_MODELS
            else {"max_tokens": max_tokens}
        )
        t0 = time.monotonic()
        response = await self._client.chat.completions.create(
            model=model,
            messages=messages,
            **token_param,
        )
        latency = (time.monotonic() - t0) * 1000

        choice = response.choices[0] if response.choices else None
        raw_content = choice.message.content if choice else None

        # Newer models (gpt-5 family) may return content as a list of blocks
        # or put text in refusal/reasoning fields — extract whatever we can find
        if isinstance(raw_content, list):
            text = "".join(
                block.get("text", "") if isinstance(block, dict) else getattr(block, "text", "")
                for block in raw_content
                if (isinstance(block, dict) and block.get("type") == "text")
                or getattr(block, "type", "") == "text"
            )
        elif raw_content:
            text = raw_content
        else:
            # Fallback: check reasoning_content (o-series models)
            text = getattr(getattr(choice, "message", None), "reasoning_content", "") or ""
            if not text:
                import logging as _logging
                _logging.getLogger("atomics.providers.openai").warning(
                    "Empty response content for model %s — raw choices: %s",
                    model,
                    str(response.choices)[:300],
                )
        usage = response.usage
        inp = usage.prompt_tokens if usage else 0
        out = usage.completion_tokens if usage else 0

        tps = out / (latency / 1000) if latency > 0 and out > 0 else None

        return ProviderResponse(
            text=text,
            input_tokens=inp,
            output_tokens=out,
            total_tokens=inp + out,
            model=model,
            latency_ms=round(latency, 2),
            estimated_cost_usd=round(_estimate_cost(model, inp, out), 6),
            tokens_per_second=round(tps, 2) if tps is not None else None,
            raw=response.model_dump() if hasattr(response, "model_dump") else None,
        )

    async def _generate_responses(
        self, prompt: str, *, system: str, model: str, max_tokens: int
    ) -> ProviderResponse:
        """Responses API — used with OAuth tokens (ChatGPT OAuth scopes)."""
        input_items: list[dict] = []
        if system:
            input_items.append({"role": "system", "content": system})
        input_items.append({"role": "user", "content": prompt})

        t0 = time.monotonic()
        response = await self._client.responses.create(
            model=model,
            input=input_items,
            instructions=system or "You are a helpful assistant.",
            max_output_tokens=max_tokens,
            store=False,
        )
        latency = (time.monotonic() - t0) * 1000

        text = getattr(response, "output_text", "") or ""
        if not text and hasattr(response, "output"):
            for item in response.output:
                if getattr(item, "type", "") == "message":
                    for block in getattr(item, "content", []):
                        if getattr(block, "type", "") == "output_text":
                            text += getattr(block, "text", "")

        usage = getattr(response, "usage", None)
        inp = getattr(usage, "input_tokens", 0) if usage else 0
        out = getattr(usage, "output_tokens", 0) if usage else 0

        tps = out / (latency / 1000) if latency > 0 and out > 0 else None

        return ProviderResponse(
            text=text,
            input_tokens=inp,
            output_tokens=out,
            total_tokens=inp + out,
            model=model,
            latency_ms=round(latency, 2),
            estimated_cost_usd=round(_estimate_cost(model, inp, out), 6),
            tokens_per_second=round(tps, 2) if tps is not None else None,
            raw=response.model_dump() if hasattr(response, "model_dump") else None,
        )

    async def health_check(self) -> bool:
        try:
            resp = await self.generate("Say OK.", max_tokens=8)
            return len(resp.text) > 0
        except Exception:
            return False
