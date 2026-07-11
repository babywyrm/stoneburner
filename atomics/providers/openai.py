"""OpenAI / Codex provider adapter.

Requires: pip install atomics[openai]  (adds openai)
Auth: OPENAI_API_KEY, OAuth/OIDC, or Codex CLI tokens.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any, cast

from atomics.eval.outcomes import (
    ProviderOutcome,
    ProviderOutcomeKind,
    policy_block_reason,
)
from atomics.providers import pricing
from atomics.providers.base import BaseProvider, ProviderResponse, compute_tps
from atomics.validation import sanitize_error

if TYPE_CHECKING:
    from openai import AsyncOpenAI
    from openai.types.chat import ChatCompletionMessageParam
    from openai.types.responses import ResponseInputParam

    from atomics.auth import AuthStrategy

# Pricing per 1M tokens (input / output). Sourced from the central pricing
# module; re-exported here for backward compatibility.
MODEL_PRICING = pricing.OPENAI_PRICING
DEFAULT_PRICING = pricing.OPENAI_DEFAULT

# Models that require max_completion_tokens instead of max_tokens
_MAX_COMPLETION_TOKENS_MODELS = {
    "gpt-5", "gpt-5-turbo", "gpt-5.3", "gpt-5.5", "o3", "o3-pro", "o3-mini", "o4-mini",
}

# Reasoning models burn internal thinking tokens before producing visible output.
# Multiply the requested max_tokens by this factor so thinking doesn't consume
# the entire budget and leave the visible response empty.
_REASONING_TOKEN_MULTIPLIER = 8
_NONTERMINAL_RESPONSE_STATUSES = frozenset({"cancelled", "queued", "in_progress"})

def _estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    return pricing.estimate_cost(
        model, input_tokens, output_tokens, table=MODEL_PRICING, default=DEFAULT_PRICING
    )


def _is_reasoning_model(model: str) -> bool:
    """Return whether OpenAI requires reasoning-model request parameters."""

    return model in _MAX_COMPLETION_TOKENS_MODELS or model.startswith(
        ("gpt-5.", "gpt-5-")
    )


def _field(value: object, name: str, default: Any = None) -> Any:
    if isinstance(value, dict):
        return value.get(name, default)
    return getattr(value, name, default)


def _combine_response_text(visible_text: str, refusal_text: str) -> str:
    if not refusal_text or refusal_text == visible_text:
        return visible_text
    if not visible_text:
        return refusal_text
    if visible_text.endswith(f"\n{refusal_text}"):
        return visible_text
    return f"{visible_text}\n{refusal_text}"


class OpenAIProvider(BaseProvider):
    """OpenAI / Codex adapter. Uses Chat Completions with API keys,
    Responses API with OAuth tokens (required by ChatGPT OAuth scopes)."""

    def __init__(
        self,
        api_key: str = "",
        default_model: str = "gpt-4o",
        *,
        client: AsyncOpenAI | None = None,
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
            import httpx
            timeout = httpx.Timeout(60.0, connect=10.0)
            if auth is not None:
                self._client = AsyncOpenAI(api_key="oauth-managed", timeout=timeout, max_retries=2)
            else:
                self._client = AsyncOpenAI(api_key=api_key, timeout=timeout, max_retries=2)
        self._default_model = default_model

    @property
    def name(self) -> str:
        return "openai"

    @property
    def default_model(self) -> str | None:
        return self._default_model

    async def generate(
        self,
        prompt: str,
        *,
        system: str = "",
        model: str | None = None,
        max_tokens: int = 1024,
        thinking: bool | None = None,
        thinking_budget: int | None = None,
        temperature: float | None = None,
    ) -> ProviderResponse:
        model = model or self._default_model

        if self._auth is not None:
            headers = await self._auth.get_headers()
            token = headers.get("Authorization", "").removeprefix("Bearer ")
            if token:
                self._client.api_key = token

        if self._use_responses_api:
            return await self._generate_responses(
                prompt, system=system, model=model, max_tokens=max_tokens,
                thinking=thinking, thinking_budget=thinking_budget,
                temperature=temperature,
            )
        return await self._generate_completions(
            prompt, system=system, model=model, max_tokens=max_tokens,
            thinking=thinking, thinking_budget=thinking_budget,
            temperature=temperature,
        )

    async def _generate_completions(
        self, prompt: str, *, system: str, model: str, max_tokens: int,
        thinking: bool | None = None, thinking_budget: int | None = None,
        temperature: float | None = None,
    ) -> ProviderResponse:
        """Chat Completions API — used with static API keys."""
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        else:
            messages.append({"role": "system", "content": "You are a helpful assistant."})
        messages.append({"role": "user", "content": prompt})

        is_reasoning = _is_reasoning_model(model)
        token_param: dict[str, Any]
        if is_reasoning:
            multiplier = _REASONING_TOKEN_MULTIPLIER if thinking is not False else 2
            token_param = {"max_completion_tokens": max_tokens * multiplier}
        else:
            token_param = {"max_tokens": max_tokens}
        # Reasoning models (o-series/gpt-5) reject an explicit temperature; only
        # forward it for standard chat models.
        if temperature is not None and not is_reasoning:
            token_param["temperature"] = temperature
        t0 = time.monotonic()
        response = await self._client.chat.completions.create(
            model=model,
            messages=cast("list[ChatCompletionMessageParam]", messages),
            **token_param,
        )
        latency = (time.monotonic() - t0) * 1000

        choice = response.choices[0] if response.choices else None
        raw_content = choice.message.content if choice else None
        raw_refusal = getattr(getattr(choice, "message", None), "refusal", None)
        refusal = raw_refusal if isinstance(raw_refusal, str) else ""
        finish_reason = getattr(choice, "finish_reason", None)

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
            text = ""
            if not refusal:
                import logging as _logging
                _logging.getLogger("atomics.providers.openai").warning(
                    "Empty response content for model %s — raw choices: %s",
                    model,
                    str(response.choices)[:300],
                )
        text = _combine_response_text(text, refusal)
        if refusal:
            outcome_kind = ProviderOutcomeKind.REFUSED
        elif finish_reason == "content_filter":
            outcome_kind = ProviderOutcomeKind.SAFETY_BLOCKED
        elif finish_reason == "length":
            outcome_kind = ProviderOutcomeKind.TRUNCATED
        elif text:
            outcome_kind = ProviderOutcomeKind.COMPLETED
        else:
            outcome_kind = ProviderOutcomeKind.EMPTY
        outcome = ProviderOutcome(
            kind=outcome_kind,
            finish_reason=finish_reason,
            safety_reason=(
                finish_reason
                if outcome_kind is ProviderOutcomeKind.SAFETY_BLOCKED
                else None
            ),
        )
        usage = response.usage
        inp = usage.prompt_tokens if usage else 0
        out = usage.completion_tokens if usage else 0

        reasoning_tokens = 0
        if usage and hasattr(usage, "completion_tokens_details"):
            details = usage.completion_tokens_details
            reasoning_tokens = getattr(details, "reasoning_tokens", 0) or 0

        tps = compute_tps(out, latency / 1000)

        return ProviderResponse(
            text=text,
            input_tokens=inp,
            output_tokens=out,
            total_tokens=inp + out,
            model=model,
            latency_ms=round(latency, 2),
            estimated_cost_usd=round(_estimate_cost(model, inp, out), 6),
            tokens_per_second=tps,
            thinking_tokens=reasoning_tokens,
            raw=response.model_dump() if hasattr(response, "model_dump") else None,
            outcome=outcome,
            finish_reason=finish_reason,
        )

    async def _generate_responses(
        self, prompt: str, *, system: str, model: str, max_tokens: int,
        thinking: bool | None = None, thinking_budget: int | None = None,
        temperature: float | None = None,
    ) -> ProviderResponse:
        """Responses API — used with OAuth tokens (ChatGPT OAuth scopes)."""
        input_items: list[dict[str, str]] = []
        if system:
            input_items.append({"role": "system", "content": system})
        input_items.append({"role": "user", "content": prompt})

        is_reasoning = _is_reasoning_model(model)
        if is_reasoning and thinking is not False:
            effective_max = max_tokens * _REASONING_TOKEN_MULTIPLIER
        else:
            effective_max = max_tokens

        extra: dict[str, Any] = {}
        # Reasoning models reject an explicit temperature; only forward otherwise.
        if temperature is not None and not is_reasoning:
            extra["temperature"] = temperature
        t0 = time.monotonic()
        response = await self._client.responses.create(
            model=model,
            input=cast("ResponseInputParam", input_items),
            instructions=system or "You are a helpful assistant.",
            max_output_tokens=effective_max,
            store=False,
            **extra,
        )
        latency = (time.monotonic() - t0) * 1000

        text = getattr(response, "output_text", "") or ""
        refusal_parts: list[str] = []
        fallback_text = ""
        if hasattr(response, "output"):
            for item in response.output:
                item_type = _field(item, "type", "")
                if item_type == "refusal":
                    refusal_parts.append(
                        _field(item, "refusal", "") or _field(item, "text", "")
                    )
                elif item_type == "message":
                    for block in _field(item, "content", []):
                        block_type = _field(block, "type", "")
                        if block_type == "output_text":
                            fallback_text += _field(block, "text", "")
                        elif block_type == "refusal":
                            refusal_parts.append(
                                _field(block, "refusal", "")
                                or _field(block, "text", "")
                            )
        refusal = "\n".join(dict.fromkeys(part for part in refusal_parts if part))
        if not text:
            text = fallback_text
        text = _combine_response_text(text, refusal)

        status = getattr(response, "status", None)
        incomplete_details = getattr(response, "incomplete_details", None)
        error = getattr(response, "error", None)
        if status == "incomplete":
            finish_reason = _field(incomplete_details, "reason") or "incomplete"
        elif status == "failed":
            failure_message = _field(error, "message")
            finish_reason = (
                _field(error, "code")
                or (
                    sanitize_error(ValueError(str(failure_message)))
                    if failure_message
                    else None
                )
                or "failed"
            )
        else:
            finish_reason = status

        error_code = _field(error, "code")
        error_message = _field(error, "message")
        policy_reason = policy_block_reason(
            code=(
                finish_reason
                if status == "incomplete"
                else error_code
            ),
            message=error_message,
        )
        if status == "failed" and policy_reason:
            outcome_kind = ProviderOutcomeKind.SAFETY_BLOCKED
        elif status == "failed":
            outcome_kind = ProviderOutcomeKind.PROVIDER_ERROR
        elif status == "incomplete" and policy_reason:
            outcome_kind = ProviderOutcomeKind.SAFETY_BLOCKED
        elif status == "incomplete":
            outcome_kind = ProviderOutcomeKind.TRUNCATED
        elif status in _NONTERMINAL_RESPONSE_STATUSES:
            outcome_kind = ProviderOutcomeKind.PROVIDER_ERROR
        elif refusal:
            outcome_kind = ProviderOutcomeKind.REFUSED
        elif text:
            outcome_kind = ProviderOutcomeKind.COMPLETED
        else:
            outcome_kind = ProviderOutcomeKind.EMPTY

        diagnostic = " | ".join(
            str(value) for value in (error_code, error_message) if value
        )
        if not diagnostic and status in _NONTERMINAL_RESPONSE_STATUSES:
            diagnostic = status
        sanitized_diagnostic = (
            sanitize_error(ValueError(diagnostic)) if diagnostic else None
        )
        outcome = ProviderOutcome(
            kind=outcome_kind,
            finish_reason=finish_reason,
            safety_reason=(
                policy_reason
                if outcome_kind is ProviderOutcomeKind.SAFETY_BLOCKED
                else None
            ),
            error_class=(
                "OpenAIResponseError"
                if outcome_kind is ProviderOutcomeKind.PROVIDER_ERROR
                else None
            ),
            error_message=(
                sanitized_diagnostic
                if outcome_kind is ProviderOutcomeKind.PROVIDER_ERROR
                else None
            ),
        )

        usage = getattr(response, "usage", None)
        inp = getattr(usage, "input_tokens", 0) if usage else 0
        out = getattr(usage, "output_tokens", 0) if usage else 0
        # The Responses API reports reasoning under usage.output_tokens_details;
        # fall back to a direct attribute for forward/backward compatibility.
        details = getattr(usage, "output_tokens_details", None) if usage else None
        if details is not None:
            reasoning_tokens = getattr(details, "reasoning_tokens", 0) or 0
        else:
            reasoning_tokens = getattr(usage, "reasoning_tokens", 0) or 0

        tps = compute_tps(out, latency / 1000)

        return ProviderResponse(
            text=text,
            input_tokens=inp,
            output_tokens=out,
            total_tokens=inp + out,
            model=model,
            latency_ms=round(latency, 2),
            estimated_cost_usd=round(_estimate_cost(model, inp, out), 6),
            tokens_per_second=tps,
            thinking_tokens=reasoning_tokens,
            raw=response.model_dump() if hasattr(response, "model_dump") else None,
            outcome=outcome,
            finish_reason=finish_reason,
        )

    async def health_check(self) -> bool:
        try:
            resp = await self.generate("Say OK.", max_tokens=8)
            # A successful round-trip that consumed tokens proves the provider is
            # live. Don't require visible text: reasoning models can spend the
            # whole (small) budget on hidden reasoning and return empty content
            # with finish_reason="length", which made this flaky for o-series.
            return resp.total_tokens > 0
        except Exception:
            return False
