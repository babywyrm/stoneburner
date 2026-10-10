"""Groq provider adapter — fast inference via OpenAI-compatible API.

Auth: GROQ_API_KEY environment variable.
Endpoint: https://api.groq.com/openai/v1/chat/completions
"""

from __future__ import annotations

import httpx

from atomics.providers._openai_compat import HostedChatProvider

MODEL_PRICING: dict[str, tuple[float, float]] = {
    "llama-3.3-70b-versatile": (0.59, 0.79),
    "llama-3.1-8b-instant": (0.05, 0.08),
    "llama-3.2-1b-preview": (0.04, 0.04),
    "llama-3.2-3b-preview": (0.06, 0.06),
    "llama-3.2-11b-vision-preview": (0.18, 0.18),
    "llama-3.2-90b-vision-preview": (0.90, 0.90),
    "gemma2-9b-it": (0.20, 0.20),
    "mixtral-8x7b-32768": (0.24, 0.24),
    "qwen-qwq-32b": (0.29, 0.39),
    "deepseek-r1-distill-llama-70b": (0.75, 0.99),
    "meta-llama/llama-4-scout-17b-16e-instruct": (0.11, 0.34),
    "meta-llama/llama-4-maverick-17b-128e-instruct": (0.50, 0.77),
}

DEFAULT_PRICING = (0.50, 0.50)


def _estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    inp_price, out_price = MODEL_PRICING.get(model, DEFAULT_PRICING)
    return (input_tokens * inp_price + output_tokens * out_price) / 1_000_000


class GroqProvider(HostedChatProvider):
    """Groq cloud inference via OpenAI-compatible Chat Completions API."""

    def __init__(
        self,
        api_key: str,
        default_model: str = "llama-3.3-70b-versatile",
        *,
        timeout: float = 60.0,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        super().__init__(
            name="groq",
            base_url="https://api.groq.com/openai/v1",
            api_key=api_key,
            default_model=default_model,
            timeout=timeout,
            client=client,
        )

    def _tool_cost(self, model: str, input_tokens: int, output_tokens: int) -> float:
        return _estimate_cost(model, input_tokens, output_tokens)
