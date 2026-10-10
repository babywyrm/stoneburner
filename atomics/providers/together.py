"""Together AI provider adapter — OpenAI-compatible cloud inference.

Auth: TOGETHER_API_KEY environment variable.
Endpoint: https://api.together.xyz/v1/chat/completions
"""

from __future__ import annotations

import httpx

from atomics.providers._openai_compat import HostedChatProvider

MODEL_PRICING: dict[str, tuple[float, float]] = {
    "meta-llama/Llama-3.3-70B-Instruct-Turbo": (0.88, 0.88),
    "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo": (0.18, 0.18),
    "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo": (0.88, 0.88),
    "meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo": (3.50, 3.50),
    "mistralai/Mixtral-8x7B-Instruct-v0.1": (0.60, 0.60),
    "mistralai/Mistral-7B-Instruct-v0.3": (0.20, 0.20),
    "Qwen/Qwen2.5-72B-Instruct-Turbo": (1.20, 1.20),
    "Qwen/Qwen2.5-7B-Instruct-Turbo": (0.30, 0.30),
    "deepseek-ai/DeepSeek-R1": (3.00, 7.00),
    "deepseek-ai/DeepSeek-V3": (0.50, 0.90),
    "google/gemma-2-27b-it": (0.80, 0.80),
    "google/gemma-2-9b-it": (0.30, 0.30),
}

DEFAULT_PRICING = (1.00, 1.00)


def _estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    inp_price, out_price = MODEL_PRICING.get(model, DEFAULT_PRICING)
    return (input_tokens * inp_price + output_tokens * out_price) / 1_000_000


class TogetherProvider(HostedChatProvider):
    """Together AI cloud inference via OpenAI-compatible Chat Completions API."""

    def __init__(
        self,
        api_key: str,
        default_model: str = "meta-llama/Llama-3.3-70B-Instruct-Turbo",
        *,
        timeout: float = 120.0,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        super().__init__(
            name="together",
            base_url="https://api.together.xyz/v1",
            api_key=api_key,
            default_model=default_model,
            timeout=timeout,
            client=client,
        )

    def _tool_cost(self, model: str, input_tokens: int, output_tokens: int) -> float:
        return _estimate_cost(model, input_tokens, output_tokens)
