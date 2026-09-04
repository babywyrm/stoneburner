"""vLLM / OpenAI-compatible provider adapter.

Speaks the OpenAI /v1/chat/completions API, which is the wire-format
dialect exposed by vLLM, LiteLLM, and any OpenAI-compatible gateway.
This is NOT the OpenAI cloud service — it connects to a local endpoint
(e.g. a LiteLLM gateway at http://gpu-host:8000/v1).

The provider name is "vllm" to make it explicit that this is a local
inference backend, not the OpenAI company.
"""

from __future__ import annotations

import time

import httpx

from atomics.benchmark.model_classes import classify_model, supports_thinking
from atomics.providers._openai_compat import OpenAICompatibleTools
from atomics.providers.base import BaseProvider, ProviderResponse, compute_tps
from atomics.providers.effort import apply_chat_effort, normalize_effort, qwen_template_effort

_THINKING_MODEL_PREFIXES: tuple[str, ...] = ("qwen3", "deepseek-r1")


def _model_supports_thinking(model: str) -> bool:
    return any(model.startswith(p) for p in _THINKING_MODEL_PREFIXES)


def _is_qwen3(model: str) -> bool:
    return model.lower().startswith("qwen3")


def _usage_reasoning_tokens(usage: dict, thinking_text: str, text: str, out: int) -> int:
    reported = usage.get("reasoning_tokens")
    if reported is None:
        details = usage.get("completion_tokens_details") or {}
        reported = details.get("reasoning_tokens") if isinstance(details, dict) else None
    if reported is not None:
        return int(reported)
    generated_chars = len(thinking_text) + len(text)
    if generated_chars > 0 and out > 0:
        return round(out * len(thinking_text) / generated_chars)
    return 0


class VllmProvider(OpenAICompatibleTools, BaseProvider):
    """Provider for vLLM / OpenAI-compatible inference endpoints.

    Targets any server that implements POST /v1/chat/completions and
    GET /v1/models (vLLM, LiteLLM, llama.cpp server, etc.).

    Thinking mode for qwen3-family models is controlled via
    ``chat_template_kwargs.enable_thinking``. ``--effort`` is dual-written:
    top-level ``reasoning_effort`` for OpenAI-compat gateways, and
    ``chat_template_kwargs.reasoning_effort`` (low/medium/xhigh) for Qwen
    Jinja templates that ignore unknown keys. ``--thinking-budget`` is sent
    as SGLang ``custom_params.thinking_budget`` when thinking is on.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8000/v1",
        default_model: str = "qwen2.5:3b",
        api_key: str = "dummy",
        *,
        timeout: float = 300.0,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._default_model = default_model
        self._api_key = api_key
        self._timeout = timeout
        self._client = client or httpx.AsyncClient()

    @property
    def name(self) -> str:
        return "vllm"

    @property
    def default_model(self) -> str | None:
        return self._default_model

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

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
        effort: str | None = None,
        reasoning_mode: str | None = None,
    ) -> ProviderResponse:
        model = model or self._default_model
        _ = reasoning_mode

        use_thinking = thinking if thinking is not None else _model_supports_thinking(model)

        messages = [
            {"role": "system", "content": system or "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ]

        body: dict = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "stream": False,
        }
        if temperature is not None:
            body["temperature"] = temperature
        native: dict[str, object] = {}
        reasoning_request = apply_chat_effort(body, effort)
        if reasoning_request:
            native.update(reasoning_request)

        if _model_supports_thinking(model):
            template_kwargs: dict[str, object] = {"enable_thinking": use_thinking}
            if use_thinking and _is_qwen3(model):
                template_effort = qwen_template_effort(effort)
                if template_effort is not None:
                    template_kwargs["reasoning_effort"] = template_effort
            body["chat_template_kwargs"] = template_kwargs
            native["chat_template_kwargs"] = template_kwargs
            if use_thinking and _is_qwen3(model) and thinking_budget is not None:
                custom = {"thinking_budget": thinking_budget}
                body["custom_params"] = custom
                native["custom_params"] = custom

        t0 = time.monotonic()
        try:
            response = await self._client.post(
                f"{self._base_url}/chat/completions",
                json=body,
                headers=self._headers(),
                timeout=self._timeout,
            )
            response.raise_for_status()
        except httpx.ConnectError as exc:
            raise ConnectionError(
                f"Cannot connect to vLLM endpoint at {self._base_url} — is it running?"
            ) from exc

        latency_ms = round((time.monotonic() - t0) * 1000, 2)

        data = response.json()
        choice = data["choices"][0]
        text = choice["message"]["content"] or ""
        usage = data.get("usage", {})

        inp = usage.get("prompt_tokens", 0)
        out = usage.get("completion_tokens", 0)
        total = usage.get("total_tokens", inp + out)

        # OpenAI-compatible HTTP call to a local gateway: latency is end-to-end
        # (network + queue + decode), so throughput is reported on wall-clock basis.
        tps = compute_tps(out, latency_ms / 1000)

        thinking_text = ""
        thinking_tokens = 0
        if use_thinking and _model_supports_thinking(model):
            thinking_content = choice.get("message", {}).get("reasoning_content", "")
            if thinking_content:
                thinking_text = thinking_content
            thinking_tokens = _usage_reasoning_tokens(usage, thinking_text, text, out)

        return ProviderResponse(
            text=text,
            input_tokens=inp,
            output_tokens=out,
            total_tokens=total,
            model=model,
            latency_ms=latency_ms,
            estimated_cost_usd=0.0,
            tokens_per_second=tps,
            thinking_tokens=thinking_tokens,
            thinking_text=thinking_text,
            raw=data,
            effort=normalize_effort(effort),
            reasoning_request=native or None,
        )

    async def list_models(self) -> list[dict[str, str | float | bool]]:
        """Fetch available models from the OpenAI-compatible /v1/models endpoint."""
        try:
            response = await self._client.get(
                f"{self._base_url}/models",
                headers=self._headers(),
                timeout=10.0,
            )
            response.raise_for_status()
        except httpx.ConnectError as exc:
            raise ConnectionError(
                f"Cannot connect to vLLM endpoint at {self._base_url} — is it running?"
            ) from exc

        data = response.json()
        results: list[dict[str, str | float | bool]] = []
        for entry in data.get("data", []):
            name: str = entry.get("id", "")
            results.append(
                {
                    "name": name,
                    "size_gb": 0.0,
                    "parameter_size": "",
                    "family": name.split(":")[0] if ":" in name else name.split("/")[-1],
                    "model_class": classify_model(name).value,
                    "thinking": supports_thinking(name),
                }
            )
        return results

    async def health_check(self) -> bool:
        try:
            response = await self._client.get(
                f"{self._base_url}/models",
                headers=self._headers(),
                timeout=5.0,
            )
            return response.status_code == 200
        except Exception:
            return False
