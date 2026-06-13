"""Ollama (local inference) provider adapter."""

from __future__ import annotations

import re

import httpx

from atomics.model_classes import classify_model, supports_thinking
from atomics.providers.base import BaseProvider, ProviderResponse, compute_tps

_THINK_TAG_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)
_THINKING_MODEL_PREFIXES = ("qwen3", "deepseek-r1")


def _model_supports_thinking(model: str) -> bool:
    return any(model.startswith(p) for p in _THINKING_MODEL_PREFIXES)


def _strip_thinking(text: str) -> tuple[str, str]:
    """Separate <think>...</think> blocks from the visible answer."""
    thinking_parts: list[str] = []
    def _collect(m: re.Match) -> str:
        thinking_parts.append(m.group(1).strip())
        return ""
    clean = _THINK_TAG_RE.sub(_collect, text).strip()
    return clean, "\n\n".join(thinking_parts)


class OllamaProvider(BaseProvider):
    def __init__(
        self,
        host: str = "http://localhost:11434",
        default_model: str = "qwen2.5:7b",
        *,
        client: object | None = None,
    ) -> None:
        self._host = host.rstrip("/")
        self._default_model = default_model
        self._client = client or httpx.AsyncClient()

    @property
    def name(self) -> str:
        return "ollama"

    async def generate(
        self,
        prompt: str,
        *,
        system: str = "",
        model: str | None = None,
        max_tokens: int = 1024,
        thinking: bool | None = None,
        thinking_budget: int | None = None,
    ) -> ProviderResponse:
        model = model or self._default_model

        use_thinking = thinking if thinking is not None else _model_supports_thinking(model)

        options: dict = {}
        if thinking_budget and use_thinking:
            options["num_predict"] = max_tokens + thinking_budget
        if not use_thinking and _model_supports_thinking(model):
            prompt = "/no_think " + prompt

        body: dict = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "system": system or "You are a helpful assistant.",
        }
        if options:
            body["options"] = options

        try:
            response = await self._client.post(
                f"{self._host}/api/generate",
                json=body,
                timeout=120.0,
            )
            response.raise_for_status()
        except httpx.ConnectError as exc:
            raise ConnectionError(
                f"Cannot connect to Ollama at {self._host} — is it running?"
            ) from exc

        data = response.json()
        raw_text = data.get("response", "")
        out = data.get("eval_count", 0)
        inp = data.get("prompt_eval_count", 0)

        thinking_text = ""
        if use_thinking and "<think>" in raw_text:
            text, thinking_text = _strip_thinking(raw_text)
        else:
            text = raw_text

        # Ollama exposes pure decode time (eval_duration, nanoseconds), so its
        # throughput is reported on the "generation" basis rather than wall-clock.
        eval_duration = data.get("eval_duration", 0)
        tps = compute_tps(out, eval_duration / 1e9) if eval_duration else None

        total_duration = data.get("total_duration", 0)
        latency = total_duration / 1e6 if total_duration else 0.0

        # Ollama reports total generated tokens (eval_count) but no separate
        # count for the <think> reasoning span. Estimate the reasoning share by
        # character proportion of the generated text so the figure stays anchored
        # to the real token total rather than an unanchored word count.
        thinking_tokens = 0
        if thinking_text and out > 0:
            generated_chars = len(thinking_text) + len(text)
            if generated_chars > 0:
                thinking_tokens = round(out * len(thinking_text) / generated_chars)

        return ProviderResponse(
            text=text,
            input_tokens=inp,
            output_tokens=out,
            total_tokens=inp + out,
            model=model,
            latency_ms=round(latency, 2),
            estimated_cost_usd=0.0,
            tokens_per_second=tps,
            tps_basis="generation",
            thinking_tokens=thinking_tokens,
            thinking_text=thinking_text,
            raw=data,
        )

    async def list_models(self) -> list[dict[str, str | float | bool]]:
        """Fetch available models from Ollama and annotate with class/thinking metadata."""
        try:
            response = await self._client.get(
                f"{self._host}/api/tags",
                timeout=10.0,
            )
            response.raise_for_status()
        except httpx.ConnectError as exc:
            raise ConnectionError(
                f"Cannot connect to Ollama at {self._host} — is it running?"
            ) from exc

        data = response.json()
        results: list[dict[str, str | float | bool]] = []
        for entry in data.get("models", []):
            name: str = entry.get("name", "")
            size_bytes: int = entry.get("size", 0)
            details: dict = entry.get("details", {})
            results.append({
                "name": name,
                "size_gb": round(size_bytes / 1e9, 1),
                "parameter_size": details.get("parameter_size", ""),
                "family": details.get("family", ""),
                "model_class": classify_model(name).value,
                "thinking": supports_thinking(name),
            })
        return results

    async def health_check(self) -> bool:
        try:
            response = await self._client.get(
                f"{self._host}/api/tags",
                timeout=5.0,
            )
            return response.status_code == 200
        except Exception:
            return False
