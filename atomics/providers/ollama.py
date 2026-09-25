"""Ollama (local inference) provider adapter."""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any

import httpx

from atomics.benchmark.model_classes import classify_model, supports_thinking
from atomics.providers._tool_dialects import (
    openai_tool_payload,
    parse_ollama_tool_calls,
)
from atomics.providers.base import BaseProvider, ProviderResponse, compute_tps
from atomics.providers.effort import normalize_effort, ollama_think_value
from atomics.providers.outcomes import ProviderOutcome, ProviderOutcomeKind

# Ollama uses the model's full window when num_ctx is omitted. granite4.2:30b
# at 131072 tokens allocated a 52 GiB runner on a 64 GiB machine. 8192 matches
# the Ollama judge context archreview already requests and fits the short
# batteries. Callers that pass context_tokens, including a larger window,
# still override this.
DEFAULT_NUM_CTX = 8192

_THINK_TAG_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)

# ponytail: one floor for every suite. A capped answer with fewer visible
# tokens than this after reasoning is a fragment, not a review. gpt-oss left
# 7 and 17; the shortest real review it wrote at the cap was 98. Per-fixture
# minimums are the upgrade if a suite needs a different floor.
_MIN_VISIBLE_TOKENS_AT_CAP = 64


def _capped_outcome(
    done_reason: object, text: str, out: int, thinking_tokens: int
) -> ProviderOutcome | None:
    """Say why a capped generation stopped. Other stops stay with the caller."""
    if done_reason != "length":
        return None
    if thinking_tokens and out - thinking_tokens < _MIN_VISIBLE_TOKENS_AT_CAP:
        return ProviderOutcome(ProviderOutcomeKind.THINKING_BUDGET, finish_reason="length")
    if text.strip():
        return ProviderOutcome(ProviderOutcomeKind.TRUNCATED, finish_reason="length")
    return None


def _is_think_field_400(exc: httpx.HTTPStatusError) -> bool:
    """True when Ollama rejected the request because of the think field."""
    resp = exc.response
    if resp is None or resp.status_code != 400:
        return False
    try:
        text = resp.text or ""
    except Exception:
        text = ""
    return "think" in text.lower()


def _model_supports_thinking(model: str) -> bool:
    return supports_thinking(model)


def _strip_thinking(text: str) -> tuple[str, str]:
    """Separate <think> blocks and orphan closers from the visible answer."""
    thinking_parts: list[str] = []

    def _collect(m: re.Match) -> str:
        thinking_parts.append(m.group(1).strip())
        return ""

    clean = _THINK_TAG_RE.sub(_collect, text).strip()
    closer = "</think>"
    if closer in clean:
        before, after = clean.rsplit(closer, 1)
        if before.strip():
            thinking_parts.append(before.strip())
        clean = after.strip()
    return clean, "\n\n".join(p for p in thinking_parts if p)


def _visible_and_thinking(raw_text: str, native: object, out: int) -> tuple[str, str, int]:
    """Split leaked CoT from visible text and estimate thinking tokens."""
    thinking_text = native.strip() if isinstance(native, str) else ""
    text, tagged = _strip_thinking(raw_text)
    if tagged:
        thinking_text = f"{thinking_text}\n\n{tagged}".strip() if thinking_text else tagged
    thinking_tokens = 0
    if thinking_text and out > 0:
        generated_chars = len(thinking_text) + len(text)
        if generated_chars > 0:
            thinking_tokens = round(out * len(thinking_text) / generated_chars)
    return text, thinking_text, thinking_tokens


class OllamaProvider(BaseProvider):
    supports_tools = True

    def __init__(
        self,
        host: str = "http://localhost:11434",
        default_model: str = "qwen2.5:7b",
        *,
        timeout: float = 300.0,
        context_tokens: int | None = None,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._host = host.rstrip("/")
        self._default_model = default_model
        self._timeout = timeout
        self._context_tokens = context_tokens
        self._client = client or httpx.AsyncClient()

    def _num_ctx(self) -> int:
        if self._context_tokens is None:
            return DEFAULT_NUM_CTX
        return self._context_tokens

    @property
    def name(self) -> str:
        return "ollama"

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
        effort: str | None = None,
        reasoning_mode: str | None = None,
    ) -> ProviderResponse:
        model = model or self._default_model

        auto = thinking if thinking is not None else _model_supports_thinking(model)
        think_field = ollama_think_value(thinking=auto, effort=effort, model=model)
        use_thinking = think_field is not False

        options: dict = {}
        if temperature is not None:
            options["temperature"] = temperature
        if max_tokens:
            options["num_predict"] = max_tokens
        options["num_ctx"] = self._num_ctx()
        if thinking_budget and use_thinking:
            options["num_predict"] = max_tokens + thinking_budget

        body: dict = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "system": system or "You are a helpful assistant.",
            "think": think_field,
        }
        if options:
            body["options"] = options

        think_fallback: str | None = None
        try:
            response = await self._client.post(
                f"{self._host}/api/generate",
                json=body,
                timeout=self._timeout,
            )
            response.raise_for_status()
        except httpx.ConnectError as exc:
            raise ConnectionError(
                f"Cannot connect to Ollama at {self._host} — is it running?"
            ) from exc
        except httpx.HTTPStatusError as exc:
            if think_field is not False and _is_think_field_400(exc):
                body["think"] = False
                response = await self._client.post(
                    f"{self._host}/api/generate",
                    json=body,
                    timeout=self._timeout,
                )
                response.raise_for_status()
                think_field = False
                think_fallback = "400"
            else:
                raise

        data = response.json()
        raw_text = data.get("response", "")
        out = data.get("eval_count", 0)
        inp = data.get("prompt_eval_count", 0)
        text, thinking_text, thinking_tokens = _visible_and_thinking(
            raw_text, data.get("thinking"), out
        )

        # Ollama exposes pure decode time (eval_duration, nanoseconds), so its
        # throughput is reported on the "generation" basis rather than wall-clock.
        eval_duration = data.get("eval_duration", 0)
        tps = compute_tps(out, eval_duration / 1e9) if eval_duration else None

        total_duration = data.get("total_duration", 0)
        latency = total_duration / 1e6 if total_duration else 0.0
        done_reason = data.get("done_reason")

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
            outcome=_capped_outcome(done_reason, text, out, thinking_tokens),
            finish_reason=done_reason if isinstance(done_reason, str) else None,
            effort=normalize_effort(effort),
            reasoning_request=(
                {"think": think_field, "think_fallback": think_fallback}
                if think_fallback
                else {"think": think_field}
            ),
        )

    async def generate_with_tools(
        self,
        prompt: str,
        *,
        tools: Sequence[dict],
        system: str = "",
        model: str | None = None,
        max_tokens: int = 1024,
        injected_tool_output: str | None = None,
        thinking: bool | None = None,
        thinking_budget: int | None = None,
        effort: str | None = None,
        reasoning_mode: str | None = None,
    ) -> ProviderResponse:
        """Tool-calling path, on /api/chat.

        Separate from generate() by design, not by accident. /api/generate has no
        tools support at all, and its response shape is what supplies
        eval_duration, the <think> text, and the token counts that generate()
        depends on. Routing both through /api/chat, or sharing the parsing
        between them, would silently change the throughput basis and token
        accounting for every Ollama figure in the project — including the
        published leaderboard. A test pins generate() to /api/generate.
        """
        del reasoning_mode
        model = model or self._default_model

        auto = thinking if thinking is not None else _model_supports_thinking(model)
        think_field = ollama_think_value(thinking=auto, effort=effort, model=model)
        use_thinking = think_field is not False

        messages: list[dict[str, Any]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        if injected_tool_output is not None:
            # Indirect injection: the attack arrives as the result of a tool the
            # model appears to have already called.
            messages.append(
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {"function": {"name": "list_files", "arguments": {"directory": "."}}}
                    ],
                }
            )
            messages.append({"role": "tool", "content": injected_tool_output})

        options: dict[str, Any] = {"num_predict": max_tokens, "num_ctx": self._num_ctx()}
        if thinking_budget and use_thinking:
            options["num_predict"] = max_tokens + thinking_budget

        body: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": False,
            "tools": openai_tool_payload(list(tools)),
            "options": options,
            "think": think_field,
        }

        think_fallback: str | None = None
        try:
            response = await self._client.post(
                f"{self._host}/api/chat",
                json=body,
                timeout=self._timeout,
            )
            response.raise_for_status()
        except httpx.ConnectError as exc:
            raise ConnectionError(
                f"Cannot connect to Ollama at {self._host} — is it running?"
            ) from exc
        except httpx.HTTPStatusError as exc:
            if think_field is not False and _is_think_field_400(exc):
                body["think"] = False
                response = await self._client.post(
                    f"{self._host}/api/chat",
                    json=body,
                    timeout=self._timeout,
                )
                response.raise_for_status()
                think_field = False
                think_fallback = "400"
            else:
                raise

        data = response.json()
        message = data.get("message") or {}
        out = data.get("eval_count", 0)
        inp = data.get("prompt_eval_count", 0)
        native = message.get("thinking")
        if not isinstance(native, str):
            native = data.get("thinking")
        text, thinking_text, thinking_tokens = _visible_and_thinking(
            message.get("content") or "", native, out
        )

        # /api/chat reports eval_duration too, so the generation basis carries
        # over and tool-path throughput stays comparable with generate().
        eval_duration = data.get("eval_duration", 0)
        total_duration = data.get("total_duration", 0)
        done_reason = data.get("done_reason")
        tool_calls = parse_ollama_tool_calls(message)

        return ProviderResponse(
            text=text,
            input_tokens=inp,
            output_tokens=out,
            total_tokens=inp + out,
            model=model,
            latency_ms=round(total_duration / 1e6, 2) if total_duration else 0.0,
            estimated_cost_usd=0.0,
            tokens_per_second=(compute_tps(out, eval_duration / 1e9) if eval_duration else None),
            tps_basis="generation",
            thinking_tokens=thinking_tokens,
            thinking_text=thinking_text,
            raw=data,
            outcome=(
                None
                if tool_calls
                else _capped_outcome(done_reason, text, out, thinking_tokens)
            ),
            finish_reason=done_reason if isinstance(done_reason, str) else None,
            tool_calls=tool_calls,
            effort=normalize_effort(effort),
            reasoning_request=(
                {"think": think_field, "think_fallback": think_fallback}
                if think_fallback
                else {"think": think_field}
            ),
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
            results.append(
                {
                    "name": name,
                    "size_gb": round(size_bytes / 1e9, 1),
                    "parameter_size": details.get("parameter_size", ""),
                    "family": details.get("family", ""),
                    "model_class": classify_model(name).value,
                    "thinking": supports_thinking(name),
                }
            )
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
