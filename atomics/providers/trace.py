"""`--show-prompt`: print what each model call sends and how much it used.

Off unless the CLI calls `enable()`. When on, `make_provider` wraps every
provider it builds, so every suite and every judge is covered without each
runner opting in. Output goes to stderr so `--json` stdout stays clean.

Token counts from the provider are exact. The per-part split is an estimate
at four characters per token, and the remainder of the exact input count is
what the chat template, role markers, and tool formatting added.
"""

from __future__ import annotations

import itertools
import json
from collections.abc import Sequence

from rich.console import Console
from rich.text import Text

from atomics.providers.base import BaseProvider, ProviderResponse

_console: Console | None = None
_calls = itertools.count(1)


def enable(console: Console | None = None) -> None:
    global _console
    _console = console or Console(stderr=True, highlight=False)


def disable() -> None:
    global _console
    _console = None


def traced(provider: BaseProvider) -> BaseProvider:
    """Wrap `provider` when `--show-prompt` is on; otherwise return it unchanged."""
    if _console is None or isinstance(provider, TracedProvider):
        return provider
    return TracedProvider(provider)


def _est(text: str) -> int:
    return round(len(text) / 4) if text else 0


def _tool_names(tools: Sequence[dict]) -> str:
    names = [t.get("function", {}).get("name") or t.get("name") or "?" for t in tools]
    return ", ".join(names)


class TracedProvider(BaseProvider):
    def __init__(self, inner: BaseProvider) -> None:
        self._inner = inner

    @property
    def name(self) -> str:
        return self._inner.name

    @property
    def default_model(self) -> str | None:
        return self._inner.default_model

    @property
    def supports_tools(self) -> bool:  # type: ignore[override]
        return self._inner.supports_tools

    def __getattr__(self, name: str) -> object:
        if name == "_inner":
            raise AttributeError(name)
        return getattr(self._inner, name)

    async def aclose(self) -> None:
        await self._inner.aclose()

    async def health_check(self) -> bool:
        return await self._inner.health_check()

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
        call = self._inner.generate(
            prompt,
            system=system,
            model=model,
            max_tokens=max_tokens,
            thinking=thinking,
            thinking_budget=thinking_budget,
            temperature=temperature,
            effort=effort,
            reasoning_mode=reasoning_mode,
        )
        parts = {"system": system, "prompt": prompt}
        return await self._shown(call, model, parts, None, max_tokens, thinking)

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
        call = self._inner.generate_with_tools(
            prompt,
            tools=tools,
            system=system,
            model=model,
            max_tokens=max_tokens,
            injected_tool_output=injected_tool_output,
            thinking=thinking,
            thinking_budget=thinking_budget,
            effort=effort,
            reasoning_mode=reasoning_mode,
        )
        parts = {
            "system": system,
            "prompt": prompt,
            "tools": json.dumps(list(tools)),
            "tool output": injected_tool_output or "",
        }
        return await self._shown(call, model, parts, tools, max_tokens, thinking)

    async def _shown(
        self,
        call: object,
        model: str | None,
        parts: dict[str, str],
        tools: Sequence[dict] | None,
        max_tokens: int,
        thinking: bool | None,
    ) -> ProviderResponse:
        number = next(_calls)
        try:
            response: ProviderResponse = await call  # type: ignore[misc]
        except Exception as exc:
            self._print(number, model, parts, tools, max_tokens, thinking, None, exc)
            raise
        self._print(number, model, parts, tools, max_tokens, thinking, response, None)
        return response

    def _num_ctx(self) -> int | None:
        num_ctx = getattr(self._inner, "_num_ctx", None)
        value = num_ctx() if callable(num_ctx) else num_ctx
        return value if isinstance(value, int) else None

    def _print(
        self,
        number: int,
        model: str | None,
        parts: dict[str, str],
        tools: Sequence[dict] | None,
        max_tokens: int,
        thinking: bool | None,
        response: ProviderResponse | None,
        error: Exception | None,
    ) -> None:
        from atomics.prompts import names_for_system

        console = _console
        if console is None:
            return
        system = parts["system"]
        names = names_for_system(system)
        if names:
            source = "built-in " + ", ".join(names)
        else:
            source = "custom" if system else "none"
        label = f"{self._inner.name} · {model or self._inner.default_model or 'default model'}"
        out = Text()
        out.append(f"system  [{source}]  ≈{_est(system)} tok\n", style="bold")
        if system:
            out.append(system + "\n", style="cyan")
        out.append(f"prompt  ≈{_est(parts['prompt'])} tok\n", style="bold")
        out.append(parts["prompt"] + "\n")
        if tools is not None:
            out.append(f"tools   {_tool_names(tools)}  ≈{_est(parts['tools'])} tok\n", style="bold")
            if parts["tool output"]:
                out.append(f"tool output  ≈{_est(parts['tool output'])} tok\n", style="bold")
                out.append(parts["tool output"] + "\n")
        num_ctx = self._num_ctx()
        thinking_label = "provider default" if thinking is None else str(thinking).lower()
        out.append(
            f"request max_tokens={max_tokens}  thinking={thinking_label}"
            + (f"  num_ctx={num_ctx}" if num_ctx else "")
            + "\n",
            style="dim",
        )
        if error is not None:
            out.append(f"error   {type(error).__name__}\n", style="red")
        elif response is not None:
            estimated = sum(_est(v) for v in parts.values())
            overhead = response.input_tokens - estimated
            out.append(f"usage   input={response.input_tokens}", style="bold")
            if response.input_tokens and overhead > 0:
                out.append(f" (≈{estimated} text + ≈{overhead} template/formatting)")
            out.append(f"  output={response.output_tokens}")
            if response.thinking_tokens:
                out.append(f"  thinking={response.thinking_tokens}")
            out.append("\n")
            if num_ctx and response.input_tokens:
                used = response.input_tokens + response.output_tokens
                out.append(
                    f"context {used}/{num_ctx} tokens ({used / num_ctx:.1%} of the window)\n"
                )
        console.rule(Text(f"call {number} · {label}"), style="dim")
        console.print(out, end="")
