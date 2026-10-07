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
from dataclasses import dataclass

from rich import box
from rich.console import Console
from rich.table import Table
from rich.text import Text

from atomics.providers.base import BaseProvider, ChatMessage, ProviderResponse
from atomics.providers.outcomes import ProviderOutcomeKind

_console: Console | None = None
_calls = itertools.count(1)


@dataclass
class _Row:
    number: int
    label: str
    judge: bool
    prompt: str
    input_tokens: int | None
    output_tokens: int | None
    num_ctx: int | None
    note: str


_rows: list[_Row] = []


def enable(console: Console | None = None) -> None:
    global _console
    _console = console or Console(stderr=True, highlight=False)
    _rows.clear()


def disable() -> None:
    global _console
    _console = None


def traced(provider: BaseProvider) -> BaseProvider:
    """Wrap `provider` when `--show-prompt` is on; otherwise return it unchanged."""
    if _console is None or isinstance(provider, TracedProvider):
        return provider
    return TracedProvider(provider)


def print_timeline() -> None:
    """One row per traced call, so growth across a run reads at a glance."""
    console = _console
    if console is None or not _rows:
        return
    table = Table(
        title=f"Context timeline · {len(_rows)} calls", title_justify="left", box=box.SIMPLE
    )
    for column in ("#", "model", "prompt", "input", "Δ", "output", "context"):
        right = column in ("#", "input", "Δ", "output")
        table.add_column(
            column,
            justify="right" if right else "left",
            no_wrap=column != "model",
            overflow="fold" if column == "model" else "ellipsis",
            min_width=10 if column == "model" else None,
        )
    previous: dict[tuple[str, bool], int] = {}
    peak = 0.0
    for row in _rows:
        delta = ""
        key = (row.label, row.judge)
        if row.input_tokens is not None:
            if key in previous:
                delta = f"{row.input_tokens - previous[key]:+d}"
            previous[key] = row.input_tokens
        context = "—"
        if row.num_ctx and row.input_tokens:
            fill = (row.input_tokens + (row.output_tokens or 0)) / row.num_ctx
            peak = max(peak, fill)
            context = f"{fill:5.1%} " + "█" * min(10, round(fill * 10))
        table.add_row(
            str(row.number),
            Text(row.label),
            Text(row.prompt),
            "—" if row.input_tokens is None else str(row.input_tokens),
            delta,
            "—" if row.output_tokens is None else str(row.output_tokens) + row.note,
            context,
        )
    total_in = sum(r.input_tokens or 0 for r in _rows)
    total_out = sum(r.output_tokens or 0 for r in _rows)
    table.caption = (
        f"input {total_in} · output {total_out}"
        + (f" · peak context {peak:.1%}" if peak else "")
        + " · Δ is input against the same model's previous call in the same role"
        + " · * = cut off at max_tokens"
    )
    table.caption_justify = "left"
    console.print(table)


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

    async def generate_chat(
        self,
        messages: Sequence[ChatMessage],
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
        call = self._inner.generate_chat(
            messages,
            system=system,
            model=model,
            max_tokens=max_tokens,
            thinking=thinking,
            thinking_budget=thinking_budget,
            temperature=temperature,
            effort=effort,
            reasoning_mode=reasoning_mode,
        )
        parts = {"system": system, "prompt": messages[-1]["content"]}
        return await self._shown(
            call, model, parts, None, max_tokens, thinking, history=messages[:-1]
        )

    async def _shown(
        self,
        call: object,
        model: str | None,
        parts: dict[str, str],
        tools: Sequence[dict] | None,
        max_tokens: int,
        thinking: bool | None,
        *,
        history: Sequence[ChatMessage] = (),
    ) -> ProviderResponse:
        number = next(_calls)
        try:
            response: ProviderResponse = await call  # type: ignore[misc]
        except Exception as exc:
            self._print(number, model, parts, tools, max_tokens, thinking, None, exc, history)
            raise
        self._print(number, model, parts, tools, max_tokens, thinking, response, None, history)
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
        history: Sequence[ChatMessage] = (),
    ) -> None:
        from atomics.prompts import entries_for_system

        console = _console
        if console is None:
            return
        system = parts["system"]
        entries = entries_for_system(system)
        if entries:
            source = "built-in " + ", ".join(f"{e.name} @{e.fingerprint}" for e in entries)
        else:
            source = "custom" if system else "none"
        label = f"{self._inner.name} · {model or self._inner.default_model or 'default model'}"
        out = Text()
        out.append(f"system  [{source}]  ≈{_est(system)} tok\n", style="bold")
        if system:
            out.append(system + "\n", style="cyan")
        history_est = sum(_est(m["content"]) for m in history)
        if history:
            out.append(
                f"history  {len(history)} earlier messages  ≈{history_est} tok\n", style="bold"
            )
            for message in history:
                out.append(f"[{message['role']}] ", style="magenta")
                out.append(message["content"] + "\n")
        heading = "new message" if history else "prompt"
        out.append(f"{heading}  ≈{_est(parts['prompt'])} tok\n", style="bold")
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
            estimated = sum(_est(v) for v in parts.values()) + history_est
            overhead = response.input_tokens - estimated
            out.append(f"usage   input={response.input_tokens}", style="bold")
            if response.input_tokens and overhead > 0:
                out.append(f" (≈{estimated} text + ≈{overhead} template/formatting)")
            elif response.input_tokens:
                out.append(f" (≈{estimated} text estimated)")
            out.append(f"  output={response.output_tokens}")
            if response.thinking_tokens:
                out.append(f" ({response.thinking_tokens} of it thinking)")
            out.append("\n")
            kind = response.outcome.kind if response.outcome else None
            if kind == ProviderOutcomeKind.TRUNCATED or (
                kind is None and response.output_tokens >= max_tokens
            ):
                out.append(
                    f"outcome cut off at max_tokens={max_tokens}: the reply is unfinished\n",
                    style="yellow",
                )
            elif kind is not None and kind != ProviderOutcomeKind.COMPLETED:
                out.append(f"outcome {kind.value}\n", style="yellow")
            if num_ctx and response.input_tokens:
                used = response.input_tokens + response.output_tokens
                out.append(
                    f"context {used}/{num_ctx} tokens ({used / num_ctx:.1%} of the window)\n"
                )
        prompt_name = ", ".join(e.name for e in entries) or ("custom" if system else "none")
        note = ""
        if error is not None:
            prompt_name += f" ({type(error).__name__})"
        elif response is not None and "cut off" in out.plain:
            note = "*"
        _rows.append(
            _Row(
                number,
                model or self._inner.default_model or "default model",
                any(e.role == "judge" for e in entries),
                prompt_name,
                response.input_tokens if response else None,
                response.output_tokens if response else None,
                num_ctx,
                note,
            )
        )
        console.rule(Text(f"call {number} · {label}"), style="dim")
        # Wrapping would insert line breaks that are not in the prompt.
        console.print(out, end="", soft_wrap=True)
