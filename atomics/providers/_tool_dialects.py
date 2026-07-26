"""Tool request/response translation for each provider API dialect.

Three dialects cover ten providers: OpenAI's `chat/completions` shape (openai,
vllm, llamacpp, groq, together, gemini), Anthropic's (claude), and Ollama's
`/api/chat`. Keeping the translation here means one implementation per dialect
rather than one per provider.

Catalog schemas are stored in OpenAI function shape — name, description,
parameters — because six of the ten providers consume it directly.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from atomics.providers.toolcalls import ToolCall, parse_arguments


def _field(block: Any, name: str, default: Any = None) -> Any:
    """Read `name` from a dict or an SDK object.

    The Anthropic SDK returns content blocks as objects with attributes, while
    tests and raw JSON paths supply dicts. A dict-only reader passes unit tests
    and silently returns nothing in production.
    """
    if isinstance(block, dict):
        return block.get(name, default)
    return getattr(block, name, default)


def openai_tool_payload(schemas: list[dict]) -> list[dict]:
    """Wrap catalog schemas in OpenAI's function envelope."""
    return [{"type": "function", "function": schema} for schema in schemas]


def parse_openai_tool_calls(message: dict[str, Any]) -> tuple[ToolCall, ...]:
    """Extract tool calls from an OpenAI-shaped assistant message."""
    raw_calls = message.get("tool_calls") or []
    calls: list[ToolCall] = []
    for entry in raw_calls:
        function = entry.get("function") or {}
        name = function.get("name") or ""
        if not name:
            continue
        arguments, malformed = parse_arguments(function.get("arguments"))
        calls.append(
            ToolCall(name=name, arguments=arguments, malformed=malformed, raw=entry)
        )
    return tuple(calls)


def parse_ollama_tool_calls(message: dict[str, Any]) -> tuple[ToolCall, ...]:
    """Extract tool calls from an Ollama /api/chat message.

    Same envelope as OpenAI, but `arguments` arrives as an object rather than a
    JSON string. `parse_arguments` handles that via its dict branch, so this is
    a deliberate alias rather than a copy — named separately because the
    difference is real and a future divergence in Ollama's shape should have an
    obvious place to land.
    """
    return parse_openai_tool_calls(message)


def anthropic_tool_payload(schemas: list[dict]) -> list[dict]:
    """Translate catalog schemas into Anthropic's tool shape.

    Anthropic names the schema key `input_schema` where OpenAI uses `parameters`.
    """
    return [
        {
            "name": schema["name"],
            "description": schema.get("description", ""),
            "input_schema": schema.get(
                "parameters", {"type": "object", "properties": {}}
            ),
        }
        for schema in schemas
    ]


def parse_anthropic_tool_calls(content_blocks: Sequence[Any]) -> tuple[ToolCall, ...]:
    """Extract tool calls from Anthropic's content-block list.

    Anthropic returns `tool_use` blocks whose `input` is already parsed, so
    `parse_arguments` takes the dict branch here. Blocks may be dicts or SDK
    objects; see `_field`.
    """
    calls: list[ToolCall] = []
    for block in content_blocks or []:
        if _field(block, "type") != "tool_use":
            continue
        name = _field(block, "name") or ""
        if not name:
            continue
        arguments, malformed = parse_arguments(_field(block, "input"))
        calls.append(
            ToolCall(
                name=name,
                arguments=arguments,
                malformed=malformed,
                raw=block if isinstance(block, dict) else None,
            )
        )
    return tuple(calls)


def anthropic_text(content_blocks: Sequence[Any]) -> str:
    """Join the text blocks of an Anthropic response, ignoring tool_use blocks."""
    return "".join(
        _field(block, "text", "") or ""
        for block in content_blocks or []
        if _field(block, "type") == "text"
    )
