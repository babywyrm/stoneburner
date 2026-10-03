"""Live checks on one model: does it answer, how fast, does it call a tool."""

from __future__ import annotations

from atomics.eval.toolcall.runner import probe_tool_capability
from atomics.inventory import ModelRecord, ProbeResult, Reply
from atomics.providers.base import BaseProvider
from atomics.validation import sanitize_error

# Long enough that tokens per second measures decoding, not one token's overhead.
ANSWER_PROMPT = "Name the capital of France, then describe it in two sentences."
_ANSWER_MAX_TOKENS = 256


async def _reply(provider: BaseProvider, model: str, *, thinking: bool) -> Reply:
    resp = await provider.generate(
        ANSWER_PROMPT, model=model, max_tokens=_ANSWER_MAX_TOKENS, thinking=thinking
    )
    return Reply(
        answered=bool(resp.text.strip()),
        outcome=resp.outcome.kind.value if resp.outcome else "completed",
        output_tokens=resp.output_tokens,
        thinking_tokens=resp.thinking_tokens,
        latency_ms=resp.latency_ms,
        tokens_per_second=resp.tokens_per_second,
    )


async def probe_model(provider: BaseProvider, record: ModelRecord) -> None:
    """Probe one model in place. A failed step is recorded, never raised."""
    result = record.probe = ProbeResult()
    try:
        result.off = await _reply(provider, record.name, thinking=False)
    except Exception as exc:
        record.errors.append(f"probe: {sanitize_error(exc)}")
        return
    record.probed["completion"] = result.off.answered
    if record.capability("tools").value is not False:
        # ponytail: probe_tool_capability also returns False when the provider
        # errors, leaving only its warning log. Return the error from it if the
        # inventory needs to tell "never calls" from "call failed".
        record.probed["tools"] = await probe_tool_capability(
            provider, model=record.name, thinking=False
        )
