"""Live checks on one model: does it answer, how fast, does it call a tool."""

from __future__ import annotations

from atomics.eval.judge import score_response
from atomics.eval.redblue.fixtures import BLUE_FIXTURES
from atomics.eval.toolcall.runner import probe_tool_capability
from atomics.inventory import JudgeFitness, ModelRecord, ProbeResult, Reply, Verdict
from atomics.providers.base import BaseProvider
from atomics.validation import sanitize_error

# Long enough that tokens per second measures decoding, not one token's overhead.
ANSWER_PROMPT = "Name the capital of France, then describe it in two sentences."
# Room for a reasoner to finish.
_ANSWER_MAX_TOKENS = 1024
_CAPPED = frozenset({"truncated", "thinking_budget"})
# ponytail: a two-sentence answer runs 40-70 tokens. Past this, or capped, the
# model was reasoning in visible text whatever the host's parser separated.
# Hosts disagree on tagging (phi4: 565 untagged on Ollama 0.34, split on 0.32).
_BRIEF_MAX_TOKENS = 256
_INLINE_MIN_TOKENS = 64


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


def _reasoned(off: Reply) -> bool:
    """Thinking was off, and the model reasoned anyway."""
    return bool(
        off.thinking_tokens or off.outcome in _CAPPED or off.output_tokens > _BRIEF_MAX_TOKENS
    )


def thinking_verdict(off: Reply, on: Reply) -> Verdict:
    """How a model treats the thinking switch, from one reply each way."""
    if _reasoned(off):
        return "off-ignored"
    if on.thinking_tokens:
        return "off-works"
    # ponytail: "inline" is a length heuristic. Thinking-on wrote at least twice
    # the visible tokens of thinking-off, and at least 64. Reading the text for
    # reasoning is the upgrade if it misfiles terse thinkers.
    if on.output_tokens >= max(_INLINE_MIN_TOKENS, 2 * off.output_tokens):
        return "inline"
    return "no-channel"


async def probe_model(provider: BaseProvider, record: ModelRecord) -> None:
    """Probe one model in place. A failed step is recorded, never raised."""
    result = record.probe = ProbeResult()
    try:
        off = result.off = await _reply(provider, record.name, thinking=False)
    except Exception as exc:
        record.errors.append(f"probe: {sanitize_error(exc)}")
        return
    if _reasoned(off) or record.claims("thinking"):
        try:
            result.on = await _reply(provider, record.name, thinking=True)
        except Exception as exc:
            record.errors.append(f"probe thinking: {sanitize_error(exc)}")
        else:
            result.verdict = thinking_verdict(off, result.on)
            record.probed["thinking"] = result.verdict in ("off-works", "off-ignored")
    record.probed["completion"] = off.answered or bool(result.on and result.on.answered)
    if record.capability("tools").value is not False:
        try:
            record.probed["tools"] = await probe_tool_capability(
                provider, model=record.name, thinking=result.recommended == "--thinking"
            )
        except Exception as exc:
            record.errors.append(f"probe tools: {sanitize_error(exc)}")


# rb-b03 asks for Dockerfile hardening. The good answer covers every gold
# criterion; the bad one is confidently wrong.
JUDGE_FIXTURE_ID = "rb-b03"

GOOD_ANSWER = """\
Security issues:
1. It runs as root: there is no USER instruction.
2. The base image tag python:3.11 is unpinned; pin it by digest.
3. COPY . . copies everything into the image, including .env files and secrets.
4. The full python image carries compilers and tools; use a multi-stage build on a slim base.
5. pip keeps its cache and installs unpinned packages; use --no-cache-dir and --require-hashes.

Add a .dockerignore that excludes .env, .git, and tests, and copy only what runs.

```dockerfile
FROM python:3.11-slim AS build
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir --require-hashes --target /deps -r requirements.txt

FROM python:3.11-slim
RUN useradd --create-home appuser
WORKDIR /app
COPY --from=build /deps /deps
COPY app.py .
ENV PYTHONPATH=/deps
USER appuser
EXPOSE 8080
CMD ["python", "app.py"]
```
"""

BAD_ANSWER = (
    "This Dockerfile is already production-ready. The only change I would make is "
    "to expose port 80 instead of 8080 so users do not have to type a port."
)


async def probe_judge(provider: BaseProvider, record: ModelRecord) -> None:
    """Ask the model to grade a right and a wrong answer. Fit: both parse, right wins."""
    fixture = next(f for f in BLUE_FIXTURES if f.id == JUDGE_FIXTURE_ID)
    scores: list[float | None] = []
    try:
        for answer in (GOOD_ANSWER, BAD_ANSWER):
            result = await score_response(
                fixture.prompt,
                answer,
                judge_provider=provider,
                judge_model=record.name,
                gold_criteria=fixture.gold_criteria,
            )
            scores.append(None if result.parse_failed else result.score)
    except Exception as exc:
        record.errors.append(f"judge: {sanitize_error(exc)}")
        return
    record.judge = JudgeFitness(good=scores[0], bad=scores[1])
