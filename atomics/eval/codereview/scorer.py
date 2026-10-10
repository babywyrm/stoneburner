"""Typed judge scoring for secure code review."""

from __future__ import annotations

import re
from dataclasses import dataclass

from atomics.eval.codereview.fixtures import SecureCodeFixture
from atomics.eval.outcomes import (
    JudgeCallResult,
    JudgeOutcome,
    JudgeOutcomeStatus,
)
from atomics.providers.base import BaseProvider, ProviderResponse

_SYSTEM = "You are grading a security code review against ground truth. Be strict and objective."
_VULNERABLE_TEMPLATE = """\
GROUND TRUTH: the reviewed code contains this vulnerability:
  {cwe} (severity {severity})
Key points a correct review should surface:
{findings}

THE REVIEWER WROTE:
{review}

Did the reviewer correctly identify the planted vulnerability (its class/root cause)?
Reply ONLY in this format:
VERDICT: <DETECTED|MISSED>
RATIONALE: <one concise sentence>
"""
_CLEAN_TEMPLATE = """\
GROUND TRUTH: the reviewed code is SECURE — there is no real vulnerability.

THE REVIEWER WROTE:
{review}

Did the reviewer incorrectly claim a real, exploitable vulnerability exists
(a false positive)? Minor style/defensive suggestions do NOT count as a false
positive — only a claimed real security vulnerability does.
Reply ONLY in this format:
VERDICT: <CLEAN|FALSE_POSITIVE>
RATIONALE: <one concise sentence>
"""
_VULNERABLE_RE = re.compile(r"VERDICT:\s*(DETECTED|MISSED)", re.IGNORECASE)
_CLEAN_RE = re.compile(r"VERDICT:\s*(CLEAN|FALSE_POSITIVE)", re.IGNORECASE)
_RATIONALE_RE = re.compile(r"RATIONALE:\s*(.+)", re.IGNORECASE)
_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


@dataclass(frozen=True)
class ReviewVerdictResult:
    """One typed code-review grading operation."""

    verdict: str
    rationale: str
    judge_model: str
    status: JudgeOutcomeStatus
    calls: tuple[JudgeCallResult, ...]


def verdict_to_judge_outcome(result: ReviewVerdictResult) -> JudgeOutcome:
    """Convert a code-review verdict into the shared judge contract."""
    score = (
        _score_for_verdict(result.verdict) if result.status is JudgeOutcomeStatus.SCORED else None
    )
    return JudgeOutcome(
        status=result.status,
        score=score,
        label=result.verdict if result.verdict != "unknown" else None,
        rationale=result.rationale,
        judge_model=result.judge_model,
        judge_scores=(score,) if score is not None else (),
        judge_cost_usd=sum(call.estimated_cost_usd for call in result.calls),
        calls=result.calls,
        judges_expected=1,
        judges_scored=1 if score is not None else 0,
    )


async def judge_review(
    fixture: SecureCodeFixture,
    review: str,
    *,
    judge_provider: BaseProvider,
    judge_model: str | None,
) -> ReviewVerdictResult:
    """Grade one review while retaining every judge call."""
    prompt, pattern = _judge_prompt(fixture, review)
    calls: list[JudgeCallResult] = []
    for thinking, max_tokens in ((False, 192), (True, 448)):
        try:
            response = await judge_provider.generate(
                prompt,
                system=_SYSTEM,
                model=judge_model,
                max_tokens=max_tokens,
                temperature=0.0,
                thinking=thinking,
                thinking_budget=200 if thinking else None,
            )
        except Exception as exc:
            calls.append(JudgeCallResult.from_error(exc, judge_model or judge_provider.name))
            return ReviewVerdictResult(
                verdict="unknown",
                rationale="Judge provider call failed.",
                judge_model=judge_model or judge_provider.name,
                status=JudgeOutcomeStatus.PROVIDER_ERROR,
                calls=tuple(calls),
            )

        raw = _effective_text(response)
        verdict, rationale = _parse_verdict(raw, pattern)
        status = (
            JudgeOutcomeStatus.SCORED if verdict != "unknown" else JudgeOutcomeStatus.PARSE_FAILED
        )
        score = _score_for_verdict(verdict) if verdict != "unknown" else None
        calls.append(
            JudgeCallResult.from_response(
                response,
                status=status,
                judge_model=judge_model or response.model,
                effective_text=raw,
                score=score,
                label=verdict if score is not None else None,
                rationale=rationale,
            )
        )
        if verdict != "unknown":
            return ReviewVerdictResult(
                verdict=verdict,
                rationale=rationale,
                judge_model=judge_model or response.model,
                status=JudgeOutcomeStatus.SCORED,
                calls=tuple(calls),
            )
    return ReviewVerdictResult(
        verdict="unknown",
        rationale="Judge output did not contain a valid verdict.",
        judge_model=judge_model or judge_provider.name,
        status=JudgeOutcomeStatus.PARSE_FAILED,
        calls=tuple(calls),
    )


def _judge_prompt(
    fixture: SecureCodeFixture,
    review: str,
) -> tuple[str, re.Pattern[str]]:
    if fixture.is_vulnerable:
        findings = "\n".join(f"  - {finding}" for finding in fixture.expected_findings)
        return (
            _VULNERABLE_TEMPLATE.format(
                cwe=fixture.cwe,
                severity=fixture.severity,
                findings=findings,
                review=review[:3000],
            ),
            _VULNERABLE_RE,
        )
    return _CLEAN_TEMPLATE.format(review=review[:3000]), _CLEAN_RE


def _parse_verdict(
    raw: str,
    pattern: re.Pattern[str],
) -> tuple[str, str]:
    match = pattern.search(raw)
    if match is None:
        return "unknown", ""
    rationale_match = _RATIONALE_RE.search(raw)
    rationale = rationale_match.group(1).strip() if rationale_match else ""
    return match.group(1).lower(), rationale


def _score_for_verdict(verdict: str) -> float:
    return 1.0 if verdict in {"detected", "clean"} else 0.0


def _effective_text(response: ProviderResponse) -> str:
    visible = _THINK_BLOCK_RE.sub("", response.text).strip()
    return visible or response.thinking_text.strip()

