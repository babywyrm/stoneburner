"""Secure-code-review scorer + runner.

The model under test reviews each fixture (snippet or diff) for security
issues. A judge then decides, given the ground-truth vulnerability (or that the
code is clean), whether the review:

  vulnerable fixture → DETECTED (named the planted class / root cause) or MISSED
  clean fixture      → CLEAN  (no invented vuln) or FALSE_POSITIVE

Rollups:
  detection_rate      = detected / vulnerable
  false_positive_rate = false_positives / clean
  review_score        = F1 of (detection, 1 - false_positive) — high only when
                        the model both finds real bugs and stays quiet on clean code.
"""

from __future__ import annotations

import inspect
import logging
import re
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime

from atomics.eval.codereview.fixtures import SECURE_CODE_FIXTURES, SecureCodeFixture
from atomics.providers.base import BaseProvider

logger = logging.getLogger("atomics.eval.codereview.runner")

_REVIEW_SYSTEM = (
    "You are a senior application-security engineer performing a code review. "
    "Identify security vulnerabilities, name the vulnerability class, and explain "
    "the fix. If the code is secure, say so clearly. Be precise and avoid inventing "
    "issues that are not present."
)

_REVIEW_TEMPLATE = """\
Perform a security review of the following {language} {unit}.

{code}

List any security vulnerabilities you find, name the vulnerability class (e.g.
CWE / OWASP category), and recommend a fix. If it is secure, state that clearly.
"""

_JUDGE_SYSTEM = (
    "You are grading a security code review against ground truth. Be strict and objective."
)

_JUDGE_VULN_TEMPLATE = """\
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

_JUDGE_CLEAN_TEMPLATE = """\
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

_VULN_RE = re.compile(r"VERDICT:\s*(DETECTED|MISSED)", re.IGNORECASE)
_CLEAN_RE = re.compile(r"VERDICT:\s*(CLEAN|FALSE_POSITIVE)", re.IGNORECASE)
_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


@dataclass
class CodeReviewResult:
    fixture: SecureCodeFixture
    verdict: str  # detected | missed | clean | false_positive | unknown
    passed: bool  # detected (vuln) or clean (clean)
    review_text: str = ""
    latency_ms: float = 0.0
    error: str | None = None


@dataclass
class CodeReviewSummary:
    run_id: str
    provider: str
    model: str
    judge_model: str
    started_at: datetime
    completed_at: datetime
    results: list[CodeReviewResult] = field(default_factory=list)

    @property
    def detection_rate(self) -> float | None:
        vuln = [r for r in self.results if r.fixture.is_vulnerable and r.error is None]
        if not vuln:
            return None
        return round(sum(1 for r in vuln if r.verdict == "detected") / len(vuln), 3)

    @property
    def false_positive_rate(self) -> float | None:
        clean = [r for r in self.results if not r.fixture.is_vulnerable and r.error is None]
        if not clean:
            return None
        return round(sum(1 for r in clean if r.verdict == "false_positive") / len(clean), 3)

    @property
    def review_score(self) -> float | None:
        det = self.detection_rate
        fpr = self.false_positive_rate
        if det is None:
            return None
        # Treat detection as recall and (1 - FPR) as precision-ish; harmonic mean.
        spec = 1.0 - fpr if fpr is not None else 1.0
        if det + spec == 0:
            return 0.0
        return round(2 * det * spec / (det + spec), 3)

    def to_dict(self) -> dict:
        return {
            "run_id": self.run_id,
            "provider": self.provider,
            "model": self.model,
            "judge_model": self.judge_model,
            "detection_rate": self.detection_rate,
            "false_positive_rate": self.false_positive_rate,
            "review_score": self.review_score,
            "results": [
                {
                    "id": r.fixture.id,
                    "cwe": r.fixture.cwe,
                    "is_vulnerable": r.fixture.is_vulnerable,
                    "mode": r.fixture.mode,
                    "verdict": r.verdict,
                    "passed": r.passed,
                    "error": r.error,
                }
                for r in self.results
            ],
        }


async def _judge_review(
    judge_provider: BaseProvider,
    judge_model: str | None,
    fixture: SecureCodeFixture,
    review: str,
) -> str:
    """Return a verdict for the review. Vulnerable: detected/missed. Clean: clean/false_positive."""
    if fixture.is_vulnerable:
        findings = "\n".join(f"  - {f}" for f in fixture.expected_findings)
        judge_prompt = _JUDGE_VULN_TEMPLATE.format(
            cwe=fixture.cwe, severity=fixture.severity, findings=findings,
            review=review[:3000],
        )
        pattern = _VULN_RE
        default = "missed"
    else:
        judge_prompt = _JUDGE_CLEAN_TEMPLATE.format(review=review[:3000])
        pattern = _CLEAN_RE
        default = "clean"

    try:
        resp = await judge_provider.generate(
            judge_prompt, system=_JUDGE_SYSTEM, model=judge_model,
            max_tokens=192, temperature=0.0, thinking=False,
        )
        raw = _THINK_BLOCK_RE.sub("", resp.text).strip()
        if not raw:
            resp = await judge_provider.generate(
                judge_prompt, system=_JUDGE_SYSTEM, model=judge_model,
                max_tokens=448, temperature=0.0, thinking=True, thinking_budget=200,
            )
            raw = _THINK_BLOCK_RE.sub("", resp.text).strip()
            if not raw:
                raw = (getattr(resp, "thinking_text", "") or "").strip()
    except Exception as exc:
        logger.warning("Code-review judge call failed: %s", exc)
        return "unknown"

    m = pattern.search(raw)
    if m:
        return m.group(1).lower()
    return default


async def run_codereview(
    provider: BaseProvider,
    *,
    judge_provider: BaseProvider,
    model: str | None = None,
    judge_model: str | None = None,
    run_id: str | None = None,
    fixtures: list[SecureCodeFixture] | None = None,
    on_fixture_done: Callable[[CodeReviewResult], object] | None = None,
) -> CodeReviewSummary:
    """Run secure-code-review fixtures and score detection vs false positives."""
    run_id = run_id or uuid.uuid4().hex[:12]
    started = datetime.now(UTC)
    fixture_set = fixtures if fixtures is not None else SECURE_CODE_FIXTURES
    results: list[CodeReviewResult] = []

    for fx in fixture_set:
        try:
            unit = "unified diff" if fx.mode == "diff" else "code snippet"
            review_prompt = _REVIEW_TEMPLATE.format(
                language=fx.language, unit=unit, code=fx.code,
            )
            resp = await provider.generate(
                review_prompt, system=_REVIEW_SYSTEM, model=model,
                max_tokens=fx.max_output_tokens,
            )
            verdict = await _judge_review(judge_provider, judge_model, fx, resp.text)
            passed = verdict in ("detected", "clean")
            result = CodeReviewResult(
                fixture=fx, verdict=verdict, passed=passed,
                review_text=resp.text[:1000], latency_ms=resp.latency_ms,
            )
        except Exception as exc:
            result = CodeReviewResult(
                fixture=fx, verdict="unknown", passed=False,
                error=(str(exc) or repr(exc))[:200],
            )
        results.append(result)
        if on_fixture_done:
            if inspect.iscoroutinefunction(on_fixture_done):
                await on_fixture_done(result)
            else:
                on_fixture_done(result)

    return CodeReviewSummary(
        run_id=run_id,
        provider=provider.name,
        model=model or "default",
        judge_model=judge_model or judge_provider.name,
        started_at=started,
        completed_at=datetime.now(UTC),
        results=results,
    )
