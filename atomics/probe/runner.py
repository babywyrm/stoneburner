"""Probe runner — fetch artifacts, build checks, judge with quality scorer."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field

from atomics.eval.judge import score_response
from atomics.probe.checks import build_check
from atomics.probe.config import ProbeTarget
from atomics.probe.connectors import ProbeConnectorError, fetch_artifact
from atomics.providers.base import BaseProvider

logger = logging.getLogger("atomics.probe.runner")


@dataclass
class ProbeResult:
    target_name: str
    artifact_type: str
    check_id: str
    score: float | None
    prev_score: float | None
    regressed: bool
    judge_model: str
    judge_rationale: str
    thinking_enabled: bool = False
    thinking_tokens: int = 0


@dataclass
class ProbeSummary:
    results: list[ProbeResult] = field(default_factory=list)

    @property
    def fixture_results(self) -> list[ProbeResult]:
        """Alias for `results` — the convergent name used across eval suites.

        See ARCHITECTURE.md "known divergences". New code should prefer
        `fixture_results`; `results` remains for back-compat.
        """
        return self.results

    @property
    def overall_score(self) -> float | None:
        scored = [r.score for r in self.results if r.score is not None]
        return round(sum(scored) / len(scored), 3) if scored else None

    def to_dict(self) -> dict:
        """Machine-readable view of the run for --json-out / dashboards / CI."""
        return {
            "overall_score": self.overall_score,
            "total_targets": len(self.results),
            "regressions": [r.target_name for r in self.regressions],
            "results": [
                {
                    "target_name": r.target_name,
                    "artifact_type": r.artifact_type,
                    "check_id": r.check_id,
                    "score": r.score,
                    "prev_score": r.prev_score,
                    "regressed": r.regressed,
                    "judge_model": r.judge_model,
                    "judge_rationale": r.judge_rationale,
                    "thinking_tokens": r.thinking_tokens,
                }
                for r in self.results
            ],
        }

    @property
    def regressions(self) -> list[ProbeResult]:
        return [r for r in self.results if r.regressed]


async def run_probe(
    provider: BaseProvider,
    *,
    judge_provider: BaseProvider,
    targets: list[ProbeTarget],
    model: str | None = None,
    judge_model: str | None = None,
    thinking: bool | None = None,
    thinking_budget: int | None = None,
    prev_scores: dict[str, float] | None = None,
    regression_threshold: float = 0.10,
    on_result: Callable[..., object] | None = None,
) -> ProbeSummary:
    """Run LLM probe checks against a list of configured artifact targets."""
    prev_scores = prev_scores or {}
    results: list[ProbeResult] = []

    for target in targets:
        logger.info("[probe] Fetching artifact '%s' (%s)", target.name, target.artifact_type)
        try:
            content = await fetch_artifact(target)
        except ProbeConnectorError as exc:
            logger.warning("[probe] Fetch failed for '%s': %s", target.name, exc)
            result = ProbeResult(
                target_name=target.name,
                artifact_type=target.artifact_type,
                check_id="fetch_error",
                score=None,
                prev_score=prev_scores.get(target.name),
                regressed=False,
                judge_model="",
                judge_rationale=f"Fetch failed: {exc}",
            )
            results.append(result)
            if on_result:
                on_result(result)
            continue

        check = build_check(target.artifact_type, content)
        logger.info("[probe] Running check '%s' on '%s'", check["check_id"], target.name)

        try:
            resp = await provider.generate(
                check["prompt"],
                system="You are a senior security analyst. Be thorough and precise.",
                model=model,
                max_tokens=1024,
                thinking=thinking,
                thinking_budget=thinking_budget,
            )
            analysis = resp.text
            thinking_tokens = resp.thinking_tokens
        except Exception as exc:
            logger.warning("[probe] Analysis failed for '%s': %s", target.name, exc)
            result = ProbeResult(
                target_name=target.name,
                artifact_type=target.artifact_type,
                check_id=check["check_id"],
                score=None,
                prev_score=prev_scores.get(target.name),
                regressed=False,
                judge_model="",
                judge_rationale=f"Analysis failed: {exc}",
                thinking_enabled=bool(thinking),
            )
            results.append(result)
            if on_result:
                on_result(result)
            continue

        judge = await score_response(
            check["prompt"],
            analysis,
            judge_provider=judge_provider,
            judge_model=judge_model,
            gold_criteria=check.get("gold_criteria"),
        )

        prev = prev_scores.get(target.name)
        regressed = (
            prev is not None
            and judge.score is not None
            and (prev - judge.score) > regression_threshold
        )

        result = ProbeResult(
            target_name=target.name,
            artifact_type=target.artifact_type,
            check_id=check["check_id"],
            score=judge.score,
            prev_score=prev,
            regressed=regressed,
            judge_model=judge.judge_model,
            judge_rationale=judge.rationale,
            thinking_enabled=bool(thinking),
            thinking_tokens=thinking_tokens,
        )
        results.append(result)

        logger.info(
            "[probe] '%s' → %.3f (%s)%s",
            target.name, judge.score or 0, check["check_id"],
            " [REGRESSION]" if regressed else "",
        )

        if on_result:
            on_result(result)

    return ProbeSummary(results=results)
