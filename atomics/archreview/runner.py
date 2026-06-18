"""Async orchestrator for an archreview run (one model, N rounds)."""

from __future__ import annotations

import logging

from atomics.archreview.models import ArchReviewResult, RepoSpec
from atomics.archreview.pack import EvidencePack
from atomics.archreview.prompt import build_analysis_prompt, parse_findings
from atomics.archreview.scorer import score_objective, score_reasoning
from atomics.providers.base import BaseProvider

logger = logging.getLogger("atomics.archreview.runner")

_CONTEXT_EXHAUSTED_TEXT_CHARS = 40
_CONTEXT_EXHAUSTED_OUTPUT_TOKENS = 8


async def run_archreview(
    *,
    spec: RepoSpec,
    tier: str,
    pack: EvidencePack,
    under_test: BaseProvider,
    under_test_model: str | None,
    judge: BaseProvider | None,
    judge_model: str | None,
    rounds: int = 1,
    objective: bool = True,
    max_output_tokens: int = 2048,
    run_id: str = "",
) -> list[ArchReviewResult]:
    """Run the analysis `rounds` times against `under_test`, scoring each round."""
    system, task = build_analysis_prompt(pack.text)
    results: list[ArchReviewResult] = []

    for rnd in range(1, rounds + 1):
        result = ArchReviewResult(
            run_id=run_id, repo=spec.name, tier=tier,
            model=under_test_model or (under_test.default_model or ""),
            provider=under_test.name, round=rnd, findings=[],
            pack_hash=pack.content_hash, judge_model=judge_model or "",
        )
        try:
            resp = await under_test.generate(
                task, system=system, model=under_test_model,
                max_tokens=max_output_tokens,
                thinking=False, temperature=0.0,
            )
        except Exception as exc:  # noqa: BLE001 — record, never abort the batch
            result.error_class = type(exc).__name__
            result.error_message = str(exc) or repr(exc)
            results.append(result)
            continue

        result.tokens_in = resp.input_tokens
        result.tokens_out = resp.output_tokens
        result.cost_usd = resp.estimated_cost_usd
        result.latency_ms = resp.latency_ms

        raw = resp.raw or {}
        if (
            raw.get("done_reason") == "length"
            and (
                len(resp.text.strip()) <= _CONTEXT_EXHAUSTED_TEXT_CHARS
                or resp.output_tokens <= _CONTEXT_EXHAUSTED_OUTPUT_TOKENS
            )
        ):
            result.parse_failed = True
            result.error_class = "ContextExhausted"
            result.error_message = (
                "model stopped at context/output length before producing findings; "
                "try a smaller evidence tier or a larger context window"
            )
            results.append(result)
            continue

        findings = parse_findings(resp.text)
        result.findings = findings
        result.parse_failed = len(findings) == 0

        if objective:
            recall, prec, fscore, matched = score_objective(findings, spec.answer_key)
            result.objective_recall = recall
            result.objective_precision = prec
            result.objective_f = fscore
            result.matched_categories = matched

        if judge is not None:
            try:
                score, _rationale = await score_reasoning(
                    resp.text, judge=judge, judge_model=judge_model)
                result.judge_score = score
            except Exception as exc:  # noqa: BLE001
                logger.warning("judge failed on %s round %d: %s",
                               result.model, rnd, exc)

        results.append(result)

    return results
