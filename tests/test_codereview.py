"""Tests for the secure-code-review suite."""

from __future__ import annotations

import pytest

from atomics.eval.codereview import SECURE_CODE_FIXTURES, run_codereview
from atomics.providers.base import ProviderResponse

# ── Fixture set sanity ─────────────────────────────────────────────────────

def test_fixtures_have_vulnerable_and_clean():
    vuln = [f for f in SECURE_CODE_FIXTURES if f.is_vulnerable]
    clean = [f for f in SECURE_CODE_FIXTURES if not f.is_vulnerable]
    assert len(vuln) >= 4, "need several vulnerable fixtures"
    assert len(clean) >= 2, "need clean fixtures for false-positive measurement"


def test_fixtures_have_both_modes():
    modes = {f.mode for f in SECURE_CODE_FIXTURES}
    assert modes == {"snippet", "diff"}


def test_ids_unique():
    ids = [f.id for f in SECURE_CODE_FIXTURES]
    assert len(ids) == len(set(ids))


def test_vulnerable_fixtures_have_cwe_and_findings():
    for f in SECURE_CODE_FIXTURES:
        if f.is_vulnerable:
            assert f.cwe, f"{f.id}: vulnerable fixture missing CWE"
            assert f.expected_findings, f"{f.id}: vulnerable fixture missing findings"
        else:
            assert f.cwe == "" and f.severity == "NONE"


def test_diff_fixtures_look_like_diffs():
    for f in SECURE_CODE_FIXTURES:
        if f.mode == "diff":
            assert "@@" in f.code and "---" in f.code, f"{f.id}: not a unified diff"


# ── Runner + rollups (mocked providers) ────────────────────────────────────

class _FakeProvider:
    name = "fake"

    async def generate(self, prompt, *, system=None, model=None, max_tokens=256,
                       thinking=None, thinking_budget=None, temperature=None):
        return ProviderResponse(
            text="my review", input_tokens=5, output_tokens=5, total_tokens=10,
            model=model or "m", latency_ms=100.0, estimated_cost_usd=0.0,
        )


class _FakeJudge:
    """Judge that returns a verdict depending on ground truth is_vulnerable.

    perfect=True → always grades correctly (detected on vuln, clean on clean).
    perfect=False → always grades incorrectly (missed on vuln, false_positive on clean).
    """
    name = "judge"

    def __init__(self, perfect: bool):
        self._perfect = perfect

    async def generate(self, prompt, *, system=None, model=None, max_tokens=256,
                       thinking=None, thinking_budget=None, temperature=None):
        # The vuln-judge prompt contains "GROUND TRUTH: the reviewed code contains"
        is_vuln_prompt = "contains this vulnerability" in prompt
        if is_vuln_prompt:
            verdict = "DETECTED" if self._perfect else "MISSED"
        else:
            verdict = "CLEAN" if self._perfect else "FALSE_POSITIVE"
        return ProviderResponse(
            text=f"VERDICT: {verdict}\nRATIONALE: x", input_tokens=5,
            output_tokens=5, total_tokens=10, model="judge", latency_ms=50.0,
            estimated_cost_usd=0.0,
        )


@pytest.mark.asyncio
async def test_perfect_reviewer_scores_high():
    summary = await run_codereview(
        _FakeProvider(), judge_provider=_FakeJudge(perfect=True),
        model="m", judge_model="judge",
    )
    assert summary.detection_rate == 1.0
    assert summary.false_positive_rate == 0.0
    assert summary.review_score == 1.0


@pytest.mark.asyncio
async def test_bad_reviewer_scores_low():
    summary = await run_codereview(
        _FakeProvider(), judge_provider=_FakeJudge(perfect=False),
        model="m", judge_model="judge",
    )
    assert summary.detection_rate == 0.0
    assert summary.false_positive_rate == 1.0
    assert summary.review_score == 0.0


@pytest.mark.asyncio
async def test_summary_to_dict_shape():
    summary = await run_codereview(
        _FakeProvider(), judge_provider=_FakeJudge(perfect=True),
        model="m", judge_model="judge", fixtures=SECURE_CODE_FIXTURES[:2],
    )
    d = summary.to_dict()
    assert {"detection_rate", "false_positive_rate", "review_score"} <= d.keys()
    assert len(d["results"]) == 2
