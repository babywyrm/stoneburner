"""Tests for the secure-code-review suite."""

from __future__ import annotations

import httpx
import pytest

from atomics.eval.codereview import SECURE_CODE_FIXTURES, run_codereview
from atomics.eval.outcomes import JudgeOutcomeStatus, ProviderOutcomeKind, RunStatus
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


class _FailingProvider:
    name = "failing"
    default_model = "failing-model"

    async def generate(self, *_args, **_kwargs):
        raise httpx.ReadTimeout("provider timed out")


class _EmptyProvider:
    name = "empty"
    default_model = "empty-model"

    async def generate(self, *_args, **_kwargs):
        return ProviderResponse(
            text="", input_tokens=2, output_tokens=0, total_tokens=2,
            model="empty-model", latency_ms=10.0, estimated_cost_usd=0.0,
        )


class _CleanFixtureFailingJudge(_FakeJudge):
    async def generate(self, prompt, **kwargs):
        if "contains this vulnerability" not in prompt:
            raise httpx.ReadTimeout("clean-fixture judge timed out")
        return await super().generate(prompt, **kwargs)


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


@pytest.mark.asyncio
async def test_codereview_summary_exposes_complete_integrity_and_alias():
    summary = await run_codereview(
        _FakeProvider(), judge_provider=_FakeJudge(perfect=True),
        model="m", judge_model="judge", fixtures=SECURE_CODE_FIXTURES[:2],
    )

    assert summary.fixture_results is summary.results
    assert summary.integrity.status is RunStatus.COMPLETE
    payload = summary.to_dict()
    assert payload["fixture_results"] == payload["results"]
    assert payload["integrity"]["status"] == "complete"
    assert payload["results"][0]["attempts"][0]["provider_kind"] == "completed"


@pytest.mark.asyncio
async def test_codereview_judge_failure_is_indeterminate():
    summary = await run_codereview(
        _FakeProvider(), judge_provider=_FailingProvider(),
        fixtures=SECURE_CODE_FIXTURES[:1],
    )

    assert summary.review_score is None
    assert summary.integrity.status is RunStatus.INFRASTRUCTURE_INVALID
    assert summary.integrity.judge_failures == 1
    attempt = summary.results[0].attempts[0]
    assert attempt.judge is not None
    assert attempt.judge.status is JudgeOutcomeStatus.PROVIDER_ERROR


@pytest.mark.asyncio
async def test_codereview_provider_timeout_retains_failed_attempt():
    summary = await run_codereview(
        _FailingProvider(), judge_provider=_FakeJudge(perfect=True),
        fixtures=SECURE_CODE_FIXTURES[:1],
    )

    assert summary.integrity.status is RunStatus.INFRASTRUCTURE_INVALID
    assert summary.integrity.infrastructure_failures == 1
    assert summary.results[0].attempts[0].provider.kind is ProviderOutcomeKind.TIMEOUT


@pytest.mark.asyncio
async def test_empty_review_is_indeterminate():
    clean = [f for f in SECURE_CODE_FIXTURES if not f.is_vulnerable][:1]
    summary = await run_codereview(
        _EmptyProvider(), judge_provider=_FakeJudge(perfect=True), fixtures=clean,
    )

    assert summary.review_score is None
    assert summary.integrity.status is RunStatus.INFRASTRUCTURE_INVALID
    assert summary.results[0].verdict == "unknown"


@pytest.mark.asyncio
async def test_review_score_is_indeterminate_without_clean_fixture_coverage():
    vulnerable = next(f for f in SECURE_CODE_FIXTURES if f.is_vulnerable)
    clean = next(f for f in SECURE_CODE_FIXTURES if not f.is_vulnerable)

    summary = await run_codereview(
        _FakeProvider(),
        judge_provider=_CleanFixtureFailingJudge(perfect=True),
        fixtures=[vulnerable, clean],
    )

    assert summary.detection_rate == 1.0
    assert summary.false_positive_rate is None
    assert summary.review_score is None
    assert summary.integrity.status is RunStatus.PARTIAL


@pytest.mark.asyncio
async def test_codereview_runner_calls_start_and_done_callbacks():
    events: list[str] = []
    fixture = SECURE_CODE_FIXTURES[0]

    await run_codereview(
        _FakeProvider(),
        judge_provider=_FakeJudge(perfect=True),
        fixtures=[fixture],
        on_fixture_start=lambda fx: events.append(f"start:{fx.id}"),
        on_fixture_done=lambda result: events.append(f"done:{result.fixture.id}"),
    )

    assert events == [f"start:{fixture.id}", f"done:{fixture.id}"]
