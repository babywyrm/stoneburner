import pytest

from atomics.archreview.models import AnswerKey, Finding
from atomics.archreview.scorer import (
    compute_robustness,
    score_objective,
    score_reasoning,
)


def _f(cat):
    return Finding(category=cat, location="x", severity="high", rationale="y")


def test_perfect_recall_and_precision():
    ak = AnswerKey(version=1, weights={"injection": 6.0, "xss": 4.0})
    findings = [_f("injection"), _f("xss")]
    recall, prec, fscore, matched = score_objective(findings, ak)
    assert recall == 1.0
    assert prec == 1.0
    assert fscore == 1.0
    assert set(matched) == {"injection", "xss"}


def test_partial_recall_weighted():
    ak = AnswerKey(version=1, weights={"injection": 6.0, "xss": 4.0})
    findings = [_f("injection")]  # caught the heavier one only
    recall, prec, fscore, matched = score_objective(findings, ak)
    assert recall == 0.6  # 6 / 10
    assert prec == 1.0


def test_hallucinated_category_lowers_precision():
    ak = AnswerKey(version=1, weights={"injection": 6.0})
    findings = [_f("injection"), _f("ssrf")]  # ssrf not present
    recall, prec, fscore, matched = score_objective(findings, ak)
    assert recall == 1.0
    assert prec == 0.5
    assert matched == ["injection"]


def test_unknown_and_duplicate_categories_ignored():
    ak = AnswerKey(version=1, weights={"injection": 6.0, "xss": 4.0})
    findings = [_f("injection"), _f("injection"), _f("unknown")]
    recall, prec, fscore, matched = score_objective(findings, ak)
    assert recall == 0.6
    # emitted distinct present-or-not categories = {injection}; unknown excluded
    assert prec == 1.0


def test_empty_findings():
    ak = AnswerKey(version=1, weights={"injection": 6.0})
    recall, prec, fscore, matched = score_objective([], ak)
    assert recall == 0.0
    assert prec == 0.0
    assert fscore == 0.0


def test_robustness_identical_rounds_is_perfectly_stable():
    rounds = [{"injection", "xss"}, {"injection", "xss"}, {"injection", "xss"}]
    stability, recall_sd = compute_robustness(rounds, [0.7, 0.7, 0.7])
    assert stability == 1.0
    assert recall_sd == 0.0


def test_robustness_varying_rounds():
    rounds = [{"injection", "xss"}, {"injection"}, {"injection", "ssrf"}]
    stability, recall_sd = compute_robustness(rounds, [0.6, 0.4, 0.5])
    assert 0.0 < stability < 1.0
    assert recall_sd > 0.0


def test_robustness_single_round():
    stability, recall_sd = compute_robustness([{"injection"}], [0.6])
    assert stability == 1.0
    assert recall_sd == 0.0


class _StubProvider:
    name = "stub"
    default_model = "stub-judge"

    def __init__(self, reply):
        self._reply = reply

    async def generate(self, prompt, *, system="", model=None, max_tokens=1024,
                       thinking=None, thinking_budget=None, temperature=None):
        from atomics.providers.base import ProviderResponse
        return ProviderResponse(text=self._reply, input_tokens=10, output_tokens=5,
                                total_tokens=15, model="stub-judge", latency_ms=1.0,
                                estimated_cost_usd=0.0)

    async def health_check(self):
        return True


@pytest.mark.asyncio
async def test_score_reasoning_parses_rating():
    judge = _StubProvider("REASONING: 8\nRATIONALE: solid trust-boundary analysis")
    score, rationale = await score_reasoning(
        "arch summary text", judge=judge, judge_model="stub-judge")
    assert score == 0.8
    assert "trust" in rationale.lower()


@pytest.mark.asyncio
async def test_score_reasoning_unparseable_returns_half():
    judge = _StubProvider("I think it's pretty good overall")
    score, rationale = await score_reasoning(
        "arch summary", judge=judge, judge_model="stub-judge")
    assert score == 0.5
