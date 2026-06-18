from atomics.archreview.models import (
    Finding, TierConfig, AnswerKey, RepoSpec, ArchReviewResult,
)


def test_finding_construction():
    f = Finding(category="injection", location="routes/x.ts", severity="high", rationale="why")
    assert f.category == "injection"


def test_answer_key_present_categories():
    ak = AnswerKey(version=1, weights={"injection": 7.5, "xss": 6.0})
    assert set(ak.present_categories()) == {"injection", "xss"}
    assert ak.total_weight() == 13.5


def test_repo_spec_tier_lookup():
    spec = RepoSpec(
        name="demo", git_ref="abc", path_env="DEMO_PATH",
        tiers={"floor": TierConfig(budget_tokens=16000)},
        answer_key=AnswerKey(version=1, weights={"injection": 1.0}),
    )
    assert spec.tier("floor").budget_tokens == 16000


def test_archreview_result_defaults():
    r = ArchReviewResult(
        run_id="run1", repo="demo", tier="floor", model="m", provider="ollama",
        round=1, findings=[],
    )
    assert r.objective_recall == 0.0
    assert r.parse_failed is False
