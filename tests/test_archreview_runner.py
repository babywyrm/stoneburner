import pytest

from atomics.archreview.models import AnswerKey, RepoSpec, TierConfig
from atomics.archreview.pack import EvidencePack
from atomics.archreview.runner import run_archreview
from atomics.providers.base import ProviderResponse


class _Provider:
    def __init__(self, name, reply, model="m1"):
        self.name = name
        self._reply = reply
        self._model = model
        self.calls = []

    @property
    def default_model(self):
        return self._model

    async def generate(self, prompt, *, system="", model=None, max_tokens=1024,
                       thinking=None, thinking_budget=None, temperature=None):
        self.calls.append({
            "model": model, "max_tokens": max_tokens,
            "thinking": thinking, "temperature": temperature,
        })
        return ProviderResponse(text=self._reply, input_tokens=100, output_tokens=20,
                                total_tokens=120, model=self._model, latency_ms=5.0,
                                estimated_cost_usd=0.0)

    async def health_check(self):
        return True


_GOOD = ("Summary: weak boundaries.\n"
         "CATEGORY: injection | LOCATION: a.ts | SEVERITY: high | WHY: raw sql\n"
         "CATEGORY: xss | LOCATION: b.html | SEVERITY: medium | WHY: reflected\n")
_JUDGE = "REASONING: 7\nRATIONALE: decent"


def _spec():
    return RepoSpec(
        name="demo", git_ref="abc", path_env="DEMO_PATH",
        tiers={"floor": TierConfig(budget_tokens=4000)},
        answer_key=AnswerKey(version=1, weights={"injection": 6.0, "xss": 4.0}),
    )


@pytest.mark.asyncio
async def test_run_archreview_scores_objective_and_judge():
    pack = EvidencePack(text="PACK", content_hash="deadbeef", file_count=3, truncated=False)
    under_test = _Provider("ollama", _GOOD, model="qwen2.5:14b")
    judge = _Provider("claude", _JUDGE, model="claude-opus-4-7")
    results = await run_archreview(
        spec=_spec(), tier="floor", pack=pack,
        under_test=under_test, under_test_model="qwen2.5:14b",
        judge=judge, judge_model="claude-opus-4-7", rounds=2,
    )
    assert len(results) == 2
    r = results[0]
    assert r.objective_recall == 1.0
    assert r.judge_score == 0.7
    assert r.pack_hash == "deadbeef"
    assert r.tokens_in == 100
    assert not r.parse_failed
    assert under_test.calls[0]["thinking"] is False
    assert under_test.calls[0]["temperature"] == 0.0


@pytest.mark.asyncio
async def test_run_archreview_judge_only_skips_objective_when_no_key():
    pack = EvidencePack(text="PACK", content_hash="h", file_count=1, truncated=False)
    under_test = _Provider("ollama", _GOOD, model="m")
    judge = _Provider("claude", _JUDGE, model="j")
    results = await run_archreview(
        spec=_spec(), tier="floor", pack=pack,
        under_test=under_test, under_test_model="m",
        judge=judge, judge_model="j", rounds=1, objective=False,
    )
    assert results[0].judge_score == 0.7
    assert results[0].objective_recall == 0.0


@pytest.mark.asyncio
async def test_run_archreview_records_provider_error():
    class _Boom(_Provider):
        async def generate(self, *a, **k):
            raise RuntimeError("backend down")

    pack = EvidencePack(text="P", content_hash="h", file_count=1, truncated=False)
    results = await run_archreview(
        spec=_spec(), tier="floor", pack=pack,
        under_test=_Boom("ollama", "", model="m"), under_test_model="m",
        judge=None, judge_model=None, rounds=1,
    )
    assert results[0].error_message is not None
    assert results[0].error_class == "RuntimeError"
