"""The API always meters eval spend; the CLI does so only when asked.

That asymmetry is the point of these tests. A CLI user spends their own money
on a run they started; an API caller is remote, authenticated by a shared key,
and was previously bounded only by the provider account's own limits.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from atomics.api.config import ServerSettings
from atomics.api.models import (
    DEFAULT_EVAL_BUDGET_USD,
    MAX_EVAL_BUDGET_USD,
    EvalRequest,
)
from atomics.api.server import create_app
from atomics.eval.budget import GuardedProvider


class TestEvalRequestBudget:
    def test_a_budget_is_applied_by_default(self):
        assert EvalRequest(suite="accuracy", provider="ollama").budget_usd == (
            DEFAULT_EVAL_BUDGET_USD
        )

    def test_a_caller_may_lower_the_ceiling(self):
        assert EvalRequest(suite="accuracy", provider="ollama", budget_usd=0.5) is not None

    def test_a_caller_may_not_remove_the_ceiling(self):
        with pytest.raises(ValidationError):
            EvalRequest(suite="accuracy", provider="ollama", budget_usd=0)

    def test_a_negative_budget_is_rejected(self):
        with pytest.raises(ValidationError):
            EvalRequest(suite="accuracy", provider="ollama", budget_usd=-1)

    def test_the_ceiling_itself_is_capped(self):
        with pytest.raises(ValidationError):
            EvalRequest(suite="accuracy", provider="ollama", budget_usd=MAX_EVAL_BUDGET_USD + 1)

    def test_the_api_rejects_an_uncapped_eval_with_422(self, tmp_path):
        app = create_app(ServerSettings(no_auth=True, db_path=tmp_path / "b.db"))
        with TestClient(app) as client:
            res = client.post(
                "/api/v1/evals",
                json={"suite": "accuracy", "provider": "ollama", "budget_usd": 0},
            )
            assert res.status_code == 422


class TestApiProvidersAreGuarded:
    def test_both_the_model_and_the_judge_are_wrapped(self, monkeypatch):
        """Guarding only the model would leave judge traffic unmetered."""
        from atomics.api import _runners

        built = []

        def fake_provider(name, model):
            from tests.test_eval_budget import StubProvider

            provider = StubProvider(name=name)
            built.append(provider)
            return provider

        monkeypatch.setattr(_runners, "_provider_for", fake_provider)
        provider, judge = _runners._guarded_providers(
            EvalRequest(suite="accuracy", provider="ollama")
        )

        assert isinstance(provider, GuardedProvider)
        assert isinstance(judge, GuardedProvider)

    def test_they_share_one_guard(self, monkeypatch):
        from atomics.api import _runners

        def fake_provider(name, model):
            from tests.test_eval_budget import StubProvider

            return StubProvider(name=name)

        monkeypatch.setattr(_runners, "_provider_for", fake_provider)
        provider, judge = _runners._guarded_providers(
            EvalRequest(suite="accuracy", provider="ollama")
        )

        assert provider.guard is judge.guard

    def test_the_requested_ceiling_reaches_the_guard(self, monkeypatch):
        from atomics.api import _runners

        def fake_provider(name, model):
            from tests.test_eval_budget import StubProvider

            return StubProvider(name=name)

        monkeypatch.setattr(_runners, "_provider_for", fake_provider)
        provider, _ = _runners._guarded_providers(
            EvalRequest(suite="accuracy", provider="ollama", budget_usd=3.5)
        )

        assert provider.guard._config.budget_limit_usd == 3.5
