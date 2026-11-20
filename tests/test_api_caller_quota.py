"""Per-caller job accounting.

The global concurrency cap alone is first-come-first-served: whoever submits
first takes every slot, and a second key gets `429` until that work drains.
That is a denial of service requiring no malice, just one impatient script, so
these tests assert that one caller cannot consume the whole server.
"""

from __future__ import annotations

import asyncio

import pytest
from fastapi import Request
from fastapi.testclient import TestClient

from atomics.api.auth import ApiKeyAuth, NoAuth, matched_key
from atomics.api.callers import ANONYMOUS_CALLER, caller_id_from_key
from atomics.api.config import ServerSettings
from atomics.api.jobs import (
    CallerQuotaExceededError,
    JobManager,
    TooManyActiveJobsError,
)
from atomics.api.server import create_app

ALICE = "alice-key-0123456789"
BOB = "bob-key-9876543210"


def _request(key: str | None) -> Request:
    headers = [(b"x-api-key", key.encode())] if key is not None else []
    return Request({"type": "http", "headers": headers, "method": "GET", "path": "/"})


async def _blocking_work(_job_id: str) -> None:
    await asyncio.Event().wait()


class TestCallerIdentity:
    def test_a_key_maps_to_a_stable_identifier(self):
        assert caller_id_from_key(ALICE) == caller_id_from_key(ALICE)

    def test_different_keys_map_to_different_identifiers(self):
        assert caller_id_from_key(ALICE) != caller_id_from_key(BOB)

    def test_the_identifier_does_not_contain_the_key(self):
        """It lands in log files, which are handled far more casually than secrets."""
        identifier = caller_id_from_key(ALICE)
        assert ALICE not in identifier
        assert identifier not in ALICE

    def test_an_empty_key_is_anonymous(self):
        assert caller_id_from_key("") == ANONYMOUS_CALLER

    def test_the_identifier_is_short_enough_to_read_in_a_log(self):
        assert len(caller_id_from_key(ALICE)) == 12

    def test_api_key_auth_identifies_the_presented_key(self):
        auth = ApiKeyAuth({ALICE, BOB})
        assert auth.identify(_request(ALICE)) == caller_id_from_key(ALICE)
        assert auth.identify(_request(BOB)) == caller_id_from_key(BOB)

    def test_an_unrecognized_key_is_anonymous_rather_than_an_error(self):
        """Identification is not authorization; callers here are already authed."""
        auth = ApiKeyAuth({ALICE})
        assert auth.identify(_request("not-a-key")) == ANONYMOUS_CALLER

    def test_no_auth_collapses_every_caller_to_one(self):
        assert NoAuth().identify(_request(None)) == ANONYMOUS_CALLER

    def test_matched_key_returns_the_key_that_matched(self):
        assert matched_key(ALICE, {ALICE, BOB}) == ALICE
        assert matched_key("nope", {ALICE, BOB}) is None


class TestPerCallerQuota:
    @pytest.mark.asyncio
    async def test_one_caller_cannot_exceed_their_share(self):
        manager = JobManager(max_active=16, max_active_per_caller=2)
        alice = caller_id_from_key(ALICE)

        await manager.submit("run", _blocking_work, owner=alice)
        await manager.submit("run", _blocking_work, owner=alice)

        with pytest.raises(CallerQuotaExceededError):
            await manager.submit("run", _blocking_work, owner=alice)

    @pytest.mark.asyncio
    async def test_a_second_caller_is_unaffected_by_the_first(self):
        """The whole point: one busy key must not starve another."""
        manager = JobManager(max_active=16, max_active_per_caller=2)
        alice, bob = caller_id_from_key(ALICE), caller_id_from_key(BOB)

        await manager.submit("run", _blocking_work, owner=alice)
        await manager.submit("run", _blocking_work, owner=alice)
        with pytest.raises(CallerQuotaExceededError):
            await manager.submit("run", _blocking_work, owner=alice)

        bob_job = await manager.submit("run", _blocking_work, owner=bob)
        assert bob_job in manager.jobs

    @pytest.mark.asyncio
    async def test_the_global_limit_still_wins_when_the_server_is_full(self):
        """A busy server reports its own load, not the caller's quota."""
        manager = JobManager(max_active=2, max_active_per_caller=8)
        await manager.submit("run", _blocking_work, owner="a")
        await manager.submit("run", _blocking_work, owner="b")

        with pytest.raises(TooManyActiveJobsError) as excinfo:
            await manager.submit("run", _blocking_work, owner="c")
        assert not isinstance(excinfo.value, CallerQuotaExceededError)
        assert "limit 2" in str(excinfo.value)

    @pytest.mark.asyncio
    async def test_finished_jobs_release_the_caller_s_quota(self):
        async def quick(_job_id: str) -> str:
            return "done"

        manager = JobManager(max_active_per_caller=1)
        first = await manager.submit("run", quick, owner="alice")
        await manager.wait_for(first)

        second = await manager.submit("run", quick, owner="alice")
        assert second in manager.jobs

    @pytest.mark.asyncio
    async def test_active_count_for_counts_only_that_caller(self):
        manager = JobManager(max_active_per_caller=4)
        await manager.submit("run", _blocking_work, owner="alice")
        await manager.submit("run", _blocking_work, owner="alice")
        await manager.submit("run", _blocking_work, owner="bob")

        assert manager.active_count_for("alice") == 2
        assert manager.active_count_for("bob") == 1
        assert manager.active_count == 3

    def test_a_non_positive_per_caller_limit_is_rejected(self):
        with pytest.raises(ValueError, match="max_active_per_caller must be positive"):
            JobManager(max_active_per_caller=0)

    @pytest.mark.asyncio
    async def test_the_quota_error_is_still_a_too_many_jobs_error(self):
        """Routes catch the base class, so both map to 429 without route changes."""
        manager = JobManager(max_active_per_caller=1)
        await manager.submit("run", _blocking_work, owner="alice")

        with pytest.raises(TooManyActiveJobsError):
            await manager.submit("run", _blocking_work, owner="alice")


class TestQuotaOverHttp:
    def test_the_setting_reaches_the_job_manager(self, tmp_path):
        app = create_app(
            ServerSettings(
                no_auth=True,
                db_path=tmp_path / "q.db",
                max_active_jobs_per_caller=3,
            )
        )
        with TestClient(app, base_url="http://127.0.0.1") as client:
            client.get("/api/v1/health")
            assert app.state.job_manager.max_active_per_caller == 3

    def test_a_non_positive_setting_is_rejected(self, tmp_path):
        with pytest.raises(ValueError, match="max_active_jobs_per_caller"):
            ServerSettings(db_path=tmp_path / "q.db", max_active_jobs_per_caller=0)

    def test_an_authenticated_submitter_is_attributed_to_their_key(self, tmp_path, jobs_never_run):
        """Jobs must carry the submitting caller, or the quota counts nothing."""
        app = create_app(ServerSettings(api_keys={ALICE}, db_path=tmp_path / "q.db"))
        with TestClient(app, base_url="http://127.0.0.1") as client:
            res = client.post(
                "/api/v1/evals",
                json={"suite": "accuracy", "provider": "ollama"},
                headers={"X-API-Key": ALICE},
            )
            assert res.status_code == 202
            job_id = res.json()["job_id"]
            assert app.state.job_manager.jobs[job_id].owner == caller_id_from_key(ALICE)


class TestJobVisibility:
    def test_a_caller_sees_only_their_own_jobs(self, tmp_path):
        from atomics.api.jobs import Job, JobStatus

        app = create_app(ServerSettings(api_keys={ALICE, BOB}, db_path=tmp_path / "q.db"))
        with TestClient(app, base_url="http://127.0.0.1") as client:
            app.state.job_manager.jobs["j-alice"] = Job(
                job_id="j-alice",
                kind="eval",
                status=JobStatus.COMPLETED,
                created_at=0.0,
                owner=caller_id_from_key(ALICE),
            )
            alice, bob = {"X-API-Key": ALICE}, {"X-API-Key": BOB}
            assert client.get("/api/v1/jobs/j-alice", headers=alice).status_code == 200
            assert client.get("/api/v1/jobs/j-alice", headers=bob).status_code == 404
            assert [
                j["job_id"] for j in client.get("/api/v1/jobs", headers=alice).json()["jobs"]
            ] == ["j-alice"]
            assert client.get("/api/v1/jobs", headers=bob).json()["jobs"] == []


class TestPerCallerSpendCeiling:
    @pytest.mark.asyncio
    async def test_caller_is_blocked_after_reaching_cumulative_spend_ceiling(self):
        from atomics.api.jobs import CallerSpendQuotaExceededError

        async def expensive(_job_id: str) -> dict[str, float]:
            return {"total_cost_usd": 3.5}

        manager = JobManager(max_spend_usd_per_caller=5.0)
        j1 = await manager.submit("eval", expensive, owner="alice")
        await manager.wait_for(j1)
        assert manager.caller_spend_usd("alice") == pytest.approx(3.5)

        j2 = await manager.submit("eval", expensive, owner="alice")
        await manager.wait_for(j2)
        assert manager.caller_spend_usd("alice") == pytest.approx(7.0)

        with pytest.raises(CallerSpendQuotaExceededError, match="per-caller spend limit"):
            await manager.submit("eval", expensive, owner="alice")

        # Bob is unaffected by Alice exhausting her cumulative budget.
        j_bob = await manager.submit("eval", expensive, owner="bob")
        await manager.wait_for(j_bob)
        assert manager.caller_spend_usd("bob") == pytest.approx(3.5)

    @pytest.mark.asyncio
    async def test_battery_step_costs_count_toward_caller_spend(self):
        async def battery_work(_job_id: str) -> dict[str, object]:
            return {
                "steps": [
                    {"suite": "provider-test", "cost_usd": 0.25},
                    {"suite": "redblue", "result": {"total_cost_usd": 1.75}},
                ]
            }

        manager = JobManager(max_spend_usd_per_caller=2.0)
        jid = await manager.submit("battery", battery_work, owner="alice")
        await manager.wait_for(jid)
        assert manager.caller_spend_usd("alice") == pytest.approx(2.0)
        with pytest.raises(TooManyActiveJobsError, match="spend"):
            await manager.submit("battery", battery_work, owner="alice")

    def test_non_positive_caller_spend_ceiling_is_rejected(self, tmp_path):
        with pytest.raises(ValueError, match="max_spend_usd_per_caller"):
            JobManager(max_spend_usd_per_caller=0.0)
        with pytest.raises(ValueError, match="max_spend_usd_per_caller"):
            ServerSettings(db_path=tmp_path / "q.db", max_spend_usd_per_caller=-1.0)

    def test_caller_spend_persists_across_server_restarts_and_job_eviction(
        self, tmp_path, monkeypatch
    ):
        db_path = tmp_path / "spend.db"

        async def fake_eval(*_args, **_kwargs):
            return {"suite": "accuracy", "total_cost_usd": 3.0}

        monkeypatch.setattr("atomics.api.routes.run_eval_suite", fake_eval)

        settings1 = ServerSettings(
            api_keys={ALICE, BOB},
            db_path=db_path,
            max_retained_jobs=1,
            max_spend_usd_per_caller=5.0,
        )
        with TestClient(create_app(settings1), base_url="http://127.0.0.1") as client1:
            r1 = client1.post(
                "/api/v1/evals",
                json={"suite": "accuracy", "provider": "ollama"},
                headers={"X-API-Key": ALICE},
            )
            assert r1.status_code == 202
            r2 = client1.post(
                "/api/v1/evals",
                json={"suite": "accuracy", "provider": "ollama"},
                headers={"X-API-Key": ALICE},
            )
            assert r2.status_code == 202

        # Fresh server instance on the same SQLite DB: j1 was evicted and RAM cleared,
        # yet Alice's $6.00 cumulative spend remains enforced across restart.
        settings2 = ServerSettings(
            api_keys={ALICE, BOB},
            db_path=db_path,
            max_retained_jobs=1,
            max_spend_usd_per_caller=5.0,
        )
        with TestClient(create_app(settings2), base_url="http://127.0.0.1") as client2:
            blocked = client2.post(
                "/api/v1/evals",
                json={"suite": "accuracy", "provider": "ollama"},
                headers={"X-API-Key": ALICE},
            )
            assert blocked.status_code == 429
            assert "per-caller spend limit" in blocked.json()["detail"]

            allowed = client2.post(
                "/api/v1/evals",
                json={"suite": "accuracy", "provider": "ollama"},
                headers={"X-API-Key": BOB},
            )
            assert allowed.status_code == 202

    @pytest.mark.asyncio
    async def test_aborted_budget_exceeded_job_still_charges_caller_spend(self, tmp_path):
        from atomics.eval.budget import EvalBudgetExceededError

        async def aborted_work(_job_id: str) -> dict[str, object]:
            raise EvalBudgetExceededError("eval stopped: budget exceeded", spent_usd=4.25)

        manager = JobManager(max_spend_usd_per_caller=4.0, db_path=tmp_path / "abort.db")
        try:
            jid = await manager.submit("eval", aborted_work, owner="alice")
            await manager.wait_for(jid)
            assert manager.caller_spend_usd("alice") == pytest.approx(4.25)
            with pytest.raises(TooManyActiveJobsError, match="spend"):
                await manager.submit("eval", aborted_work, owner="alice")
        finally:
            manager.close()

    def test_provider_test_and_sweep_count_toward_caller_spend(self, tmp_path, monkeypatch):
        db_path = tmp_path / "pt_sweep.db"

        async def fake_pt(_payload):
            return {
                "ok": True,
                "health": True,
                "provider": "claude",
                "model": "claude-sonnet-4-6",
                "response": "4",
                "cost_usd": 0.60,
            }

        monkeypatch.setattr("atomics.api.routes.run_provider_test", fake_pt)

        settings = ServerSettings(
            api_keys={ALICE},
            db_path=db_path,
            max_spend_usd_per_caller=1.0,
        )
        with TestClient(create_app(settings), base_url="http://127.0.0.1") as client:
            first = client.post(
                "/api/v1/provider-test",
                json={"provider": "claude"},
                headers={"X-API-Key": ALICE},
            )
            assert first.status_code == 200
            second = client.post(
                "/api/v1/provider-test",
                json={"provider": "claude"},
                headers={"X-API-Key": ALICE},
            )
            assert second.status_code == 200
            third = client.post(
                "/api/v1/provider-test",
                json={"provider": "claude"},
                headers={"X-API-Key": ALICE},
            )
            assert third.status_code == 429
            assert "per-caller spend limit" in third.json()["detail"]

    def test_record_caller_spend_validates_inputs(self, tmp_path):
        from atomics.storage.repository import MetricsRepository

        repo = MetricsRepository(tmp_path / "val.db")
        try:
            with pytest.raises(ValueError, match="caller_id"):
                repo.record_caller_spend("   ", 1.0)
            with pytest.raises(ValueError, match="cost_usd"):
                repo.record_caller_spend("alice", -0.01)
            with pytest.raises(ValueError, match="cost_usd"):
                repo.record_caller_spend("alice", float("nan"))
            with pytest.raises(ValueError, match="cost_usd"):
                repo.record_caller_spend("alice", float("inf"))
            with pytest.raises(ValueError, match="cost_usd"):
                repo.record_caller_spend("alice", True)  # type: ignore[arg-type]
        finally:
            repo.close()
