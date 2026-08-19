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
        with TestClient(app) as client:
            client.get("/api/v1/health")
            assert app.state.job_manager.max_active_per_caller == 3

    def test_a_non_positive_setting_is_rejected(self, tmp_path):
        with pytest.raises(ValueError, match="max_active_jobs_per_caller"):
            ServerSettings(db_path=tmp_path / "q.db", max_active_jobs_per_caller=0)

    def test_an_authenticated_submitter_is_attributed_to_their_key(self, tmp_path):
        """Jobs must carry the submitting caller, or the quota counts nothing."""
        app = create_app(ServerSettings(api_keys={ALICE}, db_path=tmp_path / "q.db"))
        with TestClient(app) as client:
            res = client.post(
                "/api/v1/evals",
                json={"suite": "accuracy", "provider": "ollama"},
                headers={"X-API-Key": ALICE},
            )
            assert res.status_code == 202
            job_id = res.json()["job_id"]
            assert app.state.job_manager.jobs[job_id].owner == caller_id_from_key(ALICE)
