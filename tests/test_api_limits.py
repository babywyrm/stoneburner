"""Tests for the API server's resource bounds and response headers.

Job state is in-process, so an unbounded job dict or an uncapped iteration
count is a memory and cost problem rather than a correctness one. These assert
the ceilings hold and that legitimate traffic still passes.
"""

from __future__ import annotations

import asyncio

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from atomics.api.config import ServerSettings
from atomics.api.headers import dashboard_csp, new_nonce
from atomics.api.jobs import JobManager, JobStatus, TooManyActiveJobsError
from atomics.api.models import (
    MAX_FIXTURES,
    MAX_INTERVAL_SECONDS,
    MAX_ITERATIONS,
    EvalRequest,
    RunRequest,
)
from atomics.api.server import create_app


async def _instant(_job_id: str) -> str:
    return "done"


def _blocking(event: asyncio.Event):
    async def work(_job_id: str) -> str:
        await event.wait()
        return "done"

    return work


class TestJobRetention:
    @pytest.mark.asyncio
    async def test_finished_jobs_are_evicted_past_the_retention_limit(self):
        manager = JobManager(max_retained=3)
        for _ in range(10):
            job_id = await manager.submit("run", _instant)
            await manager.wait_for(job_id)
        assert len(manager.jobs) <= 3

    @pytest.mark.asyncio
    async def test_eviction_keeps_the_most_recent_jobs(self):
        manager = JobManager(max_retained=2)
        ids = []
        for _ in range(5):
            job_id = await manager.submit("run", _instant)
            await manager.wait_for(job_id)
            ids.append(job_id)
        assert ids[-1] in manager.jobs
        assert ids[0] not in manager.jobs

    @pytest.mark.asyncio
    async def test_running_jobs_are_never_evicted(self):
        manager = JobManager(max_active=8, max_retained=1)
        gate = asyncio.Event()
        running = await manager.submit("run", _blocking(gate))
        for _ in range(5):
            finished = await manager.submit("run", _instant)
            await manager.wait_for(finished)

        assert running in manager.jobs
        assert manager.jobs[running].status is JobStatus.RUNNING

        gate.set()
        await manager.wait_for(running)


class TestConcurrencyLimit:
    @pytest.mark.asyncio
    async def test_submitting_past_the_active_limit_is_rejected(self):
        manager = JobManager(max_active=2)
        gate = asyncio.Event()
        await manager.submit("run", _blocking(gate))
        await manager.submit("run", _blocking(gate))

        with pytest.raises(TooManyActiveJobsError):
            await manager.submit("run", _blocking(gate))

        gate.set()
        for job in list(manager.jobs.values()):
            await manager.wait_for(job.job_id)

    @pytest.mark.asyncio
    async def test_capacity_frees_up_once_jobs_finish(self):
        manager = JobManager(max_active=1)
        gate = asyncio.Event()
        first = await manager.submit("run", _blocking(gate))
        gate.set()
        await manager.wait_for(first)

        second = await manager.submit("run", _instant)
        await manager.wait_for(second)
        assert manager.jobs[second].status is JobStatus.COMPLETED

    def test_a_non_positive_limit_is_rejected(self):
        with pytest.raises(ValueError, match="max_active must be positive"):
            JobManager(max_active=0)
        with pytest.raises(ValueError, match="max_retained must be positive"):
            JobManager(max_retained=0)

    def test_settings_reject_non_positive_job_limits(self):
        with pytest.raises(ValueError, match="max_active_jobs must be positive"):
            ServerSettings(max_active_jobs=0)
        with pytest.raises(ValueError, match="max_retained_jobs must be positive"):
            ServerSettings(max_retained_jobs=-1)


class TestRequestCaps:
    def test_iterations_above_the_cap_are_rejected(self):
        with pytest.raises(ValidationError):
            RunRequest(provider="ollama", iterations=MAX_ITERATIONS + 1)

    def test_iterations_at_the_cap_are_accepted(self):
        assert RunRequest(provider="ollama", iterations=MAX_ITERATIONS).iterations

    def test_interval_above_the_cap_is_rejected(self):
        with pytest.raises(ValidationError):
            RunRequest(provider="ollama", interval=MAX_INTERVAL_SECONDS + 1)

    def test_too_many_fixtures_are_rejected(self):
        with pytest.raises(ValidationError):
            EvalRequest(
                suite="accuracy",
                provider="ollama",
                fixtures=["f"] * (MAX_FIXTURES + 1),
            )

    def test_a_normal_request_is_unaffected(self):
        req = RunRequest(provider="ollama", iterations=3, interval=5)
        assert (req.iterations, req.interval) == (3, 5)

    def test_the_api_rejects_an_oversized_run_with_422(self, tmp_path):
        app = create_app(ServerSettings(no_auth=True, db_path=tmp_path / "caps.db"))
        with TestClient(app) as client:
            res = client.post(
                "/api/v1/runs",
                json={"provider": "ollama", "iterations": MAX_ITERATIONS + 1},
            )
            assert res.status_code == 422


class TestSecurityHeaders:
    @pytest.fixture
    def client(self, tmp_path):
        app = create_app(
            ServerSettings(no_auth=True, with_dashboard=True, db_path=tmp_path / "hdr.db")
        )
        with TestClient(app) as tc:
            yield tc

    @pytest.mark.parametrize(
        "header,expected",
        [
            ("X-Content-Type-Options", "nosniff"),
            ("X-Frame-Options", "DENY"),
            ("Referrer-Policy", "no-referrer"),
        ],
    )
    def test_baseline_headers_are_present_on_json_routes(self, client, header, expected):
        res = client.get("/api/v1/health")
        assert res.headers[header] == expected

    def test_json_routes_get_a_locked_down_csp(self, client):
        res = client.get("/api/v1/health")
        assert "default-src 'none'" in res.headers["Content-Security-Policy"]

    def test_the_dashboard_gets_a_nonce_csp_not_unsafe_inline(self, client):
        res = client.get("/dashboard")
        csp = res.headers["Content-Security-Policy"]
        assert "'nonce-" in csp
        assert "unsafe-inline" not in csp

    def test_the_dashboard_nonce_matches_its_script_tag(self, client):
        res = client.get("/dashboard")
        csp = res.headers["Content-Security-Policy"]
        nonce = csp.split("script-src 'nonce-", 1)[1].split("'", 1)[0]
        assert f'<script nonce="{nonce}">' in res.text
        assert f'<style nonce="{nonce}">' in res.text

    def test_each_dashboard_response_gets_a_fresh_nonce(self, client):
        first = client.get("/dashboard").headers["Content-Security-Policy"]
        second = client.get("/dashboard").headers["Content-Security-Policy"]
        assert first != second

    def test_nonces_are_not_predictable(self):
        assert len({new_nonce() for _ in range(50)}) == 50

    def test_the_dashboard_policy_blocks_framing_and_base_tags(self):
        csp = dashboard_csp("abc")
        assert "frame-ancestors 'none'" in csp
        assert "base-uri 'none'" in csp
