import pytest
from pydantic import ValidationError

from atomics.api.models import EvalRequest, JobResponse, RunRequest


def test_run_request_defaults():
    req = RunRequest(provider="ollama")
    assert req.provider == "ollama"
    assert req.model is None
    assert req.tier == "ez"
    assert req.iterations == 3
    assert req.interval == 5
    assert req.save is True


def test_eval_request_defaults():
    req = EvalRequest(suite="rag", provider="ollama")
    assert req.suite == "rag"
    assert req.provider == "ollama"
    assert req.fixtures is None


def test_run_request_invalid_iterations():
    with pytest.raises(ValidationError):
        RunRequest(provider="ollama", iterations=0)


def test_job_response_result_defaults_to_none():
    resp = JobResponse(
        job_id="abc",
        status="pending",
        kind="run",
        created_at="0",
    )
    assert resp.result is None
