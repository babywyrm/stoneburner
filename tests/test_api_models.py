import pytest
from pydantic import ValidationError

from atomics.api.models import EvalRequest, JobResponse, ProviderTestRequest, RunRequest


def test_run_request_defaults():
    req = RunRequest(provider="ollama")
    assert req.provider == "ollama"
    assert req.model is None
    assert req.tier == "ez"
    assert req.iterations == 3
    assert req.interval == 5
    assert req.save is True
    assert req.thinking is None
    assert req.effort is None
    assert req.reasoning_mode is None


def test_eval_request_defaults():
    req = EvalRequest(suite="rag", provider="ollama")
    assert req.suite == "rag"
    assert req.provider == "ollama"
    assert req.fixtures is None
    assert req.host is None


def test_eval_request_normalizes_effort_aliases():
    req = EvalRequest(suite="rag", provider="ollama", effort="XL", reasoning_mode="PRO")
    assert req.effort == "xhigh"
    assert req.reasoning_mode == "pro"


def test_eval_request_rejects_unknown_effort():
    with pytest.raises(ValidationError, match="unknown effort"):
        EvalRequest(suite="rag", provider="ollama", effort="ludicrous")


def test_eval_request_rejects_unknown_reasoning_mode():
    with pytest.raises(ValidationError, match="unknown reasoning mode"):
        EvalRequest(suite="rag", provider="ollama", reasoning_mode="turbo")


def test_provider_test_request_normalizes_and_rejects_effort():
    req = ProviderTestRequest(provider="openai", effort="ultra", reasoning_mode="standard")
    assert req.effort == "max"
    assert req.reasoning_mode == "standard"
    with pytest.raises(ValidationError, match="unknown effort"):
        ProviderTestRequest(provider="openai", effort="ludicrous")
    with pytest.raises(ValidationError, match="unknown reasoning mode"):
        ProviderTestRequest(provider="openai", reasoning_mode="turbo")


def test_run_request_normalizes_effort_aliases():
    req = RunRequest(provider="ollama", effort="XL", reasoning_mode="PRO")
    assert req.effort == "xhigh"
    assert req.reasoning_mode == "pro"


def test_run_request_rejects_unknown_effort():
    with pytest.raises(ValidationError, match="unknown effort"):
        RunRequest(provider="ollama", effort="ludicrous")


def test_run_request_rejects_unknown_reasoning_mode():
    with pytest.raises(ValidationError, match="unknown reasoning mode"):
        RunRequest(provider="ollama", reasoning_mode="turbo")


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
