"""Tests for reusable CLI command primitives."""

from __future__ import annotations

import json
from types import SimpleNamespace

import click
import pytest
from rich.console import Console

from atomics.commands.common import (
    FixtureProgress,
    _make_provider,
    effective_model,
    evaluation_record_from_fixture,
    integrity_exit_code,
    write_summary_json,
)
from atomics.eval.outcomes import (
    AttemptResult,
    JudgeOutcome,
    JudgeOutcomeStatus,
    ProviderOutcome,
    ProviderOutcomeKind,
    RunIntegrity,
)


class _Summary:
    def to_dict(self) -> dict[str, object]:
        return {"status": "complete"}


def _integrity(*, scored: bool) -> RunIntegrity:
    judge = (
        JudgeOutcome(
            status=JudgeOutcomeStatus.SCORED,
            score=1.0,
            judges_expected=1,
            judges_scored=1,
        )
        if scored
        else None
    )
    attempt = AttemptResult(
        attempt_index=0,
        provider=ProviderOutcome(ProviderOutcomeKind.COMPLETED),
        response_text="response",
        latency_ms=1.0,
        estimated_cost_usd=0.0,
        input_tokens=1,
        output_tokens=1,
        thinking_tokens=0,
        judge=judge,
    )
    return RunIntegrity.from_fixture_attempts([[attempt]])


def test_effective_model_prefers_requested_model() -> None:
    provider = SimpleNamespace(default_model="fallback")
    assert effective_model("requested", provider) == "requested"


def test_effective_model_uses_provider_default() -> None:
    provider = SimpleNamespace(default_model="qwen3:14b")
    assert effective_model(None, provider) == "qwen3:14b"


def test_effective_model_falls_back_to_default_label() -> None:
    provider = SimpleNamespace(default_model=None)
    assert effective_model(None, provider) == "default"


def test_integrity_exit_policy_honors_allow_partial() -> None:
    partial = _integrity(scored=False)
    complete = _integrity(scored=True)

    assert integrity_exit_code(complete, allow_partial=False) == 0
    assert integrity_exit_code(partial, allow_partial=False) == 1
    assert integrity_exit_code(partial, allow_partial=True) == 0


def test_write_summary_json_uses_to_dict(tmp_path) -> None:
    output = tmp_path / "result.json"
    write_summary_json(_Summary(), output)

    assert json.loads(output.read_text(encoding="utf-8")) == {"status": "complete"}


def test_write_summary_json_wraps_filesystem_error(tmp_path) -> None:
    with pytest.raises(click.ClickException, match="Unable to write JSON"):
        write_summary_json(_Summary(), tmp_path)


def test_fixture_progress_formats_duration() -> None:
    progress = FixtureProgress(2, Console())
    assert progress._fmt_duration(12.2) == "12s"
    assert progress._fmt_duration(125) == "2m05s"


def test_make_provider_rejects_unknown_provider() -> None:
    settings = SimpleNamespace()
    with pytest.raises(click.ClickException, match="Unknown provider"):
        _make_provider("invalid", None, None, settings)


def test_evaluation_record_from_fixture_rolls_up_attempt_usage() -> None:
    payload: dict[str, object] = {
        "id": "fixture-1",
        "status": "complete",
        "score": 1.0,
        "generation_status": "completed",
        "judge_status": "scored",
        "latency_ms": 20.0,
        "estimated_cost_usd": 0.03,
        "attempt_count": 1,
        "generation_failures": 0,
        "infrastructure_failures": 0,
        "judge_failures": 0,
        "parse_failed": False,
        "error_class": "",
        "error_message": "",
        "attempts": [
            {
                "input_tokens": 8,
                "output_tokens": 4,
                "thinking_tokens": 2,
            }
        ],
    }

    record = evaluation_record_from_fixture(
        run_id="run-1",
        suite="refusal",
        provider="ollama",
        model="qwen",
        payload=payload,
    )

    assert record.total_tokens == 12
    assert record.thinking_tokens == 2
    assert record.result_json is payload
