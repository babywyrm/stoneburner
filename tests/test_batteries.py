"""Named batteries compose existing suites. Unknown ids must not ship."""

from __future__ import annotations

import pytest

from atomics.eval.adversarial import select_fixtures as adv_select
from atomics.eval.batteries import BATTERIES, get_battery, step_args, visible_steps
from atomics.eval.codereview.fixtures import SECURE_CODE_FIXTURES as CODE_FIXTURES
from atomics.eval.redblue.fixtures import select_fixtures as rb_select
from atomics.eval.refusal.fixtures import REFUSAL_FIXTURES
from atomics.eval.toolcall.fixtures import select_fixtures as tc_select


def test_five_named_batteries_exist():
    assert [b.id for b in BATTERIES] == [
        "desk-pass",
        "blue-capability",
        "red-capability",
        "agent-gate",
        "threat-model",
    ]


def test_get_battery_unknown_raises():
    with pytest.raises(KeyError, match="unknown battery"):
        get_battery("not-a-battery")


def test_every_fixture_id_exists():
    refusal_ids = {f.id for f in REFUSAL_FIXTURES}
    code_ids = {f.id for f in CODE_FIXTURES}
    for battery in BATTERIES:
        for step in (*battery.steps, *battery.optional_steps):
            if step.suite == "toolcall":
                tc_select(category=step.category, ids=step.fixtures or None)
            elif step.suite == "adversarial":
                assert adv_select(step.categories or None, ids=step.fixtures or None)
            elif step.suite == "redblue":
                rb_select(step.mode or "all", ids=step.fixtures or None)
            elif step.suite == "refusal":
                missing = [i for i in step.fixtures if i not in refusal_ids]
                assert not missing, f"{battery.id}: {missing}"
            elif step.suite == "codereview":
                missing = [i for i in step.fixtures if i not in code_ids]
                assert not missing, f"{battery.id}: {missing}"
            elif step.suite == "qa":
                assert step.qa_file == "qa/examples/app-gate-guardrails.yaml"
            elif step.suite == "provider-test":
                assert not step.fixtures
            elif step.suite == "archreview":
                assert step.repo
            else:
                raise AssertionError(f"{battery.id}: unknown suite {step.suite}")


def test_desk_pass_needs_no_judge():
    assert get_battery("desk-pass").needs_judge is False


def test_capability_batteries_need_a_judge():
    for name in ("blue-capability", "red-capability", "threat-model", "agent-gate"):
        assert get_battery(name).needs_judge is True


def test_desk_pass_argv():
    args = step_args(
        get_battery("desk-pass").steps[2],
        model="lfm2.5:8b",
        provider="ollama",
    )
    assert args[0] == "toolcall"
    assert "--provider" in args or "-p" in args
    assert "ollama" in args
    assert "--fixtures" in args
    assert "tc-01,tc-02" in args
    assert "--channel" in args
    assert "tools" in args
    assert "--no-thinking" in args


def test_step_args_openai_and_claude_have_no_ollama_host():
    step = get_battery("agent-gate").steps[0]
    for provider in ("openai", "claude"):
        args = step_args(step, model="MODEL", provider=provider, budget="5")
        assert provider in args
        assert "--ollama-host" not in args
        assert "--budget" in args
        assert "5" in args


def test_step_args_vllm_forwards_host_and_omits_model_when_unset():
    step = get_battery("agent-gate").steps[0]
    args = step_args(
        step,
        model=None,
        provider="vllm",
        vllm_host="http://127.0.0.1:8000/v1",
    )
    assert "vllm" in args
    assert "-m" not in args
    assert "--vllm-host" in args
    assert "http://127.0.0.1:8000/v1" in args


def test_qa_step_skipped_unless_ollama_or_profile():
    desk = get_battery("desk-pass")
    ollama_steps = visible_steps(desk, provider="ollama")
    assert any(s.suite == "qa" for s in ollama_steps)
    claude_steps = visible_steps(desk, provider="claude")
    assert not any(s.suite == "qa" for s in claude_steps)
    profiled = visible_steps(desk, provider="claude", profile="profiles/local/gate.yaml")
    assert any(s.suite == "qa" for s in profiled)


def test_no_battery_is_the_full_adversarial_suite():
    for battery in BATTERIES:
        for step in battery.steps:
            if step.suite == "adversarial":
                selected = adv_select(step.categories or None, ids=step.fixtures or None)
                assert len(selected) < 72


def test_archreview_optional_omits_thinking_flag():
    step = get_battery("threat-model").optional_steps[0]
    args = step_args(step, model="granite4.2:8b", provider="ollama")
    assert args[0] == "archreview"
    assert "--no-thinking" not in args
    assert "--models" in args
    assert "granite4.2:8b" in args
    assert "--tier" in args
    assert "floor" in args
