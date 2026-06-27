"""Tests for MCP/agentic and tool-use safety adversarial fixtures."""
from __future__ import annotations


def test_mcp_agentic_fixtures_importable():
    from atomics.eval.adversarial.mcp_agentic import MCP_AGENTIC_FIXTURES
    assert len(MCP_AGENTIC_FIXTURES) >= 10


def test_tool_safety_fixtures_importable():
    from atomics.eval.adversarial.mcp_agentic import TOOL_SAFETY_FIXTURES
    assert len(TOOL_SAFETY_FIXTURES) >= 5


def test_mcp_agentic_fixtures_all_have_required_fields():
    from atomics.eval.adversarial.mcp_agentic import MCP_AGENTIC_FIXTURES
    valid_severities = {"LOW", "MEDIUM", "HIGH", "CRITICAL"}
    valid_categories = {
        "tool_call_compliance", "authority_fabrication",
        "breakglass_injection", "context_poisoning", "agent_loop_escape",
    }
    for f in MCP_AGENTIC_FIXTURES:
        assert f.id, f"fixture missing id"
        assert f.id.startswith("mcp-"), f"{f.id}: should start with 'mcp-'"
        assert f.severity in valid_severities, f"{f.id}: bad severity {f.severity}"
        assert f.category in valid_categories, f"{f.id}: bad category {f.category}"
        assert f.prompt, f"{f.id}: empty prompt"
        assert f.attack_goal, f"{f.id}: empty attack_goal"
        assert f.resistance_criteria, f"{f.id}: empty resistance_criteria"
        assert len(f.resistance_criteria) >= 2, f"{f.id}: need at least 2 criteria"


def test_tool_safety_fixtures_all_have_required_fields():
    from atomics.eval.adversarial.mcp_agentic import TOOL_SAFETY_FIXTURES
    valid_severities = {"LOW", "MEDIUM", "HIGH", "CRITICAL"}
    for f in TOOL_SAFETY_FIXTURES:
        assert f.id, f"fixture missing id"
        assert f.id.startswith("tool-"), f"{f.id}: should start with 'tool-'"
        assert f.category == "tool_use_safety", f"{f.id}: category should be tool_use_safety"
        assert f.severity in valid_severities, f"{f.id}: bad severity {f.severity}"
        assert f.prompt, f"{f.id}: empty prompt"
        assert f.attack_goal, f"{f.id}: empty attack_goal"
        assert f.resistance_criteria, f"{f.id}: empty resistance_criteria"
        assert len(f.resistance_criteria) >= 2, f"{f.id}: need at least 2 criteria"


def test_mcp_agentic_fixture_ids_unique():
    from atomics.eval.adversarial.mcp_agentic import MCP_AGENTIC_FIXTURES, TOOL_SAFETY_FIXTURES
    all_fixtures = MCP_AGENTIC_FIXTURES + TOOL_SAFETY_FIXTURES
    ids = [f.id for f in all_fixtures]
    assert len(ids) == len(set(ids)), "Duplicate fixture IDs"


def test_no_id_collision_with_existing_fixtures():
    from atomics.eval.adversarial.fixtures import ADVERSARIAL_FIXTURES
    from atomics.eval.adversarial.mcp_agentic import MCP_AGENTIC_FIXTURES, TOOL_SAFETY_FIXTURES
    existing_ids = {f.id for f in ADVERSARIAL_FIXTURES}
    new_ids = {f.id for f in MCP_AGENTIC_FIXTURES + TOOL_SAFETY_FIXTURES}
    collision = existing_ids & new_ids
    assert not collision, f"ID collision with existing fixtures: {collision}"


def test_mcp_fixtures_registered_in_runner():
    """The runner's assembled fixture list includes the new categories."""
    from atomics.eval.adversarial.mcp_agentic import MCP_AGENTIC_FIXTURES, TOOL_SAFETY_FIXTURES
    from atomics.eval.adversarial.fixtures import ADVERSARIAL_FIXTURES
    from atomics.eval.adversarial.zerotrust import ZEROTRUST_FIXTURES
    from atomics.eval.adversarial.agentic_reasoning import AGENTIC_REASONING_FIXTURES

    total = (
        len(ADVERSARIAL_FIXTURES) + len(ZEROTRUST_FIXTURES)
        + len(AGENTIC_REASONING_FIXTURES) + len(MCP_AGENTIC_FIXTURES)
        + len(TOOL_SAFETY_FIXTURES)
    )
    assert total >= 30, f"Expected at least 30 total fixtures, got {total}"


def test_mcp_category_coverage():
    """All 5 MCP/agentic categories are represented."""
    from atomics.eval.adversarial.mcp_agentic import MCP_AGENTIC_FIXTURES
    categories = {f.category for f in MCP_AGENTIC_FIXTURES}
    expected = {
        "tool_call_compliance", "authority_fabrication",
        "breakglass_injection", "context_poisoning", "agent_loop_escape",
    }
    missing = expected - categories
    assert not missing, f"Missing categories: {missing}"


def test_tool_safety_prompts_contain_dangerous_commands():
    """Each tool-safety fixture should contain a recognizable dangerous command."""
    from atomics.eval.adversarial.mcp_agentic import TOOL_SAFETY_FIXTURES
    dangerous_indicators = ["chmod", "ssh", "id_rsa", "tar", "curl", "useradd", "kubectl delete"]
    for f in TOOL_SAFETY_FIXTURES:
        has_indicator = any(ind in f.prompt for ind in dangerous_indicators)
        assert has_indicator, f"{f.id}: prompt should contain a dangerous command indicator"
