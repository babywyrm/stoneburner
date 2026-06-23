"""Tests for zero-trust adversarial fixtures."""
from __future__ import annotations


def test_zerotrust_fixtures_load():
    from atomics.eval.adversarial.zerotrust import ZEROTRUST_FIXTURES
    assert len(ZEROTRUST_FIXTURES) == 6


def test_zerotrust_fixtures_have_required_fields():
    from atomics.eval.adversarial.zerotrust import ZEROTRUST_FIXTURES

    valid_severities = {"LOW", "MEDIUM", "HIGH", "CRITICAL"}
    valid_categories = {
        "tool_manipulation", "breakglass_extract", "credential_exfil",
        "behavior_persistence", "authority_fabrication", "blocklist_evasion",
    }
    for f in ZEROTRUST_FIXTURES:
        assert f.id.startswith("zt-"), f"{f.id}: should start with zt-"
        assert f.severity in valid_severities, f"{f.id}: bad severity"
        assert f.category in valid_categories, f"{f.id}: unknown category {f.category}"
        assert len(f.prompt) > 50, f"{f.id}: prompt too short"
        assert len(f.attack_goal) > 20, f"{f.id}: attack_goal too short"
        assert len(f.resistance_criteria) >= 3, f"{f.id}: need at least 3 resistance criteria"


def test_zerotrust_ids_unique():
    from atomics.eval.adversarial.zerotrust import ZEROTRUST_FIXTURES
    ids = [f.id for f in ZEROTRUST_FIXTURES]
    assert len(ids) == len(set(ids)), f"duplicate ids: {ids}"


def test_zerotrust_registered_in_all_fixtures():
    from atomics.eval.adversarial import ALL_FIXTURES
    zt_ids = [f.id for f in ALL_FIXTURES if f.id.startswith("zt-")]
    assert len(zt_ids) == 6, f"expected 6 zt fixtures in ALL_FIXTURES, got {len(zt_ids)}"


def test_zerotrust_categories_are_distinct_from_base():
    from atomics.eval.adversarial.fixtures import ADVERSARIAL_FIXTURES
    from atomics.eval.adversarial.zerotrust import ZEROTRUST_FIXTURES

    base_categories = {f.category for f in ADVERSARIAL_FIXTURES}
    zt_categories = {f.category for f in ZEROTRUST_FIXTURES}
    overlap = base_categories & zt_categories
    assert len(overlap) == 0, f"zt categories should not overlap base: {overlap}"
