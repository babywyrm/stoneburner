"""Tests for multilingual evaluation fixtures."""

from __future__ import annotations

from atomics.eval.multilingual import ALL_MULTILINGUAL_FIXTURES
from atomics.models import TaskComplexity


def test_all_fixtures_loaded():
    assert len(ALL_MULTILINGUAL_FIXTURES) == 10


def test_fixture_ids_are_unique():
    ids = [f.id for f in ALL_MULTILINGUAL_FIXTURES]
    assert len(ids) == len(set(ids))


def test_fixture_ids_follow_convention():
    for f in ALL_MULTILINGUAL_FIXTURES:
        assert f.id.startswith("ml-"), f"Fixture {f.id} doesn't follow ml-NN pattern"


def test_fixtures_have_gold_criteria():
    for f in ALL_MULTILINGUAL_FIXTURES:
        assert len(f.gold_criteria) >= 2, f"Fixture {f.id} has fewer than 2 gold criteria"


def test_complexity_spread():
    complexities = {f.complexity for f in ALL_MULTILINGUAL_FIXTURES}
    assert TaskComplexity.LIGHT in complexities
    assert TaskComplexity.MODERATE in complexities
    assert TaskComplexity.HEAVY in complexities


def test_prompts_contain_non_ascii():
    """At least half the fixtures should contain non-ASCII characters."""
    non_ascii = sum(
        1 for f in ALL_MULTILINGUAL_FIXTURES
        if any(ord(c) > 127 for c in f.prompt)
    )
    assert non_ascii >= 5, f"Only {non_ascii}/10 fixtures have non-ASCII prompts"


def test_language_diversity():
    """Verify prompts span multiple languages by checking for language-specific characters."""
    all_prompts = " ".join(f.prompt for f in ALL_MULTILINGUAL_FIXTURES)
    assert "¿" in all_prompts or "ñ" in all_prompts, "No Spanish"
    assert "é" in all_prompts or "ç" in all_prompts, "No French"
    assert "ä" in all_prompts or "ü" in all_prompts, "No German"
    assert "ã" in all_prompts or "ç" in all_prompts, "No Portuguese"
    assert any("\u3040" <= c <= "\u30ff" or "\u4e00" <= c <= "\u9fff" for c in all_prompts), "No Japanese/CJK"
    assert any("\uac00" <= c <= "\ud7af" for c in all_prompts), "No Korean"
    assert any("\u0600" <= c <= "\u06ff" for c in all_prompts), "No Arabic"


def test_fixtures_are_valid_eval_fixtures():
    """All multilingual fixtures should be compatible with the standard eval runner."""
    from atomics.eval.fixtures import EvalFixture

    for f in ALL_MULTILINGUAL_FIXTURES:
        assert isinstance(f, EvalFixture)
        assert f.max_output_tokens > 0
        assert len(f.prompt) > 10


def test_no_id_collision_with_eval_fixtures():
    """Multilingual IDs must not collide with standard eval fixture IDs."""
    from atomics.eval.fixtures import EVAL_FIXTURES

    eval_ids = {f.id for f in EVAL_FIXTURES}
    ml_ids = {f.id for f in ALL_MULTILINGUAL_FIXTURES}
    overlap = eval_ids & ml_ids
    assert not overlap, f"ID collision: {overlap}"


def test_cli_eval_accepts_multilingual_fixture_id():
    """The eval command should recognize ml-XX fixture IDs."""
    from click.testing import CliRunner
    from atomics.cli import cli

    runner = CliRunner(env={"ATOMICS_DB_PATH": "/tmp/test-ml.db"})
    result = runner.invoke(cli, ["eval", "--help"])
    assert result.exit_code == 0
    assert "--fixtures" in result.output
