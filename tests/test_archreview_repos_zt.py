"""Tests for zero-trust archreview answer keys (nullfield, blueprint, camazotz)."""
from __future__ import annotations

from pathlib import Path

import yaml

REPOS_DIR = Path(__file__).parent.parent / "atomics" / "archreview" / "repos"
ZT_REPOS = ["nullfield.yaml", "zero-trust-blueprint.yaml", "camazotz.yaml"]


def _load(name: str) -> dict:
    path = REPOS_DIR / name
    assert path.exists(), f"{name} not found in {REPOS_DIR}"
    return yaml.safe_load(path.read_text())


def test_all_zt_repo_files_exist():
    for name in ZT_REPOS:
        assert (REPOS_DIR / name).exists(), f"missing: {name}"


def test_repo_section_present():
    for name in ZT_REPOS:
        d = _load(name)
        assert "repo" in d, f"{name}: missing 'repo' section"
        assert "name" in d["repo"], f"{name}: repo.name missing"
        assert "path_env" in d["repo"], f"{name}: repo.path_env missing"


def test_tiers_valid():
    required_tiers = {"floor", "local", "wide", "expanded"}
    for name in ZT_REPOS:
        d = _load(name)
        assert "tiers" in d, f"{name}: missing 'tiers'"
        tiers = set(d["tiers"].keys())
        assert required_tiers <= tiers, f"{name}: missing tiers {required_tiers - tiers}"
        for tier_name, tier in d["tiers"].items():
            assert "budget_tokens" in tier, f"{name}/{tier_name}: missing budget_tokens"
            assert "priority" in tier, f"{name}/{tier_name}: missing priority"
            assert isinstance(tier["priority"], list), f"{name}/{tier_name}: priority should be list"
            assert tier["budget_tokens"] > 0, f"{name}/{tier_name}: budget must be positive"


def test_answer_key_valid():
    for name in ZT_REPOS:
        d = _load(name)
        assert "answer_key" in d, f"{name}: missing 'answer_key'"
        ak = d["answer_key"]
        assert ak.get("version") == 2, f"{name}: answer_key version should be 2"
        assert "categories" in ak, f"{name}: answer_key.categories missing"
        cats = ak["categories"]
        assert len(cats) >= 5, f"{name}: need at least 5 categories, got {len(cats)}"
        total_weight = 0
        for cat in cats:
            assert "id" in cat, f"{name}: category missing 'id'"
            assert "weight" in cat, f"{name}: category {cat.get('id')} missing 'weight'"
            assert cat["weight"] > 0, f"{name}: {cat['id']} weight must be positive"
            total_weight += cat["weight"]
        assert total_weight >= 50.0, f"{name}: total weight suspiciously low ({total_weight})"


def test_answer_key_ids_unique_per_repo():
    for name in ZT_REPOS:
        d = _load(name)
        ids = [c["id"] for c in d["answer_key"]["categories"]]
        assert len(ids) == len(set(ids)), f"{name}: duplicate category ids"


def test_budget_tokens_increase_with_tier():
    for name in ZT_REPOS:
        d = _load(name)
        tiers = d["tiers"]
        budgets = [tiers[t]["budget_tokens"] for t in ["floor", "local", "wide", "expanded"]]
        assert budgets == sorted(budgets), f"{name}: budget_tokens should increase: {budgets}"
