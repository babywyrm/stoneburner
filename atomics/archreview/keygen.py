"""Answer-key generation (from juice-shop challenges.yml) and repo-spec loading."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import yaml

from atomics.archreview.models import AnswerKey, RepoSpec, TierConfig
from atomics.archreview.taxonomy import normalize_category


def answer_key_from_challenges(challenges_path: Path) -> AnswerKey:
    """Derive an AnswerKey from a juice-shop-style challenges.yml.

    Per-category weight = sum of challenge difficulty for that category.
    Challenges whose category does not map to the taxonomy are skipped.
    """
    data = yaml.safe_load(Path(challenges_path).read_text()) or []
    weights: dict[str, float] = defaultdict(float)
    for ch in data:
        cat = normalize_category(str(ch.get("category", "")))
        if cat is None:
            continue
        weights[cat.value] += float(ch.get("difficulty", 1))
    return AnswerKey(version=1, weights={k: round(v, 3) for k, v in sorted(weights.items())})


def load_repo_spec(spec_path: Path) -> RepoSpec:
    """Load a repo spec YAML into a RepoSpec."""
    raw = yaml.safe_load(Path(spec_path).read_text())
    repo = raw["repo"]
    tiers = {
        name: TierConfig(
            budget_tokens=int(t["budget_tokens"]),
            priority=tuple(t.get("priority", []) or []),
            include=tuple(t.get("include", []) or []),
            exclude=tuple(t.get("exclude", []) or []),
        )
        for name, t in raw["tiers"].items()
    }
    ak_raw = raw["answer_key"]
    weights = {c["id"]: float(c["weight"]) for c in ak_raw["categories"]}
    answer_key = AnswerKey(version=int(ak_raw["version"]), weights=weights)
    return RepoSpec(
        name=repo["name"], git_ref=repo["git_ref"], path_env=repo["path_env"],
        tiers=tiers, answer_key=answer_key,
    )
