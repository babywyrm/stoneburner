"""Tests for task catalog, weighted selection, tier filtering, and randomization."""

from atomics.models import BurnTier, TaskCategory, TaskComplexity, TIER_COMPLEXITY_MAP
from atomics.tasks.catalog import (
    PHRASING_VARIANTS,
    TASK_CATALOG,
    get_weighted_task,
)
from atomics.tasks.topics import CROSS_POLLINATION_TOPICS, TOPIC_POOLS
from atomics.tasks.randomizer import (
    RecencyTracker,
    build_combinatoric_prompt,
    build_modified_prompt,
)


def test_catalog_is_populated():
    assert len(TASK_CATALOG) > 0


def test_every_category_has_tasks():
    categories_with_tasks = {t.category for t in TASK_CATALOG}
    for cat in TaskCategory:
        assert cat in categories_with_tasks, f"No tasks for category {cat}"


def test_every_category_has_topics():
    for cat in TaskCategory:
        assert cat in TOPIC_POOLS, f"No topics for category {cat}"
        assert len(TOPIC_POOLS[cat]) >= 30, f"Category {cat} has too few topics ({len(TOPIC_POOLS[cat])})"


def test_get_weighted_task_returns_valid():
    for _ in range(50):
        task, prompt = get_weighted_task()
        assert task.name
        assert prompt
        assert len(prompt) > 10
        assert "{topic}" not in prompt
        assert "{prompt}" not in prompt


def test_all_weights_positive():
    for task in TASK_CATALOG:
        assert task.weight > 0, f"Task {task.name} has non-positive weight"


def test_every_complexity_has_tasks():
    complexities = {t.complexity for t in TASK_CATALOG}
    for c in TaskComplexity:
        assert c in complexities, f"No tasks with complexity {c}"


def test_every_task_has_phrasing_variants():
    for task in TASK_CATALOG:
        assert task.name in PHRASING_VARIANTS, f"No phrasing variants for {task.name}"
        variants = PHRASING_VARIANTS[task.name]
        assert len(variants) >= 3, f"Task {task.name} has too few variants ({len(variants)})"


def test_cross_pollination_topics_populated():
    assert len(CROSS_POLLINATION_TOPICS) >= 20


# ── Tier filtering tests ─────────────────────────────────

def test_ez_tier_only_gets_light_tasks():
    for _ in range(100):
        task, _ = get_weighted_task(BurnTier.EZ)
        assert task.complexity == TaskComplexity.LIGHT, (
            f"EZ tier got {task.complexity.value} task: {task.name}"
        )


def test_baseline_tier_gets_light_and_moderate():
    complexities_seen = set()
    for _ in range(200):
        task, _ = get_weighted_task(BurnTier.BASELINE)
        assert task.complexity in (TaskComplexity.LIGHT, TaskComplexity.MODERATE), (
            f"BASELINE tier got {task.complexity.value} task: {task.name}"
        )
        complexities_seen.add(task.complexity)
    assert TaskComplexity.LIGHT in complexities_seen
    assert TaskComplexity.MODERATE in complexities_seen


def test_mega_tier_includes_heavy_tasks():
    complexities_seen = set()
    for _ in range(300):
        task, _ = get_weighted_task(BurnTier.MEGA)
        complexities_seen.add(task.complexity)
    assert TaskComplexity.HEAVY in complexities_seen, "MEGA tier never selected a HEAVY task"


def test_tier_complexity_map_covers_all_tiers():
    for tier in BurnTier:
        assert tier in TIER_COMPLEXITY_MAP
        assert len(TIER_COMPLEXITY_MAP[tier]) > 0


# ── Randomization tests ──────────────────────────────────

def test_prompts_vary_across_calls():
    """Same tier should produce diverse prompts."""
    prompts = set()
    for _ in range(30):
        _, prompt = get_weighted_task(BurnTier.BASELINE)
        prompts.add(prompt)
    assert len(prompts) >= 25, f"Only {len(prompts)} unique prompts in 30 calls"


def test_recency_tracker_prevents_repeats():
    tracker = RecencyTracker(window=10)
    tracker.record("topic_a")
    tracker.record("topic_b")
    assert tracker.is_recent("topic_a")
    assert tracker.is_recent("topic_b")
    assert not tracker.is_recent("topic_c")


def test_recency_tracker_window_eviction():
    tracker = RecencyTracker(window=3)
    tracker.record("a")
    tracker.record("b")
    tracker.record("c")
    tracker.record("d")  # should evict "a"
    assert not tracker.is_recent("a")
    assert tracker.is_recent("d")


def test_build_modified_prompt_adds_content():
    base = "Explain TLS 1.3"
    modified = build_modified_prompt(
        base, add_audience=True, add_constraint=True,
        add_format=True, add_perspective=True,
    )
    assert modified.startswith(base)
    # With all flags on, at least sometimes it should be longer
    results = [
        build_modified_prompt(base, add_audience=True, add_constraint=True,
                              add_format=True, add_perspective=True)
        for _ in range(20)
    ]
    longer_count = sum(1 for r in results if len(r) > len(base))
    assert longer_count > 5, "Modifiers rarely applied"


def test_build_combinatoric_prompt():
    result = build_combinatoric_prompt("TLS 1.3", "game theory")
    assert "TLS 1.3" in result
    assert "game theory" in result
    assert len(result) > 20


def test_combinatorics_produce_variety():
    """Cross-pollination should produce diverse combinations."""
    results = set()
    for _ in range(20):
        result = build_combinatoric_prompt("Kubernetes RBAC", "formal verification")
        results.add(result)
    assert len(results) >= 5, "Combinatorics not producing enough variety"


def test_mega_tier_prompt_diversity():
    """Mega tier should produce highly diverse prompts due to full modifier stack."""
    prompts = set()
    for _ in range(50):
        _, prompt = get_weighted_task(BurnTier.MEGA)
        prompts.add(prompt)
    assert len(prompts) >= 45, f"Only {len(prompts)} unique mega prompts in 50 calls"
