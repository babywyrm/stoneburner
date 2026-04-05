"""Tests for task catalog, weighted selection, and tier-based filtering."""

from atomics.models import BurnTier, TaskCategory, TaskComplexity, TIER_COMPLEXITY_MAP
from atomics.tasks.catalog import TASK_CATALOG, TOPIC_POOLS, get_weighted_task


def test_catalog_is_populated():
    assert len(TASK_CATALOG) > 0


def test_every_category_has_tasks():
    categories_with_tasks = {t.category for t in TASK_CATALOG}
    for cat in TaskCategory:
        assert cat in categories_with_tasks, f"No tasks for category {cat}"


def test_every_category_has_topics():
    for cat in TaskCategory:
        assert cat in TOPIC_POOLS, f"No topics for category {cat}"
        assert len(TOPIC_POOLS[cat]) > 0


def test_get_weighted_task_returns_valid():
    for _ in range(50):
        task, topic = get_weighted_task()
        assert task.name
        assert topic
        assert task.category in TaskCategory
        assert "{topic}" not in task.prompt_template.format(topic=topic)


def test_all_weights_positive():
    for task in TASK_CATALOG:
        assert task.weight > 0, f"Task {task.name} has non-positive weight"


def test_every_complexity_has_tasks():
    complexities = {t.complexity for t in TASK_CATALOG}
    for c in TaskComplexity:
        assert c in complexities, f"No tasks with complexity {c}"


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
