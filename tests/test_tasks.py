"""Tests for task catalog and weighted selection."""

from atomics.models import TaskCategory
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
