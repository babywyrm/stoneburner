"""Cost optimization advisor — analyzes historical runs and recommends cheaper models.

Queries task_results for quality scores across models, groups by task
category and complexity, and identifies the cheapest model meeting a
minimum quality threshold for each group.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass


@dataclass
class Recommendation:
    """A single cost optimization recommendation."""

    category: str
    complexity: str
    current_model: str
    current_quality: float
    current_cost_per_task: float
    recommended_model: str
    recommended_quality: float
    recommended_cost_per_task: float
    quality_delta: float
    cost_savings_pct: float
    task_count: int


@dataclass
class AdvisorSummary:
    """Overall advisor analysis."""

    recommendations: list[Recommendation]
    total_current_cost: float
    total_recommended_cost: float
    overall_savings_pct: float
    models_analyzed: int
    min_quality_threshold: float

    def to_dict(self) -> dict:
        return {
            "total_current_cost": round(self.total_current_cost, 6),
            "total_recommended_cost": round(self.total_recommended_cost, 6),
            "overall_savings_pct": round(self.overall_savings_pct, 1),
            "models_analyzed": self.models_analyzed,
            "min_quality_threshold": self.min_quality_threshold,
            "recommendations": [
                {
                    "category": r.category,
                    "complexity": r.complexity,
                    "current_model": r.current_model,
                    "current_quality": round(r.current_quality, 3),
                    "current_cost_per_task": round(r.current_cost_per_task, 6),
                    "recommended_model": r.recommended_model,
                    "recommended_quality": round(r.recommended_quality, 3),
                    "recommended_cost_per_task": round(r.recommended_cost_per_task, 6),
                    "quality_delta": round(r.quality_delta, 3),
                    "cost_savings_pct": round(r.cost_savings_pct, 1),
                    "task_count": r.task_count,
                }
                for r in self.recommendations
            ],
        }


def analyze_cost_optimization(
    conn: sqlite3.Connection,
    *,
    min_quality: float = 0.8,
    since_hours: float | None = None,
    current_model: str | None = None,
) -> AdvisorSummary:
    """Analyze task_results and recommend cheaper models meeting quality thresholds.

    Groups tasks by category + complexity, compares quality and cost across
    models, and identifies where a cheaper model still meets min_quality.
    """
    clauses = ["accuracy_score IS NOT NULL", "status = 'success'"]
    params: list = []
    if since_hours is not None:
        clauses.append("started_at >= datetime('now', ?)")
        params.append(f"-{since_hours} hours")

    where = f"WHERE {' AND '.join(clauses)}"

    sql = f"""
        SELECT
            category,
            model,
            COUNT(*) as task_count,
            AVG(accuracy_score) as avg_quality,
            AVG(estimated_cost_usd) as avg_cost_per_task,
            SUM(estimated_cost_usd) as total_cost
        FROM task_results
        {where}
        GROUP BY category, model
        HAVING task_count >= 2
        ORDER BY category, avg_cost_per_task ASC
    """
    rows = conn.execute(sql, params).fetchall()

    groups: dict[str, list[dict]] = {}
    all_models: set[str] = set()
    for row in rows:
        d = dict(row)
        cat = d["category"]
        groups.setdefault(cat, []).append(d)
        all_models.add(d["model"])

    recommendations: list[Recommendation] = []
    total_current = 0.0
    total_recommended = 0.0

    for cat, models in groups.items():
        most_expensive = max(models, key=lambda m: m["avg_cost_per_task"])
        if current_model:
            current = next((m for m in models if m["model"] == current_model), most_expensive)
        else:
            current = most_expensive

        cheapest_meeting_threshold = None
        for m in models:
            if m["avg_quality"] >= min_quality and m["model"] != current["model"]:
                if m["avg_cost_per_task"] < current["avg_cost_per_task"]:
                    if cheapest_meeting_threshold is None or m["avg_cost_per_task"] < cheapest_meeting_threshold["avg_cost_per_task"]:
                        cheapest_meeting_threshold = m

        if cheapest_meeting_threshold is not None:
            savings_pct = (
                (1 - cheapest_meeting_threshold["avg_cost_per_task"] / current["avg_cost_per_task"])
                * 100
                if current["avg_cost_per_task"] > 0
                else 0.0
            )
            recommendations.append(Recommendation(
                category=cat,
                complexity="mixed",
                current_model=current["model"],
                current_quality=current["avg_quality"],
                current_cost_per_task=current["avg_cost_per_task"],
                recommended_model=cheapest_meeting_threshold["model"],
                recommended_quality=cheapest_meeting_threshold["avg_quality"],
                recommended_cost_per_task=cheapest_meeting_threshold["avg_cost_per_task"],
                quality_delta=cheapest_meeting_threshold["avg_quality"] - current["avg_quality"],
                cost_savings_pct=savings_pct,
                task_count=current["task_count"],
            ))
            total_current += current["total_cost"]
            total_recommended += (
                cheapest_meeting_threshold["avg_cost_per_task"] * current["task_count"]
            )
        else:
            total_current += current["total_cost"]
            total_recommended += current["total_cost"]

    overall_savings = (
        (1 - total_recommended / total_current) * 100
        if total_current > 0
        else 0.0
    )

    return AdvisorSummary(
        recommendations=recommendations,
        total_current_cost=total_current,
        total_recommended_cost=total_recommended,
        overall_savings_pct=overall_savings,
        models_analyzed=len(all_models),
        min_quality_threshold=min_quality,
    )
