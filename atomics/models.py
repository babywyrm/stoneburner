"""Shared data models used across atomics modules."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum

from pydantic import BaseModel, Field


class TaskStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"


class TaskCategory(str, Enum):
    WEB_SUMMARY = "web_summary"
    RESEARCH = "research"
    SECURITY_NEWS = "security_news"
    PATTERN_EXTRACTION = "pattern_extraction"
    GENERAL_QA = "general_qa"


class BurnTier(str, Enum):
    EZ = "ez"
    BASELINE = "baseline"
    MEGA = "mega"


class TaskComplexity(str, Enum):
    LIGHT = "light"
    MODERATE = "moderate"
    HEAVY = "heavy"


# Which complexities each tier is allowed to run
TIER_COMPLEXITY_MAP: dict[BurnTier, list[TaskComplexity]] = {
    BurnTier.EZ: [TaskComplexity.LIGHT],
    BurnTier.BASELINE: [TaskComplexity.LIGHT, TaskComplexity.MODERATE],
    BurnTier.MEGA: [TaskComplexity.LIGHT, TaskComplexity.MODERATE, TaskComplexity.HEAVY],
}


class TaskDefinition(BaseModel):
    category: TaskCategory
    name: str
    prompt_template: str
    complexity: TaskComplexity = TaskComplexity.MODERATE
    weight: float = Field(default=1.0, ge=0.0)
    max_output_tokens: int = Field(default=1024)


class TaskResult(BaseModel):
    task_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    run_id: str
    category: TaskCategory
    task_name: str
    provider: str
    model: str
    status: TaskStatus = TaskStatus.PENDING
    prompt: str = ""
    response: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    latency_ms: float = 0.0
    estimated_cost_usd: float = 0.0
    error_class: str = ""
    error_message: str = ""
    started_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    completed_at: datetime | None = None


class RunSummary(BaseModel):
    run_id: str
    started_at: datetime
    completed_at: datetime | None = None
    total_tasks: int = 0
    successful_tasks: int = 0
    failed_tasks: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_tokens: int = 0
    total_cost_usd: float = 0.0
    avg_latency_ms: float = 0.0
