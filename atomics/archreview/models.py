"""Data models for the archreview benchmark."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Finding:
    category: str          # normalized Category value, or "unknown"
    location: str
    severity: str
    rationale: str


@dataclass(frozen=True)
class TierConfig:
    budget_tokens: int
    priority: tuple[str, ...] = ()
    include: tuple[str, ...] = ()
    exclude: tuple[str, ...] = ()


@dataclass(frozen=True)
class AnswerKey:
    version: int
    weights: dict[str, float]  # category value -> weight (present-in-repo set)

    def present_categories(self) -> list[str]:
        return list(self.weights.keys())

    def total_weight(self) -> float:
        return round(sum(self.weights.values()), 6)


@dataclass(frozen=True)
class RepoSpec:
    name: str
    git_ref: str
    path_env: str
    tiers: dict[str, TierConfig]
    answer_key: AnswerKey

    def tier(self, name: str) -> TierConfig:
        if name not in self.tiers:
            raise KeyError(f"repo '{self.name}' has no tier '{name}' "
                           f"(have: {sorted(self.tiers)})")
        return self.tiers[name]


@dataclass
class ArchReviewResult:
    run_id: str
    repo: str
    tier: str
    model: str
    provider: str
    round: int
    findings: list[Finding]
    objective_recall: float = 0.0
    objective_precision: float = 0.0
    objective_f: float = 0.0
    judge_score: float | None = None
    judge_rematch_recall: float | None = None
    matched_categories: list[str] = field(default_factory=list)
    parse_failed: bool = False
    tokens_in: int = 0
    tokens_out: int = 0
    cost_usd: float = 0.0
    latency_ms: float = 0.0
    judge_model: str = ""
    pack_hash: str = ""
    error_class: str | None = None
    error_message: str | None = None


@dataclass
class ArchReviewSummary:
    repo: str
    tier: str
    results: list[ArchReviewResult]

    def to_dict(self) -> dict:
        """Machine-readable view of the run for --json-out / dashboards / CI."""
        return {
            "repo": self.repo,
            "tier": self.tier,
            "total_rounds": len(self.results),
            "results": [
                {
                    "run_id": r.run_id,
                    "model": r.model,
                    "provider": r.provider,
                    "round": r.round,
                    "objective_recall": r.objective_recall,
                    "objective_precision": r.objective_precision,
                    "objective_f": r.objective_f,
                    "judge_score": r.judge_score,
                    "matched_categories": r.matched_categories,
                    "parse_failed": r.parse_failed,
                    "findings": [
                        {
                            "category": f.category,
                            "location": f.location,
                            "severity": f.severity,
                            "rationale": f.rationale,
                        }
                        for f in r.findings
                    ],
                    "tokens_in": r.tokens_in,
                    "tokens_out": r.tokens_out,
                    "cost_usd": r.cost_usd,
                    "latency_ms": r.latency_ms,
                    "judge_model": r.judge_model,
                    "error": r.error_message,
                }
                for r in self.results
            ],
        }
