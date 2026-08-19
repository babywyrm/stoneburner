"""Burn tier profiles — ez, baseline, mega.

Each tier defines cadence, budget, rate limits, and which task complexities
are eligible. The engine uses these to override settings at runtime.
"""

from __future__ import annotations

from dataclasses import dataclass

from atomics.models import BurnTier


@dataclass(frozen=True)
class TierProfile:
    tier: BurnTier
    description: str
    loop_interval_seconds: int
    loop_jitter_seconds: int
    max_tokens_per_hour: int
    max_requests_per_minute: int
    budget_limit_usd: float
    preferred_model: str | None = None


TIER_PROFILES: dict[BurnTier, TierProfile] = {
    BurnTier.EZ: TierProfile(
        tier=BurnTier.EZ,
        description="Lightweight — short tasks, low frequency, minimal spend",
        loop_interval_seconds=300,
        loop_jitter_seconds=30,
        max_tokens_per_hour=15_000,
        max_requests_per_minute=8,
        budget_limit_usd=5.0,
        preferred_model="claude-haiku-4-5-20251001",
    ),
    BurnTier.BASELINE: TierProfile(
        tier=BurnTier.BASELINE,
        description="Standard — mixed tasks, moderate frequency, balanced spend",
        loop_interval_seconds=120,
        loop_jitter_seconds=15,
        max_tokens_per_hour=100_000,
        max_requests_per_minute=30,
        budget_limit_usd=50.0,
    ),
    BurnTier.MEGA: TierProfile(
        tier=BurnTier.MEGA,
        description="Heavy — deep research, high frequency, aggressive spend",
        loop_interval_seconds=30,
        loop_jitter_seconds=5,
        max_tokens_per_hour=500_000,
        max_requests_per_minute=60,
        budget_limit_usd=250.0,
    ),
}


def get_tier_profile(tier: BurnTier) -> TierProfile:
    return TIER_PROFILES[tier]
