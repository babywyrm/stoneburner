"""Tests for burn tier profiles."""

from atomics.models import BurnTier
from atomics.tiers import TIER_PROFILES, get_tier_profile


def test_all_tiers_have_profiles():
    for tier in BurnTier:
        profile = get_tier_profile(tier)
        assert profile.tier == tier
        assert profile.budget_limit_usd > 0
        assert profile.max_tokens_per_hour > 0
        assert profile.max_requests_per_minute > 0
        assert profile.loop_interval_seconds > 0


def test_ez_is_cheapest():
    ez = get_tier_profile(BurnTier.EZ)
    baseline = get_tier_profile(BurnTier.BASELINE)
    mega = get_tier_profile(BurnTier.MEGA)
    assert ez.budget_limit_usd < baseline.budget_limit_usd < mega.budget_limit_usd


def test_mega_is_fastest():
    ez = get_tier_profile(BurnTier.EZ)
    baseline = get_tier_profile(BurnTier.BASELINE)
    mega = get_tier_profile(BurnTier.MEGA)
    assert mega.loop_interval_seconds < baseline.loop_interval_seconds < ez.loop_interval_seconds


def test_mega_has_highest_throughput():
    ez = get_tier_profile(BurnTier.EZ)
    baseline = get_tier_profile(BurnTier.BASELINE)
    mega = get_tier_profile(BurnTier.MEGA)
    assert mega.max_tokens_per_hour > baseline.max_tokens_per_hour > ez.max_tokens_per_hour
    assert (
        mega.max_requests_per_minute > baseline.max_requests_per_minute > ez.max_requests_per_minute
    )


def test_ez_prefers_haiku():
    ez = get_tier_profile(BurnTier.EZ)
    assert ez.preferred_model is not None
    assert "haiku" in ez.preferred_model


def test_profile_descriptions_populated():
    for profile in TIER_PROFILES.values():
        assert len(profile.description) > 10
