"""Tests for configuration loading."""

import os

from atomics.config import AtomicsSettings, load_settings


def test_defaults():
    settings = AtomicsSettings(anthropic_api_key="test-key")
    assert settings.max_tokens_per_hour == 100_000
    assert settings.budget_limit_usd == 50.0
    assert settings.loop_interval_seconds == 120


def test_load_settings_returns_instance():
    settings = load_settings()
    assert isinstance(settings, AtomicsSettings)
