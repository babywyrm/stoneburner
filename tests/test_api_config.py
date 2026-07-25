from atomics.api.config import ServerSettings


def test_server_settings_defaults():
    settings = ServerSettings()
    assert settings.host == "127.0.0.1"
    assert settings.port == 8000
    assert settings.no_auth is False
    assert settings.api_keys == set()


def test_server_settings_invalid_port():
    import pytest
    with pytest.raises(ValueError):
        ServerSettings(port=0)


def test_worker_absence_threshold_defaults_to_four_missed_heartbeats():
    assert ServerSettings().worker_absent_after_seconds == 120.0


def test_a_non_positive_absence_threshold_is_rejected():
    """Zero or negative would mark every worker absent the moment it registers."""
    import pytest
    with pytest.raises(ValueError, match="must be positive"):
        ServerSettings(worker_absent_after_seconds=0)
