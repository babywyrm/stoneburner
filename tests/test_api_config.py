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
