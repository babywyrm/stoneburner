from importlib.metadata import version

from fastapi.testclient import TestClient

from atomics.api.auth import ApiKeyAuth
from atomics.api.config import ServerSettings
from atomics.api.server import create_app


def test_create_app_no_auth():
    app = create_app(settings=ServerSettings(no_auth=True))
    with TestClient(app) as client:
        resp = client.get("/api/v1/health")
        assert resp.status_code == 200


def test_create_app_api_key():
    app = create_app(settings=ServerSettings(api_keys={"secret"}))
    with TestClient(app) as client:
        assert isinstance(app.state.auth, ApiKeyAuth)
        # Health is public; protected routes require the API key.
        assert client.get("/api/v1/health").status_code == 200
        resp = client.post("/api/v1/runs", json={"provider": "ollama"})
        assert resp.status_code == 401


def test_create_app_with_log_level():
    app = create_app(settings=ServerSettings(log_level="debug"))
    assert app.state.settings.log_level == "debug"


def test_the_api_advertises_the_installed_package_version():
    """Guards against the drift this replaced: the API had its own hardcoded
    version, so it kept announcing 0.12.0 while the package moved on, and
    atomics.__version__ sat at 0.1.0 through eleven releases."""
    app = create_app(settings=ServerSettings(no_auth=True))
    assert app.version == version("stoneburner-atomics")
