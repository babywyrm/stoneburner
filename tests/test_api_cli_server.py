from click.testing import CliRunner

from atomics.cli import cli


def test_server_command_help():
    runner = CliRunner()
    result = runner.invoke(cli, ["server", "--help"])
    assert result.exit_code == 0
    assert "server" in result.output.lower()


def test_server_help_includes_dashboard_flag():
    runner = CliRunner()
    result = runner.invoke(cli, ["server", "--help"])
    assert result.exit_code == 0
    assert "--with-dashboard" in result.output


def test_server_help_includes_db_path_option():
    runner = CliRunner()
    result = runner.invoke(cli, ["server", "--help"])
    assert result.exit_code == 0
    assert "--db-path" in result.output


def test_server_cli_forwards_job_bounds_to_server_settings(monkeypatch, tmp_path):
    captured: dict = {}

    def fake_create_app(settings=None, **_kwargs):
        captured["settings"] = settings
        return object()

    monkeypatch.setattr("atomics.api.server.create_app", fake_create_app)
    monkeypatch.setattr("uvicorn.run", lambda *_a, **_k: None)

    result = CliRunner().invoke(
        cli,
        [
            "server",
            "--no-auth",
            "--db-path",
            str(tmp_path / "srv.db"),
            "--max-active-jobs",
            "8",
            "--max-retained-jobs",
            "64",
            "--max-active-jobs-per-caller",
            "2",
            "--max-request-bytes",
            "65536",
            "--max-caller-rpm",
            "30",
        ],
    )
    assert result.exit_code == 0, result.output
    s = captured["settings"]
    assert s.max_active_jobs == 8
    assert s.max_retained_jobs == 64
    assert s.max_active_jobs_per_caller == 2
    assert s.max_request_body_bytes == 65536
    assert s.max_requests_per_minute_per_caller == 30


def test_server_cli_forwards_allowed_hosts_and_fails_closed_on_empty(monkeypatch, tmp_path):
    captured: dict = {}

    def fake_create_app(settings=None, **_kwargs):
        captured["settings"] = settings
        return object()

    monkeypatch.setattr("atomics.api.server.create_app", fake_create_app)
    monkeypatch.setattr("uvicorn.run", lambda *_a, **_k: None)

    res = CliRunner().invoke(
        cli,
        [
            "server",
            "--no-auth",
            "--db-path",
            str(tmp_path / "srv.db"),
            "--allowed-host",
            "http://127.0.0.1:11434",
            "--allowed-hosts",
            "https://example.com:8000/v1,http://localhost:8080",
        ],
    )
    assert res.exit_code == 0, res.output
    assert captured["settings"].allowed_hosts == {
        "http://127.0.0.1:11434",
        "https://example.com:8000/v1",
        "http://localhost:8080",
    }

    # Empty string (e.g. unset shell var in --allowed-hosts "$ALLOW") must fail closed.
    bad = CliRunner().invoke(
        cli,
        [
            "server",
            "--no-auth",
            "--db-path",
            str(tmp_path / "srv.db"),
            "--allowed-hosts",
            "   ",
        ],
    )
    assert bad.exit_code != 0
