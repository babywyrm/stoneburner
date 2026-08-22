"""atomics repl resolves URL/key the same way as atomics mcp."""

from __future__ import annotations

from click.testing import CliRunner

from atomics.cli import cli
from atomics.mcp.client import API_URL_ENV


def test_help_lists_api_flags() -> None:
    result = CliRunner().invoke(cli, ["repl", "--help"])
    assert result.exit_code == 0
    assert "--api-url" in result.output
    assert "--api-key" in result.output


def test_command_is_registered() -> None:
    assert "repl" in cli.commands


def test_flags_take_precedence_over_environment(monkeypatch) -> None:
    captured: dict = {}

    def fake_run(client, **kwargs):
        captured["client"] = client
        return 0

    import atomics.commands.repl as repl_mod

    monkeypatch.setattr(repl_mod, "run_repl", fake_run)
    monkeypatch.setenv(API_URL_ENV, "http://from-env:1")
    result = CliRunner().invoke(cli, ["repl", "--api-url", "http://from-flag:2", "--api-key", "k"])
    assert result.exit_code == 0
    assert captured["client"].base_url == "http://from-flag:2"
