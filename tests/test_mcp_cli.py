"""Tests for the `atomics mcp` command.

The command resolves configuration and starts a server, so these check the
resolution order and the one hazard specific to stdio transport: stdout carries
the JSON-RPC frames, and anything else written there corrupts the session.
"""

from __future__ import annotations

import pytest
from click.testing import CliRunner

from atomics.cli import cli
from atomics.mcp.client import API_KEY_ENV, API_URL_ENV, DEFAULT_API_URL


class FakeServer:
    def __init__(self):
        self.ran = False
        self.run_kwargs = None

    def run(self, **kwargs):
        self.ran = True
        self.run_kwargs = kwargs


@pytest.fixture
def started(monkeypatch):
    """Capture what the command builds without starting a real server."""
    # The command imports the SDK lazily; this fixture is the first thing that
    # needs it. Skip here so `--help` still runs without the extra.
    pytest.importorskip("mcp")
    captured: dict = {}

    def build_server(client=None):
        captured["client"] = client
        captured["server"] = FakeServer()
        return captured["server"]

    def setup_logging(level, **kwargs):
        captured["logging"] = (level, kwargs)

    import atomics.commands.common as common_module
    import atomics.mcp.server as server_module

    monkeypatch.setattr(server_module, "build_server", build_server)
    monkeypatch.setattr(common_module, "setup_logging", setup_logging)
    monkeypatch.delenv(API_URL_ENV, raising=False)
    monkeypatch.delenv(API_KEY_ENV, raising=False)
    return captured


def test_help_describes_the_proxy():
    result = CliRunner().invoke(cli, ["mcp", "--help"])
    assert result.exit_code == 0
    assert "--api-url" in result.output
    assert "--api-key" in result.output


def test_defaults_to_loopback_api(started):
    result = CliRunner().invoke(cli, ["mcp", "--api-key", "k"])
    assert result.exit_code == 0
    assert started["client"].base_url == DEFAULT_API_URL
    assert started["server"].ran is True


def test_serves_stdio_only(started):
    """No HTTP transport is offered. This process holds an API key with spend
    authority and nothing at the MCP layer would authenticate a network caller,
    so a listening port would hand that authority to anyone who could reach it.
    """
    assert CliRunner().invoke(cli, ["mcp", "--api-key", "k"]).exit_code == 0
    assert started["server"].run_kwargs == {}

    rejected = CliRunner().invoke(cli, ["mcp", "--transport", "streamable-http"])
    assert rejected.exit_code != 0


def test_flags_take_precedence_over_environment(started, monkeypatch):
    monkeypatch.setenv(API_URL_ENV, "http://from-env:1")
    result = CliRunner().invoke(cli, ["mcp", "--api-url", "http://from-flag:2", "--api-key", "k"])
    assert result.exit_code == 0
    assert started["client"].base_url == "http://from-flag:2"


def test_environment_is_used_when_no_flag(started, monkeypatch):
    monkeypatch.setenv(API_URL_ENV, "http://from-env:1")
    result = CliRunner().invoke(cli, ["mcp", "--api-key", "k"])
    assert result.exit_code == 0
    assert started["client"].base_url == "http://from-env:1"


def test_missing_api_key_warns_rather_than_failing(started):
    """Anonymous works against a --no-auth server, so this is a warning. Silence
    would leave a 401 against an authenticated server unexplained."""
    result = CliRunner().invoke(cli, ["mcp"])
    assert result.exit_code == 0
    assert "No API key" in result.stderr


def test_stdout_stays_clean_for_the_protocol(started):
    """On stdio, stdout is the JSON-RPC channel. Status output belongs on
    stderr; a single stray line here would break every MCP session."""
    result = CliRunner().invoke(cli, ["mcp", "--api-key", "k"])
    assert result.exit_code == 0
    assert result.stdout == ""
    assert "atomics MCP server" in result.stderr


def test_logging_is_forced_to_plain_so_records_go_to_stderr(started):
    """The CLI group installs a Rich handler that writes to stdout. Leaving it
    in place would let any warning corrupt the protocol stream."""
    result = CliRunner().invoke(cli, ["mcp", "--api-key", "k"])
    assert result.exit_code == 0
    assert started["logging"][1]["plain"] is True
