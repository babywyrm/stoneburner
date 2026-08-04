"""How the `atomics server` command configures logging.

Every case here is a bug found by running a real server and reading the log
file. None of them are visible through TestClient, which never starts uvicorn
and never routes a record through a handler, so they are pinned explicitly.
"""

from __future__ import annotations

import logging
from unittest.mock import patch

from click.testing import CliRunner

from atomics.commands.api import server
from atomics.commands.common import setup_logging


def _run_server(tmp_path, *extra):
    """Invoke the CLI with uvicorn.run stubbed, returning the captured call."""
    with patch("uvicorn.run") as mock_run:
        result = CliRunner().invoke(
            server,
            [
                "--api-key", "k",
                "--db-path", str(tmp_path / "s.db"),
                *extra,
            ],
        )
    assert result.exit_code == 0, result.output
    return mock_run


class TestUvicornAccessLog:
    def test_uvicorn_s_own_access_log_is_disabled(self, tmp_path):
        """It writes the raw request line, query string included.

        A key passed as `?api_key=` landed in the log despite the middleware
        deliberately omitting query strings. Ours replaces it and carries the
        correlation ID and caller besides.
        """
        mock_run = _run_server(tmp_path)
        assert mock_run.call_args.kwargs["access_log"] is False


class TestServerLogHandler:
    def test_the_server_configures_plain_logging(self, tmp_path):
        """Without this, atomics loggers never reach a handler at INFO.

        The access log and every job line vanished silently, so a correlation
        ID correlated nothing.
        """
        with patch("uvicorn.run"), patch(
            "atomics.commands.common.setup_logging"
        ) as mock_setup:
            result = CliRunner().invoke(
                server, ["--api-key", "k", "--db-path", str(tmp_path / "s.db")]
            )

        assert result.exit_code == 0, result.output
        mock_setup.assert_called_once()
        assert mock_setup.call_args.kwargs["plain"] is True

    def test_the_requested_level_is_honored(self, tmp_path):
        with patch("uvicorn.run"), patch(
            "atomics.commands.common.setup_logging"
        ) as mock_setup:
            CliRunner().invoke(
                server,
                [
                    "--api-key", "k",
                    "--db-path", str(tmp_path / "s.db"),
                    "--log-level", "debug",
                ],
            )

        assert mock_setup.call_args.args[0] == "debug"


class TestPlainLogging:
    """Rich wraps to 80 columns when redirected, splitting one access log entry
    across four lines and leaving it unparseable by grep or any aggregator."""

    def test_a_long_record_stays_on_one_line(self, capsys):
        setup_logging("info", plain=True)
        try:
            logging.getLogger("atomics.test").info(
                "request_id=%s caller=%s method=%s path=%s status=%d duration_ms=%.1f",
                "c2efdc5eea0f408b",
                "d7013327e170",
                "GET",
                "/api/v1/distributed/runs/0e6e566871ee",
                200,
                27.4,
            )
        finally:
            logging.basicConfig(force=True)

        written = capsys.readouterr().err.strip()
        assert len(written.splitlines()) == 1
        assert "request_id=c2efdc5eea0f408b" in written
        assert "duration_ms=27.4" in written

    def test_the_line_carries_a_timestamp_and_logger_name(self, capsys):
        setup_logging("info", plain=True)
        try:
            logging.getLogger("atomics.test").info("hello")
        finally:
            logging.basicConfig(force=True)

        written = capsys.readouterr().err.strip()
        assert "atomics.test" in written
        assert "INFO" in written
        assert written.startswith("20")

    def test_rich_is_still_the_default_for_interactive_commands(self):
        from rich.logging import RichHandler

        setup_logging("info")
        try:
            handlers = logging.getLogger().handlers
            assert any(isinstance(h, RichHandler) for h in handlers)
        finally:
            logging.basicConfig(force=True)

    def test_plain_mode_uses_no_rich_handler(self):
        from rich.logging import RichHandler

        setup_logging("info", plain=True)
        try:
            handlers = logging.getLogger().handlers
            assert not any(isinstance(h, RichHandler) for h in handlers)
        finally:
            logging.basicConfig(force=True)

    def test_third_party_loggers_stay_quiet(self):
        setup_logging("debug", plain=True)
        try:
            assert logging.getLogger("atomics").level == logging.DEBUG
            assert logging.getLogger().level == logging.WARNING
        finally:
            logging.basicConfig(force=True)
