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
