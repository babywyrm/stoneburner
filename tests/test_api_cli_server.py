from click.testing import CliRunner

from atomics.cli import cli


def test_server_command_help():
    runner = CliRunner()
    result = runner.invoke(cli, ["server", "--help"])
    assert result.exit_code == 0
    assert "server" in result.output.lower()
