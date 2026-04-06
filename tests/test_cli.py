"""Tests for CLI commands (non-network)."""

from click.testing import CliRunner

from atomics.cli import cli


def test_cli_help():
    runner = CliRunner()
    result = runner.invoke(cli, ["--help"])
    assert result.exit_code == 0
    assert "Atomics" in result.output


def test_cli_version():
    runner = CliRunner()
    result = runner.invoke(cli, ["--version"])
    assert result.exit_code == 0
    assert "0.1.0" in result.output


def test_cli_tiers_command():
    runner = CliRunner()
    result = runner.invoke(cli, ["tiers"])
    assert result.exit_code == 0
    assert "EZ" in result.output
    assert "BASELINE" in result.output
    assert "MEGA" in result.output


def test_cli_schedule_crontab():
    runner = CliRunner()
    result = runner.invoke(cli, ["schedule", "--format", "crontab", "--tier", "mega"])
    assert result.exit_code == 0
    assert "--tier mega" in result.output
    assert "crontab" in result.output


def test_cli_schedule_systemd():
    runner = CliRunner()
    result = runner.invoke(cli, ["schedule", "--format", "systemd", "--tier", "ez"])
    assert result.exit_code == 0
    assert "[Service]" in result.output
    assert "--tier ez" in result.output


def test_cli_schedule_launchd():
    runner = CliRunner()
    result = runner.invoke(cli, ["schedule", "--format", "launchd"])
    assert result.exit_code == 0
    assert "com.babywyrm.atomics" in result.output
    assert "RunAtLoad" in result.output


def test_cli_run_no_api_key():
    runner = CliRunner(env={"ANTHROPIC_API_KEY": ""})
    result = runner.invoke(cli, ["run", "-n", "1"])
    assert result.exit_code != 0
    assert "ANTHROPIC_API_KEY" in result.output


def test_cli_report(tmp_path):
    runner = CliRunner(env={"ATOMICS_DB_PATH": str(tmp_path / "test.db")})
    result = runner.invoke(cli, ["report"])
    assert result.exit_code == 0
    assert "No runs" in result.output
