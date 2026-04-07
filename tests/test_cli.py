"""Tests for CLI commands (non-network)."""

from types import SimpleNamespace

import pytest
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


def test_cli_schedule_install_and_uninstall(monkeypatch):
    runner = CliRunner()

    monkeypatch.setattr(
        "atomics.scheduler.cron.install_crontab", lambda _: "Crontab entry installed"
    )
    monkeypatch.setattr(
        "atomics.scheduler.cron.uninstall_crontab",
        lambda: "Atomics crontab entries removed",
    )

    install = runner.invoke(
        cli,
        ["schedule", "--format", "crontab", "--install", "--tier", "ez"],
    )
    assert install.exit_code == 0
    assert "installed" in install.output.lower()

    uninstall = runner.invoke(
        cli,
        ["schedule", "--format", "crontab", "--uninstall", "--tier", "ez"],
    )
    assert uninstall.exit_code == 0
    assert "removed" in uninstall.output.lower()


def test_cli_run_with_mocked_claude(monkeypatch, tmp_path):
    runner = CliRunner(
        env={
            "ANTHROPIC_API_KEY": "fake",
            "ATOMICS_DB_PATH": str(tmp_path / "db.sqlite"),
        }
    )
    calls = {"ran": False}

    class DummyEngine:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        async def run(self, max_iterations=None):
            calls["ran"] = True
            calls["max_iterations"] = max_iterations

    class DummyRepo:
        def __init__(self, _):
            pass

        def close(self):
            pass

    class DummyClaude:
        def __init__(self, api_key, default_model):
            self.api_key = api_key
            self.default_model = default_model
            self.name = "claude"

    monkeypatch.setattr("atomics.core.engine.LoopEngine", DummyEngine)
    monkeypatch.setattr("atomics.storage.repository.MetricsRepository", DummyRepo)
    monkeypatch.setattr("atomics.providers.claude.ClaudeProvider", DummyClaude)

    result = runner.invoke(
        cli,
        ["run", "--tier", "ez", "--provider", "claude", "-n", "2", "-i", "1"],
    )
    assert result.exit_code == 0
    assert calls["ran"] is True
    assert calls["max_iterations"] == 2


def test_cli_run_with_mocked_bedrock(monkeypatch, tmp_path):
    runner = CliRunner(env={"ATOMICS_DB_PATH": str(tmp_path / "db.sqlite")})
    calls = {"ran": False}

    class DummyEngine:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        async def run(self, max_iterations=None):
            calls["ran"] = True

    class DummyRepo:
        def __init__(self, _):
            pass

        def close(self):
            pass

    class DummyBedrock:
        name = "bedrock"

        def __init__(self, region, model_id):
            self.region = region
            self.model_id = model_id

    monkeypatch.setattr("atomics.core.engine.LoopEngine", DummyEngine)
    monkeypatch.setattr("atomics.storage.repository.MetricsRepository", DummyRepo)
    monkeypatch.setattr("atomics.providers.bedrock.BedrockProvider", DummyBedrock)

    result = runner.invoke(
        cli,
        ["run", "--provider", "bedrock", "--region", "us-east-1", "-n", "1"],
    )
    assert result.exit_code == 0
    assert calls["ran"] is True


@pytest.mark.unit
def test_cli_provider_test_missing_api_key():
    runner = CliRunner(env={"ANTHROPIC_API_KEY": ""})
    result = runner.invoke(cli, ["provider-test"])
    assert result.exit_code != 0
    assert "ANTHROPIC_API_KEY" in result.output


@pytest.mark.unit
def test_cli_run_keyboard_interrupt(monkeypatch, tmp_path):
    """KeyboardInterrupt during the async loop should print the friendly message."""
    runner = CliRunner(
        env={
            "ANTHROPIC_API_KEY": "fake",
            "ATOMICS_DB_PATH": str(tmp_path / "db.sqlite"),
        }
    )

    class InterruptEngine:
        def __init__(self, **_kwargs):
            pass

        async def run(self, max_iterations=None):
            raise KeyboardInterrupt

    class DummyRepo:
        def __init__(self, _):
            pass

        def close(self):
            pass

    class DummyClaude:
        name = "claude"

        def __init__(self, api_key, default_model):
            self.api_key = api_key
            self.default_model = default_model

    monkeypatch.setattr("atomics.core.engine.LoopEngine", InterruptEngine)
    monkeypatch.setattr("atomics.storage.repository.MetricsRepository", DummyRepo)
    monkeypatch.setattr("atomics.providers.claude.ClaudeProvider", DummyClaude)

    result = runner.invoke(cli, ["run", "-n", "1"])
    assert result.exit_code == 0
    assert "Interrupted" in result.output


def test_cli_doctor(tmp_path):
    runner = CliRunner(env={"ATOMICS_DB_PATH": str(tmp_path / "doctor.db")})
    result = runner.invoke(cli, ["doctor"])
    assert result.exit_code == 0
    assert "Python" in result.output


def test_cli_export_jsonl_empty_db(tmp_path):
    runner = CliRunner(env={"ATOMICS_DB_PATH": str(tmp_path / "empty.db")})
    result = runner.invoke(cli, ["export", "--format", "jsonl"])
    assert result.exit_code == 0


def test_cli_export_csv_with_row(tmp_path):
    from datetime import UTC, datetime

    from atomics.models import TaskCategory, TaskResult, TaskStatus
    from atomics.storage.repository import MetricsRepository

    db = tmp_path / "e.db"
    repo = MetricsRepository(db)
    repo.create_run("r1")
    repo.save_task_result(
        TaskResult(
            run_id="r1",
            category=TaskCategory.GENERAL_QA,
            task_name="x",
            provider="claude",
            model="m",
            status=TaskStatus.SUCCESS,
            total_tokens=5,
            estimated_cost_usd=0.0,
            started_at=datetime.now(UTC),
            completed_at=datetime.now(UTC),
        )
    )
    repo.close()
    runner = CliRunner(env={"ATOMICS_DB_PATH": str(db)})
    result = runner.invoke(cli, ["export", "--format", "csv"])
    assert result.exit_code == 0
    assert "task_name" in result.output


def test_cli_completion_zsh():
    runner = CliRunner()
    result = runner.invoke(cli, ["completion", "zsh"])
    assert result.exit_code == 0
    assert "compdef" in result.output or "_atomics_completion" in result.output


def test_cli_run_post_hook(monkeypatch, tmp_path):
    runner = CliRunner(
        env={
            "ANTHROPIC_API_KEY": "fake",
            "ATOMICS_DB_PATH": str(tmp_path / "db.sqlite"),
        }
    )
    hook_fired: list[tuple[str, dict]] = []

    def capture_hook(cmd, env):
        hook_fired.append((cmd, dict(env)))
        return 0

    class DummyEngine:
        def __init__(self, **kwargs):
            pass

        async def run(self, max_iterations=None):
            from datetime import UTC, datetime

            from atomics.models import RunSummary

            return RunSummary(
                run_id="hookrun",
                started_at=datetime.now(UTC),
                completed_at=datetime.now(UTC),
                total_tasks=1,
                successful_tasks=1,
                failed_tasks=0,
                total_tokens=1,
                total_cost_usd=0.0,
            )

    class DummyRepo:
        def __init__(self, _):
            pass

        def close(self):
            pass

    class DummyClaude:
        name = "claude"

        def __init__(self, api_key, default_model):
            pass

    monkeypatch.setattr("atomics.core.engine.LoopEngine", DummyEngine)
    monkeypatch.setattr("atomics.storage.repository.MetricsRepository", DummyRepo)
    monkeypatch.setattr("atomics.providers.claude.ClaudeProvider", DummyClaude)
    monkeypatch.setattr("atomics.hooks.run_post_hook", capture_hook)
    monkeypatch.setattr("atomics.hooks.notify_run_complete", lambda *_a, **_k: None)

    result = runner.invoke(
        cli,
        ["run", "--tier", "ez", "--provider", "claude", "-n", "1", "-i", "1", "--hook", "echo ok"],
    )
    assert result.exit_code == 0
    assert hook_fired
    assert hook_fired[0][0] == "echo ok"
    assert hook_fired[0][1]["ATOMICS_RUN_ID"] == "hookrun"


def test_cli_provider_test_success(monkeypatch):
    runner = CliRunner(env={"ANTHROPIC_API_KEY": "fake"})

    class DummyProvider:
        name = "claude"

        def __init__(self, api_key, default_model):
            self.api_key = api_key
            self.default_model = default_model

        async def health_check(self):
            return True

        async def generate(self, *_args, **_kwargs):
            return SimpleNamespace(
                text="4",
                input_tokens=1,
                output_tokens=1,
                total_tokens=2,
                latency_ms=1.0,
                estimated_cost_usd=0.0,
            )

    monkeypatch.setattr("atomics.providers.claude.ClaudeProvider", DummyProvider)
    result = runner.invoke(cli, ["provider-test"])
    assert result.exit_code == 0
    assert "passed" in result.output.lower()
