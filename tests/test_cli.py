"""Tests for CLI commands (non-network)."""

from datetime import UTC
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
    assert "version" in result.output.lower()


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


def test_cli_run_openai_no_api_key(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "atomics.auth.codex._default_codex_auth_path",
        lambda: tmp_path / "nonexistent.json",
    )
    monkeypatch.setattr("atomics.auth.store._default_auth_dir", lambda: tmp_path)
    runner = CliRunner(env={"OPENAI_API_KEY": ""})
    result = runner.invoke(cli, ["run", "--provider", "openai", "-n", "1"])
    assert result.exit_code != 0
    assert "No OpenAI credentials" in result.output


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


def test_cli_run_with_mocked_openai(monkeypatch, tmp_path):
    runner = CliRunner(
        env={
            "OPENAI_API_KEY": "fake",
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

    class DummyOpenAI:
        name = "openai"

        def __init__(self, api_key, default_model):
            self.api_key = api_key
            self.default_model = default_model

    monkeypatch.setattr("atomics.core.engine.LoopEngine", DummyEngine)
    monkeypatch.setattr("atomics.storage.repository.MetricsRepository", DummyRepo)
    monkeypatch.setattr("atomics.providers.openai.OpenAIProvider", DummyOpenAI)

    result = runner.invoke(
        cli,
        ["run", "--provider", "openai", "--tier", "ez", "-n", "3", "-i", "1"],
    )
    assert result.exit_code == 0
    assert calls["ran"] is True
    assert calls["max_iterations"] == 3


def test_cli_provider_test_openai_missing_key(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "atomics.auth.codex._default_codex_auth_path",
        lambda: tmp_path / "nonexistent.json",
    )
    monkeypatch.setattr("atomics.auth.store._default_auth_dir", lambda: tmp_path)
    runner = CliRunner(env={"OPENAI_API_KEY": ""})
    result = runner.invoke(cli, ["provider-test", "--provider", "openai"])
    assert result.exit_code != 0
    assert "No OpenAI credentials" in result.output


def test_cli_provider_test_bedrock_success(monkeypatch):
    runner = CliRunner()

    class DummyProvider:
        name = "bedrock"

        def __init__(self, region, model_id):
            pass

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
                tokens_per_second=None,
                tps_basis="wall_clock",
                thinking_tokens=0,
                cache_read_tokens=0,
                cache_write_tokens=0,
            )

    monkeypatch.setattr("atomics.providers.bedrock.BedrockProvider", DummyProvider)
    result = runner.invoke(cli, ["provider-test", "--provider", "bedrock"])
    assert result.exit_code == 0
    assert "passed" in result.output.lower()


def test_cli_provider_test_openai_success(monkeypatch):
    runner = CliRunner(env={"OPENAI_API_KEY": "fake"})

    class DummyProvider:
        name = "openai"

        def __init__(self, api_key, default_model):
            pass

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
                tokens_per_second=None,
                tps_basis="wall_clock",
                thinking_tokens=0,
                cache_read_tokens=0,
                cache_write_tokens=0,
            )

    monkeypatch.setattr("atomics.providers.openai.OpenAIProvider", DummyProvider)
    result = runner.invoke(cli, ["provider-test", "--provider", "openai"])
    assert result.exit_code == 0
    assert "passed" in result.output.lower()


def test_cli_schedule_crontab_with_provider():
    runner = CliRunner()
    result = runner.invoke(
        cli, ["schedule", "--format", "crontab", "--tier", "ez", "--provider", "openai"]
    )
    assert result.exit_code == 0
    assert "--provider openai" in result.output


def test_cli_schedule_systemd_with_provider():
    runner = CliRunner()
    result = runner.invoke(
        cli, ["schedule", "--format", "systemd", "--tier", "baseline", "--provider", "bedrock"]
    )
    assert result.exit_code == 0
    assert "--provider bedrock" in result.output


def test_cli_schedule_launchd_with_provider():
    runner = CliRunner()
    result = runner.invoke(
        cli, ["schedule", "--format", "launchd", "--provider", "openai"]
    )
    assert result.exit_code == 0
    assert "openai" in result.output


def test_cli_schedule_status_empty(tmp_path):
    runner = CliRunner(env={"ATOMICS_DB_PATH": str(tmp_path / "status.db")})
    result = runner.invoke(cli, ["schedule-status"])
    assert result.exit_code == 0
    assert "No schedules" in result.output


def test_cli_schedule_status_with_entries(tmp_path, monkeypatch):
    from atomics.storage.repository import MetricsRepository

    db_path = tmp_path / "status2.db"
    repo = MetricsRepository(db_path)
    repo.save_schedule(
        schedule_id="launchd.ez.bedrock",
        format="launchd",
        tier="ez",
        provider="bedrock",
        model=None,
        interval_minutes=30,
        max_iterations=10,
    )
    repo.close()

    monkeypatch.setattr(
        "atomics.scheduler.cron.check_schedule_health",
        lambda fmt, tier, **kw: True,
    )
    runner = CliRunner(env={"ATOMICS_DB_PATH": str(db_path)})
    result = runner.invoke(cli, ["schedule-status"])
    assert result.exit_code == 0
    assert "Installed Schedules" in result.output
    assert "bedrock" in result.output or "bedr" in result.output


def test_cli_compare_empty(tmp_path):
    runner = CliRunner(env={"ATOMICS_DB_PATH": str(tmp_path / "cmp.db")})
    result = runner.invoke(cli, ["compare"])
    assert result.exit_code == 0
    assert "No data" in result.output


def test_cli_compare_with_data(tmp_path):
    from datetime import UTC, datetime

    from atomics.models import TaskCategory, TaskResult, TaskStatus
    from atomics.storage.repository import MetricsRepository

    db_path = tmp_path / "cmp2.db"
    repo = MetricsRepository(db_path)
    for provider, run_id in [("claude", "c1"), ("bedrock", "c2")]:
        repo.create_run(run_id, provider=provider, tier="ez")
        for i in range(2):
            result = TaskResult(
                task_id=f"{run_id}-t{i}",
                run_id=run_id,
                category=TaskCategory.GENERAL_QA,
                task_name="q",
                provider=provider,
                model="m",
                status=TaskStatus.SUCCESS,
                total_tokens=100,
                latency_ms=200.0,
                estimated_cost_usd=0.01,
                started_at=datetime.now(UTC),
                completed_at=datetime.now(UTC),
            )
            repo.save_task_result(result)
        repo.complete_run(run_id)
    repo.close()

    runner = CliRunner(env={"ATOMICS_DB_PATH": str(db_path)})
    result = runner.invoke(cli, ["compare"])
    assert result.exit_code == 0
    assert "Comparison" in result.output


def test_cli_compare_by_model(tmp_path):
    from datetime import UTC, datetime

    from atomics.models import TaskCategory, TaskResult, TaskStatus
    from atomics.storage.repository import MetricsRepository

    db_path = tmp_path / "cmp3.db"
    repo = MetricsRepository(db_path)
    repo.create_run("m1", provider="openai")
    for model, tid in [("gpt-4o", "t1"), ("gpt-4o-mini", "t2")]:
        result = TaskResult(
            task_id=tid,
            run_id="m1",
            category=TaskCategory.GENERAL_QA,
            task_name="q",
            provider="openai",
            model=model,
            status=TaskStatus.SUCCESS,
            total_tokens=50,
            latency_ms=100.0,
            estimated_cost_usd=0.005,
            started_at=datetime.now(UTC),
            completed_at=datetime.now(UTC),
        )
        repo.save_task_result(result)
    repo.complete_run("m1")
    repo.close()

    runner = CliRunner(env={"ATOMICS_DB_PATH": str(db_path)})
    result = runner.invoke(cli, ["compare", "--by", "model"])
    assert result.exit_code == 0
    assert "Model" in result.output


def test_cli_compare_with_filters(tmp_path):
    runner = CliRunner(env={"ATOMICS_DB_PATH": str(tmp_path / "cmpf.db")})
    result = runner.invoke(cli, ["compare", "--since-hours", "24", "--tier", "ez"])
    assert result.exit_code == 0


def test_cli_compare_warns_on_mixed_throughput_basis(tmp_path):
    """A wall_clock provider next to a generation provider must trigger the basis warning."""
    from datetime import UTC, datetime

    from atomics.models import TaskCategory, TaskResult, TaskStatus
    from atomics.storage.repository import MetricsRepository

    db_path = tmp_path / "cmp_basis.db"
    repo = MetricsRepository(db_path)
    specs = [
        ("openai", "gpt-4o", "wall_clock", "ob1"),
        ("ollama", "qwen2.5:7b", "generation", "ol1"),
    ]
    for provider, model, basis, run_id in specs:
        repo.create_run(run_id, provider=provider, tier="ez")
        repo.save_task_result(
            TaskResult(
                task_id=f"{run_id}-t",
                run_id=run_id,
                category=TaskCategory.GENERAL_QA,
                task_name="q",
                provider=provider,
                model=model,
                status=TaskStatus.SUCCESS,
                total_tokens=100,
                latency_ms=200.0,
                tokens_per_second=80.0,
                tps_basis=basis,
                estimated_cost_usd=0.0,
                started_at=datetime.now(UTC),
                completed_at=datetime.now(UTC),
            )
        )
        repo.complete_run(run_id)
    repo.close()

    runner = CliRunner(env={"ATOMICS_DB_PATH": str(db_path)})
    result = runner.invoke(cli, ["compare"])
    assert result.exit_code == 0
    # The warning only fires when tps_bases is populated from the DB and parsed
    # into more than one distinct basis across the compared groups.
    assert "Mixed throughput bases" in result.output


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
                tokens_per_second=None,
                tps_basis="wall_clock",
                thinking_tokens=0,
                cache_read_tokens=0,
                cache_write_tokens=0,
            )

    monkeypatch.setattr("atomics.providers.claude.ClaudeProvider", DummyProvider)
    result = runner.invoke(cli, ["provider-test"])
    assert result.exit_code == 0
    assert "passed" in result.output.lower()


def test_cli_models_command(monkeypatch):
    """atomics models should list Ollama models with class/thinking annotations."""

    mock_models = [
        {"name": "qwen2.5:7b", "size_gb": 4.7, "parameter_size": "7.6B",
         "family": "qwen2.5", "model_class": "mid", "thinking": False},
        {"name": "deepseek-r1:14b", "size_gb": 9.0, "parameter_size": "14.8B",
         "family": "deepseek", "model_class": "mid", "thinking": True},
    ]

    class FakeOllama:
        def __init__(self, **_kw):
            pass
        async def list_models(self):
            return mock_models

    monkeypatch.setattr("atomics.providers.ollama.OllamaProvider", FakeOllama)
    runner = CliRunner()
    result = runner.invoke(cli, ["models", "--host", "http://fake:11434"])
    assert result.exit_code == 0
    assert "qwen2.5:7b" in result.output
    assert "deepseek-r1:14b" in result.output
    assert "mid" in result.output


def test_cli_models_connection_error(monkeypatch):
    """atomics models should handle connection errors gracefully."""
    class FakeOllama:
        def __init__(self, **_kw):
            pass
        async def list_models(self):
            raise ConnectionError("Cannot connect to Ollama at http://fake:11434")

    monkeypatch.setattr("atomics.providers.ollama.OllamaProvider", FakeOllama)
    runner = CliRunner()
    result = runner.invoke(cli, ["models", "--host", "http://fake:11434"])
    assert result.exit_code == 1
    assert "Cannot connect" in result.output


def test_cli_sweep_command(monkeypatch):
    """atomics sweep should run eval across multiple models."""
    from atomics.sweep import ModelSweepResult

    mock_results = [
        ModelSweepResult(
            model="qwen2.5:1.5b", fixtures_run=2, overall_quality=0.85,
            avg_latency_ms=150.0, total_tokens=500, total_cost_usd=0.0,
            value_score=850.0, eval_summary=None,
        ),
        ModelSweepResult(
            model="mistral:7b", fixtures_run=2, overall_quality=0.72,
            avg_latency_ms=300.0, total_tokens=800, total_cost_usd=0.0,
            value_score=720.0, eval_summary=None,
        ),
    ]

    async def fake_sweep(**kwargs):
        cb = kwargs.get("on_model_done")
        for r in mock_results:
            if cb:
                cb(r)
        return mock_results

    monkeypatch.setattr("atomics.sweep.run_model_sweep", fake_sweep)
    runner = CliRunner()
    result = runner.invoke(cli, [
        "sweep", "--models", "qwen2.5:1.5b,mistral:7b",
        "--host", "http://fake:11434",
    ])
    assert result.exit_code == 0
    assert "qwen2.5:1.5b" in result.output
    assert "mistral:7b" in result.output
    assert "85" in result.output  # 85% quality


def test_cli_sweep_cloud_provider(monkeypatch):
    """atomics sweep --provider claude should work with cloud models."""
    from atomics.sweep import ModelSweepResult

    mock_results = [
        ModelSweepResult(
            model="claude-sonnet-4-6", fixtures_run=2, overall_quality=0.92,
            avg_latency_ms=800.0, total_tokens=1200, total_cost_usd=0.036,
            value_score=30.7, eval_summary=None,
        ),
    ]

    factory_models: list[str] = []

    async def fake_sweep(**kwargs):
        factory = kwargs["provider_factory"]
        for m in kwargs["models"]:
            p = factory(m)
            factory_models.append(p._default_model if hasattr(p, "_default_model") else m)
        cb = kwargs.get("on_model_done")
        for r in mock_results:
            if cb:
                cb(r)
        return mock_results

    monkeypatch.setattr("atomics.sweep.run_model_sweep", fake_sweep)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-fake-key")
    runner = CliRunner()
    result = runner.invoke(cli, [
        "sweep", "--provider", "claude",
        "--models", "claude-sonnet-4-6",
    ])
    assert result.exit_code == 0
    assert "claude-sonnet-4-6" in result.output
    assert "92" in result.output
    assert "$0.036" in result.output


def test_cli_sweep_openai_provider(monkeypatch):
    """atomics sweep --provider openai should work with OpenAI models."""
    from atomics.sweep import ModelSweepResult

    mock_results = [
        ModelSweepResult(
            model="gpt-4o", fixtures_run=2, overall_quality=0.88,
            avg_latency_ms=600.0, total_tokens=900, total_cost_usd=0.027,
            value_score=32.6, eval_summary=None,
        ),
    ]

    async def fake_sweep(**kwargs):
        cb = kwargs.get("on_model_done")
        for r in mock_results:
            if cb:
                cb(r)
        return mock_results

    monkeypatch.setattr("atomics.sweep.run_model_sweep", fake_sweep)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-proj-test-fake-key")
    runner = CliRunner()
    result = runner.invoke(cli, [
        "sweep", "--provider", "openai",
        "--models", "gpt-4o",
    ])
    assert result.exit_code == 0
    assert "gpt-4o" in result.output


def test_cli_sweep_requires_models_or_all_local(monkeypatch):
    """atomics sweep without --models or --all-local should fail."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-fake-key")
    runner = CliRunner()
    result = runner.invoke(cli, ["sweep", "--provider", "claude"])
    assert result.exit_code == 1


def test_cli_sweep_verbose_shows_replies(monkeypatch):
    """atomics sweep --verbose should print actual model replies."""
    from datetime import UTC, datetime

    from atomics.eval.fixtures import EvalFixture
    from atomics.eval.judge import JudgeResult
    from atomics.eval.runner import EvalRunSummary, FixtureResult
    from atomics.models import TaskCategory, TaskComplexity, TaskResult, TaskStatus
    from atomics.sweep import ModelSweepResult

    fixture = EvalFixture(
        id="ev-01", prompt="What is X?", complexity=TaskComplexity.LIGHT,
        gold_criteria="Explain X", max_output_tokens=512,
    )
    task = TaskResult(
        run_id="test", category=TaskCategory.GENERAL_QA,
        task_name="ev-01", provider="claude", model="claude-sonnet-4-6",
        prompt="What is X?", response="X is a fantastic thing that does Y and Z.",
        status=TaskStatus.SUCCESS, input_tokens=50, output_tokens=100,
        total_tokens=150, latency_ms=800.0, estimated_cost_usd=0.003,
    )
    judge = JudgeResult(score=0.9, accuracy=4, completeness=3, format_score=3,
                        rationale="Good answer", judge_model="qwen2.5:7b")
    fr = FixtureResult(fixture=fixture, task_result=task, judge=judge)
    summary = EvalRunSummary(
        run_id="test", provider="claude", model="claude-sonnet-4-6",
        judge_provider="ollama", judge_model="qwen2.5:7b",
        started_at=datetime.now(UTC), completed_at=datetime.now(UTC),
        fixture_results=[fr],
    )
    mock_results = [
        ModelSweepResult(
            model="claude-sonnet-4-6", fixtures_run=1, overall_quality=0.9,
            avg_latency_ms=800.0, total_tokens=150, total_cost_usd=0.003,
            value_score=300.0, eval_summary=summary,
        ),
    ]

    async def fake_sweep(**kwargs):
        fixture_cb = kwargs.get("on_fixture_done")
        model_cb = kwargs.get("on_model_done")
        for r in mock_results:
            if fixture_cb and r.eval_summary:
                for fix_r in r.eval_summary.fixture_results:
                    fixture_cb(fix_r)
            if model_cb:
                model_cb(r)
        return mock_results

    monkeypatch.setattr("atomics.sweep.run_model_sweep", fake_sweep)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-fake-key")
    runner = CliRunner()
    result = runner.invoke(cli, [
        "sweep", "--provider", "claude", "--models", "claude-sonnet-4-6", "--verbose",
    ])
    assert result.exit_code == 0
    assert "X is a fantastic thing" in result.output
    assert "ev-01" in result.output


def test_cli_sweep_no_verbose_hides_replies(monkeypatch):
    """Without --verbose, sweep should NOT print full replies."""
    from datetime import UTC, datetime

    from atomics.eval.fixtures import EvalFixture
    from atomics.eval.judge import JudgeResult
    from atomics.eval.runner import EvalRunSummary, FixtureResult
    from atomics.models import TaskCategory, TaskComplexity, TaskResult, TaskStatus
    from atomics.sweep import ModelSweepResult

    fixture = EvalFixture(
        id="ev-01", prompt="What is X?", complexity=TaskComplexity.LIGHT,
        gold_criteria="Explain X", max_output_tokens=512,
    )
    task = TaskResult(
        run_id="test", category=TaskCategory.GENERAL_QA,
        task_name="ev-01", provider="claude", model="claude-sonnet-4-6",
        prompt="What is X?", response="X is a fantastic thing that does Y and Z.",
        status=TaskStatus.SUCCESS, input_tokens=50, output_tokens=100,
        total_tokens=150, latency_ms=800.0, estimated_cost_usd=0.003,
    )
    judge = JudgeResult(score=0.9, accuracy=4, completeness=3, format_score=3,
                        rationale="Good answer", judge_model="qwen2.5:7b")
    fr = FixtureResult(fixture=fixture, task_result=task, judge=judge)
    summary = EvalRunSummary(
        run_id="test", provider="claude", model="claude-sonnet-4-6",
        judge_provider="ollama", judge_model="qwen2.5:7b",
        started_at=datetime.now(UTC), completed_at=datetime.now(UTC),
        fixture_results=[fr],
    )
    mock_results = [
        ModelSweepResult(
            model="claude-sonnet-4-6", fixtures_run=1, overall_quality=0.9,
            avg_latency_ms=800.0, total_tokens=150, total_cost_usd=0.003,
            value_score=300.0, eval_summary=summary,
        ),
    ]

    async def fake_sweep(**kwargs):
        fixture_cb = kwargs.get("on_fixture_done")
        model_cb = kwargs.get("on_model_done")
        for r in mock_results:
            if fixture_cb and r.eval_summary:
                for fix_r in r.eval_summary.fixture_results:
                    fixture_cb(fix_r)
            if model_cb:
                model_cb(r)
        return mock_results

    monkeypatch.setattr("atomics.sweep.run_model_sweep", fake_sweep)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-fake-key")
    runner = CliRunner()
    result = runner.invoke(cli, [
        "sweep", "--provider", "claude", "--models", "claude-sonnet-4-6",
    ])
    assert result.exit_code == 0
    assert "X is a fantastic thing" not in result.output


# ---------------------------------------------------------------------------
# vllm provider — CLI path tests
# ---------------------------------------------------------------------------

def test_cli_run_with_mocked_vllm(monkeypatch, tmp_path):
    """atomics run --provider vllm should construct VllmProvider and run."""
    from types import SimpleNamespace
    from unittest.mock import AsyncMock, MagicMock

    from atomics.models import BurnTier

    fake_resp = SimpleNamespace(
        text="ok", input_tokens=10, output_tokens=20, total_tokens=30,
        model="qwen2.5:3b", latency_ms=120.0, estimated_cost_usd=0.0,
        tokens_per_second=100.0, thinking_tokens=0, thinking_text="",
        tps_basis="wall_clock", cache_read_tokens=0, cache_write_tokens=0,
    )
    fake_provider = MagicMock()
    fake_provider.name = "vllm"
    fake_provider.generate = AsyncMock(return_value=fake_resp)
    fake_provider.health_check = AsyncMock(return_value=True)

    class FakeVllm:
        def __init__(self, **_kw):
            self._default_model = "qwen2.5:3b"
        async def generate(self, *a, **kw):
            return fake_resp
        async def health_check(self):
            return True
        @property
        def name(self):
            return "vllm"

    monkeypatch.setattr("atomics.providers.vllm.VllmProvider", FakeVllm)

    from datetime import datetime

    from atomics.core.engine import LoopEngine
    from atomics.models import RunSummary
    fake_summary = RunSummary(
        run_id="test-vllm-001",
        started_at=datetime.now(UTC),
        total_tasks=1, total_tokens=30, total_cost_usd=0.0,
        avg_tokens_per_task=30.0, avg_latency_ms=120.0,
        tier=BurnTier.EZ, provider="vllm",
    )

    async def fake_run(*a, **kw):
        return fake_summary

    monkeypatch.setattr(LoopEngine, "run", fake_run)
    monkeypatch.chdir(tmp_path)

    runner = CliRunner()
    result = runner.invoke(cli, [
        "run", "--provider", "vllm",
        "--vllm-host", "http://fake:8000/v1",
        "-m", "qwen2.5:3b", "-n", "1",
    ])
    assert result.exit_code == 0


def test_cli_provider_test_vllm_success(monkeypatch, tmp_path):
    """atomics provider-test --provider vllm should connect and generate."""
    from types import SimpleNamespace

    fake_resp = SimpleNamespace(
        text="ok", input_tokens=5, output_tokens=10, total_tokens=15,
        model="qwen2.5:3b", latency_ms=80.0, estimated_cost_usd=0.0,
        tokens_per_second=125.0, thinking_tokens=0, thinking_text="",
        tps_basis="wall_clock", cache_read_tokens=0, cache_write_tokens=0,
    )

    class FakeVllm:
        def __init__(self, **_kw):
            self._default_model = "qwen2.5:3b"
        async def generate(self, *a, **kw):
            return fake_resp
        async def health_check(self):
            return True
        @property
        def name(self):
            return "vllm"

    monkeypatch.setattr("atomics.providers.vllm.VllmProvider", FakeVllm)
    monkeypatch.chdir(tmp_path)

    runner = CliRunner()
    result = runner.invoke(cli, [
        "provider-test", "--provider", "vllm",
        "--vllm-host", "http://fake:8000/v1",
        "-m", "qwen2.5:3b",
    ])
    assert result.exit_code == 0
    assert "vllm" in result.output.lower()


def test_cli_sweep_vllm_provider(monkeypatch, tmp_path):
    """atomics sweep --provider vllm should use VllmProvider via _make_provider."""
    from atomics.sweep import ModelSweepResult

    constructed: list[str] = []

    class FakeVllm:
        def __init__(self, base_url="http://localhost:8000/v1", default_model="qwen2.5:3b", **_kw):
            self._default_model = default_model
            constructed.append(default_model)
        @property
        def name(self):
            return "vllm"

    mock_results = [
        ModelSweepResult(
            model="qwen2.5:3b", fixtures_run=1, overall_quality=0.90,
            avg_latency_ms=300.0, total_tokens=200, total_cost_usd=0.0,
            value_score=900.0, eval_summary=None,
        ),
    ]

    async def fake_sweep(**kwargs):
        factory = kwargs["provider_factory"]
        factory("qwen2.5:3b")
        cb = kwargs.get("on_model_done")
        for r in mock_results:
            if cb:
                cb(r)
        return mock_results

    monkeypatch.setattr("atomics.providers.vllm.VllmProvider", FakeVllm)
    monkeypatch.setattr("atomics.sweep.run_model_sweep", fake_sweep)
    monkeypatch.chdir(tmp_path)

    runner = CliRunner()
    result = runner.invoke(cli, [
        "sweep", "--provider", "vllm",
        "--vllm-host", "http://fake:8000/v1",
        "--models", "qwen2.5:3b",
    ])
    assert result.exit_code == 0
    assert "qwen2.5:3b" in result.output
    assert "qwen2.5:3b" in constructed


def test_cli_eval_vllm_provider(monkeypatch, tmp_path):
    """atomics eval --provider vllm should construct VllmProvider for both model and judge.

    Regression: eval's local _build_provider had no vllm branch and the command
    lacked a --vllm-host flag, so `eval --provider vllm` died with 'Unknown provider: vllm'.
    """
    from types import SimpleNamespace

    constructed: list[tuple[str, str]] = []

    class FakeVllm:
        def __init__(self, base_url="http://localhost:8000/v1", default_model="qwen2.5:3b", **_kw):
            self._default_model = default_model
            constructed.append((base_url, default_model))

        @property
        def name(self):
            return "vllm"

    fake_summary = SimpleNamespace(
        overall_accuracy=0.9, value_score=900.0, avg_latency_ms=120.0,
        total_tokens=100, total_cost_usd=0.0, fixture_results=[],
        parse_failure_rate=0.0,
    )

    async def fake_run_eval(*_a, **_kw):
        return fake_summary

    monkeypatch.setattr("atomics.providers.vllm.VllmProvider", FakeVllm)
    monkeypatch.setattr("atomics.eval.runner.run_eval", fake_run_eval)
    monkeypatch.chdir(tmp_path)

    runner = CliRunner()
    result = runner.invoke(cli, [
        "eval", "--provider", "vllm",
        "--vllm-host", "http://fake:8000/v1",
        "-m", "qwen3:0.6b",
        "--judge-provider", "vllm",
        "--judge-model", "qwen2.5:3b",
        "--no-save",
    ])
    assert result.exit_code == 0, result.output
    assert "Unknown provider" not in result.output
    # Model under test and judge both built through VllmProvider with the supplied host
    assert ("http://fake:8000/v1", "qwen3:0.6b") in constructed
    assert ("http://fake:8000/v1", "qwen2.5:3b") in constructed


def test_cli_eval_vllm_missing_host_uses_config(monkeypatch, tmp_path):
    """atomics eval --provider vllm without --vllm-host falls back to settings.vllm_host."""
    from types import SimpleNamespace

    constructed: list[str] = []

    class FakeVllm:
        def __init__(self, base_url="http://localhost:8000/v1", default_model="qwen2.5:3b", **_kw):
            constructed.append(base_url)

        @property
        def name(self):
            return "vllm"

    fake_summary = SimpleNamespace(
        overall_accuracy=None, value_score=None, avg_latency_ms=0.0,
        total_tokens=0, total_cost_usd=0.0, fixture_results=[],
        parse_failure_rate=0.0,
    )

    async def fake_run_eval(*_a, **_kw):
        return fake_summary

    monkeypatch.setenv("ATOMICS_VLLM_HOST", "http://config-host:8000/v1")
    monkeypatch.setattr("atomics.providers.vllm.VllmProvider", FakeVllm)
    monkeypatch.setattr("atomics.eval.runner.run_eval", fake_run_eval)
    monkeypatch.chdir(tmp_path)

    runner = CliRunner()
    result = runner.invoke(cli, [
        "eval", "--provider", "vllm", "-m", "qwen3:0.6b",
        "--judge-provider", "vllm", "--judge-model", "qwen2.5:3b",
        "--no-save",
    ])
    assert result.exit_code == 0, result.output
    assert "http://config-host:8000/v1" in constructed


def test_cli_models_vllm_provider(monkeypatch, tmp_path):
    """atomics models --provider vllm should list models from VllmProvider."""
    mock_models = [
        {"name": "qwen2.5:1.5b", "size_gb": 0.0, "parameter_size": "",
         "family": "qwen2.5", "model_class": "light", "thinking": False},
        {"name": "qwen3.5:0.8b", "size_gb": 0.0, "parameter_size": "",
         "family": "qwen3.5", "model_class": "light", "thinking": True},
    ]

    class FakeVllm:
        def __init__(self, **_kw):
            pass
        async def list_models(self):
            return mock_models

    monkeypatch.setattr("atomics.providers.vllm.VllmProvider", FakeVllm)
    monkeypatch.chdir(tmp_path)

    runner = CliRunner()
    result = runner.invoke(cli, [
        "models", "--provider", "vllm",
        "--vllm-host", "http://fake:8000/v1",
    ])
    assert result.exit_code == 0
    assert "qwen2.5:1.5b" in result.output
    assert "qwen3.5:0.8b" in result.output
    assert "light" in result.output


def test_cli_models_vllm_connection_error(monkeypatch, tmp_path):
    """atomics models --provider vllm should surface connection errors."""
    class FakeVllm:
        def __init__(self, **_kw):
            pass
        async def list_models(self):
            raise ConnectionError("Cannot connect to vLLM endpoint at http://fake:8000/v1")

    monkeypatch.setattr("atomics.providers.vllm.VllmProvider", FakeVllm)
    monkeypatch.chdir(tmp_path)

    runner = CliRunner()
    result = runner.invoke(cli, [
        "models", "--provider", "vllm",
        "--vllm-host", "http://fake:8000/v1",
    ])
    assert result.exit_code == 1
    assert "Cannot connect" in result.output


# ---------------------------------------------------------------------------
# baselines command
# ---------------------------------------------------------------------------

def test_cli_baselines_empty(tmp_path, monkeypatch):
    """atomics baselines should handle empty database gracefully."""
    monkeypatch.chdir(tmp_path)
    runner = CliRunner()
    result = runner.invoke(cli, ["baselines"])
    assert result.exit_code == 0
    assert "No baselines" in result.output


def test_cli_baselines_with_records(tmp_path, monkeypatch):
    """atomics baselines should display saved baseline records."""
    from types import SimpleNamespace

    fake_records = [
        SimpleNamespace(
            name="qwen3-stable", suite="soak", model="qwen3:4b",
            avg_tps=85.3, avg_p95_ms=1200.0, verdict="STABLE",
            timestamp="2026-06-04T18:00:00",
        ),
    ]

    monkeypatch.setattr("atomics.regression.list_baselines", lambda _conn: fake_records)
    monkeypatch.chdir(tmp_path)

    runner = CliRunner()
    result = runner.invoke(cli, ["baselines"])
    assert result.exit_code == 0
    assert "qwen3-stable" in result.output
    assert "STABLE" in result.output
