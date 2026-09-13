from click.testing import CliRunner

from atomics.cli import cli


def test_battery_help_on_root():
    result = CliRunner().invoke(cli, ["battery", "--help"])
    assert result.exit_code == 0
    assert "list" in result.output
    assert "show" in result.output
    assert "run" in result.output


def test_battery_list_names_all_five():
    result = CliRunner().invoke(cli, ["battery", "list"])
    assert result.exit_code == 0
    for name in (
        "desk-pass",
        "blue-capability",
        "red-capability",
        "agent-gate",
        "threat-model",
    ):
        assert name in result.output
    assert "cheap" in result.output


def test_battery_show_desk_pass_prints_commands():
    result = CliRunner().invoke(
        cli, ["battery", "show", "desk-pass", "-m", "lfm2.5:8b"]
    )
    assert result.exit_code == 0
    assert "provider-test" in result.output
    assert "qa/examples/app-gate-guardrails.yaml" in result.output
    assert "tc-01,tc-02" in result.output
    assert "FUNCTION_COMPATIBLE" in result.output
    assert "Not walkthrough" in result.output or "Not walkthrough-compatible" in result.output


def test_battery_show_unknown_exits_nonzero():
    result = CliRunner().invoke(cli, ["battery", "show", "nope"])
    assert result.exit_code != 0
    assert "desk-pass" in result.output


def test_battery_show_openai_omits_raw_qa():
    result = CliRunner().invoke(
        cli,
        ["battery", "show", "desk-pass", "-p", "openai", "-m", "gpt-4.1", "--budget", "5"],
    )
    assert result.exit_code == 0
    assert "openai" in result.output
    assert "qa/examples/app-gate-guardrails.yaml" not in result.output
    assert "provider-test" in result.output
    assert "toolcall" in result.output


def test_battery_show_claude_agent_gate():
    result = CliRunner().invoke(
        cli,
        [
            "battery",
            "show",
            "agent-gate",
            "-p",
            "claude",
            "-m",
            "claude-sonnet-4-6",
            "--judge-provider",
            "claude",
            "--judge-model",
            "claude-sonnet-4-6",
            "--budget",
            "5",
        ],
    )
    assert result.exit_code == 0
    assert "claude" in result.output
    assert "--budget" in result.output


def test_battery_show_lab_vllm():
    result = CliRunner().invoke(
        cli,
        [
            "battery",
            "show",
            "agent-gate",
            "-p",
            "vllm",
            "--vllm-host",
            "http://127.0.0.1:8000/v1",
        ],
    )
    assert result.exit_code == 0
    assert "vllm" in result.output
    assert "--vllm-host" in result.output


def test_battery_run_unknown_exits_nonzero():
    result = CliRunner().invoke(cli, ["battery", "run", "nope"])
    assert result.exit_code != 0
    assert "desk-pass" in result.output


def test_battery_run_judged_requires_judge_model():
    result = CliRunner().invoke(
        cli, ["battery", "run", "blue-capability", "-m", "x"]
    )
    assert result.exit_code == 2
    assert "judge" in result.output.lower()


def test_battery_run_desk_pass_invokes_steps(monkeypatch):
    seen: list[list[str]] = []

    def fake_invoke(args: list[str]) -> int:
        seen.append(args)
        return 0

    monkeypatch.setattr("atomics.commands.battery.invoke_atomics", fake_invoke)
    result = CliRunner().invoke(
        cli, ["battery", "run", "desk-pass", "-m", "lfm2.5:8b"]
    )
    assert result.exit_code == 0
    assert [args[0] for args in seen] == ["provider-test", "qa", "toolcall"]
    assert "lfm2.5:8b" in seen[0]


def test_battery_run_stops_on_first_failure(monkeypatch):
    seen: list[str] = []

    def fake_invoke(args: list[str]) -> int:
        seen.append(args[0])
        return 1 if args[0] == "qa" else 0

    monkeypatch.setattr("atomics.commands.battery.invoke_atomics", fake_invoke)
    result = CliRunner().invoke(
        cli, ["battery", "run", "desk-pass", "-m", "lfm2.5:8b"]
    )
    assert result.exit_code != 0
    assert seen == ["provider-test", "qa"]


def test_battery_run_keep_going(monkeypatch):
    seen: list[str] = []

    def fake_invoke(args: list[str]) -> int:
        seen.append(args[0])
        return 1 if args[0] == "qa" else 0

    monkeypatch.setattr("atomics.commands.battery.invoke_atomics", fake_invoke)
    result = CliRunner().invoke(
        cli, ["battery", "run", "desk-pass", "-m", "lfm2.5:8b", "--keep-going"]
    )
    assert result.exit_code != 0
    assert seen == ["provider-test", "qa", "toolcall"]
