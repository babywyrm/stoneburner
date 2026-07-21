"""CLI smoke test for labcompare (mocks the orchestrator)."""
from __future__ import annotations

from unittest.mock import patch

from click.testing import CliRunner

from atomics.cli import cli
from atomics.labcompare import CellResult


def test_labcompare_cli_renders_table():
    fake_cells = [
        CellResult("laptop", "http://a:11434", "m", tokens_per_second=48.0,
                   latency_ms=200.0, vram_fit_pct=1.0, quality_score=0.94),
        CellResult("brainbox", "http://b:11434", "m", tokens_per_second=7.0,
                   latency_ms=1400.0, vram_fit_pct=0.65, quality_score=0.94),
    ]
    with patch("atomics.commands.benchmark._run_labcompare_sync", return_value=fake_cells):
        runner = CliRunner()
        result = runner.invoke(cli, [
            "labcompare",
            "--host", "laptop=http://a:11434",
            "--host", "brainbox=http://b:11434",
            "--models", "m",
            "--no-save",
        ])
    assert result.exit_code == 0, result.output
    assert "laptop" in result.output
    assert "brainbox" in result.output
    assert "6.9" in result.output  # speedup ratio rendered


def test_labcompare_cli_requires_two_hosts():
    runner = CliRunner()
    result = runner.invoke(cli, [
        "labcompare", "--host", "only=http://a:11434", "--models", "m", "--no-save",
    ])
    assert result.exit_code != 0
    assert "at least two" in result.output.lower()
