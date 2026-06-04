"""Tests for JSONL/CSV export helpers."""

import io

from atomics.exporters import write_csv, write_jsonl, write_tasks_export


def test_write_jsonl_roundtrip_keys():
    buf = io.StringIO()
    rows = [{"a": 1, "b": "x"}, {"a": 2, "b": "y"}]
    write_jsonl(rows, buf)
    assert '"a": 1' in buf.getvalue()
    assert "\n" in buf.getvalue()


def test_write_csv_headers():
    buf = io.StringIO()
    rows = [{"z": 1, "a": 2}, {"a": 3, "z": 4}]
    write_csv(rows, buf)
    out = buf.getvalue()
    assert "a" in out and "z" in out


def test_write_tasks_export_dispatches():
    buf = io.StringIO()
    write_tasks_export([{"k": "v"}], "jsonl", buf)
    assert buf.getvalue()


# ── export CLI suite flag ──────────────────────────────────────────────────────

def _tmp_repo_with_data(tmp_path):
    from atomics.storage.repository import MetricsRepository
    from atomics.storage.schema import init_db
    from types import SimpleNamespace

    conn = init_db(tmp_path / "db.sqlite")
    conn.close()
    repo = MetricsRepository(tmp_path / "db.sqlite")

    # insert a stress result
    from atomics.stress import StressResult
    sr = StressResult(model="qwen2.5:7b", host="http://localhost:11434",
                      peak_tps=100.0, saturation_concurrency=4,
                      duration_seconds=60.0, total_tokens=5000, total_requests=20)
    repo.save_stress_result(sr)

    # insert a sweep result
    sweep_r = SimpleNamespace(
        model="qwen2.5:7b", provider="ollama", overall_quality=0.9,
        avg_latency_ms=1500.0, total_tokens=3000, total_cost_usd=0.0, fixtures_run=6,
    )
    repo.save_sweep_result(sweep_r)

    return repo


def test_write_generic_export_jsonl():
    from atomics.cli import _write_generic_export
    rows = [{"a": 1, "b": 2}, {"a": 3, "b": 4}]
    buf = io.StringIO()
    _write_generic_export(rows, "jsonl", buf)
    lines = [l for l in buf.getvalue().strip().split("\n") if l]
    assert len(lines) == 2
    import json
    assert json.loads(lines[0])["a"] == 1


def test_write_generic_export_csv():
    from atomics.cli import _write_generic_export
    rows = [{"x": 10, "y": 20}]
    buf = io.StringIO()
    _write_generic_export(rows, "csv", buf)
    out = buf.getvalue()
    assert "x" in out and "10" in out


def test_write_generic_export_empty():
    from atomics.cli import _write_generic_export
    buf = io.StringIO()
    _write_generic_export([], "jsonl", buf)
    assert buf.getvalue() == ""


def test_export_cli_stress_suite(tmp_path, monkeypatch):
    from click.testing import CliRunner
    from atomics.cli import cli
    from atomics.stress import StressResult
    from atomics.storage.repository import MetricsRepository

    monkeypatch.chdir(tmp_path)
    repo = MetricsRepository(tmp_path / "db.sqlite")
    sr = StressResult(model="qwen2.5:7b", host="http://localhost",
                      peak_tps=100.0, saturation_concurrency=2,
                      duration_seconds=30.0, total_tokens=1000, total_requests=5)
    repo.save_stress_result(sr)
    repo.close()

    monkeypatch.setenv("ATOMICS_DB_PATH", str(tmp_path / "db.sqlite"))
    runner = CliRunner()
    result = runner.invoke(cli, ["export", "--suite", "stress"])
    assert result.exit_code == 0, result.output
    assert "qwen2.5:7b" in result.output


def test_export_cli_sweep_suite(tmp_path, monkeypatch):
    from click.testing import CliRunner
    from atomics.cli import cli
    from types import SimpleNamespace
    from atomics.storage.repository import MetricsRepository

    monkeypatch.chdir(tmp_path)
    repo = MetricsRepository(tmp_path / "db.sqlite")
    sweep_r = SimpleNamespace(
        model="qwen3:14b", provider="ollama", overall_quality=0.92,
        avg_latency_ms=2000.0, total_tokens=4000, total_cost_usd=0.0, fixtures_run=6,
    )
    repo.save_sweep_result(sweep_r)
    repo.close()

    monkeypatch.setenv("ATOMICS_DB_PATH", str(tmp_path / "db.sqlite"))
    runner = CliRunner()
    result = runner.invoke(cli, ["export", "--suite", "sweep"])
    assert result.exit_code == 0, result.output
    assert "qwen3:14b" in result.output


# ── _json_default datetime / write_csv empty rows ────────────────────────────

def test_json_default_datetime():
    from datetime import datetime, UTC
    from atomics.exporters import _json_default
    dt = datetime(2026, 1, 15, 12, 0, 0, tzinfo=UTC)
    assert _json_default(dt) == dt.isoformat()


def test_json_default_non_datetime():
    from atomics.exporters import _json_default
    assert _json_default(object()) != ""  # falls through to str()


def test_write_csv_empty_rows():
    import io
    from atomics.exporters import write_csv
    out = io.StringIO()
    write_csv([], out)
    assert out.getvalue() == ""  # early return, nothing written


def test_write_csv_nonempty():
    import io
    from atomics.exporters import write_csv
    out = io.StringIO()
    write_csv([{"a": 1, "b": "x"}, {"a": 2, "b": "y"}], out)
    content = out.getvalue()
    assert "a" in content
    assert "1" in content
