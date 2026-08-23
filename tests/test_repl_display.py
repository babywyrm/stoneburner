"""Quiet wait lines: in-flight, fixture scores, completed headline."""

from __future__ import annotations

from atomics.repl.display import (
    QuietWait,
    format_completed,
    format_fixture_row,
    format_in_flight,
    format_submitted,
)


def test_sweep_in_flight_line() -> None:
    assert format_in_flight({"model": "qwen3:14b", "suite": "eval"}) == (
        "  qwen3:14b  eval"
    )


def test_in_flight_line() -> None:
    assert format_in_flight(
        {"fixture_id": "ev-25", "phase": "judge", "model": "llama3.2:1b"}
    ) == "  ev-25  judge     llama3.2:1b"


def test_fixture_row_verbose_includes_reply() -> None:
    text = format_fixture_row(
        {
            "id": "ev-01",
            "score": 0.6,
            "status": "success",
            "tokens": 170,
            "latency_ms": 812.4,
            "response": "Paris is the capital.",
        },
        verbose=True,
    )
    assert "170 tok  812ms" in text
    assert "    Paris is the capital." in text


def test_fixture_row_line() -> None:
    assert (
        format_fixture_row(
            {"id": "ev-18", "score": 0.0, "status": "success", "tokens": 166}
        )
        == "  ev-18  0.00  success  166 tok"
    )


def test_submitted_headline() -> None:
    text = format_submitted(
        {
            "job_id": "abc",
            "status": "pending",
            "kind": "eval",
            "request": {
                "suite": "toolcall",
                "model": "llama3.2:1b",
                "host": "http://192.168.1.79:11434",
            },
        }
    )
    assert text == (
        "eval  toolcall  llama3.2:1b  http://192.168.1.79:11434\n"
        "abc  pending\n"
    )
    assert '"job_id"' not in text


def test_completed_headline() -> None:
    text = format_completed(
        {
            "status": "completed",
            "request": {
                "suite": "accuracy",
                "model": "llama3.2:1b",
                "host": "http://192.168.1.79:11434",
            },
            "result": {
                "overall_accuracy": 0.844,
                "fixtures_run": 25,
                "total_tokens": 16450,
                "total_cost_usd": 0.0,
            },
            "progress": {"current": 25, "total": 25},
        }
    )
    assert "accuracy" in text
    assert "llama3.2:1b" in text
    assert "192.168.1.79" in text
    assert "0.844" in text
    assert "25/25" in text
    assert "16450" in text


def test_quiet_wait_emits_phase_then_row_then_headline() -> None:
    lines: list[str] = []
    view = QuietWait(lines.append, color=False)
    view.update(
        {
            "status": "running",
            "progress": {
                "current": 0,
                "total": 1,
                "in_flight": {"fixture_id": "ev-01", "phase": "generate", "model": "m"},
            },
            "result": None,
        }
    )
    view.update(
        {
            "status": "running",
            "progress": {
                "current": 0,
                "total": 1,
                "in_flight": {"fixture_id": "ev-01", "phase": "judge", "model": "m"},
            },
            "result": None,
        }
    )
    view.update(
        {
            "status": "completed",
            "request": {"suite": "accuracy", "model": "m", "host": "h"},
            "progress": {"current": 1, "total": 1, "in_flight": None},
            "result": {
                "overall_accuracy": 0.6,
                "fixtures_run": 1,
                "total_tokens": 153,
                "total_cost_usd": 0.0,
                "fixtures": [
                    {"id": "ev-01", "score": 0.6, "status": "success", "tokens": 153}
                ],
            },
        }
    )
    text = "".join(lines)
    assert "generate" in text
    assert "judge" in text
    assert "ev-01  0.60" in text
    assert "0.600" in text or "0.6" in text
    assert text.count("ev-01  0.60") == 1


def test_quiet_wait_still_running_after_cap() -> None:
    lines: list[str] = []
    view = QuietWait(lines.append, color=False)
    body = {
        "status": "running",
        "progress": {
            "current": 24,
            "total": 25,
            "in_flight": {"fixture_id": "ev-25", "phase": "judge", "model": "m"},
        },
    }
    view.update(body)
    view.finish(body)
    text = "".join(lines)
    assert "judge" in text
    assert "still running  24/25" in text


def test_stress_in_flight_line() -> None:
    assert format_in_flight({"concurrency": 2, "phase_seconds": 5.0}) == "  c=2  5s"


def test_soak_in_flight_line() -> None:
    assert format_in_flight({"elapsed_seconds": 10, "concurrency": 1}) == "  10s  c=1"


def test_completed_stress_headline() -> None:
    text = format_completed(
        {
            "status": "completed",
            "kind": "stress",
            "request": {
                "model": "llama3.2:1b",
                "host": "http://192.168.1.79:11434",
            },
            "result": {
                "peak_tps": 435.2,
                "saturation_concurrency": 2,
                "total_tokens": 5736,
                "phases": [
                    {"concurrency": 1, "requests": 7, "aggregate_tps": 337.0},
                    {"concurrency": 2, "requests": 10, "aggregate_tps": 435.2},
                ],
            },
        }
    )
    assert "stress  llama3.2:1b  http://192.168.1.79:11434" in text
    assert "435" in text
    assert "sat=2" in text
    assert "2 phases" in text
    assert text.splitlines()[1].startswith("-") is False


def test_completed_soak_headline() -> None:
    text = format_completed(
        {
            "status": "completed",
            "kind": "soak",
            "request": {
                "model": "llama3.2:1b",
                "host": "http://192.168.1.79:11434",
            },
            "result": {
                "verdict": "STABLE",
                "throughput_drift_pct": -1.2,
                "latency_drift_pct": 0.4,
                "total_tokens": 6656,
                "samples": [
                    {"elapsed_seconds": 10, "requests": 8, "aggregate_tps": 256.0},
                    {"elapsed_seconds": 20, "requests": 9, "aggregate_tps": 240.0},
                ],
            },
        }
    )
    assert "soak  llama3.2:1b  http://192.168.1.79:11434" in text
    assert "STABLE" in text
    assert "2 samples" in text
    assert "-  0" not in text


def test_quiet_wait_walks_stress_phases() -> None:
    lines: list[str] = []
    view = QuietWait(lines.append, color=False)
    view.update(
        {
            "status": "running",
            "progress": {
                "current": 0,
                "total": 2,
                "in_flight": {"concurrency": 1, "phase_seconds": 5.0},
            },
            "result": None,
        }
    )
    view.update(
        {
            "status": "running",
            "progress": {
                "current": 1,
                "total": 2,
                "in_flight": {"concurrency": 2, "phase_seconds": 5.0},
            },
            "result": {
                "phases": [
                    {
                        "concurrency": 1,
                        "requests": 7,
                        "failed": 0,
                        "aggregate_tps": 337.0,
                    }
                ]
            },
        }
    )
    view.update(
        {
            "status": "completed",
            "kind": "stress",
            "request": {"model": "m", "host": "h"},
            "progress": {"current": 2, "total": 2, "in_flight": None},
            "result": {
                "peak_tps": 435.0,
                "saturation_concurrency": 2,
                "phases": [
                    {
                        "concurrency": 1,
                        "requests": 7,
                        "failed": 0,
                        "aggregate_tps": 337.0,
                    },
                    {
                        "concurrency": 2,
                        "requests": 10,
                        "failed": 0,
                        "aggregate_tps": 435.0,
                    },
                ],
            },
        }
    )
    text = "".join(lines)
    assert "c=1  5s" in text
    assert "c=2  5s" in text
    assert "c=1  337" in text
    assert "sat=2" in text
    assert text.count("c=1  337") == 1


def test_color_off_has_no_ansi() -> None:
    line = format_fixture_row(
        {"id": "ev-01", "score": 1.0, "status": "success", "tokens": 10},
        color=True,
    )
    assert "\033[" in line
    plain = format_fixture_row(
        {"id": "ev-01", "score": 1.0, "status": "success", "tokens": 10},
        color=False,
    )
    assert "\033[" not in plain
