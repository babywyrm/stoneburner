"""Quiet wait lines: in-flight, fixture scores, completed headline."""

from __future__ import annotations

from atomics.repl.display import QuietWait, format_completed, format_fixture_row, format_in_flight


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
