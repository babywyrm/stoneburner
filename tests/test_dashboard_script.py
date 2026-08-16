"""Run the dashboard script in a fake DOM.

TestClient checks prove the HTML and APIs. This executes the page script so a
regression that assigns `.innerHTML` or writes `data.result` into the DOM
fails here, not in an operator's browser.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from atomics.api.dashboard import _DASHBOARD_HTML

HARNESS = Path(__file__).resolve().parent / "dashboard_script_harness.mjs"
NODE = shutil.which("node")


@pytest.mark.skipif(NODE is None, reason="node not installed")
def test_dashboard_script_renders_run_and_job_via_textcontent(tmp_path: Path) -> None:
    script = _DASHBOARD_HTML.split("<script>", 1)[1].rsplit("</script>", 1)[0]
    script_path = tmp_path / "dashboard.js"
    script_path.write_text(script, encoding="utf-8")
    report_path = tmp_path / "report.json"

    completed = subprocess.run(
        [NODE, str(HARNESS), str(script_path), str(report_path)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr + completed.stdout
    report = json.loads(report_path.read_text(encoding="utf-8"))

    assert report["innerHTMLWrites"] == []
    assert "SECRET_RESULT" not in report["allText"]
    assert "do not leak" not in report["allText"]
    assert "<img src=x onerror=alert(1)>" in report["allText"]
    assert "rf-01" in report["allText"]
    assert "0.75" in report["allText"]
    assert "ollama" in report["allText"]
    assert "eval-job" in report["allText"] or "completed" in report["allText"]
    assert report["jobDetailVisible"] is True
    # hashchange to #job= hides the run panel; the fixture rows must still
    # have been written via textContent before that.
    assert "rf-01" in report["allText"]
