"""Tests for the npm worker bridge and CLI."""

from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest
from click.testing import CliRunner

from atomics.commands.worker_npm import worker_npm

NPM_DIR = Path(__file__).resolve().parent.parent / "atomics" / "workers" / "npm"


class MockCoordinatorHandler(BaseHTTPRequestHandler):
    """Minimal coordinator that lets a Node.js worker register and poll."""

    calls: list[tuple[str, dict | None]] = []
    worker_id: str = "w-npm-123"
    next_assignment: dict | None = None

    def _respond(self, status: int, body: dict) -> None:
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(body).encode())

    def _read_body(self) -> dict | None:
        length = int(self.headers.get("Content-Length", 0))
        if length == 0:
            return None
        return json.loads(self.rfile.read(length).decode())

    def log_message(self, *_args) -> None:
        pass

    def do_POST(self) -> None:
        body = self._read_body()
        if self.path.endswith("/register"):
            MockCoordinatorHandler.calls.append(("register", body))
            self._respond(200, {"worker_id": self.worker_id})
        elif self.path.endswith("/heartbeat"):
            MockCoordinatorHandler.calls.append(("heartbeat", body))
            self._respond(200, {"status": "ok"})
        elif "/jobs/" in self.path and self.path.endswith("/result"):
            MockCoordinatorHandler.calls.append(("submit", body))
            self._respond(200, {"status": "completed"})
        else:
            self._respond(404, {})

    def do_GET(self) -> None:
        if self.path.endswith("/jobs/next"):
            MockCoordinatorHandler.calls.append(("poll", None))
            self._respond(200, self.next_assignment or {})
        else:
            self._respond(404, {})


@pytest.fixture
def coordinator_server(tmp_path):
    MockCoordinatorHandler.calls = []
    MockCoordinatorHandler.next_assignment = None
    server = HTTPServer(("127.0.0.1", 0), MockCoordinatorHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    url = f"http://127.0.0.1:{server.server_port}"
    try:
        yield url
    finally:
        server.shutdown()


def test_npm_worker_registers_and_polls(coordinator_server):
    """The Node.js worker registers, heartbeats, and polls the coordinator."""
    if not shutil.which("node"):
        pytest.skip("node not installed")

    env = os.environ.copy()
    env["ATOMICS_COORDINATOR_URL"] = coordinator_server
    env["ATOMICS_WORKER_API_KEY"] = "npm-key"
    env["ATOMICS_WORKER_LABELS"] = "box=239"
    env["ATOMICS_WORKER_CAPABILITIES"] = "node"
    env["ATOMICS_WORKER_HEARTBEAT_INTERVAL"] = "1"

    proc = subprocess.Popen(
        ["node", "worker.js"],
        cwd=str(NPM_DIR),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        time.sleep(2.5)
        proc.send_signal(signal.SIGTERM)
        proc.wait(timeout=5)
    except Exception:
        proc.kill()
        raise

    calls = [c[0] for c in MockCoordinatorHandler.calls]
    assert "register" in calls
    assert "heartbeat" in calls
    assert "poll" in calls


def test_worker_npm_cli_requires_api_key():
    runner = CliRunner()
    result = runner.invoke(worker_npm, [])
    assert result.exit_code != 0
    assert "api-key" in result.output


def test_worker_npm_cli_rejects_malformed_label():
    runner = CliRunner()
    result = runner.invoke(worker_npm, ["--api-key", "k", "--label", "badlabel"])
    assert result.exit_code != 0
    assert "key=value" in result.output


def test_worker_npm_cli_rejects_missing_node(monkeypatch):
    monkeypatch.setattr("atomics.commands.worker_npm.shutil.which", lambda _: None)
    runner = CliRunner()
    result = runner.invoke(worker_npm, ["--api-key", "k"])
    assert result.exit_code != 0
    assert "node is required" in result.output


def test_npm_worker_executes_and_submits_assignment(coordinator_server):
    """The Node.js worker executes a polled assignment via the bridge and submits."""
    if not shutil.which("node"):
        pytest.skip("node not installed")

    MockCoordinatorHandler.next_assignment = {
        "assignment_id": "a-npm-1",
        "job_id": "j-npm-1",
        "task_spec": {"task_name": "quick_question", "prompt": "hello", "runtime": "node"},
    }

    env = os.environ.copy()
    env["ATOMICS_COORDINATOR_URL"] = coordinator_server
    env["ATOMICS_WORKER_API_KEY"] = "npm-key"
    env["ATOMICS_WORKER_CAPABILITIES"] = "node"
    env["ATOMICS_WORKER_HEARTBEAT_INTERVAL"] = "1"

    proc = subprocess.Popen(
        ["node", "worker.js"],
        cwd=str(NPM_DIR),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        time.sleep(2.5)
        proc.send_signal(signal.SIGTERM)
        proc.wait(timeout=5)
    except Exception:
        proc.kill()
        raise

    submit_calls = [c for c in MockCoordinatorHandler.calls if c[0] == "submit"]
    assert submit_calls, "worker should have submitted a result"
    body = submit_calls[0][1]
    assert body["status"] == "completed"
    result = json.loads(body["result_json"])
    assert result["status"] == "ok"
