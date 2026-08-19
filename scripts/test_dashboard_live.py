"""Live end-to-end test of the dashboard and new list endpoints."""

from __future__ import annotations

import subprocess
import sys
import time
import urllib.error
from pathlib import Path
from urllib.request import Request, urlopen

PORT = 8765
DB = Path(__file__).with_name("dash_live.db")


def http(method: str, path: str, body: bytes | None = None) -> tuple[int, bytes]:
    url = f"http://127.0.0.1:{PORT}{path}"
    req = Request(url, method=method, data=body)
    if body:
        req.add_header("Content-Type", "application/json")
    try:
        with urlopen(req, timeout=5) as resp:
            return resp.status, resp.read()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read()


def wait_for_server(timeout: float = 10.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            status, _ = http("GET", "/api/v1/health")
            if status == 200:
                return
        except Exception:
            pass
        time.sleep(0.2)
    raise RuntimeError("server did not start")


def main() -> int:
    if DB.exists():
        DB.unlink()

    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "atomics",
            "server",
            "--no-auth",
            "--with-dashboard",
            "--port",
            str(PORT),
            "--worker-absent-after",
            "5",
            "--db-path",
            str(DB),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        wait_for_server()

        status, body = http("GET", "/dashboard")
        assert status == 200, f"dashboard returned {status}: {body[:200]}"
        assert b"atomics dashboard" in body, "dashboard HTML missing title"
        print("GET /dashboard -> 200 OK, HTML looks correct")

        status, body = http("GET", "/api/v1/distributed/runs")
        assert status == 200, f"list jobs returned {status}: {body[:200]}"
        assert b'"jobs":[]' in body or b'"jobs": []' in body, (
            f"expected empty jobs list: {body[:200]}"
        )
        print("GET /api/v1/distributed/runs -> 200 OK, empty list")

        status, body = http("GET", "/api/v1/workers")
        assert status == 200, f"list workers returned {status}: {body[:200]}"
        assert b'"workers":[]' in body or b'"workers": []' in body, (
            f"expected empty workers list: {body[:200]}"
        )
        print("GET /api/v1/workers -> 200 OK, empty list")

        status, body = http(
            "POST",
            "/api/v1/distributed/runs",
            b'{"mode": "split", "run_request": {"iterations": 1}}',
        )
        assert status == 202, f"create run returned {status}: {body[:200]}"
        print(f"POST /api/v1/distributed/runs -> {status}")

        status, body = http("GET", "/api/v1/distributed/runs")
        assert status == 200, f"list jobs after create returned {status}: {body[:200]}"
        assert b'"jobs":' in body and b'"mode"' in body, f"expected jobs: {body[:200]}"
        print("GET /api/v1/distributed/runs -> 200 OK, one job visible")

        print("All live dashboard checks passed.")
        return 0
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
        if DB.exists():
            DB.unlink()


if __name__ == "__main__":
    sys.exit(main())
