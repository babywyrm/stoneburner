"""Local smoke test: a real coordinator and real worker processes.

The pytest suite drives FastAPI's `TestClient`, which is an in-process ASGI
shim, against a `FakeProvider`. That covers the logic but proves nothing about
the pieces working as separate processes over real sockets — the first run of
this script found the fleet status table truncating every worker id to
`5b5fc...` at the default terminal width, which 1833 green tests had missed.

Two phases:

1. Happy path — auth is enforced, two workers register with labels, a fleet run
   broadcasts to both, the rollup reports per-host numbers, the table renders.
2. Host loss — a worker is killed mid-run. The coordinator must notice the
   silence, mark it offline, fail its pinned slice, and let the job resolve to
   `partial` instead of waiting forever on a host that is not coming back.

Usage: uv run python scripts/smoke_fleet.py

Needs no credentials and no model: inference is served by the OpenAI-compatible
stub in tests/. Nothing touches the real database — on macOS the data directory
is `data/` relative to the working directory, so each phase runs in its own
scratch directory.
"""

from __future__ import annotations

import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

import httpx

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "tests"))
from inference_stub import RecordedRequest, StubInferenceServer  # noqa: E402

ATOMICS = str(REPO / "1" / "bin" / "atomics")
KEY = "smoke-key-not-a-real-credential"
SCRATCH = Path("/tmp/atomics-smoke-fleet")

failures: list[str] = []


def check(label: str, ok: bool, detail: str = "") -> None:
    print(f"  {'PASS' if ok else 'FAIL'}  {label}{' — ' + detail if detail else ''}")
    if not ok:
        failures.append(label)


def wait_for(fn, timeout=30.0, interval=0.3):
    """Poll until `fn` returns something truthy. Returns None on timeout."""
    deadline = time.time() + timeout
    last_error: Exception | None = None
    while time.time() < deadline:
        try:
            if value := fn():
                return value
        except Exception as exc:  # noqa: BLE001 - a not-yet-listening socket
            last_error = exc
        time.sleep(interval)
    if last_error is not None:
        print(f"       (last error while waiting: {last_error!r})")
    return None


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def worker_rows(db: Path) -> list[tuple[str, str]]:
    import sqlite3

    conn = sqlite3.connect(db)
    try:
        return list(conn.execute("SELECT worker_id, status FROM workers"))
    finally:
        conn.close()


class Fleet:
    """A coordinator process plus its workers, in an isolated scratch directory."""

    def __init__(self, name: str, stub: StubInferenceServer, **server_flags: str):
        self.work = SCRATCH / name
        shutil.rmtree(self.work, ignore_errors=True)
        (self.work / "data").mkdir(parents=True)
        self.db = self.work / "data" / "atomics.db"
        self.port = free_port()
        self.url = f"http://127.0.0.1:{self.port}"
        self.auth = {"X-API-Key": KEY}
        self.env = {
            **os.environ,
            "ATOMICS_VLLM_HOST": stub.openai_base_url,
            "ATOMICS_VLLM_MODEL": "stub-model",
        }
        self.procs: list[subprocess.Popen] = []
        flags = [f for pair in server_flags.items() for f in pair]
        self._spawn([ATOMICS, "server", "--api-key", KEY, "--port", str(self.port), *flags])

    def _spawn(self, argv: list[str]) -> subprocess.Popen:
        proc = subprocess.Popen(
            argv, cwd=self.work, env=self.env,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        self.procs.append(proc)
        return proc

    def wait_healthy(self) -> bool:
        return bool(
            wait_for(
                lambda: httpx.get(f"{self.url}/api/v1/health", timeout=2).status_code == 200
            )
        )

    def add_worker(self, *labels: str, heartbeat: int = 5) -> subprocess.Popen:
        label_flags = [f for label in labels for f in ("--label", label)]
        return self._spawn([
            ATOMICS, "worker", "--coordinator", self.url, "--api-key", KEY,
            *label_flags, "-p", "vllm", "--heartbeat-interval", str(heartbeat),
        ])

    def wait_for_workers(self, count: int) -> bool:
        return bool(wait_for(lambda: len(worker_rows(self.db)) >= count, timeout=45))

    def submit_fleet(self, iterations: int, selector: dict | None = None) -> httpx.Response:
        payload: dict = {
            "mode": "fleet",
            "run_request": {"iterations": iterations, "tier": "ez"},
        }
        if selector:
            payload["worker_selector"] = selector
        return httpx.post(
            f"{self.url}/api/v1/distributed/runs",
            json=payload, headers=self.auth, timeout=15,
        )

    def job(self, job_id: str) -> dict:
        return httpx.get(
            f"{self.url}/api/v1/distributed/runs/{job_id}",
            headers=self.auth, timeout=10,
        ).json()

    def wait_terminal(self, job_id: str, timeout=120.0) -> dict | None:
        terminal = {"completed", "partial", "failed"}

        def poll():
            job = self.job(job_id)
            return job if job.get("status") in terminal else None

        return wait_for(poll, timeout=timeout, interval=0.5)

    def status_output(self, job_id: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            [ATOMICS, "distributed", "status", job_id,
             "--coordinator", self.url, "--api-key", KEY],
            cwd=self.work, env=self.env, capture_output=True, text=True, timeout=60,
        )

    def stop(self) -> None:
        for proc in self.procs:
            if proc.poll() is None:
                proc.terminate()
        for proc in self.procs:
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()


def phase_happy_path(stub: StubInferenceServer) -> None:
    print("\n=== Phase 1: a real two-host fleet run ===")
    fleet = Fleet("happy", stub)
    try:
        check("coordinator answers /health over a real socket", fleet.wait_healthy())

        anon = httpx.post(
            f"{fleet.url}/api/v1/distributed/runs",
            json={"mode": "split", "run_request": {"iterations": 1}}, timeout=10,
        )
        check("anonymous submit rejected", anon.status_code == 401, f"got {anon.status_code}")

        fleet.add_worker("box=alpha", "site=lab")
        fleet.add_worker("box=beta", "site=lab")
        check("both worker processes registered", fleet.wait_for_workers(2))

        submit = fleet.submit_fleet(2, selector={"site": "lab"})
        check("fleet run accepted", submit.status_code == 202,
              f"got {submit.status_code}: {submit.text[:160]}")
        if submit.status_code != 202:
            return
        job_id = submit.json()["job_id"]

        done = fleet.wait_terminal(job_id, timeout=90)
        check("job reached a terminal status", bool(done),
              (done or {}).get("status", "timed out"))
        if not done:
            return
        check("job completed cleanly", done["status"] == "completed", done["status"])
        check("stub served real inference over HTTP", len(stub.chat_completions()) >= 4,
              f"{len(stub.chat_completions())} chat completions")

        summary = json.loads(done["summary_json"])
        hosts = {w["labels"].get("box"): w for w in summary["workers"]}
        check("rollup has a row per host", set(hosts) == {"alpha", "beta"}, str(set(hosts)))
        check("each host ran the full task set",
              all(h["completed"] == 2 for h in hosts.values()),
              str({k: v["completed"] for k, v in hosts.items()}))

        table = fleet.status_output(job_id)
        check("distributed status exits 0", table.returncode == 0, table.stderr[-160:])
        # Full ids, not "5b5fc...": a truncated table cannot say which host won.
        check("both worker ids readable in the table",
              all(w["worker_id"] in table.stdout for w in summary["workers"]))
        check("labels readable in the table",
              "box=alpha" in table.stdout and "box=beta" in table.stdout)
        print("\n" + table.stdout)
    finally:
        fleet.stop()


def phase_host_loss(stub: StubInferenceServer) -> None:
    """Kill a worker mid-run and require the job to resolve rather than hang."""
    print("\n=== Phase 2: a host disappears mid-run ===")
    absent_after = 8
    # Tight thresholds keep this under a minute; the shipped defaults are a 30s
    # heartbeat and a 120s window, which would make the same assertions a
    # three-minute wait.
    fleet = Fleet(
        "host-loss", stub,
        **{"--worker-absent-after": str(absent_after)},
    )
    try:
        check("coordinator answers /health over a real socket", fleet.wait_healthy())
        fleet.add_worker("box=survivor", heartbeat=2)
        doomed = fleet.add_worker("box=doomed", heartbeat=2)
        check("both worker processes registered", fleet.wait_for_workers(2))

        before = {wid for wid, _ in worker_rows(fleet.db)}
        submit = fleet.submit_fleet(6)
        check("fleet run accepted", submit.status_code == 202,
              f"got {submit.status_code}: {submit.text[:160]}")
        if submit.status_code != 202:
            return
        job_id = submit.json()["job_id"]

        # SIGKILL, not terminate: a graceful shutdown could deregister, and the
        # case under test is a host that vanishes without saying anything.
        wait_for(lambda: len(stub.chat_completions()) >= 2, timeout=30)
        doomed.send_signal(signal.SIGKILL)
        doomed.wait(timeout=10)
        print(f"  killed the 'doomed' worker; absence threshold is {absent_after}s")

        offline = wait_for(
            lambda: [wid for wid, status in worker_rows(fleet.db) if status == "offline"],
            timeout=absent_after + 40,
        )
        check("the silent host is marked offline", bool(offline),
              "still online after the threshold elapsed")

        done = fleet.wait_terminal(job_id, timeout=120)
        check("the job resolves instead of waiting on a dead host", bool(done),
              (done or {}).get("status", "timed out"))
        if not done:
            return
        check("job resolved to partial", done["status"] == "partial", done["status"])

        summary = json.loads(done["summary_json"])
        hosts = {w["labels"].get("box"): w for w in summary["workers"]}
        check("the surviving host finished its whole slice",
              hosts.get("survivor", {}).get("completed") == 6,
              str(hosts.get("survivor", {}).get("completed")))
        check("the dead host's losses are recorded, not hidden",
              summary["failed"] >= 1, f"{summary['failed']} failed")
        check("every assignment is accounted for",
              summary["completed"] + summary["failed"] == 12,
              f"{summary['completed']} + {summary['failed']}")
        check("no worker vanished from the registry",
              {wid for wid, _ in worker_rows(fleet.db)} == before)

        table = fleet.status_output(job_id)
        check("status renders a partial fleet run", table.returncode == 0,
              table.stderr[-160:])
        print("\n" + table.stdout)
    finally:
        fleet.stop()


def main() -> int:
    SCRATCH.mkdir(parents=True, exist_ok=True)

    fast = StubInferenceServer().start()
    print(f"Stub inference (fast) on {fast.openai_base_url}")
    try:
        phase_happy_path(fast)
    finally:
        fast.stop()

    def slow(request: RecordedRequest) -> str:
        # Long enough that a worker is still busy when it gets killed.
        time.sleep(3.0)
        return "a considered answer"

    slow_stub = StubInferenceServer(slow).start()
    print(f"\nStub inference (3s per call) on {slow_stub.openai_base_url}")
    try:
        phase_host_loss(slow_stub)
    finally:
        slow_stub.stop()

    print()
    if failures:
        print(f"FAILURES ({len(failures)}): " + "; ".join(failures))
        return 1
    print("ALL LOCAL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
