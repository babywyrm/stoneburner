# submit_battery Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `POST /batteries` + MCP `submit_battery` run a named battery as one metered job.

**Architecture:** `BatteryRequest` expands `get_battery(name)` into ordered steps. A new `run_battery_from_request` runs each step through the existing `run_eval_suite` path under one shared `BudgetMeter`. `progress.total` is the step count; `result.steps` grows per finished step.

**Tech Stack:** FastAPI, Pydantic, existing `atomics.api` runners/jobs, `atomics.eval.batteries`, MCP stdio proxy.

Spec: `docs/superpowers/archive/specs/2026-09-18-submit-battery-design.md`

---

### Task 1: BatteryRequest model

**Files:**
- Modify: `atomics/api/models.py` (after `SweepRequest`, ~line 180)
- Test: `tests/test_api_models.py`

- [ ] **Step 1: failing test**

```python
def test_battery_request_unknown_name_rejected():
    from atomics.api.models import BatteryRequest
    import pytest
    with pytest.raises(Exception):
        BatteryRequest(name="nope", provider="ollama", model="x", budget_usd=5)


def test_battery_request_requires_budget():
    from atomics.api.models import BatteryRequest
    import pytest
    with pytest.raises(Exception):
        BatteryRequest(name="desk-pass", provider="ollama", model="x")


def test_battery_request_defaults():
    from atomics.api.models import BatteryRequest
    req = BatteryRequest(name="desk-pass", provider="ollama", model="x", budget_usd=5)
    assert req.runs == 1
    assert req.thinking is None
```

- [ ] **Step 2: run, expect fail** — `uv run pytest tests/test_api_models.py -q -k battery` → `ImportError: BatteryRequest`.

- [ ] **Step 3: implement**

```python
class BatteryRequest(BaseModel):
    """Run a named battery as one job. Budget is required (paid judge possible)."""

    model_config = ConfigDict(extra="forbid")

    name: str
    provider: str
    model: str | None = None
    budget_usd: float = Field(gt=0, le=MAX_EVAL_BUDGET_USD)
    judge_model: str | None = None
    judge_host: str | None = None
    host: str | None = None
    thinking: bool | None = None
    effort: str | None = None
    reasoning_mode: str | None = None
    runs: int = Field(default=1, ge=1, le=MAX_SWEEP_RUNS)
    profile: str | None = None

    @field_validator("effort")
    @classmethod
    def _known_effort(cls, value: str | None) -> str | None:
        return _normalize_effort_field(value)

    @field_validator("reasoning_mode")
    @classmethod
    def _known_reasoning_mode(cls, value: str | None) -> str | None:
        return _normalize_reasoning_mode_field(value)

    @field_validator("name")
    @classmethod
    def _known_battery(cls, value: str) -> str:
        from atomics.eval.batteries import get_battery
        try:
            get_battery(value)
        except KeyError as exc:
            raise ValueError(str(exc)) from exc
        return value
```

- [ ] **Step 4: run, expect pass.**

- [ ] **Step 5: commit** — `feat(api): BatteryRequest model`

---

### Task 2: battery runner

**Files:**
- Create: `atomics/api/_battery.py`
- Test: `tests/test_api_battery.py`

- [ ] **Step 1: failing test**

```python
import pytest
from atomics.api._battery import run_battery_from_request
from atomics.api.models import BatteryRequest


@pytest.mark.asyncio
async def test_battery_expands_steps_in_order(monkeypatch):
    seen: list[str] = []

    async def fake_run_eval_suite(payload, job=None):
        seen.append(payload.suite)
        return {"suite": payload.suite, "fixtures": []}

    monkeypatch.setattr("atomics.api._battery.run_eval_suite", fake_run_eval_suite)
    req = BatteryRequest(name="desk-pass", provider="ollama", model="x", budget_usd=5)
    result = await run_battery_from_request(req)
    assert [s["suite"] for s in result["steps"]] == seen
    assert seen  # desk-pass has steps
```

- [ ] **Step 2: run, expect fail** — module does not exist.

- [ ] **Step 3: implement** `atomics/api/_battery.py`

```python
"""Run a named battery as one metered API job."""

from __future__ import annotations

from typing import Any

from fastapi import HTTPException

from atomics.api._runners import run_eval_suite
from atomics.api.jobs import Job
from atomics.api.models import BatteryRequest, EvalRequest
from atomics.eval.batteries import get_battery, visible_steps
from atomics.eval.budget import EvalBudgetExceededError

# provider-test and qa are not eval suites; they run through their own paths.
# v1 runs only the eval-suite steps and records provider-test/qa as skipped.
_SKIP = frozenset({"provider-test", "qa", "archreview"})


def _step_to_eval(payload: BatteryRequest, suite: str, step: Any) -> EvalRequest:
    return EvalRequest(
        suite=suite,
        provider=payload.provider,
        model=payload.model,
        judge_model=payload.judge_model,
        host=payload.host,
        judge_host=payload.judge_host,
        budget_usd=payload.budget_usd,
        thinking=payload.thinking,
        effort=payload.effort,
        reasoning_mode=payload.reasoning_mode,
        runs=payload.runs,
        fixtures=list(step.fixtures) or None,
        channel=(step.extra[step.extra.index("--channel") + 1] if "--channel" in step.extra else None),
    )


async def run_battery_from_request(
    payload: BatteryRequest, job: Job | None = None
) -> dict[str, Any]:
    battery = get_battery(payload.name)
    steps_out: list[dict[str, Any]] = []
    for step in visible_steps(battery, provider=payload.provider, profile=payload.profile):
        if step.suite in _SKIP:
            steps_out.append({"suite": step.suite, "skipped": True})
            continue
        try:
            result = await run_eval_suite(_step_to_eval(payload, step.suite, step), job=job)
        except EvalBudgetExceededError:
            raise
        except (HTTPException, ValueError, RuntimeError) as exc:
            steps_out.append({"suite": step.suite, "ok": False, "error": str(exc)})
            raise HTTPException(status_code=400, detail=f"{payload.name}:{step.suite}: {exc}") from exc
        steps_out.append({"suite": step.suite, "ok": True, "result": result})
    return {
        "battery": payload.name,
        "provider": payload.provider,
        "model": payload.model,
        "budget_usd": payload.budget_usd,
        "steps": steps_out,
    }
```

- [ ] **Step 4: run, expect pass.**

- [ ] **Step 5: commit** — `feat(api): battery runner`

---

### Task 3: route + progress

**Files:**
- Modify: `atomics/api/routes.py` (add `POST /batteries`)
- Modify: `atomics/api/job_progress.py` (`battery_job_total`, `initial_battery_progress`)
- Test: `tests/test_api_battery.py`

- [ ] **Step 1: failing test** — POST `/batteries` with a mocked job manager returns 202 and kind `battery`; unknown name 400.

- [ ] **Step 2: run, expect 404.**

- [ ] **Step 3: implement**

In `job_progress.py`:

```python
def battery_job_total(payload: BatteryRequest) -> int:
    from atomics.eval.batteries import get_battery, visible_steps
    battery = get_battery(payload.name)
    return len(visible_steps(battery, provider=payload.provider, profile=payload.profile))


def initial_battery_progress(payload: BatteryRequest) -> dict[str, Any]:
    return {"current": 0, "total": battery_job_total(payload), "in_flight": None, "trail": []}
```

In `routes.py`:

```python
@router.post("/batteries", response_model=JobResponse, status_code=status.HTTP_202_ACCEPTED)
async def start_battery(
    payload: BatteryRequest,
    job_manager: JobManager = Depends(get_job_manager),
    caller: str = Depends(require_auth),
) -> JobResponse:
    """Run a named battery as one metered job. Budget is required."""
    try:
        job_id = await job_manager.submit(
            "battery",
            lambda jid: run_battery_from_request(payload, job=job_manager.jobs[jid]),
            owner=caller,
            request=payload_request(payload, load_settings()),
            progress=initial_battery_progress(payload),
        )
    except TooManyActiveJobsError as exc:
        raise HTTPException(status_code=status.HTTP_429_TOO_MANY_REQUESTS, detail=str(exc)) from exc
    job = job_manager.jobs[job_id]
    return _job_to_response(job)
```

Import `BatteryRequest`, `run_battery_from_request`, `initial_battery_progress`.

- [ ] **Step 4: run, expect pass.**

- [ ] **Step 5: commit** — `feat(api): POST /batteries`

---

### Task 4: MCP submit_battery

**Files:**
- Modify: `atomics/mcp/client.py` (`submit_battery`)
- Modify: `atomics/mcp/server.py` (`submit_battery` tool)
- Test: `tests/test_mcp_client.py`, `tests/test_mcp_server.py`

- [ ] **Step 1: failing test** — client posts to `/batteries`; server tool forwards fields and returns job id.

- [ ] **Step 2: run, expect fail** — no attribute.

- [ ] **Step 3: implement**

Client:

```python
def submit_battery(
    self,
    *,
    name: str,
    provider: str,
    budget_usd: float,
    model: str | None = None,
    judge_model: str | None = None,
    judge_host: str | None = None,
    host: str | None = None,
    thinking: bool | None = None,
    effort: str | None = None,
    reasoning_mode: str | None = None,
    runs: int = 1,
    profile: str | None = None,
) -> Any:
    payload: dict[str, Any] = {
        "name": name,
        "provider": provider,
        "budget_usd": budget_usd,
        "runs": runs,
    }
    if model is not None:
        payload["model"] = model
    if judge_model is not None:
        payload["judge_model"] = judge_model
    if judge_host is not None:
        payload["judge_host"] = judge_host
    if host is not None:
        payload["host"] = host
    if thinking is not None:
        payload["thinking"] = thinking
    if effort is not None:
        payload["effort"] = effort
    if reasoning_mode is not None:
        payload["reasoning_mode"] = reasoning_mode
    if profile is not None:
        payload["profile"] = profile
    return self._request("POST", "/batteries", json=payload)
```

Server tool mirrors `submit_sweep`, annotated `SPENDS`, docstring: budget required, one named battery, poll `get_job`.

- [ ] **Step 4: run, expect pass.**

- [ ] **Step 5: commit** — `feat(mcp): submit_battery`

---

### Task 5: docs + changelog

**Files:**
- Modify: `CHANGELOG.md` (Unreleased → Added)
- Modify: `docs/API_SERVER.md` and `docs/REPL.md` if they list endpoints/tools
- Modify: `ROADMAP.md` — mark `submit_battery` done under Next

- [ ] **Step 1:** add CHANGELOG entry and endpoint/tool rows.
- [ ] **Step 2:** run full `pytest -q --no-cov`, `ruff check`, `mypy` on touched files.
- [ ] **Step 3:** commit — `feat: API/MCP submit_battery`

---

## Self-review

- Spec coverage: route, MCP, model, runner, budget, thinking/effort forward, unknown-name 400, no archreview, no keep-going. All present.
- provider-test/qa are not eval suites; v1 records them skipped rather than running them (documented in `_SKIP`).
- `payload_request` for a battery: confirm it accepts `BatteryRequest` (it is generic over payloads; if it is typed to specific models, add a branch).
