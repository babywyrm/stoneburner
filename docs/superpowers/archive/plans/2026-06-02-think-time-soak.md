# Think-Time / Arrival Delay for `atomics soak` Implementation Plan

> **STATUS: COMPLETED** — shipped in v0.6.0 (`atomics soak --think-time SECONDS`). This document is retained as an implementation record; all steps below are checked off.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `--think-time` option to `atomics soak` that inserts a per-worker sleep between requests, enabling realistic simulation of users with think time between calls.

**Architecture:** Add `think_time_seconds: float = 0.0` to all three soak runner functions (`run_soak`, `run_soak_provider`, `run_soak_profile`) in `soak.py`. Each worker sleeps `think_time_seconds` after a successful request before firing the next. The CLI adds `--think-time` (float, seconds, default 0) and surfaces effective simulated user count in the header. Tests verify the sleep is called with the right delay and that 0 is a no-op.

**Tech Stack:** Python 3.11+, asyncio, httpx, click, pytest, unittest.mock

---

## File Map

| File | Change |
|------|--------|
| `atomics/soak.py` | Add `think_time_seconds` param to `run_soak`, `run_soak_provider`, `run_soak_profile`; `asyncio.sleep` after each successful request |
| `atomics/cli.py` | Add `--think-time` flag to `soak` command; print simulated user count in header |
| `tests/test_soak.py` | Add tests: think-time sleep called, zero is no-op, header shows user count |

---

## Task 1: Add `think_time_seconds` to `run_soak`

**Files:**
- Modify: `atomics/soak.py` — `run_soak()` function

### What changes

In the `_worker` coroutine inside `run_soak`, after a successful request, sleep `think_time_seconds` before looping:

```python
async def run_soak(
    host: str,
    model: str,
    concurrency: int = 4,
    duration_seconds: float = 1800,
    sample_interval: int = 30,
    num_predict: int = 2048,
    think_time_seconds: float = 0.0,
    on_sample: Callable[[SoakSample], None] | None = None,
) -> SoakResult:
    ...
    async def _worker(client, worker_id):
        ...
        while not stop_event.is_set():
            ...
            try:
                out_tok, _in_tok, lat_ms, _tps = await _single_request(...)
                async with window_lock:
                    ...
                if think_time_seconds > 0 and not stop_event.is_set():
                    await asyncio.sleep(think_time_seconds)
            except Exception:
                async with window_lock:
                    ...
```

- [x] **Step 1: Write the failing tests**

```python
# in tests/test_soak.py

class TestThinkTime:
    @pytest.mark.asyncio
    async def test_think_time_sleep_called(self):
        """Worker sleeps think_time_seconds after each successful request."""
        from atomics.soak import run_soak

        async def _fake_request(client, host, model, prompt, num_predict):
            return 100, 10, 500.0, 50.0

        sleep_calls: list[float] = []

        async def _fake_sleep(seconds):
            sleep_calls.append(seconds)

        with patch("atomics.stress._single_request", side_effect=_fake_request), \
             patch("atomics.stress._get_vram_used_mb", return_value=None), \
             patch("asyncio.sleep", side_effect=_fake_sleep):
            await run_soak(
                host="http://fake:11434",
                model="test",
                concurrency=1,
                duration_seconds=0.05,
                sample_interval=999,
                think_time_seconds=1.5,
            )

        think_sleeps = [s for s in sleep_calls if s == 1.5]
        assert len(think_sleeps) >= 1

    @pytest.mark.asyncio
    async def test_zero_think_time_no_sleep(self):
        """Default think_time=0 does not call asyncio.sleep for think time."""
        from atomics.soak import run_soak

        async def _fake_request(client, host, model, prompt, num_predict):
            return 100, 10, 500.0, 50.0

        sleep_calls: list[float] = []

        async def _fake_sleep(seconds):
            sleep_calls.append(seconds)

        with patch("atomics.stress._single_request", side_effect=_fake_request), \
             patch("atomics.stress._get_vram_used_mb", return_value=None), \
             patch("asyncio.sleep", side_effect=_fake_sleep):
            await run_soak(
                host="http://fake:11434",
                model="test",
                concurrency=1,
                duration_seconds=0.05,
                sample_interval=999,
                think_time_seconds=0.0,
            )

        think_sleeps = [s for s in sleep_calls if s > 0.5]
        assert len(think_sleeps) == 0

    @pytest.mark.asyncio
    async def test_think_time_profile_sleep_called(self):
        """run_soak_profile also sleeps think_time_seconds."""
        from atomics.profiles import TargetProfile
        from atomics.soak import run_soak_profile

        async def _fake_profile_request(client, profile, prompt):
            return "ok", 500.0, None

        sleep_calls: list[float] = []

        async def _fake_sleep(seconds):
            sleep_calls.append(seconds)

        profile = TargetProfile(
            name="t", type="ollama",
            ollama_host="http://fake:11434",
            model="test",
            prompts=["hello"],
        )

        with patch("atomics.profiles._single_request_profile", side_effect=_fake_profile_request), \
             patch("asyncio.sleep", side_effect=_fake_sleep):
            await run_soak_profile(
                profile=profile,
                concurrency=1,
                duration_seconds=0.05,
                sample_interval=999,
                think_time_seconds=2.0,
            )

        think_sleeps = [s for s in sleep_calls if s == 2.0]
        assert len(think_sleeps) >= 1
```

- [x] **Step 2: Run tests, confirm they fail**

```bash
uv run pytest tests/test_soak.py::TestThinkTime -v --no-cov
```

Expected: 3 failures (function signature mismatch / sleep not called)

- [x] **Step 3: Implement in `soak.py`**

Add `think_time_seconds: float = 0.0` parameter to `run_soak`, `run_soak_provider`, and `run_soak_profile`. In each `_worker`, after a successful request block, add:

```python
if think_time_seconds > 0 and not stop_event.is_set():
    await asyncio.sleep(think_time_seconds)
```

For `run_soak_provider` (no `stop_event`, uses a time-bound loop), use:

```python
if think_time_seconds > 0:
    await asyncio.sleep(think_time_seconds)
```

- [x] **Step 4: Run tests, confirm they pass**

```bash
uv run pytest tests/test_soak.py::TestThinkTime -v --no-cov
```

Expected: 3 passed

- [x] **Step 5: Run full suite to check for regressions**

```bash
uv run pytest tests/ -q --no-cov
```

Expected: 578+ passed, 0 failed

---

## Task 2: Add `--think-time` to the CLI

**Files:**
- Modify: `atomics/cli.py` — `soak` command

### What changes

Add `--think-time` option and pass it through to the runner. Surface effective simulated user count in the header.

```python
@click.option("--think-time", type=float, default=0.0, show_default=True,
              help="Seconds to wait between requests per worker (simulates user think time).")
```

In the header block, add (after the existing lines):
```python
if think_time_seconds > 0:
    # effective users = concurrency * (1 + think_time / avg_request_time)
    # we don't know avg_request_time yet, so just show the setting
    console.print(f"Think time: [bold]{think_time_seconds}s[/bold] per worker "
                  f"(simulates ~{concurrency} users with {think_time_seconds}s pauses)\n")
```

Pass `think_time_seconds=think_time` to all three runner branches.

- [x] **Step 1: Write the failing CLI test**

```python
# in tests/test_soak.py, add to existing CLI tests or new class

class TestThinkTimeCLI:
    def test_think_time_flag_accepted(self):
        """--think-time flag is accepted without error."""
        from click.testing import CliRunner
        from atomics.cli import cli

        runner = CliRunner()
        result = runner.invoke(cli, [
            "soak", "--help"
        ])
        assert "--think-time" in result.output

    def test_think_time_passed_to_runner(self):
        """--think-time value is forwarded to run_soak."""
        from click.testing import CliRunner
        from atomics.cli import cli
        from unittest.mock import patch, AsyncMock
        from atomics.soak import SoakResult

        fake_result = SoakResult(model="test", host="http://fake:11434")
        fake_result.samples = []

        async def _fake_run(**kwargs):
            assert kwargs.get("think_time_seconds") == 5.0
            return fake_result

        runner = CliRunner()
        with patch("atomics.soak.run_soak", side_effect=_fake_run):
            result = runner.invoke(cli, [
                "soak", "--model", "test",
                "--ollama-host", "http://fake:11434",
                "--duration", "1m",
                "--think-time", "5.0",
                "--no-save",
            ])
        assert result.exit_code == 0
```

- [x] **Step 2: Run tests, confirm they fail**

```bash
uv run pytest tests/test_soak.py::TestThinkTimeCLI -v --no-cov
```

Expected: `--think-time` not in help, `think_time_seconds` not forwarded

- [x] **Step 3: Implement in `cli.py`**

Add option to `soak` command decorator:
```python
@click.option("--think-time", type=float, default=0.0, show_default=True,
              help="Seconds to wait between requests per worker (simulates user think time).")
```

Add `think_time: float` to the function signature. In the header section, after printing duration/concurrency, add:
```python
if think_time > 0:
    console.print(
        f"Think time: [bold]{think_time}s[/bold] per worker — "
        f"simulates ~{concurrency} users with natural pauses\n"
    )
```

Pass to runners:
```python
# ollama branch
result = asyncio.run(run_soak(
    ...
    think_time_seconds=think_time,
))
# provider branch
result = asyncio.run(run_soak_provider(
    ...
    think_time_seconds=think_time,
))
# profile branch
result = asyncio.run(run_soak_profile(
    ...
    think_time_seconds=think_time,
))
```

- [x] **Step 4: Run tests, confirm they pass**

```bash
uv run pytest tests/test_soak.py::TestThinkTimeCLI -v --no-cov
```

Expected: 2 passed

- [x] **Step 5: Run full suite**

```bash
uv run pytest tests/ -q --no-cov
```

Expected: 580+ passed, 0 failed

- [x] **Step 6: Commit**

```bash
git add atomics/soak.py atomics/cli.py tests/test_soak.py
git commit -m "feat(soak): add --think-time for user think time simulation"
git push
```

---

## Self-Review

**Spec coverage:**
- `think_time_seconds` in all 3 runners ✓
- `--think-time` CLI flag ✓
- Header shows think-time setting ✓
- Tests: sleep called, zero is no-op, CLI flag accepted, value forwarded ✓
- Full suite regression check ✓

**Placeholder scan:** No TBDs, all code shown.

**Type consistency:** `think_time_seconds: float` throughout, `think_time: float` at CLI boundary.
