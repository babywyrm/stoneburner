# Per-Pass Live Lines Implementation Plan

> **For agentic workers:** Execute inline in this session. TDD per suite.

**Goal:** Operators watching `redblue --runs 3` or `adversarial --runs 3` see each pass, not only the mean.

**Architecture:** Same `on_run_done` hook as toolcall. Fire after every pass including failures. CLI prints only when `--runs > 1`.

**Tech Stack:** existing pytest / Click / Rich runners.

---

### Task 1: Redblue runner hook

**Files:**
- Modify: `atomics/eval/redblue/runner.py`
- Test: `tests/test_redblue.py`

- [ ] Failing tests for fire-per-pass, failed generate, awaitable hook
- [ ] Implement `on_run_done` on `run_redblue`

### Task 2: Redblue CLI lines

**Files:**
- Modify: `atomics/commands/security/cmd_redblue.py`
- Test: `tests/test_redblue.py`

- [ ] Failing CLI test: `--runs 3` prints `run 2/3`; `--runs 1` does not
- [ ] Wire `on_run` callback

### Task 3: Adversarial runner hook

**Files:**
- Modify: `atomics/eval/adversarial/runner.py`
- Test: `tests/test_adversarial.py`

- [ ] Same three runner tests, using `_single_fixture`

### Task 4: Adversarial CLI lines

**Files:**
- Modify: `atomics/commands/security/cmd_adversarial.py`
- Test: `tests/test_cli.py`

- [ ] CLI `--runs 3` / `--runs 1`; compare path receives the hook

### Task 5: Docs + changelog

**Files:**
- Modify: `docs/SECURITY_SUITES.md`, `docs/CLI_REFERENCE.md`, `CHANGELOG.md`
