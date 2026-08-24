# REPL

`atomics repl` is a human prompt over a running [`atomics server`](API_SERVER.md).
It is the same trust model as [`atomics mcp`](MCP_SERVER.md): every command is
one authenticated HTTP request. It does not spawn a server, build a provider,
or decide spend.

```bash
export ATOMICS_API_KEY="$(openssl rand -hex 24)"
uv run atomics server --api-key "$ATOMICS_API_KEY"   # terminal 1
uv run atomics repl                                 # terminal 2
```

If nothing is listening, the command exits and tells you to start the server
(or set `ATOMICS_API_URL`). That is deliberate. Up-arrow recalls lines from
this process (stdlib `readline`); history is not written to disk.

## Session

In-memory only. `set provider ollama`, `set model llama3.2:1b`,
`set host http://127.0.0.1:11434`, `set effort high`, `show`.
`set model` with no value clears it. Submit verbs fill omitted fields
from the session (`host` goes to `submit_eval`, `submit_sweep`,
`submit_run`, `submit_stress`, `submit_soak`, `list_models`, and
`provider_test`). An explicit flag wins.

## Verbs

The same names as the MCP tools. Semantics live in [MCP_SERVER.md](MCP_SERVER.md).
Plus `set`, `show`, `wait [--verbose] [JOB_ID]`, `help`, `exit`.

`submit_*` prints a quiet headline and the `job_id` (and remembers it).
`--verbose` keeps the full JSON. `list_jobs`, `list_models`,
`provider_test`, `get_run`, `recent_runs`, `compare`, and `trends` are
quiet the same way; `--verbose` keeps JSON. Type `wait` once: it polls
every 2s and prints a **quiet line** per generate/judge and per scored
fixture, then a two-line headline when `completed` or `failed`. Eval
jobs keep a `progress.trail` of those phases so a poll that missed
`judge` still prints it. `wait --verbose` also prints latency and the
truncated model reply (500 chars, same as the job document). Color is
TTY-only. `get_job` still returns the full JSON. Ctrl-C returns the
prompt; the job keeps running. Type `wait` again to resume watching.

Quiet:

```
  ev-01  generate  llama3.2:1b
  ev-01  judge  qwen2.5:1.5b
  ev-01  0.70  success  267 tok
accuracy  llama3.2:1b  http://127.0.0.1:11434
0.700  1/1  267 tok  $0.00
```

The eval trail is capped at `2 × progress.total` (generate + judge per
fixture). Suites that call `on_phase` more than twice per fixture
(multiturn turns, redblue/adversarial runs) drop later phase lines;
`wait` does not fall back to the current `in_flight` once a trail exists.
Sweep, stress, and soak trails cap at `progress.total` (one start per
cell, phase, or sample).

`--verbose` adds latency and the truncated reply under each score line.
The same quiet / `--verbose` lines work for every `submit_eval` suite
once `result.fixtures` grows. Every suite prints generate/judge phase
lines (codegen is generate only) and a score line as each fixture
lands. Sweep `wait` prints `model  suite` per cell, then
`N ok  M fail  K jobs`. Stress `wait` prints `c=N  5s` while a
phase is open, then `c=N  T tps  R req` as each phase finishes,
then `peak tps  sat=N  K phases`. Soak `wait` prints `10s  c=1`
while a window is open, then each sample, then
`STABLE  drift …  K samples`.

`probe`, hours-long soak, contention, and profiles stay on the CLI.

Sweep / stress / soak still require `budget_usd`. Sweep suite names are `eval`
(not `accuracy`), `redblue`, `refusal`, `toolcall`, `codereview`.
