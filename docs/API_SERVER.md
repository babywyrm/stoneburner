# API Server Mode

Run atomics as a local HTTP service for CI/CD, dashboards, remote scheduling,
or as the backend for [`atomics mcp`](MCP_SERVER.md), which exposes the same
authenticated surface to LLM agents.

## Install

```bash
uv sync --extra api
```

## Start the server

```bash
# local development with no auth (do not use in production)
uv run atomics server --no-auth

# production with API keys — pass --api-key once per accepted key
export ATOMICS_API_KEY="$(openssl rand -hex 24)"
uv run atomics server --api-key "$ATOMICS_API_KEY"
```

## Authentication

API routes (except health) require an `X-API-Key` header when API keys are configured.

Examples below read the key from `$ATOMICS_API_KEY` rather than inlining it, so
keys stay out of shell history and out of anything you paste into an issue.

```bash
curl -H "X-API-Key: $ATOMICS_API_KEY" http://127.0.0.1:8000/api/v1/runs \
  -H "Content-Type: application/json" \
  -d '{"provider": "ollama", "iterations": 3}'
```

### Separate worker keys

Workers register, poll, and submit results with an `X-API-Key` too. Without
`--worker-api-key` they share the submitter keys, which means a worker
credential also authorizes run and eval submission. Give workers their own keys
whenever they run on hosts you do not control:

```bash
uv run atomics server \
  --api-key "$ATOMICS_API_KEY" \
  --worker-api-key "$ATOMICS_WORKER_API_KEY"
```

A key passed only to `--worker-api-key` is rejected on `/api/v1/runs` and
`/api/v1/evals`, and a submitter key is rejected on the worker endpoints.

### `--no-auth` is loopback-only

`--no-auth` disables authentication entirely, so the server refuses to start if
it is combined with a non-loopback `--host`. Binding `0.0.0.0` without a key
would expose eval submission to the network unauthenticated.

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/v1/health` | Liveness — is this process serving (public) |
| GET | `/api/v1/ready` | Readiness — can it serve real work (public) |
| POST | `/api/v1/runs` | Start a benchmark run |
| POST | `/api/v1/evals` | Start an eval suite |
| POST | `/api/v1/sweeps` | Start a bounded multi-model, multi-suite campaign (budget required) |
| POST | `/api/v1/stress` | Ramp concurrency to find saturation (budget required, c≤8, phase ≤15s) |
| POST | `/api/v1/soak` | Hold concurrency and classify drift (budget required, 30–300s, c≤4) |
| GET | `/api/v1/jobs` | List in-memory API jobs (no result payload) |
| GET | `/api/v1/jobs/{job_id}` | Poll job status/result |
| GET | `/api/v1/models` | List Ollama or vLLM tags (`?provider=&host=`) |
| POST | `/api/v1/provider-test` | Health + fixed 2+2 generate (spends a few tokens) |
| GET | `/api/v1/compare` | Compare providers/models |
| GET | `/api/v1/reports/recent-runs` | Recent run report |
| GET | `/api/v1/reports/trends` | Hourly token/cost series (`?hours=1..168`) |
| GET | `/api/v1/runs/{run_id}` | One persisted run and sanitized fixtures |
| POST | `/api/v1/workers/register` | Register a worker |
| POST | `/api/v1/workers/{worker_id}/heartbeat` | Worker heartbeat |
| GET | `/api/v1/workers/{worker_id}/jobs/next` | Claim next task assignment |
| POST | `/api/v1/workers/{worker_id}/jobs/{assignment_id}/result` | Submit task result |
| POST | `/api/v1/distributed/runs` | Start a distributed run (split, fleet, or full) |
| GET | `/api/v1/distributed/runs/{job_id}` | Distributed run status |
| GET | `/api/v1/distributed/runs` | List recent distributed runs |
| GET | `/api/v1/workers` | List registered workers |
| GET | `/dashboard` | Optional web dashboard (requires `--with-dashboard`) |

### Liveness vs readiness

Point liveness probes at `/health` and readiness probes at `/ready`. They
answer different questions and conflating them causes the wrong repair.

`/health` returns `200 {"status": "ok"}` whenever the process can serve a
request, and checks nothing external. Liveness answers "should this process be
restarted", and restarting the API server does not fix an unreachable
database — it just kills in-flight jobs and removes the endpoint that could
have told you what was wrong.

`/ready` checks that the coordinator's database actually answers, returning
`503` when it does not, with a per-check breakdown:

```json
{"status": "not_ready",
 "checks": [{"name": "database", "ok": false, "detail": "OperationalError: disk I/O error"}]}
```

That is the signal a load balancer should use to take an instance out of
rotation. Before this split, `/health` answered `ok` unconditionally, so a
server stayed in rotation while every request it received was going to fail.

Both endpoints are public and need no API key: a probe should not require a
credential, and neither reveals run data.

### Correlation IDs

Every response carries an `X-Request-ID`. Send your own to thread a trace
through — it is honored when it matches `[A-Za-z0-9._-]{1,64}`, and replaced
with a generated ID otherwise, since an ID containing newlines can forge log
lines.

The ID appears in one access log line per request and, crucially, in the logs
of any job that request started:

```
request_id=9f2c1a caller=7bafaa96aa23 method=POST path=/api/v1/evals status=202 duration_ms=41.2
job_submitted job_id=3d8e... kind=eval caller=7bafaa96aa23 request_id=9f2c1a
job_finished  job_id=3d8e... kind=eval caller=7bafaa96aa23 request_id=9f2c1a status=failed duration_ms=91847.0
```

Runs and evals are async, so the submitting request returns long before the
work finishes. The shared ID is what connects a failure to whoever asked for
it. Query strings and request bodies are never logged.

`caller` is a twelve-character digest of the API key, never the key itself.

Server logs are one plain line per record — no Rich wrapping — so `grep`,
journald, and log aggregators can parse them. Uvicorn's own access log is
disabled, since it writes the raw request line including the query string.

### Per-caller capacity

`max_active_jobs` bounds the whole server; `max_active_jobs_per_caller`
(default `4`) bounds any one key. Without the second, the first is
first-come-first-served — one impatient script takes every slot and every other
key gets `429` until its work drains.

Both limits return `429`; the message says which one you hit. Under `--no-auth`
there is no credential to tell callers apart, so per-caller quotas collapse to
the global limit and the server warns about it at startup.

## Example: start a run and poll

```bash
JOB_ID=$(curl -s -H "X-API-Key: $ATOMICS_API_KEY" -H "Content-Type: application/json" \
  -d '{"provider": "ollama", "model": "qwen3:14b", "tier": "ez", "iterations": 3}' \
  http://127.0.0.1:8000/api/v1/runs | jq -r '.job_id')

sleep 2
curl -s -H "X-API-Key: $ATOMICS_API_KEY" http://127.0.0.1:8000/api/v1/jobs/$JOB_ID | jq
```

## Dashboard

The server can serve an optional, read-only web dashboard at `/dashboard`. Enable it with `--with-dashboard`:

```bash
uv run atomics server --no-auth --with-dashboard
```

Open `http://127.0.0.1:8000/dashboard`. When authentication is enabled the page
prompts for a key and keeps it in `sessionStorage`. You can still pass
`?api_key=YOUR_KEY` to prefill it — the dashboard saves it and immediately
strips it from the address bar so it does not linger in browser history,
referrer headers, or proxy access logs.

The dashboard auto-refreshes every 10 seconds and shows:

- Recent benchmark runs with status, tokens, and cost. Click a run id (or
  open `#run=<id>`) to see fixture scores. Prompts and raw judge JSON are
  not sent to the browser.
- Hourly token trends (eval and benchmark fixtures)
- In-memory API jobs. Click a job id (or `#job=<id>`) to watch status on
  the 10s refresh. The list omits `result`; the panel never renders it.
- Active distributed jobs and their mode
- Registered workers and their capabilities/labels
- Provider/model success-rate comparison bars

The dashboard is purely visual: it only reads from the existing API endpoints and does not change any server behavior.

All values are written to the page with `textContent`. Worker labels and
capabilities are caller-supplied, so rendering them as markup would let anyone
who can register a worker run script in an operator's browser.

## Eval suites

POST `/api/v1/evals` accepts these `suite` values. `atomics mcp` `submit_eval`
forwards the same names. The job result's `overall_score` is the suite
headline; for `toolcall` that headline is the dangerous-call rate (higher is
worse).

| `suite` | Headline | Measures |
|---------|----------|----------|
| `accuracy` | accuracy | Quality vs gold answers |
| `rag` | RAG score | Grounding / faithfulness |
| `multiturn` | conversation score | Context retention |
| `adversarial` | resilience | Resistance to manipulation |
| `codegen` | pass rate | Generated tests that pass |
| `refusal` | calibration | Over- vs under-refusal |
| `redblue` | quality | Offensive / defensive capability |
| `toolcall` | dangerous-call rate | Tool-channel leaks (higher is worse) |
| `codereview` | review score | Planted-vuln detection vs false positives |

`probe` is CLI-only. Load tests have their own endpoints, not `suite` values.

### Sweeps

`POST /api/v1/sweeps` runs the overnight driver (`eval.gauntlet`) as a job.
Unlike a single eval, **`budget_usd` has no default** — omit it and the
request is `422`. Name the models; there is no `--all-local` / discover-
everything flag (call `GET /models` first). Caps: 8 models, 3 runs, suites
from `eval`, `redblue`, `refusal`, `toolcall`, `codereview` (note `eval`,
not `accuracy`).

```bash
curl -H "X-API-Key: $ATOMICS_API_KEY" -H "Content-Type: application/json" \
  -d '{"provider":"ollama","models":["qwen3:14b","granite4.1:8b"],"suites":["redblue","refusal"],"runs":3,"budget_usd":25}' \
  http://127.0.0.1:8000/api/v1/sweeps
```

### Stress and soak

`POST /api/v1/stress` ramps concurrency against one named model.
`POST /api/v1/soak` holds concurrency and returns STABLE / DEGRADED /
UNSTABLE. Both require `budget_usd`. These are not the CLI's hours-long
path: concurrency is 1–8 (stress) or 1–4 (soak), each stress phase is at
most 15 seconds, soak duration is 30–300 seconds, and `num_predict` is
fixed at 256. No contention mode, no profile YAML, no baselines.

```bash
curl -H "X-API-Key: $ATOMICS_API_KEY" -H "Content-Type: application/json" \
  -d '{"provider":"ollama","model":"qwen3:14b","budget_usd":2,"max_concurrency":4}' \
  http://127.0.0.1:8000/api/v1/stress

curl -H "X-API-Key: $ATOMICS_API_KEY" -H "Content-Type: application/json" \
  -d '{"provider":"ollama","model":"qwen3:14b","budget_usd":2,"duration_seconds":120,"concurrency":2}' \
  http://127.0.0.1:8000/api/v1/soak
```

The `codegen` suite executes model-generated Python. It runs in a child
interpreter with a scrubbed environment (no provider credentials), a scratch
working directory, address-space and CPU limits, blocked network calls, and a
wall-clock kill. See [`atomics/eval/codegen/sandbox.py`](../atomics/eval/codegen/sandbox.py).
That boundary assumes a model producing wrong code, not an attacker with a
kernel exploit — restrict who can reach this endpoint accordingly.

```bash
curl -H "X-API-Key: $ATOMICS_API_KEY" -H "Content-Type: application/json" \
  -d '{"suite": "rag", "provider": "ollama", "model": "qwen3:14b"}' \
  http://127.0.0.1:8000/api/v1/evals
```

### Spend ceiling

Every API-triggered eval is metered. `budget_usd` defaults to `10.0` and is a
single ceiling shared by the model under test and every judge, so a run cannot
quietly cost a multiple of what was asked for:

```bash
curl -H "X-API-Key: $ATOMICS_API_KEY" -H "Content-Type: application/json" \
  -d '{"suite": "adversarial", "provider": "claude", "budget_usd": 2.50}' \
  http://127.0.0.1:8000/api/v1/evals
```

You can lower the ceiling but not remove it. `0`, negative values, and anything
above `1000` are rejected with `422`. A run that hits the ceiling stops rather
than continuing to spend, and the job reports `EvalBudgetExceededError` with the
amount spent — distinct from a `400`, which means the request itself was bad.

The CLI defaults the other way: no ceiling unless you pass `--budget`, because
a local operator is spending their own money on a run they chose to start. See
[SECURITY.md](../SECURITY.md#spend-ceilings-on-eval-suites).

## Distributed Runs

Spread benchmark work across multiple worker processes that poll a coordinator (the atomics API server). Workers claim, execute, and report assignments; two modes decide how the work is divided:

- **split** — the coordinator divides one run into task assignments and any worker takes the next. Use it to finish a run faster.
- **fleet** — every worker matching a label selector receives the identical task set. Use it to compare hosts.
- **full** — one worker runs an entire benchmark end-to-end. Use it to delegate a full run to a specific host or capability.

### Coordinator / worker model

1. Start the API server as the coordinator.
2. Start one or more `atomics worker` processes. Each registers, heartbeats, polls for assignments, and submits results.
3. Submit a distributed run with `atomics distributed run` (or `POST /api/v1/distributed/runs`).
4. Poll status with `atomics distributed status` (or `GET /api/v1/distributed/runs/{job_id}`).

### Worker endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/v1/workers/register` | Register a worker; returns `worker_id` |
| POST | `/api/v1/workers/{worker_id}/heartbeat` | Keep the worker alive |
| GET | `/api/v1/workers/{worker_id}/jobs/next` | Claim the next pending assignment (or empty) |
| POST | `/api/v1/workers/{worker_id}/jobs/{assignment_id}/result` | Submit assignment result or error |

Every worker endpoint, registration included, requires worker authentication via `X-API-Key` (pluggable `WorkerAuth`; uses the server API keys when auth is enabled). A worker has no id yet at registration, so the worker key is the only credential it can present.

Workers register with `capabilities` (e.g., `["python"]` for the built-in Python worker, `["node"]` for the npm bridge). The coordinator uses capabilities to route assignments: a task with `runtime: "node"` is only claimed by workers that advertise the `node` capability. Python workers advertise `python` by default; the npm worker advertises `node` by default. A worker with no capabilities is treated as `python` for backwards compatibility.

A worker that stops sending heartbeats for 120 seconds is marked offline. It is then excluded from new fleet runs, and its pinned work is failed rather than left for a host that is not coming back. The window is `atomics server --worker-absent-after SECONDS`; raise it above roughly four times your workers' `--heartbeat-interval`, or hosts that are heartbeating exactly as configured will be declared absent and lose their fleet slice.

### Distributed run endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/v1/distributed/runs` | Start a distributed run (`202` + job body) |
| GET | `/api/v1/distributed/runs/{job_id}` | Job status, assignments, and aggregated progress |
| GET | `/api/v1/distributed/runs` | List recent distributed runs |
| GET | `/api/v1/workers` | List registered workers |

Both require client authentication via `X-API-Key`. Submitting a run spends GPU time and cloud budget, so neither is anonymous.

`POST /api/v1/distributed/runs` accepts `mode`, `run_request` (tier/iterations, plus optional provider/model), and for fleet mode an optional `worker_selector`.

| `mode` | Behavior |
|--------|----------|
| `split` | One assignment per task, claimed by whichever worker asks first. `worker_selector` is **rejected with `400`**, since each task goes to the next available worker and a selector could not be honored. |
| `fleet` | Every worker matching `worker_selector` receives the identical task set, for cross-host comparison. |
| `full` | One assignment containing the whole `run_request` is pinned to the first matching worker (or left unclaimed if no selector is given). The chosen worker executes the full `LoopEngine` locally. |

A `provider`/`model` in `run_request` pins every task to that provider; omit them and each worker uses its own configured provider.

For fleet runs, a worker must match every pair in `worker_selector`; an absent or empty selector matches all online workers. A selector matching no online worker is **rejected with `400`** rather than creating a job that can never progress. The matching workers are snapshotted at submit time.

When a job reaches a terminal status, `summary_json` carries a per-worker rollup: completed and failed counts, tokens, mean and p95 latency, throughput, and estimated cost, plus job totals.

```bash
# Fleet run across every 4090 in the lab
curl -X POST "$COORDINATOR/api/v1/distributed/runs" \
  -H "X-API-Key: $ATOMICS_API_KEY" \
  -H 'Content-Type: application/json' \
  -d '{"mode": "fleet",
       "run_request": {"tier": "baseline", "iterations": 20},
       "worker_selector": {"gpu": "4090", "site": "lab"}}'

# Full run: delegate a complete run to one box=239 worker
curl -X POST "$COORDINATOR/api/v1/distributed/runs" \
  -H "X-API-Key: $ATOMICS_API_KEY" \
  -H 'Content-Type: application/json' \
  -d '{"mode": "full",
       "run_request": {"tier": "ez", "iterations": 5},
       "worker_selector": {"box": "239"}}'
```

### Example: local three-terminal setup

All three terminals share one key; generate it once with
`export ATOMICS_API_KEY="$(openssl rand -hex 24)"`.

```bash
# Terminal 1 — coordinator
uv run atomics server --api-key "$ATOMICS_API_KEY"

# Terminal 2 — worker
ATOMICS_WORKER_API_KEY="$ATOMICS_API_KEY" uv run atomics worker --label gpu=1

# Terminal 3 — submit and poll
uv run atomics distributed run -p ollama -t baseline -n 4
uv run atomics distributed status <job_id>

# Or compare every labelled host on identical work
uv run atomics distributed run --mode fleet --label gpu=1 -n 20
uv run atomics distributed status <job_id>
```
