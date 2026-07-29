# API Server Mode

Run atomics as a local HTTP service for CI/CD, dashboards, or remote scheduling.

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

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/v1/health` | Health check (public) |
| POST | `/api/v1/runs` | Start a benchmark run |
| POST | `/api/v1/evals` | Start an eval suite |
| GET | `/api/v1/jobs/{job_id}` | Poll job status/result |
| GET | `/api/v1/compare` | Compare providers/models |
| GET | `/api/v1/reports/recent-runs` | Recent run report |
| POST | `/api/v1/workers/register` | Register a worker |
| POST | `/api/v1/workers/{worker_id}/heartbeat` | Worker heartbeat |
| GET | `/api/v1/workers/{worker_id}/jobs/next` | Claim next task assignment |
| POST | `/api/v1/workers/{worker_id}/jobs/{assignment_id}/result` | Submit task result |
| POST | `/api/v1/distributed/runs` | Start a distributed run (split, fleet, or full) |
| GET | `/api/v1/distributed/runs/{job_id}` | Distributed run status |

## Example: start a run and poll

```bash
JOB_ID=$(curl -s -H "X-API-Key: $ATOMICS_API_KEY" -H "Content-Type: application/json" \
  -d '{"provider": "ollama", "model": "qwen3:14b", "tier": "ez", "iterations": 3}' \
  http://127.0.0.1:8000/api/v1/runs | jq -r '.job_id')

sleep 2
curl -s -H "X-API-Key: $ATOMICS_API_KEY" http://127.0.0.1:8000/api/v1/jobs/$JOB_ID | jq
```

## Eval suites

POST `/api/v1/evals` accepts `"suite": "accuracy" | "rag" | "multiturn" | "adversarial" | "codegen"`.

```bash
curl -H "X-API-Key: $ATOMICS_API_KEY" -H "Content-Type: application/json" \
  -d '{"suite": "rag", "provider": "ollama", "model": "qwen3:14b"}' \
  http://127.0.0.1:8000/api/v1/evals
```

## Distributed Runs

Spread benchmark work across multiple worker processes that poll a coordinator (the atomics API server). Workers claim, execute, and report assignments; two modes decide how the work is divided:

- **split** — the coordinator divides one run into task assignments and any worker takes the next. Use it to finish a run faster.
- **fleet** — every worker matching a label selector receives the identical task set. Use it to compare hosts.

`full` (one worker runs an entire run) is declared but unimplemented and rejected.

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

A worker that stops sending heartbeats for 120 seconds is marked offline. It is then excluded from new fleet runs, and its pinned work is failed rather than left for a host that is not coming back. The window is `atomics server --worker-absent-after SECONDS`; raise it above roughly four times your workers' `--heartbeat-interval`, or hosts that are heartbeating exactly as configured will be declared absent and lose their fleet slice.

### Distributed run endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/v1/distributed/runs` | Start a distributed run (`202` + job body) |
| GET | `/api/v1/distributed/runs/{job_id}` | Job status, assignments, and aggregated progress |

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
