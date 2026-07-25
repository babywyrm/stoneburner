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
| POST | `/api/v1/distributed/runs` | Start a split-mode distributed run |
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

Split benchmark work across multiple worker processes that poll a coordinator (the atomics API server). Phase 1 supports **split mode** only: the coordinator divides a run into task assignments; workers claim, execute, and report results. Full-run and fleet modes are future work.

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

Heartbeat, claim, and result endpoints require worker authentication via `X-API-Key` (pluggable `WorkerAuth`; uses the server API keys when auth is enabled).

### Distributed run endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/v1/distributed/runs` | Start a split-mode run (`202` + job body) |
| GET | `/api/v1/distributed/runs/{job_id}` | Job status, assignments, and aggregated progress |

`POST /api/v1/distributed/runs` accepts a body with `mode` (`split`) and `run_request` (tier/iterations, plus optional provider/model). Only `split` mode is accepted in Phase 1.

A `provider`/`model` in `run_request` pins every task to that provider; omit them and each worker uses its own configured provider. `worker_selector` is **rejected with `400`** in Phase 1 — split mode assigns each task to the next available worker, so a selector cannot be honored.

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
```
