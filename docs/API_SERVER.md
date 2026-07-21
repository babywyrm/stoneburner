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

# production with API key
uv run atomics server --api-key sk-abc123 --api-key sk-xyz789
```

## Authentication

API routes (except health) require an `X-API-Key` header when API keys are configured.

```bash
curl -H "X-API-Key: sk-abc123" http://127.0.0.1:8000/api/v1/runs \
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
JOB_ID=$(curl -s -H "X-API-Key: sk-abc123" -H "Content-Type: application/json" \
  -d '{"provider": "ollama", "model": "qwen3:14b", "tier": "ez", "iterations": 3}' \
  http://127.0.0.1:8000/api/v1/runs | jq -r '.job_id')

sleep 2
curl -s -H "X-API-Key: sk-abc123" http://127.0.0.1:8000/api/v1/jobs/$JOB_ID | jq
```

## Eval suites

POST `/api/v1/evals` accepts `"suite": "accuracy" | "rag" | "multiturn" | "adversarial" | "codegen"`.

```bash
curl -H "X-API-Key: sk-abc123" -H "Content-Type: application/json" \
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

`POST /api/v1/distributed/runs` accepts a body with `mode` (`split`), `run_request` (provider/tier/iterations/model), and optional `worker_selector` labels. Only `split` mode is accepted in Phase 1.

### Example: local three-terminal setup

```bash
# Terminal 1 — coordinator
uv run atomics server --api-key sk-abc123

# Terminal 2 — worker
ATOMICS_WORKER_API_KEY=sk-abc123 uv run atomics worker --label gpu=1

# Terminal 3 — submit and poll
ATOMICS_API_KEY=sk-abc123 uv run atomics distributed run -p ollama -t baseline -n 4 --label gpu=1
ATOMICS_API_KEY=sk-abc123 uv run atomics distributed status <job_id>
```
