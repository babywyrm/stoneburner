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
