# API `models` and `provider-test`, then MCP

**Date:** 2026-08-16
**Status:** Approved 2026-08-16

## Goal

An agent talking to `atomics mcp` can discover what is loaded and check that
it answers, without inventing those tools in the proxy. The API grows the
two endpoints first; MCP only forwards them.

## Why now

`atomics mcp` is a proxy over `atomics server`. `models` and `provider-test`
are the two CLI commands an agent needs before `submit_eval`, and they have
no route. ROADMAP already named them. Sweep / stress / soak stay CLI-only.

## Scope

1. `GET /api/v1/models?provider=ollama|vllm&host=` — list tags. Auth. Sync.
2. `POST /api/v1/provider-test` — health + one fixed generate. Auth. Sync.
3. MCP tools `list_models` (read-only) and `provider_test` (spends).
4. Contract tests so a renamed route fails against the real app.
5. Docs: API_SERVER, MCP_SERVER, SECURITY (tool table). No house IPs.

It will not:

- add a job for either call (listing and `2+2` are seconds);
- accept a caller-supplied prompt (spend amplification);
- expose the CLI's keyless OpenAI auto-detect;
- add sweep / stress / soak / probe endpoints;
- change auth, budgets, or `--no-auth` loopback rules.

## Architecture

Reuse `make_provider`. Hosts go through `validate_endpoint_url`.

`models` is GET because it only reads the inference server. `provider-test`
is POST because it spends tokens; ROADMAP said "read-only" for both, which
is true of `models` and false of a generate. MCP labels them honestly.

Fixed probe: `"What is 2+2? Reply with just the number."`, `max_tokens=32`.
Same text as the CLI.

`models` accepts only `ollama` and `vllm` (the two providers that implement
`list_models`). Anything else is `400`. Connection failure is `502`.

`provider-test` accepts any factory provider. Config errors are `400`.
A finished probe is `200` with `ok: true|false` so an agent can read a
failed health check instead of treating it as transport death.

## Tests

- Auth required (401 without key).
- `models` lists from a mocked provider; unknown provider 400; connect 502.
- `provider-test` returns health + generate fields; no prompt field on the
  request model.
- MCP registers the two tools with the right read-only hints.
- Contract replay against the real FastAPI app.
