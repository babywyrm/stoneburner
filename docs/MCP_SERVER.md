# MCP Server

`atomics mcp` exposes atomics to LLM agents over the [Model Context
Protocol](https://modelcontextprotocol.io/). It is a proxy: every tool call is
one authenticated HTTP request to a running `atomics server`.

```bash
uv sync --extra mcp --extra api

export ATOMICS_API_KEY="$(openssl rand -hex 24)"
uv run atomics server --api-key "$ATOMICS_API_KEY"   # terminal 1
uv run atomics mcp                                 # terminal 2, or spawned by your MCP client
```

## Why a proxy and not a native server

An MCP client is an LLM agent: a remote, automated caller. That is the same
trust position a remote HTTP caller occupies, and the opposite of the CLI's,
which assumes a local operator spending their own money deliberately.

Routing through the API means the agent inherits the guardrails the API already
enforces, in one place:

- **API-key authentication** on every route except `/health`.
- **A per-eval dollar ceiling** (`DEFAULT_EVAL_BUDGET_USD`, 10 USD), applied to
  the model and its judge together.
- **Bounds on iterations and fixtures**, so one call cannot pin the server.
- **Async job scheduling**, so nothing blocks on model work.

The MCP server holds no provider, storage, or budget logic of its own. There is
no second copy of a spending decision to keep in sync, and the copy an agent
talks to is the one that was already audited.

## Configuration

| Setting | Source | Default |
|---------|--------|---------|
| API base URL | `--api-url` or `$ATOMICS_API_URL` | `http://127.0.0.1:8000` |
| API key | `--api-key` or `$ATOMICS_API_KEY` | none (anonymous) |

`ATOMICS_API_KEY` is the same variable the CLI reference and
[API_SERVER.md](API_SERVER.md) already use, so an MCP client needs no new
credential. Without a key the server still starts and warns: anonymous works
only against `atomics server --no-auth`.

### Client configuration

Most MCP clients spawn the server over stdio. The equivalent of:

```json
{
  "mcpServers": {
    "atomics": {
      "command": "uv",
      "args": ["run", "atomics", "mcp"],
      "env": {
        "ATOMICS_API_URL": "http://127.0.0.1:8000",
        "ATOMICS_API_KEY": "your-key"
      }
    }
  }
}
```

On stdio, **stdout carries the JSON-RPC frames**. The command writes all status
output and log records to stderr for that reason; nothing else may print to
stdout.

### stdio only

There is no HTTP transport option. This process holds an API key with spend
authority, and nothing at the MCP layer would authenticate a network caller, so
a listening port would hand that authority to anyone who could reach it — the
opposite of the guardrails the proxy exists to inherit.

To drive atomics from another host, expose the API server, which does
authenticate, and run `atomics mcp` locally against it:

```bash
ATOMICS_API_URL="https://atomics.internal:8000" uv run atomics mcp
```

## Tools

| Tool | Read-only | What it does |
|------|-----------|--------------|
| `health` | yes | Check the API server is reachable |
| `list_models` | yes | List tags on Ollama or vLLM |
| `list_jobs` | yes | In-memory API jobs (no `result`; poll `get_job`) |
| `get_job` | yes | Status, and result once finished, for a submitted job |
| `get_run` | yes | One persisted run and its fixtures (prompts omitted) |
| `compare` | yes | Compare recorded results by provider or model |
| `recent_runs` | yes | List recent recorded runs |
| `trends` | yes | Hourly token and cost series |
| `provider_test` | **no** | Health + a fixed 2+2 generate — spends a few tokens |
| `submit_run` | **no** | Start a benchmark run — spends tokens |
| `submit_eval` | **no** | Start an eval suite — spends tokens |

The spending tools are annotated as not read-only so a client can treat them
as costly. The read tools are annotated read-only so an agent is not
discouraged from the cheap calls.

### Asynchronous by design

`submit_run` and `submit_eval` return a job id immediately:

```json
{"job_id": "3f2a...", "status": "pending", "kind": "eval"}
```

Poll `get_job` with that id until `status` is finished, then read `result`. No
tool blocks on model work, which is what keeps a long eval from timing out an
agent's tool call. The server's `instructions` tell the agent to do this.

### Errors

A failed call carries the API's own `detail` — unknown suite, budget exceeded,
no such job — rather than a bare status code, so the agent can act on it. An
unreachable API names the missing piece instead of reporting a generic
connection error.

## Scope

The tool surface is bounded by what the API exposes. The CLI can do more —
`sweep`, `stress`, `soak`, `probe` — but those have no endpoint, and a proxy
should not invent one. Exposing any of them means adding an API endpoint first,
with the authentication and bounds that implies, and reaching MCP from there.

Note the symmetry worth keeping in mind: atomics ships an adversarial suite that
tests models against MCP and agentic manipulation (`--category mcp`). A server
that can spend money and start load is exactly the kind of target that suite
describes. Keep the tool surface narrow and the API key scoped.
