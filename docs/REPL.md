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
(or set `ATOMICS_API_URL`). That is deliberate.

## Session

In-memory only. `set provider ollama`, `set model gpt-oss:20b`, `set effort high`,
`show`. `set model` with no value clears it. Submit verbs fill omitted fields
from the session. An explicit flag wins.

## Verbs

The same names as the MCP tools. Semantics live in [MCP_SERVER.md](MCP_SERVER.md).
Plus `set`, `show`, `wait [JOB_ID]`, `help`, `exit`.

`submit_*` prints the job body and remembers `job_id`. `wait` polls until
`status` is `completed` (not `finished`), then prints the job JSON. A 60-second
poll cap or Ctrl-C stops the poll; the job keeps running.

`probe`, hours-long soak, contention, and profiles stay on the CLI.

Sweep / stress / soak still require `budget_usd`. Sweep suite names are `eval`
(not `accuracy`), `redblue`, `refusal`, `toolcall`, `codereview`.
