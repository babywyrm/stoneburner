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

In-memory only. `set provider ollama`, `set model llama3.2:1b`,
`set host http://192.168.1.79:11434`, `set effort high`, `show`.
`set model` with no value clears it. Submit verbs fill omitted fields
from the session (`host` goes to `submit_eval`, `list_models`, and
`provider_test`). An explicit flag wins.

## Verbs

The same names as the MCP tools. Semantics live in [MCP_SERVER.md](MCP_SERVER.md).
Plus `set`, `show`, `wait [--verbose] [JOB_ID]`, `help`, `exit`.

`submit_*` prints the job body (including resolved `request`) and remembers
`job_id`. Type `wait` once: it polls every 2s and prints a **quiet line**
per generate/judge and per scored fixture, then a two-line headline when
`completed`. `wait --verbose` also prints latency and the truncated model
reply (500 chars, same as the job document). Color is TTY-only. `get_job`
still returns the full JSON. A 60-second cap or Ctrl-C returns the prompt;
the job keeps running. Type `wait` again only if it is still `running`
after the cap.

Quiet:

```
  ev-01  generate  llama3.2:1b
  ev-01  0.70  success  267 tok
accuracy  llama3.2:1b  http://192.168.1.79:11434
0.700  1/1  267 tok  $0.00
```

`--verbose` adds latency and the truncated reply under each score line.

`probe`, hours-long soak, contention, and profiles stay on the CLI.

Sweep / stress / soak still require `budget_usd`. Sweep suite names are `eval`
(not `accuracy`), `redblue`, `refusal`, `toolcall`, `codereview`.
