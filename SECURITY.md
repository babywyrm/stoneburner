# Security

Reporting a vulnerability comes first; everything after it documents the
operational security decisions in stoneburner that users and operators should
be aware of.

## Reporting a vulnerability

**Do not open a public issue for a security problem.** Use GitHub's private
reporting instead:

[Report a vulnerability](https://github.com/babywyrm/stoneburner/security/advisories/new)

If private reporting is unavailable to you, open a public issue containing only
"security report, requesting contact" and no details, and a private channel
will be arranged.

### What to include

A working proof of concept helps more than anything else. Beyond that: the
version or commit, which component is affected (CLI, API server, coordinator,
worker, dashboard, an eval suite), and what an attacker gains.

### Scope

This is a benchmarking tool, not a hosted service, so severity depends heavily
on deployment. In scope, roughly in order of interest:

- Anything reachable over the API server or coordinator endpoints without the
  corresponding key, or that lets one key act with another's privileges
- Escapes from the codegen sandbox (see below for what it does and does not
  claim)
- Credential disclosure — provider keys, OAuth tokens, keychain contents —
  through logs, exports, error messages, or the database
- Injection into the terminal, the dashboard, or persisted records

Out of scope: findings that require an attacker who already has local code
execution as the operator, the deliberate risks documented later in this file
(post-run hooks, custom OIDC issuers), and resource exhaustion by an
already-authenticated caller, which is a known gap being tracked rather than a
new finding.

### What to expect

This is a single-maintainer project, so the honest answer is best effort rather
than a contractual SLA: acknowledgement within about a week, a fix or a written
decision not to fix before any public disclosure, and credit in the changelog
unless you would rather not be named. If you plan to publish, say so and a
timeline can be agreed rather than discovered.

## Secret scanning

Run both scanners before pushing. The git-history scans are the ones that
matter — they cover exactly what leaves the machine, and ignore untracked local
files such as `profiles/local/`, `qa/local/`, and `.env`.

```bash
# Committed history
trufflehog git file://. --results=verified,unknown --fail \
  --exclude-detectors=Polygon,Pastebin,URI
gitleaks git --no-banner --config .gitleaks.toml

# Working tree, including staged-but-uncommitted changes
trufflehog filesystem . --exclude-paths=.trufflehogignore --fail \
  --exclude-detectors=Polygon,Pastebin,URI
gitleaks dir --no-banner --config .gitleaks.toml
```

Both should report no findings. Configuration notes:

- `Polygon`, `Pastebin`, and `URI` are excluded as high-noise TruffleHog
  detectors: they fire on public endpoint URLs in fixtures and documentation.
- TruffleHog's `--config` accepts only custom detector definitions, so path and
  detector exclusions must be CLI flags. Paths live in `.trufflehogignore`.
- `.gitleaks.toml` must be passed explicitly; gitleaks does not auto-discover
  it. It extends the upstream ruleset with path exclusions and an allowlist for
  documented placeholders that remain in git history.
- Examples in `docs/` read keys from `$ATOMICS_API_KEY` rather than inlining
  them. Keep it that way: a literal `sk-`-prefixed placeholder trips
  credential detectors and trains readers to paste keys into shell history.

## Post-run hooks (`--hook` / `ATOMICS_POST_RUN_HOOK`)

The `atomics run --hook "command"` flag (or `ATOMICS_POST_RUN_HOOK` env var)
executes an arbitrary shell command after each burn-loop iteration. This is
intentional — it enables notification scripts, log rotation, and custom
integrations.

**Risks:**
- The command runs with the same privileges as the atomics process.
- On shared machines or in CI, any process that can set environment variables
  or modify the hook flag can achieve code execution.
- The hook is passed to `subprocess.run(..., shell=True)`.

**Mitigations:**
- Do not use hooks on shared/untrusted machines without restricting who can
  set the environment or pass CLI flags.
- Consider running atomics in a sandboxed container when hooks are enabled.
- Hooks are opt-in — they do not run unless explicitly configured.

## Custom OIDC issuer (`atomics login --issuer`)

The `--issuer` flag allows specifying a custom OpenID Connect provider for
OAuth-based authentication. This is intended for enterprise environments with
private identity providers.

**Risks:**
- A malicious `--issuer` URL could point the local OAuth callback server at an
  attacker-controlled OIDC endpoint, potentially capturing tokens.
- The device authorization flow polls the issuer's token endpoint repeatedly.

**Mitigations:**
- Only use `--issuer` with URLs you trust and control.
- Built-in profiles (no `--issuer`) connect only to known providers.
- Token storage uses `0o600` file permissions and OS keychain when available.

## Secrets storage

Secrets stored via `atomics secrets set` use the OS keychain (macOS Keychain,
Linux secret-service, Windows Credential Locker). Values are never written to
disk as plaintext, never logged, and never included in exported results.

`atomics secrets get` masks values by default; use `--show` to reveal.

## URL validation

All `--ollama-host`, `--vllm-host`, `--judge-host`, and `--host` endpoints are
validated to require `http://` or `https://` schemes. Embedded credentials,
`file://` URIs, and path traversal are rejected.

## API server exposure

`atomics server` was designed for a trusted operator on a trusted network. Its
authorization model is a shared key, not per-caller identity, so treat reaching
the port as equivalent to holding the key.

- `--no-auth` is refused on any non-loopback `--host`. It disables
  authentication for every route, including eval submission.
- Workers authenticate with `--worker-api-key` when supplied. Leave it unset and
  workers share the submitter keys, which means a worker credential also
  authorizes runs and evals. Set it whenever workers run on hosts you do not
  control; the server warns at startup when they are shared.
- A worker may only submit results for an assignment it currently holds and has
  not already finished. Other submissions are rejected with `409`.
- Keys are compared with `hmac.compare_digest`.
- Every response carries `nosniff`, `X-Frame-Options: DENY`,
  `Referrer-Policy: no-referrer`, and a restrictive CSP. The dashboard's CSP
  uses a per-response nonce rather than `unsafe-inline`.

Resource bounds, all configurable on `ServerSettings`:

| Bound | Default | What it prevents |
|-------|---------|------------------|
| `max_active_jobs` | 16 | Unbounded concurrent runs; excess gets `429` |
| `max_retained_jobs` | 256 | The job dict growing until the process dies |
| `iterations` | 1000 | A single request running indefinitely |
| `interval` | 3600s | The same, via a long sleep between tasks |
| `fixtures` | 500 | An oversized eval request |
| `budget_usd` | 10.0 | One eval spending without limit; capped at 1000 |
| `max_active_jobs_per_caller` | 4 | One key taking every slot and starving the rest |

These cap how much *one server* will do at once. The per-caller bound is the
one that makes the others fair: a global limit alone is first-come-first-served,
so whoever submits first holds all sixteen slots and every other key gets `429`
until that work drains. Under `--no-auth` there is no credential distinguishing
callers, so the per-caller bound collapses into the global one.

## Logging

Access logs record the correlation ID, a caller digest, method, path, status,
and duration. They deliberately omit query strings and request bodies, both of
which have carried API keys.

Callers appear as a twelve-character SHA-256 prefix of their key, never the key.
Logs are read, shipped, and retained far more casually than credentials are, so
the identifier is stable enough to attribute activity and useless as a
credential.

A caller-supplied `X-Request-ID` is accepted only as `[A-Za-z0-9._-]{1,64}`.
Anything else is replaced with a generated ID, since an identifier containing
newlines or control characters can forge log entries.

Note that prompts and responses are still persisted to SQLite unencrypted, and
the eval runner logs prompt prefixes at INFO. Treat the results database and
logs as sensitive.

## Spend ceilings on eval suites

Benchmark runs have always been metered: `LoopEngine` builds a `RateBudgetGuard`
from the tier profile. Eval suites had no equivalent, which mattered because
`POST /api/v1/evals` is reachable by any holder of an API key and an adversarial
run with extra judges is the most expensive operation this tool performs.

Every eval provider is now wrapped in a guard that meters spend, hourly tokens,
requests per minute, and consecutive failures. The model under test and every
judge share **one** ceiling, so a run with `--extra-judges` cannot quietly cost
a multiple of what was asked for. Judge traffic is metered too — that is where
consensus scoring actually spends.

The two surfaces default differently, deliberately:

| Surface | Default | Why |
|---------|---------|-----|
| `POST /api/v1/evals` | Always metered, `$10` | The caller is remote and holds a shared key |
| `atomics <suite>` | No ceiling unless `--budget` | A local operator is spending their own money on a run they started |

An API caller may lower `budget_usd` but cannot remove it; `0` and negative
values are rejected with `422`, and the ceiling itself is capped at `$1000`.
When a run hits a ceiling it stops with `EvalBudgetExceededError` rather than
continuing to spend, and the job records that as the failure reason. Per-minute
request pressure is waited out instead, since it clears on its own.

Two limits worth stating plainly:

- This bounds a **single run**. It is not per-caller accounting across runs: an
  authenticated caller can still submit repeatedly, and each submission gets its
  own ceiling. Per-caller budget accounting remains unimplemented.
- A dollar ceiling only binds where calls cost money. Local providers (Ollama,
  vLLM, llama.cpp) report `$0.00`, so `budget_usd` never trips for them. The
  guard's request-rate and hourly-token limits still apply, but if your threat
  model is "someone pins the GPU on a local endpoint", the spend ceiling is not
  the control that stops it.

## Generated code execution (codegen suite)

The `codegen` suite executes code written by the model under test, and is
reachable over HTTP via `POST /api/v1/evals`. Each snippet runs in a child
interpreter with:

- an environment stripped of provider credentials
- a temporary working directory, discarded afterwards
- address-space, CPU, and file-size limits
- `socket` calls blocked
- a wall-clock timeout that kills the whole process group, which also covers
  module-level statements

This is a real boundary but not a jail. It assumes a model producing incorrect
or careless code, not an adversary with a sandbox escape. Do not expose the
codegen suite to untrusted callers, and prefer running it in a container.

## LLM output rendering

LLM responses and judge rationale are escaped before Rich terminal rendering
to prevent markup injection (e.g., a model outputting `[bold red]FAKE[/]`).

## Error message persistence

Exception strings stored in the database (`error_message` columns) are
sanitized to strip common credential patterns (Bearer tokens, API keys,
AWS access keys) before persistence and export.

## Evaluation evidence and exports

Adversarial, refusal, and code-review JSON exports contain raw model responses
and judge-call evidence, including visible response text, thinking text, the
effective text that was parsed, and sanitized error diagnostics. Refusal and
code-review persist the same canonical evidence in
`evaluation_results.result_json`. These fields can still contain prompts,
generated secrets, personal data, source code, or other sensitive content.

Treat evaluation exports and the Stoneburner metrics database as sensitive.
Store them with restrictive file permissions, limit access to trusted
operators, and avoid committing or uploading them to public systems. Use
`--no-save` when durable local evidence is unnecessary; this does not redact
explicit `--json-out` output.
