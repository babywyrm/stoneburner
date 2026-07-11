# Security Considerations

This document covers operational security decisions in stoneburner that users
and operators should be aware of.

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

## LLM output rendering

LLM responses and judge rationale are escaped before Rich terminal rendering
to prevent markup injection (e.g., a model outputting `[bold red]FAKE[/]`).

## Error message persistence

Exception strings stored in the database (`error_message` columns) are
sanitized to strip common credential patterns (Bearer tokens, API keys,
AWS access keys) before persistence and export.

## Adversarial evaluation exports

Adversarial JSON exports contain raw model responses and complete judge-call
evidence, including visible response text, thinking text, the effective text
that was parsed, and sanitized error diagnostics. These fields can still
contain prompts, generated secrets, personal data, or other sensitive content.

Treat adversarial exports as sensitive. Store them with restrictive file
permissions, limit access to trusted operators, and avoid committing or
uploading them to public systems.
