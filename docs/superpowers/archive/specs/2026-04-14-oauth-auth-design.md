# OAuth/OIDC Auth for Atomics

## Problem

The OpenAI provider requires a static `OPENAI_API_KEY`. Users who authenticate
via OAuth (Codex CLI, corporate SSO) can't use atomics without generating a
separate key. We need flexible auth that works with OpenAI's ChatGPT OAuth,
reuses Codex CLI tokens when available, and supports arbitrary OIDC providers
for corporate environments.

## Design

### Auth Strategy Pattern

`AuthStrategy` ABC injected into providers. Three implementations:

| Strategy | Source | Refresh | Use case |
|----------|--------|---------|----------|
| `ApiKeyAuth` | Static env var | None | Current default, unchanged |
| `OAuthPKCEAuth` | Browser PKCE flow | Auto refresh_token | Standalone login, generic OIDC |
| `CodexTokenAuth` | `~/.codex/auth.json` | Reuses Codex refresh_token | Convenience for Codex CLI users |

### Auto-Detection Order

1. `OPENAI_API_KEY` set → `ApiKeyAuth`
2. `~/.codex/auth.json` exists with unexpired tokens → `CodexTokenAuth`
3. Atomics cached tokens from `atomics login` → `OAuthPKCEAuth`
4. Nothing → error: "run `atomics login` or set OPENAI_API_KEY"

Override with `--auth apikey|oauth|codex` on any command.

### PKCE Flow

Uses httpx (existing dependency). No new packages.

1. Generate `code_verifier` + `code_challenge` (S256)
2. Start temporary `localhost:19274` callback server
3. Open browser → OIDC authorization endpoint
4. User authenticates, redirect back with `code`
5. Exchange code + verifier for tokens at token endpoint
6. Cache to disk, auto-refresh on expiry

Headless fallback: device code flow (print URL + code for manual entry).

### Built-in OIDC Profiles

`openai` profile baked in with OpenAI's issuer, client_id, and scopes.
Custom providers configured via env vars:

| Variable | Default | Purpose |
|----------|---------|---------|
| `ATOMICS_AUTH_MODE` | `auto` | Force auth strategy |
| `ATOMICS_OIDC_ISSUER` | — | Custom OIDC issuer URL |
| `ATOMICS_OIDC_CLIENT_ID` | — | Custom OIDC client ID |
| `ATOMICS_OIDC_SCOPES` | — | Space-separated scopes |

### CLI Commands

| Command | Purpose |
|---------|---------|
| `atomics login` | Browser OIDC login, cache tokens |
| `atomics login --issuer URL --client-id ID` | Custom OIDC provider |
| `atomics logout` | Clear cached tokens |
| `atomics whoami` | Show auth mode + identity |

### Token Storage

- Linux: `$XDG_DATA_HOME/atomics/auth.json` (default `~/.local/share/atomics/`)
- macOS: `~/Library/Application Support/atomics/auth.json`
- Separate from metrics DB. File permissions `0600`.

### Provider Changes

`OpenAIProvider.__init__` gains optional `auth: AuthStrategy`. When present,
uses `auth.get_headers()` instead of static API key. Claude and Bedrock
providers unchanged.

### File Layout

```
atomics/auth/
├── __init__.py      # AuthStrategy ABC, auto_detect_auth()
├── apikey.py        # ApiKeyAuth
├── oauth.py         # OAuthPKCEAuth (PKCE + device code + refresh)
├── codex.py         # CodexTokenAuth (reads ~/.codex/auth.json)
├── store.py         # Token persistence (XDG/macOS paths)
└── profiles.py      # Built-in OIDC configs
```

### Testing

- Unit: mock OIDC endpoints via httpx, test token caching/refresh/expiry,
  PKCE generation, auto-detection logic
- CLI: login/logout/whoami with mocked browser + callback; auth mode
  selection in run/provider-test
- Integration: `@pytest.mark.network` for real OAuth flows (opt-in)
