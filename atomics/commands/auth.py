"""Auth and secrets CLI commands."""

from __future__ import annotations

import asyncio

import click
from rich.console import Console

from atomics.config import load_settings


@click.command("login")
@click.option(
    "--profile",
    "oidc_profile",
    type=str,
    default="openai",
    help="Built-in OIDC profile name",
)
@click.option("--issuer", type=str, default=None, help="Custom OIDC issuer URL")
@click.option("--client-id", type=str, default=None, help="Custom OIDC client ID")
@click.option("--scopes", type=str, default=None, help="Space-separated OIDC scopes")
@click.option("--headless", is_flag=True, help="Use device code flow (no browser)")
def login(
    oidc_profile: str,
    issuer: str | None,
    client_id: str | None,
    scopes: str | None,
    headless: bool,
) -> None:
    """Log in via OAuth/OIDC (opens browser or prints device code)."""
    from atomics.auth.oauth import OAuthPKCEAuth
    from atomics.auth.profiles import OIDCProfile, get_profile
    from atomics.auth.store import TokenStore

    console = Console()

    if issuer and client_id:
        profile = OIDCProfile(
            name="custom",
            issuer=issuer,
            authorization_endpoint=f"{issuer.rstrip('/')}/authorize",
            token_endpoint=f"{issuer.rstrip('/')}/oauth/token",
            device_authorization_endpoint=f"{issuer.rstrip('/')}/oauth/device/code",
            client_id=client_id,
            scopes=scopes.split() if scopes else ["openid", "profile", "email"],
        )
    else:
        profile = get_profile(oidc_profile)

    store = TokenStore()
    auth = OAuthPKCEAuth(profile=profile, store=store)

    async def _login():
        tokens = await auth.login(headless=headless)
        console.print(f"[green]Logged in via {profile.name}[/green]")
        console.print(f"[dim]Tokens cached at {store.path}[/dim]")
        return tokens

    asyncio.run(_login())


@click.command("logout")
def logout() -> None:
    """Clear cached OAuth tokens."""
    from atomics.auth.store import TokenStore

    console = Console()
    store = TokenStore()
    store.clear()
    console.print("[green]Logged out — cached tokens cleared.[/green]")


@click.command("whoami")
def whoami() -> None:
    """Show current auth mode and identity."""
    from atomics.auth.store import TokenStore

    console = Console()
    settings = load_settings()

    if settings.openai_api_key:
        masked = settings.openai_api_key[:8] + "..."
        console.print(f"[cyan]Auth mode:[/cyan] API key ({masked})")
        return

    from atomics.auth.codex import CodexTokenAuth

    codex = CodexTokenAuth()
    if codex.tokens_available():
        console.print("[cyan]Auth mode:[/cyan] Codex CLI API key (~/.codex/auth.json)")
        return
    if codex.codex_installed():
        console.print(
            "[yellow]Codex CLI detected[/yellow] but its ChatGPT tokens can't access "
            "the OpenAI API. Create a key at https://platform.openai.com/api-keys"
        )

    store = TokenStore()
    tokens = store.load()
    if tokens.access_token and not tokens.expired:
        console.print(f"[cyan]Auth mode:[/cyan] OAuth ({tokens.profile_name or 'unknown'})")
        # Decode identity from id_token if available
        if tokens.id_token:
            try:
                import base64
                import json

                parts = tokens.id_token.split(".")
                payload = parts[1] + "=" * (4 - len(parts[1]) % 4)
                claims = json.loads(base64.urlsafe_b64decode(payload))
                email = claims.get("email", "")
                name = claims.get("name", "")
                if name or email:
                    console.print(f"[dim]Identity:[/dim] {name} ({email})" if name else f"[dim]Identity:[/dim] {email}")
            except Exception:
                pass
        return

    console.print("[yellow]Not authenticated.[/yellow] Run [bold]atomics login[/bold] or set OPENAI_API_KEY.")


@click.group("secrets")
def secrets_group():
    """Manage API keys and secrets in the OS keychain.

    Secrets stored here are used as the last-resort fallback when an API key
    is not set via environment variable or .env file.

    Resolution order (first found wins):
      1. Environment variable (e.g. ANTHROPIC_API_KEY=...)
      2. .env file in the project directory
      3. OS keychain (managed by this command)
    """


@secrets_group.command("set")
@click.argument("key")
@click.option("--force", is_flag=True, default=False,
              help="Allow storing keys not in the standard set.")
def secrets_set(key: str, force: bool):
    """Store a secret in the OS keychain.

    The value is prompted interactively (hidden input — not echoed to terminal).

    Example: atomics secrets set ANTHROPIC_API_KEY
    """
    from atomics.secrets import KNOWN_KEYS, keychain_available, set_secret

    if not keychain_available():
        click.echo("Error: no OS keychain backend available.", err=True)
        raise SystemExit(1)

    key = key.upper()
    if key not in KNOWN_KEYS and not force:
        click.echo(
            f"Error: unknown key {key!r}. Valid keys: {', '.join(sorted(KNOWN_KEYS))}\n"
            f"Use --force to store a custom key.",
            err=True,
        )
        raise SystemExit(1)

    value = click.prompt(f"Enter value for {key}", hide_input=True, confirmation_prompt=True)
    if not value.strip():
        click.echo("Error: empty value — not stored.", err=True)
        raise SystemExit(1)

    set_secret(key, value.strip())
    click.echo(f"Stored: {key}")


@secrets_group.command("get")
@click.argument("key")
@click.option("--show", is_flag=True,
              help="Print the secret value to stdout. Off by default so keys are "
                   "not exposed in terminal scrollback, logs, or shell history.")
def secrets_get(key: str, show: bool):
    """Check a secret in the OS keychain. Prints the value only with --show.

    By default this reports whether the key is present (and a masked preview)
    without exposing the value. Use --show to print the raw value for piping:

      atomics secrets get ANTHROPIC_API_KEY            # presence + masked
      atomics secrets get ANTHROPIC_API_KEY --show     # raw value
    """
    from atomics.secrets import get_secret

    key = key.upper()
    value = get_secret(key)
    if value is None:
        click.echo(f"Not found: {key}", err=True)
        raise SystemExit(1)
    if show:
        click.echo(value)
        return
    # Masked preview: never reveal more than the last 4 chars, and only for
    # values long enough that a tail can't reconstruct the secret.
    preview = f"****{value[-4:]}" if len(value) >= 12 else "****"
    click.echo(f"{key}: set ({len(value)} chars, {preview}) — use --show to reveal")


@secrets_group.command("list")
def secrets_list():
    """List secret keys stored in the OS keychain (names only, never values)."""
    from atomics.secrets import keychain_available, list_secrets

    if not keychain_available():
        click.echo("No OS keychain backend available.")
        return

    stored = list_secrets()
    if not stored:
        click.echo("No secrets stored in the keychain.")
        return

    click.echo(f"{len(stored)} secret(s) in keychain:")
    for key in stored:
        click.echo(f"  {key}")


@secrets_group.command("delete")
@click.argument("key")
def secrets_delete(key: str):
    """Remove a secret from the OS keychain.

    Example: atomics secrets delete ANTHROPIC_API_KEY
    """
    from atomics.secrets import delete_secret

    key = key.upper()
    if delete_secret(key):
        click.echo(f"Deleted: {key}")
    else:
        click.echo(f"Not found or could not delete: {key}", err=True)
        raise SystemExit(1)
