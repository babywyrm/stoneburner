"""OS-keychain-backed secrets storage for atomics.

Resolution order (first found wins):
  1. Env var (ANTHROPIC_API_KEY=...)
  2. .env file (handled by pydantic-settings)
  3. OS keychain via this module

Uses the `keyring` package which auto-detects the OS backend:
  - macOS: Keychain
  - Linux: secret-service (GNOME Keyring / KWallet)
  - Windows: Credential Locker
  - CI/headless: falls back gracefully (returns None)

Never logs secret values. The `list` function returns key names only, and the
`get` CLI command masks the value unless `--show` is passed explicitly.
"""

from __future__ import annotations

import logging

logger = logging.getLogger("atomics.secrets")

SERVICE_NAME = "atomics"

# Known secret keys that load_settings() will check
KNOWN_KEYS = frozenset({
    "ANTHROPIC_API_KEY",
    "OPENAI_API_KEY",
    "OLLAMA_API_KEY",
    "BEDROCK_ACCESS_KEY",
    "BEDROCK_SECRET_KEY",
})


def set_secret(key: str, value: str) -> None:
    """Store a secret in the OS keychain."""
    import keyring

    keyring.set_password(SERVICE_NAME, key, value)
    logger.debug("stored secret: %s", key)


def get_secret(key: str) -> str | None:
    """Retrieve a secret from the OS keychain.

    Returns None if the key is not stored or the keychain backend is unavailable.
    Never raises on missing keys or backend failures — graceful degradation.
    """
    try:
        import keyring

        value = keyring.get_password(SERVICE_NAME, key)
        if value:
            logger.debug("resolved secret from keychain: %s", key)
        return value
    except Exception:
        return None


def delete_secret(key: str) -> bool:
    """Remove a secret from the OS keychain. Returns True if deleted."""
    try:
        import keyring

        keyring.delete_password(SERVICE_NAME, key)
        logger.debug("deleted secret: %s", key)
        return True
    except Exception:
        return False


def list_secrets() -> list[str]:
    """List stored secret key names (never values).

    Note: keyring doesn't have a native 'list all' API, so we check
    KNOWN_KEYS and return which ones have stored values.
    """
    stored = []
    for key in sorted(KNOWN_KEYS):
        if get_secret(key) is not None:
            stored.append(key)
    return stored


def has_secrets() -> bool:
    """Return True if any secrets are stored in the keychain."""
    return len(list_secrets()) > 0


def keychain_available() -> bool:
    """Return True if the OS keychain backend is functional."""
    try:
        import keyring

        backend = keyring.get_keyring()
        # The 'fail' backend means no real keychain is available
        return "fail" not in backend.name.lower()
    except Exception:
        return False
