"""Authentication strategies for LLM providers."""

from __future__ import annotations

from abc import ABC, abstractmethod

__all__ = ["AuthStrategy", "auto_detect_auth"]


class AuthStrategy(ABC):
    """Base class for all auth strategies."""

    @abstractmethod
    async def get_headers(self) -> dict[str, str]:
        """Return HTTP headers (typically Authorization) for an API request."""

    @abstractmethod
    async def validate(self) -> bool:
        """Return True if the current credentials are usable."""

    @property
    @abstractmethod
    def description(self) -> str:
        """Human-readable label for CLI output (e.g. 'API key', 'OAuth (openai)')."""


def auto_detect_auth(
    *,
    api_key: str = "",
    oidc_profile: str = "openai",
    oidc_issuer: str | None = None,
    oidc_client_id: str | None = None,
    oidc_scopes: str | None = None,
) -> AuthStrategy:
    """Pick the best available auth strategy in priority order.

    1. Static API key (if provided)
    2. Codex CLI tokens (~/.codex/auth.json)
    3. Atomics cached OAuth tokens
    4. Raise with instructions
    """
    if api_key:
        from atomics.auth.apikey import ApiKeyAuth

        return ApiKeyAuth(api_key)

    from atomics.auth.codex import CodexTokenAuth

    codex = CodexTokenAuth()
    if codex.tokens_available():
        return codex

    from atomics.auth.oauth import OAuthPKCEAuth
    from atomics.auth.store import TokenStore

    store = TokenStore()
    if store.has_valid_tokens():
        if oidc_issuer and oidc_client_id:
            from atomics.auth.profiles import OIDCProfile

            profile = OIDCProfile(
                name="custom",
                issuer=oidc_issuer,
                client_id=oidc_client_id,
                scopes=oidc_scopes.split() if oidc_scopes else ["openid", "profile"],
            )
        else:
            from atomics.auth.profiles import get_profile

            profile = get_profile(oidc_profile)
        return OAuthPKCEAuth(profile=profile, store=store)

    hint = ""
    if codex.codex_installed():
        hint = (
            "\n\nNote: Codex CLI is installed but its ChatGPT OAuth tokens "
            "cannot access the OpenAI developer API (missing scopes). "
            "A separate API key is required."
        )

    raise RuntimeError(
        "No OpenAI credentials found. Either:\n"
        "  1. export OPENAI_API_KEY=sk-...  (from https://platform.openai.com/api-keys)\n"
        "  2. run: atomics login" + hint
    )
