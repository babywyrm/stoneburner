"""Built-in OIDC provider profiles."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class OIDCProfile:
    name: str
    issuer: str
    authorization_endpoint: str = ""
    token_endpoint: str = ""
    device_authorization_endpoint: str = ""
    client_id: str = ""
    scopes: list[str] = field(default_factory=lambda: ["openid", "profile", "email"])
    callback_port: int = 19274


OPENAI_PROFILE = OIDCProfile(
    name="openai",
    issuer="https://auth0.openai.com/",
    authorization_endpoint="https://auth.openai.com/authorize",
    token_endpoint="https://auth0.openai.com/oauth/token",
    device_authorization_endpoint="https://auth0.openai.com/oauth/device/code",
    client_id="app_EMoamEEZ73f0CkXaXp7hrann",
    scopes=["openid", "profile", "email", "offline_access"],
)

_PROFILES: dict[str, OIDCProfile] = {
    "openai": OPENAI_PROFILE,
}


def get_profile(name: str) -> OIDCProfile:
    """Look up a built-in OIDC profile by name."""
    if name not in _PROFILES:
        available = ", ".join(sorted(_PROFILES))
        raise ValueError(f"Unknown OIDC profile {name!r}. Available: {available}")
    return _PROFILES[name]


def list_profiles() -> list[str]:
    return sorted(_PROFILES)
