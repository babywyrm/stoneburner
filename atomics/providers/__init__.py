"""Provider adapters for LLM APIs."""

from atomics.providers.base import BaseProvider, ProviderResponse
from atomics.providers.claude import ClaudeProvider

__all__ = [
    "BaseProvider",
    "BedrockProvider",
    "ClaudeProvider",
    "OpenAIProvider",
    "ProviderResponse",
]


def __getattr__(name: str):
    if name == "BedrockProvider":
        from atomics.providers.bedrock import BedrockProvider

        return BedrockProvider
    if name == "OpenAIProvider":
        from atomics.providers.openai import OpenAIProvider

        return OpenAIProvider
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
