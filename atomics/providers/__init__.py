"""Provider adapters for LLM APIs."""

from atomics.providers.base import BaseProvider, ProviderResponse
from atomics.providers.claude import ClaudeProvider

__all__ = [
    "BaseProvider",
    "BedrockProvider",
    "BrainGatewayProvider",
    "ClaudeProvider",
    "OllamaProvider",
    "OpenAIProvider",
    "ProviderResponse",
]


def __getattr__(name: str):
    if name == "BedrockProvider":
        from atomics.providers.bedrock import BedrockProvider

        return BedrockProvider
    if name == "BrainGatewayProvider":
        from atomics.providers.brain_gateway import BrainGatewayProvider

        return BrainGatewayProvider
    if name == "OllamaProvider":
        from atomics.providers.ollama import OllamaProvider

        return OllamaProvider
    if name == "OpenAIProvider":
        from atomics.providers.openai import OpenAIProvider

        return OpenAIProvider
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
