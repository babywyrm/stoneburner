"""Build provider adapters from settings.

This lives in the providers layer rather than the command layer because the API
server and distributed workers need to construct providers too, and reaching up
into `atomics.commands` to do it inverted the dependency direction: a FastAPI
worker with a missing credential ended up raising a Click exception.

Adapter modules are imported inside each branch so that installing one optional
extra does not require the others.
"""

from __future__ import annotations

from atomics.config import AtomicsSettings
from atomics.providers.base import BaseProvider
from atomics.validation import validate_endpoint_url

PROVIDER_NAMES: tuple[str, ...] = (
    "claude",
    "bedrock",
    "openai",
    "ollama",
    "vllm",
    "brain-gateway",
    "groq",
    "together",
    "gemini",
    "llamacpp",
)


class ProviderConfigError(Exception):
    """A provider could not be built from the supplied configuration.

    Covers a missing credential, an unusable endpoint URL, and an unknown
    provider name. Each caller renders it in its own idiom: the CLI re-raises it
    as a `ClickException`, the API returns 400, and a distributed worker fails
    the assignment.
    """


def make_provider(
    name: str,
    model: str | None,
    host: str | None,
    settings: AtomicsSettings,
    *,
    vllm_host: str | None = None,
    region: str = "us-east-1",
    context_tokens: int | None = None,
    inference_timeout: int | None = None,
    host_label: str = "host",
    vllm_host_label: str = "vllm host",
) -> BaseProvider:
    """Build a provider instance, or raise `ProviderConfigError`.

    `host_label` and `vllm_host_label` name the offending setting in endpoint
    validation errors. The CLI passes its flag names; other callers leave the
    neutral defaults so a server does not claim a command-line flag was wrong.
    """
    if host:
        try:
            host = validate_endpoint_url(host, label=host_label)
        except ValueError as exc:
            raise ProviderConfigError(str(exc)) from exc
    if vllm_host:
        try:
            vllm_host = validate_endpoint_url(vllm_host, label=vllm_host_label)
        except ValueError as exc:
            raise ProviderConfigError(str(exc)) from exc

    if name == "claude":
        if not settings.anthropic_api_key:
            raise ProviderConfigError(
                "ANTHROPIC_API_KEY not set. Export it or add to .env"
            )
        from atomics.providers.claude import ClaudeProvider

        return ClaudeProvider(
            api_key=settings.anthropic_api_key,
            default_model=model or settings.default_model,
        )
    if name == "bedrock":
        from atomics.providers.bedrock import BedrockProvider

        return BedrockProvider(
            region=region,
            model_id=model or "us.anthropic.claude-sonnet-4-6",
        )
    if name == "openai":
        if not settings.openai_api_key:
            raise ProviderConfigError(
                "OPENAI_API_KEY not set. Export it or install with: "
                "uv sync --extra openai"
            )
        from atomics.providers.openai import OpenAIProvider

        return OpenAIProvider(
            api_key=settings.openai_api_key,
            default_model=model or "gpt-4o",
        )
    if name == "vllm":
        from atomics.providers.vllm import VllmProvider

        return VllmProvider(
            base_url=vllm_host or settings.vllm_host,
            default_model=model or settings.vllm_model,
            timeout=inference_timeout or settings.vllm_timeout,
        )
    if name == "brain-gateway":
        from atomics.providers.brain_gateway import BrainGatewayProvider

        return BrainGatewayProvider(
            url=host or settings.brain_gateway_url,
            default_model=model,
        )
    if name == "ollama":
        from atomics.providers.ollama import OllamaProvider

        return OllamaProvider(
            host=host or settings.ollama_host,
            default_model=model or settings.ollama_model,
            timeout=inference_timeout or settings.ollama_timeout,
            context_tokens=context_tokens,
        )
    if name == "llamacpp":
        from atomics.providers.llamacpp import LlamaCppProvider

        return LlamaCppProvider(
            base_url=host or settings.llamacpp_host,
            default_model=model or "local",
        )
    if name == "groq":
        if not settings.groq_api_key:
            raise ProviderConfigError(
                "GROQ_API_KEY not set. Get one at https://console.groq.com/keys"
            )
        from atomics.providers.groq import GroqProvider

        return GroqProvider(
            api_key=settings.groq_api_key,
            default_model=model or "llama-3.3-70b-versatile",
        )
    if name == "together":
        if not settings.together_api_key:
            raise ProviderConfigError(
                "TOGETHER_API_KEY not set. "
                "Get one at https://api.together.xyz/settings/api-keys"
            )
        from atomics.providers.together import TogetherProvider

        return TogetherProvider(
            api_key=settings.together_api_key,
            default_model=model or "meta-llama/Llama-3.3-70B-Instruct-Turbo",
        )
    if name == "gemini":
        if not settings.gemini_api_key:
            raise ProviderConfigError(
                "GEMINI_API_KEY not set. Get one at https://aistudio.google.com/apikey"
            )
        from atomics.providers.gemini import GeminiProvider

        return GeminiProvider(
            api_key=settings.gemini_api_key,
            default_model=model or "gemini-2.5-flash",
        )
    raise ProviderConfigError(
        f"Unknown provider: {name!r}. Valid: {', '.join(PROVIDER_NAMES)}"
    )
