"""Tests for the provider factory's configuration contract.

The factory moved out of the command layer so the API server and distributed
workers could build providers without importing Click. The point of these tests
is that it reports failures in its own vocabulary: a caller that is not a CLI
must never receive a `ClickException`, and an endpoint error must not name a
command-line flag the caller never offered.
"""

from __future__ import annotations

from types import SimpleNamespace

import click
import pytest

from atomics.providers.factory import (
    PROVIDER_NAMES,
    ProviderConfigError,
    make_provider,
)


def test_unknown_provider_raises_domain_error_not_a_cli_error() -> None:
    """The whole reason the factory exists: no Click leaking into other layers."""
    with pytest.raises(ProviderConfigError) as excinfo:
        make_provider("invalid", None, None, SimpleNamespace())

    assert "Unknown provider" in str(excinfo.value)
    assert not isinstance(excinfo.value, click.ClickException)


def test_unknown_provider_message_lists_every_supported_name() -> None:
    """Guards the advertised list against drifting from the accepted one.

    The name list used to be duplicated between the error message and the CLI's
    click.Choice, so adding a provider meant remembering three places.
    """
    with pytest.raises(ProviderConfigError) as excinfo:
        make_provider("invalid", None, None, SimpleNamespace())

    message = str(excinfo.value)
    for name in PROVIDER_NAMES:
        assert name in message


@pytest.mark.parametrize(
    ("provider", "settings_attr", "expected_env_var"),
    [
        ("claude", "anthropic_api_key", "ANTHROPIC_API_KEY"),
        ("openai", "openai_api_key", "OPENAI_API_KEY"),
        ("groq", "groq_api_key", "GROQ_API_KEY"),
        ("together", "together_api_key", "TOGETHER_API_KEY"),
        ("gemini", "gemini_api_key", "GEMINI_API_KEY"),
    ],
)
def test_missing_credential_raises_domain_error(
    provider: str, settings_attr: str, expected_env_var: str
) -> None:
    settings = SimpleNamespace(**{settings_attr: ""})
    with pytest.raises(ProviderConfigError) as excinfo:
        make_provider(provider, None, None, settings)

    assert expected_env_var in str(excinfo.value)
    assert not isinstance(excinfo.value, click.ClickException)


def test_bad_endpoint_uses_a_neutral_label_by_default() -> None:
    """A server must not blame a CLI flag its caller never typed."""
    with pytest.raises(ProviderConfigError) as excinfo:
        make_provider("ollama", None, "file:///etc/passwd", SimpleNamespace())

    message = str(excinfo.value)
    assert message.startswith("host:")
    assert "--ollama-host" not in message


def test_vllm_endpoint_is_labelled_separately_from_the_generic_host() -> None:
    with pytest.raises(ProviderConfigError) as excinfo:
        make_provider(
            "vllm",
            None,
            None,
            SimpleNamespace(),
            vllm_host="file:///etc/passwd",
        )

    assert str(excinfo.value).startswith("vllm host:")


def test_caller_supplied_labels_reach_the_error_message() -> None:
    with pytest.raises(ProviderConfigError) as excinfo:
        make_provider(
            "ollama",
            None,
            "ftp://nope",
            SimpleNamespace(),
            host_label="--my-flag",
        )

    assert str(excinfo.value).startswith("--my-flag:")


def test_ollama_uses_control_file_when_host_and_model_unset(tmp_path, monkeypatch):
    from atomics.config import AtomicsSettings
    from atomics.providers.ollama import OllamaProvider

    env_path = tmp_path / "inference.env"
    env_path.write_text(
        "INFERENCE_BACKEND=ollama\nINFERENCE_URL=http://127.0.0.1:9999\nINFERENCE_MODEL=from-file\n"
    )
    monkeypatch.setenv("INFERENCE_ENV", str(env_path))
    monkeypatch.delenv("BRAIN_ENV", raising=False)
    monkeypatch.delenv("ATOMICS_OLLAMA_HOST", raising=False)
    monkeypatch.delenv("ATOMICS_OLLAMA_MODEL", raising=False)
    provider = make_provider("ollama", None, None, AtomicsSettings())
    assert isinstance(provider, OllamaProvider)
    assert provider._host == "http://127.0.0.1:9999"
    assert provider._default_model == "from-file"


def test_ollama_cli_host_wins_over_control_file(tmp_path, monkeypatch):
    from atomics.config import AtomicsSettings
    from atomics.providers.ollama import OllamaProvider

    env_path = tmp_path / "inference.env"
    env_path.write_text(
        "INFERENCE_BACKEND=ollama\nINFERENCE_URL=http://127.0.0.1:9999\nINFERENCE_MODEL=from-file\n"
    )
    monkeypatch.setenv("INFERENCE_ENV", str(env_path))
    monkeypatch.delenv("BRAIN_ENV", raising=False)
    provider = make_provider("ollama", None, "http://localhost:11434", AtomicsSettings())
    assert isinstance(provider, OllamaProvider)
    assert provider._host == "http://localhost:11434"
