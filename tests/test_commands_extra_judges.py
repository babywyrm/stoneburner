from types import SimpleNamespace

from atomics.commands.common import parse_extra_judges


def test_empty_spec_is_no_panel() -> None:
    assert parse_extra_judges(None, build=lambda *_a, **_k: None) == []
    assert parse_extra_judges("", build=lambda *_a, **_k: None) == []


def test_parses_provider_model_and_optional_host() -> None:
    built: list[tuple] = []

    def build(name, model, host):
        built.append((name, model, host))
        return SimpleNamespace(name=name)

    pairs = parse_extra_judges(
        "claude:claude-sonnet-4-6,ollama:deepseek-r1:14b@http://gpu-host:11434",
        build=build,
        default_host="http://fallback:11434",
    )
    assert [p[1] for p in pairs] == ["claude-sonnet-4-6", "deepseek-r1:14b"]
    assert built[0] == ("claude", "claude-sonnet-4-6", "http://fallback:11434")
    assert built[1] == ("ollama", "deepseek-r1:14b", "http://gpu-host:11434")
