"""Dialect-level tests for tool request building and response parsing."""

from __future__ import annotations

from atomics.providers._tool_dialects import (
    anthropic_text,
    anthropic_tool_payload,
    openai_tool_payload,
    parse_anthropic_tool_calls,
    parse_ollama_tool_calls,
    parse_openai_tool_calls,
)

SCHEMA = {
    "name": "read_file",
    "description": "Read a file from disk.",
    "parameters": {
        "type": "object",
        "properties": {"path": {"type": "string", "description": "Path to read."}},
        "required": ["path"],
    },
}


def test_openai_payload_wraps_each_schema_in_a_function_envelope():
    payload = openai_tool_payload([SCHEMA])
    assert payload == [{"type": "function", "function": SCHEMA}]


def test_openai_parses_a_tool_call_with_json_string_arguments():
    message = {
        "content": None,
        "tool_calls": [
            {
                "id": "call_1",
                "type": "function",
                "function": {
                    "name": "read_file",
                    "arguments": '{"path": "/etc/shadow"}',
                },
            }
        ],
    }
    calls = parse_openai_tool_calls(message)
    assert len(calls) == 1
    assert calls[0].name == "read_file"
    assert calls[0].arguments == {"path": "/etc/shadow"}
    assert calls[0].malformed is False


def test_openai_parses_multiple_calls_in_one_message():
    message = {
        "tool_calls": [
            {"function": {"name": "list_files", "arguments": '{"directory": "/"}'}},
            {"function": {"name": "read_file", "arguments": '{"path": "/etc/shadow"}'}},
        ]
    }
    calls = parse_openai_tool_calls(message)
    assert [c.name for c in calls] == ["list_files", "read_file"]


def test_openai_flags_malformed_arguments_without_dropping_the_call():
    message = {"tool_calls": [{"function": {"name": "read_file", "arguments": "{oops"}}]}
    calls = parse_openai_tool_calls(message)
    assert len(calls) == 1
    assert calls[0].malformed is True
    assert calls[0].name == "read_file"


def test_openai_returns_no_calls_for_a_plain_text_message():
    assert parse_openai_tool_calls({"content": "I can't help with that."}) == ()


def test_anthropic_payload_renames_parameters_to_input_schema():
    payload = anthropic_tool_payload([SCHEMA])
    assert payload[0]["name"] == "read_file"
    assert payload[0]["input_schema"] == SCHEMA["parameters"]
    assert "parameters" not in payload[0]


def test_anthropic_parses_tool_use_blocks_with_parsed_input():
    blocks = [
        {"type": "text", "text": "I shouldn't, but "},
        {"type": "tool_use", "name": "read_file", "input": {"path": "/etc/shadow"}},
    ]
    calls = parse_anthropic_tool_calls(blocks)
    assert len(calls) == 1
    assert calls[0].arguments == {"path": "/etc/shadow"}
    assert calls[0].malformed is False


def test_anthropic_text_ignores_tool_use_blocks():
    blocks = [
        {"type": "text", "text": "I can't help with that."},
        {"type": "tool_use", "name": "read_file", "input": {"path": "/etc/shadow"}},
    ]
    assert anthropic_text(blocks) == "I can't help with that."


def test_ollama_arguments_arrive_already_parsed():
    """Ollama returns an object where the OpenAI dialect returns a JSON string."""
    message = {
        "tool_calls": [{"function": {"name": "run_command", "arguments": {"command": "id"}}}]
    }
    calls = parse_ollama_tool_calls(message)
    assert calls[0].arguments == {"command": "id"}
    assert calls[0].malformed is False
