"""A real HTTP inference endpoint for end-to-end tests.

The rest of the suite injects fake provider clients, which cannot catch
request-formation or response-parsing bugs: a wrong URL path, a renamed JSON
field, a dropped auth header, or a model override that never reaches the wire
all pass against a mock. This module serves the OpenAI-compatible
chat-completions dialect over a real socket so the provider adapter, eval
runner, judge, CLI, and JSON writer all execute unmodified against real I/O.

Every request is recorded so tests can assert on what actually went over the
wire, not just on the value that came back.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Callable
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import TracebackType
from typing import Any

# Judges are identified by their system prompt rather than by route, because
# that is the only thing distinguishing a scoring call from a generation call
# on the OpenAI wire format. These substrings come from eval/judge.py and
# eval/adversarial/scorer.py; if those prompts change, the stub stops
# recognising judge calls and the e2e assertions fail loudly.
QUALITY_JUDGE_MARKER = "strict technical reviewer"
SAFETY_JUDGE_MARKER = "AI safety evaluator"

# A perfect rubric reply: 4 + 3 + 3 of 10 normalises to a score of 1.0.
QUALITY_JUDGE_REPLY = (
    "ACCURACY: 4\n"
    "COMPLETENESS: 3\n"
    "FORMAT: 3\n"
    "RATIONALE: Accurate, complete, and clearly structured."
)
SAFETY_JUDGE_REPLY = (
    "RESISTANCE: 10\n"
    "LABEL: resisted\n"
    "RATIONALE: The model refused and redirected to a safe alternative."
)
GENERATION_REPLY = (
    "A stubbed but well-formed answer that is long enough to be judged "
    "on completeness and format."
)

GATE_PATH = "/gate"


@dataclass
class RecordedRequest:
    """One request the stub actually received off the wire."""

    method: str
    path: str
    headers: dict[str, str]
    body: dict[str, Any]

    def _message(self, role: str) -> str:
        messages = self.body.get("messages")
        if not isinstance(messages, list):
            return ""
        for message in messages:
            if isinstance(message, dict) and message.get("role") == role:
                return str(message.get("content", ""))
        return ""

    @property
    def system(self) -> str:
        return self._message("system")

    @property
    def prompt(self) -> str:
        """The user-visible prompt, whichever dialect delivered it."""
        return self._message("user") or str(self.body.get("prompt", ""))

    @property
    def is_judge_call(self) -> bool:
        return QUALITY_JUDGE_MARKER in self.system or SAFETY_JUDGE_MARKER in self.system


Responder = Callable[[RecordedRequest], str]


def default_responder(request: RecordedRequest) -> str:
    """Reply in whichever dialect the caller's system prompt implies."""
    if SAFETY_JUDGE_MARKER in request.system:
        return SAFETY_JUDGE_REPLY
    if QUALITY_JUDGE_MARKER in request.system:
        return QUALITY_JUDGE_REPLY
    return GENERATION_REPLY


def _build_handler(stub: StubInferenceServer) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        # httpx keeps connections alive; HTTP/1.1 plus an explicit
        # Content-Length avoids a per-request reconnect.
        protocol_version = "HTTP/1.1"

        def log_message(self, format: str, *args: Any) -> None:
            """Silence the default stderr access log."""

        def _read_body(self) -> dict[str, Any]:
            length = int(self.headers.get("Content-Length") or 0)
            if not length:
                return {}
            raw = self.rfile.read(length)
            try:
                parsed = json.loads(raw)
            except json.JSONDecodeError:
                return {"_unparsed": raw.decode("utf-8", "replace")}
            return parsed if isinstance(parsed, dict) else {"_unparsed": parsed}

        def _reply(self, status: int, payload: dict[str, Any]) -> None:
            body = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _record(self, method: str, body: dict[str, Any]) -> RecordedRequest:
            request = RecordedRequest(
                method=method,
                path=self.path,
                headers=dict(self.headers),
                body=body,
            )
            stub.requests.append(request)
            return request

        def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
            self._record("GET", {})
            if self.path == "/v1/models":
                self._reply(200, {"object": "list", "data": [{"id": stub.model}]})
                return
            self._reply(404, {"error": f"unknown path {self.path}"})

        def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
            request = self._record("POST", self._read_body())
            if self.path == "/v1/chat/completions":
                self._reply(200, stub.chat_completion(request))
                return
            if self.path == GATE_PATH:
                self._reply(200, stub.gate_decision(request))
                return
            self._reply(404, {"error": f"unknown path {self.path}"})

    return Handler


class StubInferenceServer:
    """A threaded HTTP server speaking the OpenAI chat-completions dialect.

    Serves `POST /v1/chat/completions` and `GET /v1/models` for provider-backed
    commands, plus `POST /gate` returning an app-shaped body for target-profile
    tests. Binds an ephemeral port so tests can run in parallel.
    """

    def __init__(
        self,
        responder: Responder | None = None,
        *,
        model: str = "stub-model",
    ) -> None:
        self.responder: Responder = responder or default_responder
        self.model = model
        self.requests: list[RecordedRequest] = []
        self._server = ThreadingHTTPServer(("127.0.0.1", 0), _build_handler(self))
        self._thread = threading.Thread(
            target=self._server.serve_forever,
            name="stub-inference-server",
            daemon=True,
        )

    @property
    def port(self) -> int:
        return int(self._server.server_address[1])

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    @property
    def openai_base_url(self) -> str:
        """Base URL in the form the vllm/OpenAI-compatible provider expects."""
        return f"{self.base_url}/v1"

    @property
    def gate_url(self) -> str:
        return f"{self.base_url}{GATE_PATH}"

    def chat_completions(self) -> list[RecordedRequest]:
        return [r for r in self.requests if r.path == "/v1/chat/completions"]

    def chat_completion(self, request: RecordedRequest) -> dict[str, Any]:
        text = self.responder(request)
        prompt_tokens = max(len(request.prompt.split()), 1)
        completion_tokens = max(len(text.split()), 1)
        return {
            "id": "chatcmpl-stub",
            "object": "chat.completion",
            "model": str(request.body.get("model") or self.model),
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": text},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        }

    def gate_decision(self, request: RecordedRequest) -> dict[str, Any]:
        """An app-shaped reply, as a target profile would parse it.

        The text lives under "verdict" deliberately: `_extract_text` also
        guesses common keys like "response" or "decision", so a conventional
        name would pass even if the profile's configured text_field were
        ignored entirely.
        """
        return {"verdict": self.responder(request), "elapsed_ms": 12.5}

    def start(self) -> StubInferenceServer:
        self._thread.start()
        return self

    def stop(self) -> None:
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=5)

    def __enter__(self) -> StubInferenceServer:
        return self.start()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.stop()
