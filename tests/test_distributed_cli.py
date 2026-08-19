import json

import pytest


def test_worker_help():
    from click.testing import CliRunner

    from atomics.commands.worker import worker

    runner = CliRunner()
    result = runner.invoke(worker, ["--help"])
    assert result.exit_code == 0
    assert "coordinator" in result.output


def test_distributed_help():
    from click.testing import CliRunner

    from atomics.commands.distributed import distributed

    runner = CliRunner()
    result = runner.invoke(distributed, ["--help"])
    assert result.exit_code == 0
    assert "run" in result.output
    assert "status" in result.output


def test_distributed_status_outputs_clean_json(monkeypatch):
    """atomics distributed status should print clean JSON without Rich markup."""
    import json
    from unittest.mock import MagicMock

    from click.testing import CliRunner

    from atomics.commands.distributed import distributed

    fake_response = MagicMock()
    fake_response.json.return_value = {
        "job_id": "abc123",
        "status": "completed",
        "mode": "split",
    }
    fake_response.raise_for_status.return_value = None

    def fake_get(url, *, headers=None):
        assert url == "http://coordinator:8000/api/v1/distributed/runs/abc123"
        assert headers == {"X-API-Key": "client-key"}
        return fake_response

    monkeypatch.setattr("atomics.commands.distributed.httpx.get", fake_get)

    runner = CliRunner()
    result = runner.invoke(
        distributed,
        [
            "status",
            "--coordinator",
            "http://coordinator:8000",
            "--api-key",
            "client-key",
            "abc123",
        ],
    )
    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert data["job_id"] == "abc123"
    assert data["status"] == "completed"


def _capture_run_payload(monkeypatch) -> dict:
    """Invoke `distributed run` against a stubbed coordinator and return the payload."""
    from unittest.mock import MagicMock

    captured: dict = {}
    fake_response = MagicMock()
    fake_response.json.return_value = {"job_id": "job-1"}
    fake_response.raise_for_status.return_value = None

    def fake_post(url, *, json=None, headers=None):
        captured.update(json or {})
        return fake_response

    monkeypatch.setattr("atomics.commands.distributed.httpx.post", fake_post)
    return captured


def test_distributed_run_sends_pinned_provider(monkeypatch):
    from click.testing import CliRunner

    from atomics.commands.distributed import distributed

    captured = _capture_run_payload(monkeypatch)
    result = CliRunner().invoke(
        distributed,
        [
            "run",
            "--api-key",
            "client-key",
            "--provider",
            "vllm",
            "--model",
            "qwen3:14b",
        ],
    )
    assert result.exit_code == 0, result.output
    assert captured["run_request"]["provider"] == "vllm"
    assert captured["run_request"]["model"] == "qwen3:14b"


def test_distributed_run_omits_provider_when_unset(monkeypatch):
    """Without -p the request must stay silent so workers keep their own provider."""
    from click.testing import CliRunner

    from atomics.commands.distributed import distributed

    captured = _capture_run_payload(monkeypatch)
    result = CliRunner().invoke(distributed, ["run", "--api-key", "client-key"])
    assert result.exit_code == 0, result.output
    assert "provider" not in captured["run_request"]
    assert "model" not in captured["run_request"]


def test_distributed_run_rejects_label_in_split_mode(monkeypatch):
    """Split assigns to the next free worker, so a selector still cannot be honored."""
    from click.testing import CliRunner

    from atomics.commands.distributed import distributed

    captured = _capture_run_payload(monkeypatch)
    result = CliRunner().invoke(
        distributed,
        [
            "run",
            "--api-key",
            "client-key",
            "--label",
            "gpu=1",
        ],
    )
    assert result.exit_code != 0
    assert "fleet" in result.output
    assert captured == {}, "no request should be submitted when --label is rejected"


def _stub_status_response(monkeypatch, payload: dict) -> None:
    from unittest.mock import MagicMock

    fake_response = MagicMock()
    fake_response.json.return_value = payload
    fake_response.raise_for_status.return_value = None
    monkeypatch.setattr(
        "atomics.commands.distributed.httpx.get",
        lambda url, *, headers=None: fake_response,
    )


_FLEET_STATUS = {
    "job_id": "job-1",
    "status": "partial",
    "mode": "fleet",
    "summary_json": json.dumps(
        {
            "workers": [
                {
                    # Realistic 12-hex id: short synthetic names hid the fact
                    # that Rich was truncating every column to unreadability.
                    "worker_id": "5b5fc1a2d3e4",
                    "labels": {"gpu": "4090"},
                    "completed": 4,
                    "failed": 0,
                    "input_tokens": 40,
                    "output_tokens": 80,
                    "mean_latency_ms": 120.5,
                    "p95_latency_ms": 200.0,
                    "mean_tokens_per_second": 55.0,
                    "estimated_cost_usd": 0.02,
                    "provider": "ollama",
                    "model": "qwen3:14b",
                },
                {
                    "worker_id": "683d3f5b7c8a",
                    "labels": {"gpu": "3060"},
                    "completed": 2,
                    "failed": 2,
                    "input_tokens": 20,
                    "output_tokens": 30,
                    "mean_latency_ms": 480.0,
                    "p95_latency_ms": 600.0,
                    "mean_tokens_per_second": 12.0,
                    "estimated_cost_usd": 0.01,
                    "provider": "ollama",
                    "model": "qwen3:14b",
                },
            ],
            "completed": 6,
            "failed": 2,
            "total_input_tokens": 60,
            "total_output_tokens": 110,
            "estimated_cost_usd": 0.03,
        }
    ),
}


def test_fleet_status_prints_a_row_per_host(monkeypatch):
    from click.testing import CliRunner

    from atomics.commands.distributed import distributed

    _stub_status_response(monkeypatch, _FLEET_STATUS)
    result = CliRunner().invoke(distributed, ["status", "job-1", "--api-key", "client-key"])

    assert result.exit_code == 0, result.output
    # Identifiers must survive intact at the default 80-column width. Truncated
    # to "5b5fc…" and "box=a…" the table cannot answer which host was faster,
    # which is the only reason fleet mode exists.
    assert "5b5fc1a2d3e4" in result.output
    assert "683d3f5b7c8a" in result.output
    assert "gpu=4090" in result.output
    assert "gpu=3060" in result.output
    # The comparison is the point: both hosts' latencies must be visible.
    assert "120.5" in result.output
    assert "480.0" in result.output
    # Rendered, not dumped: raw JSON already contains every value above, so the
    # test would pass against the old output without these two assertions.
    assert "mean_latency_ms" not in result.output
    with pytest.raises(json.JSONDecodeError):
        json.loads(result.output)


def test_a_host_that_ran_nothing_does_not_cost_the_table_a_column(monkeypatch):
    """The one run where the numbers matter most must not be the narrowest.

    A worker killed mid-run reports no model, which read as "the hosts disagree
    about the model" and brought the Model column back, squeezing the rest until
    a label rendered as `box=sur` / `vivor` across two lines.
    """
    from click.testing import CliRunner

    from atomics.commands.distributed import distributed

    payload = json.loads(_FLEET_STATUS["summary_json"])
    dead = payload["workers"][1]
    dead.update({"completed": 0, "failed": 4, "model": None, "provider": None})
    status = {**_FLEET_STATUS, "summary_json": json.dumps(payload)}

    _stub_status_response(monkeypatch, status)
    result = CliRunner().invoke(distributed, ["status", "job-1", "--api-key", "client-key"])

    assert result.exit_code == 0, result.output
    assert "Model" not in result.output
    assert "model: qwen3:14b" in result.output
    # Both identifiers still whole, which is what the extra room buys.
    assert "5b5fc1a2d3e4" in result.output
    assert "683d3f5b7c8a" in result.output
    assert "gpu=3060" in result.output


def test_split_status_still_prints_plain_json(monkeypatch):
    """Split jobs keep their existing machine-readable output."""
    from click.testing import CliRunner

    from atomics.commands.distributed import distributed

    _stub_status_response(monkeypatch, {"job_id": "job-1", "status": "completed", "mode": "split"})
    result = CliRunner().invoke(distributed, ["status", "job-1", "--api-key", "client-key"])

    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["mode"] == "split"


def test_status_json_out_writes_the_rollup(monkeypatch, tmp_path):
    from click.testing import CliRunner

    from atomics.commands.distributed import distributed

    _stub_status_response(monkeypatch, _FLEET_STATUS)
    out = tmp_path / "fleet.json"
    result = CliRunner().invoke(
        distributed,
        ["status", "job-1", "--api-key", "client-key", "--json-out", str(out)],
    )

    assert result.exit_code == 0, result.output
    written = json.loads(out.read_text())
    assert written["job_id"] == "job-1"
    assert len(written["summary"]["workers"]) == 2


def test_fleet_mode_sends_the_label_selector(monkeypatch):
    from click.testing import CliRunner

    from atomics.commands.distributed import distributed

    captured = _capture_run_payload(monkeypatch)
    result = CliRunner().invoke(
        distributed,
        [
            "run",
            "--api-key",
            "client-key",
            "--mode",
            "fleet",
            "--label",
            "gpu=4090",
            "--label",
            "site=lab",
        ],
    )
    assert result.exit_code == 0, result.output
    assert captured["mode"] == "fleet"
    assert captured["worker_selector"] == {"gpu": "4090", "site": "lab"}


def test_fleet_mode_without_labels_sends_no_selector(monkeypatch):
    """An absent selector means every online worker, so omit the key entirely."""
    from click.testing import CliRunner

    from atomics.commands.distributed import distributed

    captured = _capture_run_payload(monkeypatch)
    result = CliRunner().invoke(
        distributed,
        [
            "run",
            "--api-key",
            "client-key",
            "--mode",
            "fleet",
        ],
    )
    assert result.exit_code == 0, result.output
    assert captured["mode"] == "fleet"
    assert "worker_selector" not in captured


def test_fleet_mode_rejects_a_malformed_label(monkeypatch):
    from click.testing import CliRunner

    from atomics.commands.distributed import distributed

    captured = _capture_run_payload(monkeypatch)
    result = CliRunner().invoke(
        distributed,
        [
            "run",
            "--api-key",
            "client-key",
            "--mode",
            "fleet",
            "--label",
            "gpu",
        ],
    )
    assert result.exit_code != 0
    assert "key=value" in result.output
    assert captured == {}


def test_distributed_run_rejects_unknown_provider(monkeypatch):
    from click.testing import CliRunner

    from atomics.commands.distributed import distributed

    _capture_run_payload(monkeypatch)
    result = CliRunner().invoke(
        distributed,
        [
            "run",
            "--api-key",
            "client-key",
            "--provider",
            "not-a-provider",
        ],
    )
    assert result.exit_code != 0
    assert "not-a-provider" in result.output


def test_worker_cli_passes_provider_model_host(monkeypatch):
    """atomics worker should pass provider/model/host to WorkerClient."""
    import asyncio
    from unittest.mock import MagicMock

    from click.testing import CliRunner

    from atomics.commands.worker import worker

    fake_client = MagicMock()
    fake_client.shutdown = MagicMock()
    captured: dict = {}

    def fake_worker_client(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return fake_client

    monkeypatch.setattr("atomics.commands.worker.WorkerClient", fake_worker_client)
    monkeypatch.setattr(asyncio, "run", lambda coro: None)

    runner = CliRunner()
    result = runner.invoke(
        worker,
        [
            "--coordinator",
            "http://coordinator:8000",
            "--api-key",
            "worker-key",
            "--provider",
            "brain-gateway",
            "--host",
            "http://gpu-host:30080",
            "--model",
            "qwen3:4b",
            "--label",
            "box=239",
        ],
    )
    assert result.exit_code == 0, result.output
    assert captured, "WorkerClient was not instantiated"
    kwargs = captured["kwargs"]
    assert kwargs["provider_name"] == "brain-gateway"
    assert kwargs["host"] == "http://gpu-host:30080"
    assert kwargs["model"] == "qwen3:4b"


def test_full_mode_sends_run_request_and_selector(monkeypatch):
    """Full mode is accepted and can target a worker by label."""
    from click.testing import CliRunner

    from atomics.commands.distributed import distributed

    captured = _capture_run_payload(monkeypatch)
    result = CliRunner().invoke(
        distributed,
        [
            "run",
            "--api-key",
            "client-key",
            "--mode",
            "full",
            "--label",
            "box=239",
            "--provider",
            "brain-gateway",
            "--model",
            "qwen3:4b",
        ],
    )
    assert result.exit_code == 0, result.output
    assert captured["mode"] == "full"
    assert captured["worker_selector"] == {"box": "239"}
    assert captured["run_request"]["provider"] == "brain-gateway"
    assert captured["run_request"]["model"] == "qwen3:4b"


_FULL_STATUS = {
    "job_id": "job-2",
    "status": "completed",
    "mode": "full",
    "summary_json": json.dumps(
        {
            "summary": {
                "total_tasks": 5,
                "successful_tasks": 5,
                "failed_tasks": 0,
                "total_tokens": 1234,
                "total_cost_usd": 0.012,
                "avg_latency_ms": 145.5,
            }
        }
    ),
}


def test_full_status_prints_a_summary_table(monkeypatch):
    """Full mode status should render a compact summary rather than raw JSON."""
    from click.testing import CliRunner

    from atomics.commands.distributed import distributed

    _stub_status_response(monkeypatch, _FULL_STATUS)
    result = CliRunner().invoke(distributed, ["status", "job-2", "--api-key", "client-key"])
    assert result.exit_code == 0, result.output
    assert "Full run" in result.output
    assert "5" in result.output
    assert "1234" in result.output
    with pytest.raises(json.JSONDecodeError):
        json.loads(result.output)


def test_full_status_without_summary_prints_plain_json(monkeypatch):
    """A full-mode job with no result yet should fall back to plain JSON."""
    from click.testing import CliRunner

    from atomics.commands.distributed import distributed

    _stub_status_response(monkeypatch, {"job_id": "job-2", "status": "pending", "mode": "full"})
    result = CliRunner().invoke(distributed, ["status", "job-2", "--api-key", "client-key"])
    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["mode"] == "full"


def test_distributed_run_sends_runtime_override(monkeypatch):
    """The --runtime flag reaches the coordinator run request."""
    from click.testing import CliRunner

    from atomics.commands.distributed import distributed

    captured = _capture_run_payload(monkeypatch)
    result = CliRunner().invoke(
        distributed,
        [
            "run",
            "--api-key",
            "client-key",
            "--runtime",
            "node",
        ],
    )
    assert result.exit_code == 0, result.output
    assert captured["run_request"]["runtime"] == "node"
