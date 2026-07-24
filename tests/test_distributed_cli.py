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
    result = runner.invoke(distributed, [
        "status",
        "--coordinator", "http://coordinator:8000",
        "--api-key", "client-key",
        "abc123",
    ])
    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert data["job_id"] == "abc123"
    assert data["status"] == "completed"


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
    result = runner.invoke(worker, [
        "--coordinator", "http://coordinator:8000",
        "--api-key", "worker-key",
        "--provider", "brain-gateway",
        "--host", "http://nuc:30080",
        "--model", "qwen3:4b",
        "--label", "box=239",
    ])
    assert result.exit_code == 0, result.output
    assert captured, "WorkerClient was not instantiated"
    kwargs = captured["kwargs"]
    assert kwargs["provider_name"] == "brain-gateway"
    assert kwargs["host"] == "http://nuc:30080"
    assert kwargs["model"] == "qwen3:4b"
