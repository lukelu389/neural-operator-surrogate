"""Exercise the real MLflow file backend without a tracking server."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from nop.train.logging import LocalRunLogger


@pytest.fixture
def local_client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("MLFLOW_ALLOW_FILE_STORE", "true")
    monkeypatch.setenv("MLFLOW_DISABLE_TELEMETRY", "true")
    from mlflow.tracking import MlflowClient

    return MlflowClient(tracking_uri=(tmp_path / "mlruns").as_uri())


def _logger(tmp_path: Path, **identity: object) -> LocalRunLogger:
    return LocalRunLogger(
        tmp_path / "output",
        {"seed": 101, "config_hash": "example-hash", **identity},
        "sanity",
        tmp_path / "mlruns",
    )


def test_logs_identity_metrics_and_artifacts_locally(tmp_path: Path, local_client) -> None:
    logger = _logger(tmp_path, config={"optimizer": "AdamW", "batch_size": 100})
    logger.log_epoch(0, {"train_loss": 0.02, "validation_MAE_vol_points": 0.1})
    artifact = tmp_path / "checkpoint.json"
    artifact.write_text('{"checkpoint_sha256": "abc"}\n')
    logger.log_artifact(artifact)
    logger.finish()

    run = local_client.get_run(logger.run_id)
    assert run.info.status == "FINISHED"
    assert run.data.params["seed"] == "101"
    assert run.data.params["config_hash"] == "example-hash"
    assert run.data.metrics == {"train_loss": 0.02, "validation_MAE_vol_points": 0.1}
    artifacts = {entry.path for entry in local_client.list_artifacts(logger.run_id)}
    assert artifacts == {"checkpoint.json", "mlflow_identity.json"}
    path = Path(local_client.download_artifacts(logger.run_id, "mlflow_identity.json"))
    assert json.loads(path.read_text())["config"] == {"optimizer": "AdamW", "batch_size": 100}
    assert run.info.artifact_uri.startswith(logger.tracking_uri + "/")


def test_reopening_reuses_run_and_deduplicates_metric_steps(tmp_path: Path, local_client) -> None:
    first = _logger(tmp_path)
    first.log_epoch(0, {"loss": 0.1})
    first.finish()
    resumed = _logger(tmp_path)
    assert resumed.run_id == first.run_id
    resumed.log_epoch(0, {"loss": 0.1})
    resumed.log_epoch(1, {"loss": 0.01})
    resumed.log_epoch(1, {"loss": 0.01})
    resumed.finish()
    assert [
        (point.step, point.value) for point in local_client.get_metric_history(first.run_id, "loss")
    ] == [
        (0, 0.1),
        (1, 0.01),
    ]


def test_conflicting_replayed_metric_fails_before_logging(tmp_path: Path, local_client) -> None:
    logger = _logger(tmp_path)
    logger.log_epoch(1, {"loss": 0.1})
    resumed = _logger(tmp_path)
    with pytest.raises(ValueError, match="committed epoch 1"):
        resumed.log_epoch(1, {"new_metric": 2.0, "loss": 0.2})
    assert local_client.get_metric_history(logger.run_id, "new_metric") == []
    assert [point.value for point in local_client.get_metric_history(logger.run_id, "loss")] == [
        0.1
    ]


def test_retry_recovers_a_partially_written_metric_batch(
    tmp_path: Path, local_client, monkeypatch: pytest.MonkeyPatch
) -> None:
    logger = _logger(tmp_path)
    log_batch = logger._client.log_batch

    def interrupted_batch(run_id, metrics, synchronous):
        log_batch(run_id, metrics=metrics[:1], synchronous=synchronous)
        raise OSError("interrupted after first metric")

    monkeypatch.setattr(logger._client, "log_batch", interrupted_batch)
    with pytest.raises(OSError, match="after first metric"):
        logger.log_epoch(0, {"loss": 0.1, "learning_rate": 0.001})
    monkeypatch.setattr(logger._client, "log_batch", log_batch)
    logger.log_epoch(0, {"loss": 0.1, "learning_rate": 0.001})
    assert [point.value for point in local_client.get_metric_history(logger.run_id, "loss")] == [
        0.1
    ]
    assert [
        point.value for point in local_client.get_metric_history(logger.run_id, "learning_rate")
    ] == [0.001]


@pytest.mark.parametrize("changed", ["identity", "store", "experiment"])
def test_reopening_rejects_changed_run_identity(tmp_path: Path, local_client, changed: str) -> None:
    _logger(tmp_path)
    with pytest.raises(ValueError, match="different run identity or store"):
        LocalRunLogger(
            tmp_path / "output",
            {"seed": 102 if changed == "identity" else 101, "config_hash": "example-hash"},
            "changed" if changed == "experiment" else "sanity",
            tmp_path / ("changed-store" if changed == "store" else "mlruns"),
        )
    assert not (tmp_path / "changed-store").exists()


def test_nonfinite_metrics_are_tagged_and_never_stored_as_numbers(
    tmp_path: Path, local_client
) -> None:
    logger = _logger(tmp_path)
    logger.log_epoch(3, {"loss": float("nan"), "tail": float("inf"), "learning_rate": 0.001})
    logger.finish("FAILED")
    run = local_client.get_run(logger.run_id)
    assert run.info.status == "FAILED"
    assert run.data.metrics == {"learning_rate": 0.001}
    assert run.data.tags["nop.failure"] == "nonfinite_metric"
    assert run.data.tags["nop.nonfinite_metric_epoch"] == "3"
    assert json.loads(run.data.tags["nop.nonfinite_metrics"]) == ["loss", "tail"]


def test_existing_remote_artifact_experiment_is_rejected(tmp_path: Path, local_client) -> None:
    local_client.create_experiment("sanity", artifact_location="https://example.invalid/artifacts")
    with pytest.raises(ValueError, match="inside the local tracking directory"):
        _logger(tmp_path)
    assert not (tmp_path / "output").exists()


def test_saved_run_id_cannot_be_redirected_to_another_run(tmp_path: Path, local_client) -> None:
    logger = _logger(tmp_path)
    experiment_id = local_client.get_run(logger.run_id).info.experiment_id
    other = local_client.create_run(experiment_id)
    metadata_path = tmp_path / "output" / "mlflow_run.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["run_id"] = other.info.run_id
    metadata_path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="does not match"):
        _logger(tmp_path)


def test_scoped_settings_restore_original_environment(
    tmp_path: Path, local_client, monkeypatch
) -> None:
    monkeypatch.setenv("MLFLOW_ALLOW_FILE_STORE", "false")
    monkeypatch.setenv("MLFLOW_DISABLE_TELEMETRY", "false")
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "https://example.invalid/tracking")
    logger = _logger(tmp_path)
    logger.log_epoch(0, {"loss": 1.0})
    logger.finish()
    assert os.environ["MLFLOW_ALLOW_FILE_STORE"] == "false"
    assert os.environ["MLFLOW_DISABLE_TELEMETRY"] == "false"
    assert os.environ["MLFLOW_TRACKING_URI"] == "https://example.invalid/tracking"


def test_fresh_import_and_logging_never_attempt_network(tmp_path: Path) -> None:
    code = """
import os
from pathlib import Path
import socket
import sys

attempts = []
def reject_network(*args, **kwargs):
    attempts.append(repr(args))
    raise AssertionError('local logger attempted a network connection')
socket.socket.connect = reject_network
socket.socket.connect_ex = reject_network
socket.create_connection = reject_network
os.environ['MLFLOW_TRACKING_URI'] = 'https://example.invalid/tracking'
os.environ['MLFLOW_REGISTRY_URI'] = 'https://example.invalid/registry'
os.environ['MLFLOW_DISABLE_TELEMETRY'] = 'false'
os.environ['MLFLOW_ALLOW_FILE_STORE'] = 'false'
from nop.train.logging import LocalRunLogger
path = Path(sys.argv[1])
logger = LocalRunLogger(path / 'output', {'seed': 101}, 'offline', path / 'mlruns')
logger.log_epoch(0, {'loss': 0.1})
logger.log_artifact(path / 'output' / 'mlflow_run.json')
logger.finish()
assert not attempts, attempts
"""
    result = subprocess.run(
        [sys.executable, "-c", code, str(tmp_path)],
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("epoch", [-1, True, 0.5])
def test_invalid_epoch_is_rejected(tmp_path: Path, local_client, epoch: object) -> None:
    logger = _logger(tmp_path)
    with pytest.raises(ValueError, match="nonnegative integer"):
        logger.log_epoch(epoch, {"loss": 1.0})
