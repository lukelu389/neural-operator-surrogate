"""Synchronous, local-only MLflow logging with resumable run identities.

Callers supply JSON-serializable provenance (configs, seeds, hashes, environment)
and explicitly log their checkpoints/reports as artifacts. One writer owns each
output directory; this logger does not coordinate concurrent training processes.
"""

import json
import math
import os
import time
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from hashlib import sha256
from pathlib import Path
from urllib.parse import unquote, urlparse


@contextmanager
def _local_mlflow_settings() -> Iterator[None]:
    # The frozen protocol requires FileStore, which current MLflow disables by
    # default. Disable telemetry before importing MLflow and during every call.
    settings = {"MLFLOW_ALLOW_FILE_STORE": "true", "MLFLOW_DISABLE_TELEMETRY": "true"}
    previous = {key: os.environ.get(key) for key in settings}
    os.environ.update(settings)
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _write_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


class LocalRunLogger:
    """Log to an explicit file store, reusing a run only for identical provenance."""

    def __init__(
        self,
        output_dir: Path,
        identity: Mapping[str, object],
        experiment_name: str,
        tracking_dir: Path,
    ) -> None:
        canonical = json.dumps(
            dict(identity), sort_keys=True, separators=(",", ":"), allow_nan=False
        )
        identity_hash = sha256(canonical.encode()).hexdigest()
        self.tracking_dir = tracking_dir.resolve()
        self.tracking_uri = self.tracking_dir.as_uri()
        self._history: dict[str, dict[int, float]] = {}
        output_dir = output_dir.resolve()
        metadata_path = output_dir / "mlflow_run.json"
        expected = {
            "version": 1,
            "tracking_uri": self.tracking_uri,
            "experiment_name": experiment_name,
            "identity_sha256": identity_hash,
        }
        metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else None
        if metadata is not None and (
            not isinstance(metadata, dict)
            or any(metadata.get(key) != value for key, value in expected.items())
            or not isinstance(metadata.get("run_id"), str)
        ):
            raise ValueError("MLflow output directory belongs to a different run identity or store")
        if not experiment_name.strip():
            raise ValueError("experiment_name must be nonempty")

        with _local_mlflow_settings():
            from mlflow.tracking import MlflowClient

            self._client = MlflowClient(
                tracking_uri=self.tracking_uri,
                registry_uri=self.tracking_uri,
                workspace_store_uri=self.tracking_uri,
            )
            experiment = self._client.get_experiment_by_name(experiment_name)
            if experiment is None:
                experiment_id = self._client.create_experiment(experiment_name)
            else:
                self._require_local_artifacts(experiment.artifact_location)
                if experiment.lifecycle_stage != "active":
                    raise ValueError("MLflow experiment is not active")
                experiment_id = experiment.experiment_id
            if metadata is None:
                run = self._client.create_run(
                    experiment_id,
                    tags={"nop.identity_sha256": identity_hash},
                )
            else:
                run = self._client.get_run(metadata["run_id"])
                if (
                    run.info.experiment_id != experiment_id
                    or run.data.tags.get("nop.identity_sha256") != identity_hash
                    or run.info.lifecycle_stage != "active"
                ):
                    raise ValueError("Saved MLflow run does not match its identity or experiment")
            self._require_local_artifacts(run.info.artifact_uri)
            self.run_id: str = run.info.run_id
            output_dir.mkdir(parents=True, exist_ok=True)
            _write_json(metadata_path, {**expected, "run_id": self.run_id})
            identity_path = output_dir / "mlflow_identity.json"
            _write_json(identity_path, json.loads(canonical))
            self._client.log_artifact(self.run_id, str(identity_path))
            self._client.log_param(self.run_id, "identity_sha256", identity_hash, synchronous=True)
            for key, value in identity.items():
                if value is None or isinstance(value, (str, bool, int, float)):
                    self._client.log_param(self.run_id, key, value, synchronous=True)
            self._client.update_run(self.run_id, status="RUNNING")

    def _require_local_artifacts(self, uri: str) -> None:
        parsed = urlparse(uri)
        path = Path(unquote(parsed.path)).resolve()
        if (
            parsed.scheme != "file"
            or parsed.netloc
            or parsed.query
            or parsed.fragment
            or not path.is_relative_to(self.tracking_dir)
        ):
            raise ValueError("MLflow artifact URI must stay inside the local tracking directory")

    def log_epoch(self, epoch: int, metrics: Mapping[str, float]) -> None:
        """Log finite metrics once per epoch; reject conflicting replayed values.

        Nonfinite values are omitted from MLflow's numeric histories and recorded
        in failure tags. The training loop must also fail its invalid trial.
        """
        if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0:
            raise ValueError("epoch must be a nonnegative integer")
        with _local_mlflow_settings():
            from mlflow.entities import Metric

            pending = []
            nonfinite = []
            timestamp = int(time.time() * 1000)
            for key, value in metrics.items():
                value = float(value)
                if not math.isfinite(value):
                    nonfinite.append(key)
                    continue
                if key not in self._history:
                    history = self._client.get_metric_history(self.run_id, key)
                    self._history[key] = {point.step: point.value for point in history}
                if epoch in self._history[key]:
                    if self._history[key][epoch] != value:
                        raise ValueError(
                            f"MLflow metric {key!r} differs at committed epoch {epoch}"
                        )
                else:
                    pending.append(Metric(key, value, timestamp, epoch))
            if nonfinite:
                self._client.set_tag(self.run_id, "nop.failure", "nonfinite_metric")
                self._client.set_tag(self.run_id, "nop.nonfinite_metric_epoch", str(epoch))
                self._client.set_tag(self.run_id, "nop.nonfinite_metrics", json.dumps(nonfinite))
            if pending:
                try:
                    self._client.log_batch(self.run_id, metrics=pending, synchronous=True)
                except BaseException:
                    # FileStore can commit part of a batch before interruption.
                    # Reload its actual history on retry, including in-process
                    # retries, instead of writing already committed points twice.
                    for metric in pending:
                        self._history.pop(metric.key, None)
                    raise
                for metric in pending:
                    self._history[metric.key][epoch] = metric.value

    def log_artifact(self, path: Path) -> None:
        """Copy a caller-specified artifact into this run's local artifact store."""
        if not path.is_file():
            raise ValueError(f"MLflow artifact must be an existing file: {path}")
        with _local_mlflow_settings():
            self._require_local_artifacts(self._client.get_run(self.run_id).info.artifact_uri)
            self._client.log_artifact(self.run_id, str(path.resolve()))

    def finish(self, status: str = "FINISHED") -> None:
        """Record completion or failure without selecting a global active run."""
        if status not in {"FINISHED", "FAILED"}:
            raise ValueError("status must be FINISHED or FAILED")
        with _local_mlflow_settings():
            self._client.set_terminated(self.run_id, status=status)
