"""Frozen-budget trial selection and verified development-data preparation."""

import fcntl
import hashlib
import json
import math
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
import torch

from nop.data.dataset import load_split, sha256_file
from nop.data.normalize import TrainNormalization
from nop.pricing.iv import IVStatus
from nop.protocol import read_json_object, verify_frozen_files
from nop.train.engine import TrainingData, TrainSettings


def paired_trial_draws(policy: dict[str, Any]) -> list[dict[str, Any]]:
    """Draw the declared LR then decay for each trial, identically for every model.

    The architecture never enters the random stream. A failed trial retains its
    identifier and consumes its existing draw; this function creates no extras.
    """
    tuning = policy["tuning"]
    if (
        tuning["sampler"] != "numpy_Generator_PCG64"
        or tuning["draw_order"] != "trial_then_learning_rate_then_weight_decay"
        or tuning["paired_identical_draws_across_models"] is not True
        or tuning["failed_trials"] != "count_against_budget_no_replacement"
    ):
        raise ValueError("unsupported shared tuning policy")
    count = tuning["trials_per_model"]
    if type(count) is not int or count <= 0:
        raise ValueError("trial budget must be a positive integer")
    bounds: dict[str, tuple[float, float]] = {}
    for name in ("learning_rate", "weight_decay"):
        setting = tuning[name]
        low, high = float(setting["low"]), float(setting["high"])
        if (
            setting["distribution"] != "log_uniform"
            or not math.isfinite(low)
            or not math.isfinite(high)
            or not 0 < low < high
        ):
            raise ValueError(f"invalid log-uniform bounds for {name}")
        bounds[name] = (math.log(low), math.log(high))
    generator = np.random.Generator(np.random.PCG64(tuning["random_seed"]))
    return [
        {
            "trial_id": f"trial_{index:02d}",
            **{
                name: float(np.exp(generator.uniform(*bounds[name])))
                for name in ("learning_rate", "weight_decay")
            },
        }
        for index in range(count)
    ]


def build_train_settings(
    policy: dict[str, Any],
    learning_rate: float,
    weight_decay: float,
    seed: int,
    device: str = "cpu",
) -> TrainSettings:
    """Translate the shared frozen training policy without extra run overrides."""
    training = policy["training"]
    early = training["early_stopping"]
    if (
        training["dtype"] != "float32"
        or training["mixed_precision"] is not False
        or training["optimizer"] != "AdamW"
        or training["schedule"] != "linear_warmup_then_cosine"
        or training["schedule_step"] != "each_optimizer_update"
        or early["metric"] != "validation_MAE_vol_points"
        or early["every_epochs"] != 1
        or early["ties"] != "retain_earlier_epoch"
        or early["restore"] != "best_validation_epoch"
    ):
        raise ValueError("unsupported shared training policy")
    for name, value in (("learning_rate", learning_rate), ("weight_decay", weight_decay)):
        bounds = policy["tuning"][name]
        if not bounds["low"] <= value <= bounds["high"]:
            raise ValueError(f"{name} lies outside the frozen search range")
    if seed not in [policy["tuning"]["training_seed"], *training["seeds"]]:
        raise ValueError("seed is outside the declared tuning/headline seeds")
    return TrainSettings(
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        seed=seed,
        max_epochs=training["max_epochs"],
        batch_size=training["batch_size"],
        warmup_epochs=training["warmup_epochs"],
        gradient_clip_norm=training["gradient_clip_global_norm"],
        minimum_learning_rate_fraction=training["minimum_learning_rate_fraction"],
        patience_epochs=early["patience_epochs"],
        minimum_improvement_vol_points=early["minimum_improvement_vol_points"],
        betas=tuple(training["betas"]),
        epsilon=training["epsilon"],
        device=device,
    )


def select_trial(records: list[dict[str, Any]], policy: dict[str, Any]) -> dict[str, Any]:
    """Select only after every declared trial completes or fails, without replacement.

    Records contain trial_id, status, learning_rate, weight_decay,
    parameter_count, and validation_metrics. The optional config_id defaults
    to trial_id for the final lexical tie break. Failed records may have null
    validation_metrics and never compete with successfully completed trials.
    """
    tuning = policy["tuning"]
    if tuning["selection_metric"] != "validation_MAE_vol_points" or tuning["tie_breaks"] != [
        "validation_p99_abs_error",
        "lower_parameter_count",
        "lexical_config_id",
    ]:
        raise ValueError("unsupported trial-selection policy")
    draws = {record["trial_id"]: record for record in paired_trial_draws(policy)}
    identifiers = [record["trial_id"] for record in records]
    if len(records) != len(draws) or set(identifiers) != set(draws):
        raise ValueError("selection requires exactly the complete declared trial budget")
    candidates: list[tuple[tuple[float, float, int, str], dict[str, Any]]] = []
    for record in records:
        expected = draws[record["trial_id"]]
        if any(record[name] != expected[name] for name in ("learning_rate", "weight_decay")):
            raise ValueError("trial hyperparameters differ from the declared paired draw")
        if record["status"] not in ("completed", "early_stopped", "failed"):
            raise ValueError("selection requires terminal trial records")
        count = record["parameter_count"]
        config_id = record.get("config_id", record["trial_id"])
        if type(count) is not int or count <= 0 or not isinstance(config_id, str) or not config_id:
            raise ValueError("trials require a positive parameter count and lexical config ID")
        if record["status"] == "failed":
            continue
        metrics = record["validation_metrics"]
        if not isinstance(metrics, dict):
            raise ValueError("successful trials require finite validation metrics")
        values: list[float] = []
        for name in ("mean_absolute_error", "p99_absolute_error"):
            value = metrics.get(name)
            if (
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not math.isfinite(value)
                or value < 0
            ):
                raise ValueError("successful trials require finite validation metrics")
            values.append(float(value))
        if metrics.get("prediction_failure_count", 0):
            raise ValueError("successful trials require finite validation metrics")
        candidates.append(((values[0], values[1], count, config_id), record))
    if not candidates:
        raise ValueError("every declared trial failed; no configuration can be selected")
    return min(candidates, key=lambda candidate: candidate[0])[1]


def load_development_data(
    root: Path, dataset_path: Path
) -> tuple[TrainingData, TrainingData, dict[str, Any]]:
    """Load only verified train/validation, reusing the saved training moments.

    Verify Gate 1, its complete dataset manifest, every used chunk, and the
    exact normalization/grid archives before exposing standardized float32
    arrays. Returned provenance and grid coordinates are JSON serializable.
    No statistics are fitted, and no holdout path is opened.
    """
    frozen = verify_frozen_files(root)
    if not dataset_path.is_absolute():
        dataset_path = root / dataset_path
    manifest_path = dataset_path / "manifest.json"
    manifest = read_json_object(manifest_path)
    evidence = frozen["evidence"]["dataset"]
    hashes = [
        chunk["sha256"]
        for split in ("train", "validation")
        for chunk in manifest["splits"][split]["chunks"]
    ] + [manifest["normalization_sha256"], manifest["grid_sha256"]]
    combined = hashlib.sha256(json.dumps(hashes).encode()).hexdigest()
    if (
        not manifest["complete"]
        or manifest["scope"] != "train_validation_only"
        or set(manifest["splits"]) != {"train", "validation"}
        or manifest["dataset_hash"] != evidence["dataset_hash"]
        or combined != manifest["dataset_hash"]
    ):
        raise ValueError("training requires the verified Gate 1 dataset identity")
    metadata = {"seed": manifest["seed"], "fit_split": "train", **manifest["metadata"]}
    archives: dict[str, dict[str, torch.Tensor]] = {}
    for name in ("normalization", "grid"):
        path = dataset_path / f"{name}.npz"
        if sha256_file(path) != manifest[f"{name}_sha256"]:
            raise ValueError(f"corrupt {name} archive")
        with np.load(path, allow_pickle=False) as arrays:
            if json.loads(str(arrays["metadata"].item())) != metadata:
                raise ValueError(f"{name} metadata disagrees with training provenance")
            archives[name] = {
                key: torch.from_numpy(arrays[key]) for key in arrays.files if key != "metadata"
            }
    moments = archives["normalization"]
    if set(moments) != {"parameter_mean", "parameter_std", "iv_mean", "iv_std", "iv_count"}:
        raise ValueError("normalization archive has unexpected fields")
    normalization = TrainNormalization(**moments)
    shape = normalization.iv_mean.shape
    if (
        len(shape) != 2
        or normalization.parameter_mean.shape != (5,)
        or normalization.parameter_std.shape != (5,)
        or normalization.iv_std.shape != shape
        or normalization.iv_count.shape != shape
        or any(not torch.isfinite(value).all() for value in moments.values())
        or (normalization.parameter_std <= 0).any()
        or (normalization.iv_std <= 0).any()
        or (normalization.iv_count <= 0).any()
    ):
        raise ValueError("invalid saved training normalization")
    grid = archives["grid"]
    if set(grid) != {"maturity", "z", "k", "strike"} or any(
        value.shape != shape or not torch.isfinite(value).all() for value in grid.values()
    ):
        raise ValueError("saved grid disagrees with normalization")
    prepared: list[TrainingData] = []
    for split in ("train", "validation"):
        loaded = load_split(dataset_path, split)
        valid = loaded["status"] == IVStatus.SUCCESS
        if not torch.equal(torch.isfinite(loaded["iv"]), valid):
            raise ValueError("IV failure mask disagrees with stored labels")
        if split == "train" and not torch.equal(valid.sum(dim=0), normalization.iv_count):
            raise ValueError("normalization counts disagree with training reference mask")
        prepared.append(
            TrainingData(
                parameters=normalization.transform_parameters(loaded["parameters"]).float(),
                targets=normalization.transform_iv(loaded["iv"]).float(),
                valid=valid,
                target_mean=normalization.iv_mean.float(),
                target_std=normalization.iv_std.float(),
            )
        )
    return (
        prepared[0],
        prepared[1],
        {
            "dataset_hash": manifest["dataset_hash"],
            "dataset_manifest_sha256": sha256_file(manifest_path),
            "normalization_sha256": manifest["normalization_sha256"],
            "grid_sha256": manifest["grid_sha256"],
            "normalization_metadata": metadata,
            "grid": {key: value.tolist() for key, value in grid.items()},
        },
    )


@contextmanager
def run_lock(path: Path) -> Iterator[None]:
    """Reject concurrent writers; the operating system releases locks on crashes.

    Keep the lockfile inode stable across invocations: unlinking after unlock
    could let concurrent processes lock different inodes for the same path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as stream:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f"another experiment process holds {path}") from error
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
