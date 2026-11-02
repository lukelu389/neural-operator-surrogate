"""Shared deterministic AdamW training, validation selection, and resumable cache.

Inputs are already standardized using training-only moments. Checkpoints commit
at epoch boundaries, including Python, NumPy and torch RNG states. An interrupted
epoch is replayed from its previous committed boundary, never started as a new
trial. Supply the same initialized model, settings, metadata, and data to resume.
"""

import hashlib
import inspect
import json
import math
import os
import random
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from nop.eval.metrics import iv_metrics
from nop.reproducibility import seed_everything
from nop.train.losses import masked_mse


@dataclass(frozen=True)
class TrainingData:
    """CPU float32 parameters/standardized targets with an explicit quote mask.

    Targets have shape ``[surface, *grid]``; mean/std have shape ``[*grid]``.
    Masked targets may be NaN. Optional fixed weights have the targets' shape.
    This container neither reads a dataset nor estimates normalization moments.
    """

    parameters: torch.Tensor
    targets: torch.Tensor
    valid: torch.Tensor
    target_mean: torch.Tensor
    target_std: torch.Tensor
    weights: torch.Tensor | None = None

    def __post_init__(self) -> None:
        values = [self.parameters, self.targets, self.target_mean, self.target_std]
        if self.weights is not None:
            values.append(self.weights)
        if any(value.device.type != "cpu" or value.dtype != torch.float32 for value in values):
            raise ValueError("training arrays and moments must be CPU float32 tensors")
        if any(value.requires_grad for value in values):
            raise ValueError("training arrays and moments must not require gradients")
        if self.parameters.ndim != 2 or self.parameters.shape[0] == 0:
            raise ValueError("parameters must have nonempty shape [surface, parameter]")
        if self.targets.ndim < 2 or self.targets.shape[0] != self.parameters.shape[0]:
            raise ValueError("targets must have shape [surface, *grid]")
        if (
            self.valid.dtype != torch.bool
            or self.valid.device.type != "cpu"
            or self.valid.shape != self.targets.shape
        ):
            raise ValueError("valid must be a CPU Boolean tensor matching targets")
        if not self.valid.any():
            raise ValueError("a split must contain at least one valid reference quote")
        if self.target_mean.shape != self.targets.shape[1:] or (
            self.target_std.shape != self.target_mean.shape
        ):
            raise ValueError("target moments must match the native output grid")
        if not all(torch.isfinite(value).all() for value in values[:1] + values[2:4]):
            raise ValueError("parameters and target moments must be finite")
        if (self.target_std <= 0).any():
            raise ValueError("target standard deviations must be positive")
        reference = self.targets * self.target_std + self.target_mean
        if not torch.isfinite(reference[self.valid]).all() or (reference[self.valid] <= 0).any():
            raise ValueError("valid reference quotes must have finite positive IV")
        if self.weights is not None:
            if self.weights.shape != self.targets.shape:
                raise ValueError("weights must match targets")
            selected = self.weights[self.valid]
            if not torch.isfinite(selected).all() or (selected < 0).any():
                raise ValueError("valid quote weights must be finite and nonnegative")


@dataclass(frozen=True)
class TrainSettings:
    """One fixed training configuration; callers supply the declared trial draws.

    Defaults follow the frozen shared protocol. Overrides are useful for the
    separately labeled sanity fixtures, not additional architecture tuning.
    ``max_optimizer_steps`` is the optional sanity cap. Devices must reproduce
    the same deterministic execution path when resuming.
    """

    learning_rate: float
    weight_decay: float
    seed: int
    max_epochs: int = 100
    batch_size: int = 256
    warmup_epochs: int = 5
    gradient_clip_norm: float = 1.0
    minimum_learning_rate_fraction: float = 0.01
    patience_epochs: int = 10
    minimum_improvement_vol_points: float = 0.0
    betas: tuple[float, float] = (0.9, 0.999)
    epsilon: float = 1e-8
    device: str = "cpu"
    max_optimizer_steps: int | None = None

    def __post_init__(self) -> None:
        for name in ("max_epochs", "batch_size", "patience_epochs"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if (
            isinstance(self.seed, bool)
            or not isinstance(self.seed, int)
            or not 0 <= self.seed < 2**32
        ):
            raise ValueError("seed must be an integer in [0, 2**32 - 1]")
        if (
            isinstance(self.warmup_epochs, bool)
            or not isinstance(self.warmup_epochs, int)
            or not 0 <= self.warmup_epochs <= self.max_epochs
        ):
            raise ValueError("warmup_epochs must be an integer between zero and max_epochs")
        if self.max_optimizer_steps is not None and (
            isinstance(self.max_optimizer_steps, bool)
            or not isinstance(self.max_optimizer_steps, int)
            or self.max_optimizer_steps < 1
        ):
            raise ValueError("max_optimizer_steps must be a positive integer")
        for name in ("learning_rate", "gradient_clip_norm", "epsilon"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        for name in ("weight_decay", "minimum_improvement_vol_points"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if not 0 < self.minimum_learning_rate_fraction <= 1:
            raise ValueError("minimum_learning_rate_fraction must lie in (0, 1]")
        if len(self.betas) != 2 or not all(0 <= beta < 1 for beta in self.betas):
            raise ValueError("AdamW betas must lie in [0, 1)")


class InvalidPredictionError(ValueError):
    """A failed trial; it cannot be replaced by a fresh run in the same cache."""


def learning_rate_factor(update: int, total: int, warmup: int, minimum: float) -> float:
    """Multiplier for a zero-based optimizer update; skipped batches do not count.

    Warmup uses ``1/warmup, ..., 1``. Subsequent updates decay to ``minimum``
    at the final planned update. Without warmup the first update uses full LR.
    Empty batches do not advance this fixed planned-update schedule.
    """
    if total < 1 or not 0 <= warmup <= total or update < 0 or not 0 < minimum <= 1:
        raise ValueError("invalid learning-rate schedule arguments")
    if update < warmup:
        return (update + 1) / warmup
    if warmup == total:
        return 1.0
    denominator = total - warmup if warmup else max(total - 1, 1)
    progress = min((update - warmup + bool(warmup)) / denominator, 1.0)
    return minimum + (1.0 - minimum) * (1.0 + math.cos(math.pi * progress)) / 2.0


def _tensor_identity(value: torch.Tensor) -> str:
    array = value.detach().cpu().contiguous().numpy()
    digest = hashlib.sha256(str((str(array.dtype), array.shape)).encode())
    digest.update(array.tobytes())
    return digest.hexdigest()


def _file_hash(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _identity(
    model: nn.Module,
    training: TrainingData,
    validation: TrainingData,
    settings: TrainSettings,
    metadata: dict[str, Any],
) -> tuple[str, dict[str, Any]]:
    model_source = inspect.getsourcefile(type(model))
    identity: dict[str, Any] = {
        "settings": asdict(settings),
        "metadata": metadata,
        "model_class": f"{type(model).__module__}.{type(model).__qualname__}",
        "model_repr": repr(model),
        "model_source_hash": _file_hash(Path(model_source)) if model_source else None,
        "initial_model_state": {
            key: _tensor_identity(value) for key, value in model.state_dict().items()
        },
        "engine_source_hash": _file_hash(Path(__file__)),
        "loss_source_hash": _file_hash(Path(inspect.getfile(masked_mse))),
        "metric_source_hash": _file_hash(Path(inspect.getfile(iv_metrics))),
        "torch_version": str(torch.__version__),
        "numpy_version": np.__version__,
        "torch_threads": torch.get_num_threads(),
    }
    for name, data in (("training", training), ("validation", validation)):
        identity[name] = {
            key: _tensor_identity(value)
            for key in ("parameters", "targets", "valid", "target_mean", "target_std", "weights")
            if (value := getattr(data, key)) is not None
        }
    encoded = json.dumps(identity, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode()).hexdigest(), json.loads(encoded)


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def _rng_state() -> dict[str, Any]:
    numpy_state = np.random.get_state()
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": [
            numpy_state[0],
            torch.tensor(numpy_state[1].astype(np.int64)),
            numpy_state[2],
            numpy_state[3],
            numpy_state[4],
        ],
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }
    if torch.backends.mps.is_available():
        state["mps"] = torch.mps.get_rng_state()
    return state


def _restore_rng(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    saved = state["numpy"]
    np.random.set_state((saved[0], saved[1].numpy().astype(np.uint32), *saved[2:]))
    torch.set_rng_state(state["torch"])
    if state["cuda"]:
        torch.cuda.set_rng_state_all(state["cuda"])
    if "mps" in state:
        torch.mps.set_rng_state(state["mps"])


def _log_record(logger: Callable[[dict[str, Any]], None], record: dict[str, Any]) -> None:
    # External logging must not change the random sequence used by optimization.
    state = _rng_state()
    try:
        logger(record)
    finally:
        _restore_rng(state)


def _load_checkpoint(output: Path, reference: dict[str, str], config_hash: str) -> dict[str, Any]:
    name = reference["path"]
    if Path(name).name != name:
        raise ValueError("checkpoint manifest must name a local checkpoint")
    path = output / name
    if not path.is_file() or _file_hash(path) != reference["sha256"]:
        raise ValueError(f"checkpoint hash mismatch: {name}")
    payload: dict[str, Any] = torch.load(path, map_location="cpu", weights_only=True)
    if payload["config_hash"] != config_hash:
        raise ValueError("checkpoint configuration mismatch")
    return payload


def _commit_checkpoint(
    output: Path,
    payload: dict[str, Any],
    previous: dict[str, Any],
    is_best: bool,
) -> dict[str, Any]:
    # Unique generations make the manifest update the single atomic commit point.
    path = output / f"epoch-{payload['completed_epochs']:06d}.pt"
    temporary = path.with_suffix(".pt.tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)
    reference = {"path": path.name, "sha256": _file_hash(path)}
    manifest = {
        "config_hash": payload["config_hash"],
        "last": reference,
        "best": reference if is_best else previous.get("best"),
    }
    _atomic_json(output / "checkpoint.json", manifest)
    keep = {ref["path"] for ref in (manifest["last"], manifest["best"]) if ref is not None}
    for ref in (previous.get("last"), previous.get("best")):
        if ref is not None and ref["path"] not in keep:
            (output / ref["path"]).unlink(missing_ok=True)
    return manifest


def _check_prediction(prediction: torch.Tensor, valid: torch.Tensor, data: TrainingData) -> None:
    if prediction.shape != valid.shape or prediction.dtype != torch.float32:
        raise InvalidPredictionError("model output must be float32 and match the reference grid")
    iv = prediction * data.target_std.to(prediction.device) + data.target_mean.to(prediction.device)
    failures = (~torch.isfinite(iv[valid]) | (iv[valid] <= 0)).sum().item()
    if failures:
        raise InvalidPredictionError(
            f"{failures} nonfinite or nonpositive IV predictions on valid quotes"
        )


def _validate(model: nn.Module, data: TrainingData, settings: TrainSettings) -> dict[str, Any]:
    model.eval()
    predictions = []
    with torch.no_grad():
        for start in range(0, len(data.parameters), settings.batch_size):
            stop = start + settings.batch_size
            prediction = model(data.parameters[start:stop].to(settings.device))
            predictions.append(prediction.cpu())
    prediction = torch.cat(predictions)
    _check_prediction(prediction, data.valid, data)
    return iv_metrics(
        prediction * data.target_std + data.target_mean,
        data.targets * data.target_std + data.target_mean,
        data.valid,
    )


def fit(
    model: nn.Module,
    train_data: TrainingData,
    validation_data: TrainingData,
    settings: TrainSettings,
    output: Path,
    metadata: dict[str, Any],
    logger: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Fit one declared trial, restore its earliest best validation model, or resume.

    Cache identities include data bytes, initial weights, source and settings.
    Changed inputs or corrupt checkpoints fail explicitly instead of retraining.
    Logging runs after each checkpoint commit and is excluded from cache identity.
    Existing history is replayed to the logger on resume/cache access; callbacks
    should accept identical repeated epoch records without creating duplicates.
    Invalid predictions persist a failed-trial report and raise on every replay.
    """
    if train_data.targets.shape[1:] != validation_data.targets.shape[1:]:
        raise ValueError("training and validation must use the same output grid")
    if train_data.parameters.shape[1] != validation_data.parameters.shape[1]:
        raise ValueError("training and validation must use the same parameter features")
    if not torch.equal(train_data.target_mean, validation_data.target_mean) or not torch.equal(
        train_data.target_std, validation_data.target_std
    ):
        raise ValueError("validation must reuse training target moments")
    if any(
        value.is_floating_point() and value.dtype != torch.float32
        for value in model.state_dict().values()
    ):
        raise ValueError("model parameters and floating buffers must be float32")
    config_hash, identity = _identity(model, train_data, validation_data, settings, metadata)
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "checkpoint.json"
    report_path = output / "report.json"
    manifest: dict[str, Any] = {}
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest["config_hash"] != config_hash:
            raise ValueError("stale training cache: inputs or configuration changed")
        _load_checkpoint(output, manifest["last"], config_hash)
        if manifest["best"] is not None:
            _load_checkpoint(output, manifest["best"], config_hash)
    elif report_path.exists():
        raise ValueError("training report exists without its checkpoint manifest")
    if report_path.exists():
        report: dict[str, Any] = json.loads(report_path.read_text())
        best_hash = manifest["best"]["sha256"] if manifest["best"] is not None else None
        if (
            report["config_hash"] != config_hash
            or report["checkpoint_manifest"] != manifest
            or report["checkpoint_sha256"] != best_hash
        ):
            raise ValueError("cached report does not match its checkpoints")
        if logger is not None:
            for record in report["history"]:
                _log_record(logger, record)
        if report["status"] == "failed":
            raise InvalidPredictionError(report["failure"])
        model.load_state_dict(_load_checkpoint(output, manifest["best"], config_hash)["model"])
        model.to(settings.device).eval()
        return report

    seed_everything(settings.seed)
    model.to(settings.device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=settings.learning_rate,
        weight_decay=settings.weight_decay,
        betas=settings.betas,
        eps=settings.epsilon,
    )
    batches = math.ceil(len(train_data.parameters) / settings.batch_size)
    total = settings.max_epochs * batches
    if settings.max_optimizer_steps is not None:
        total = min(total, settings.max_optimizer_steps)
    warmup = min(settings.warmup_epochs * batches, total)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda update: learning_rate_factor(
            update, total, warmup, settings.minimum_learning_rate_fraction
        ),
    )
    progress: dict[str, Any] = {
        "completed_epochs": 0,
        "optimizer_steps": 0,
        "best_epoch": None,
        "best_validation_MAE_vol_points": None,
        "bad_epochs": 0,
        "history": [],
        "elapsed_seconds": 0.0,
    }

    def payload() -> dict[str, Any]:
        return {
            **progress,
            "config_hash": config_hash,
            "identity": identity,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "rng": _rng_state(),
        }

    if manifest:
        saved = _load_checkpoint(output, manifest["last"], config_hash)
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        progress = {key: saved[key] for key in progress}
        if logger is not None:
            for record in progress["history"]:
                _log_record(logger, record)
        _restore_rng(saved["rng"])
    else:
        manifest = _commit_checkpoint(output, payload(), {}, False)

    started = time.monotonic()
    previous_elapsed = progress["elapsed_seconds"]
    status = "completed"
    try:
        for epoch in range(progress["completed_epochs"], settings.max_epochs):
            if progress["bad_epochs"] >= settings.patience_epochs:
                status = "early_stopped"
                break
            if settings.max_optimizer_steps is not None and progress["optimizer_steps"] >= total:
                status = "step_limit"
                break
            model.train()
            generator = torch.Generator().manual_seed((settings.seed + epoch) % 2**32)
            permutation = torch.randperm(len(train_data.parameters), generator=generator)
            loss_sum, quote_count, skipped = 0.0, 0, 0
            learning_rate = float(optimizer.param_groups[0]["lr"])
            for start in range(0, len(permutation), settings.batch_size):
                indices = permutation[start : start + settings.batch_size]
                valid = train_data.valid[indices].to(settings.device)
                count = int(valid.sum().item())
                if count == 0:
                    skipped += 1
                    continue
                optimizer.zero_grad(set_to_none=True)
                prediction = model(train_data.parameters[indices].to(settings.device))
                _check_prediction(prediction, valid, train_data)
                weights = (
                    train_data.weights[indices].to(settings.device)
                    if train_data.weights is not None
                    else None
                )
                try:
                    loss = masked_mse(
                        prediction, train_data.targets[indices].to(settings.device), valid, weights
                    )
                except FloatingPointError as error:
                    raise InvalidPredictionError(str(error)) from error
                if not torch.isfinite(loss):
                    raise InvalidPredictionError("nonfinite training loss")
                loss.backward()  # type: ignore[no-untyped-call]  # Missing PyTorch annotation.
                norm = nn.utils.clip_grad_norm_(model.parameters(), settings.gradient_clip_norm)
                if not torch.isfinite(norm):
                    raise InvalidPredictionError("nonfinite training gradient")
                learning_rate = float(optimizer.param_groups[0]["lr"])
                optimizer.step()
                scheduler.step()
                progress["optimizer_steps"] += 1
                loss_sum += float(loss.detach().cpu()) * count
                quote_count += count
                if (
                    settings.max_optimizer_steps is not None
                    and progress["optimizer_steps"] >= total
                ):
                    break
            validation = _validate(model, validation_data, settings)
            mae = float(validation["mean_absolute_error"])
            best = progress["best_validation_MAE_vol_points"]
            improved = best is None or mae < best - settings.minimum_improvement_vol_points
            progress["bad_epochs"] = 0 if improved else progress["bad_epochs"] + 1
            if improved:
                progress["best_epoch"] = epoch + 1
                progress["best_validation_MAE_vol_points"] = mae
            record = {
                "epoch": epoch + 1,
                "train_loss": loss_sum / quote_count if quote_count else None,
                "train_valid_quote_count": quote_count,
                "skipped_empty_batches": skipped,
                "validation_metrics": validation,
                "learning_rate": learning_rate,
                "optimizer_steps": progress["optimizer_steps"],
            }
            progress["history"].append(record)
            progress["completed_epochs"] = epoch + 1
            progress["elapsed_seconds"] = previous_elapsed + time.monotonic() - started
            manifest = _commit_checkpoint(output, payload(), manifest, improved)
            if logger is not None:
                _log_record(logger, record)
        if progress["bad_epochs"] >= settings.patience_epochs:
            status = "early_stopped"
        elif settings.max_optimizer_steps is not None and progress["optimizer_steps"] >= total:
            status = "step_limit"
    except InvalidPredictionError as error:
        failure_report = {
            **progress,
            "config_hash": config_hash,
            "status": "failed",
            "failure": str(error),
            "checkpoint_manifest": manifest,
            "checkpoint_sha256": manifest["best"]["sha256"] if manifest["best"] else None,
        }
        _atomic_json(report_path, failure_report)
        raise

    report = {
        **progress,
        "status": status,
        "config_hash": config_hash,
        "checkpoint_manifest": manifest,
        "checkpoint_sha256": manifest["best"]["sha256"],
    }
    _atomic_json(report_path, report)
    best_state = _load_checkpoint(output, manifest["best"], config_hash)
    model.load_state_dict(best_state["model"])
    model.eval()
    return report
