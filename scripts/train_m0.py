"""Run the fixed M0 tuning budget and three-seed validation gate, with resumable logs."""

import argparse
import json
import platform
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from nop.data.dataset import current_commit, sha256_file, write_arrays, write_json
from nop.eval.validation_report import validation_summary, write_validation_figures
from nop.models.mlp import MLP
from nop.protocol import verify_frozen_files
from nop.reproducibility import seed_everything
from nop.train.engine import InvalidPredictionError, TrainingData, fit
from nop.train.experiments import (
    build_train_settings,
    load_development_data,
    paired_trial_draws,
    run_lock,
    select_trial,
)
from nop.train.logging import LocalRunLogger


def _log_epoch(logger: LocalRunLogger, row: dict[str, Any]) -> None:
    values = {
        name: float(row[name])
        for name in (
            "train_loss",
            "train_valid_quote_count",
            "learning_rate",
            "optimizer_steps",
            "skipped_empty_batches",
        )
        if row[name] is not None
    }
    values.update(
        {
            f"validation_{key}": float(value)
            for key, value in row["validation_metrics"].items()
            if value is not None
        }
    )
    logger.log_epoch(row["epoch"], values)


def run_one(
    root: Path,
    config: dict[str, Any],
    model_config: dict[str, Any],
    policy: dict[str, Any],
    identity: dict[str, Any],
    train: TrainingData,
    validation: TrainingData,
    draw: dict[str, Any],
    seed: int,
    run_id: str,
) -> tuple[dict[str, Any], torch.Tensor | None, list[dict[str, Any]]]:
    """Run/reuse exactly one named trial or seed, retaining terminal numerical failures."""
    seed_everything(seed)
    model = MLP(tuple(model_config["hidden_widths"]), train.target_mean, train.target_std)
    count = sum(parameter.numel() for parameter in model.parameters())
    if count != model_config["trainable_parameters"]:
        raise ValueError("M0 parameter count disagrees with its pretrial configuration")
    settings = build_train_settings(
        policy, draw["learning_rate"], draw["weight_decay"], seed, device=config["device"]
    )
    output = root / config["output"] / run_id
    metadata = {**identity, "run_id": run_id, "seed": seed, "hyperparameters": draw}
    logger = LocalRunLogger(
        output, metadata, config["experiment_name"], root / config["tracking_directory"]
    )
    record = {
        **draw,
        "config_id": draw["trial_id"],
        "run_id": run_id,
        "seed": seed,
        "parameter_count": count,
        "mlflow_run_id": logger.run_id,
    }
    try:
        trained = fit(
            model,
            train,
            validation,
            settings,
            output,
            metadata,
            logger=lambda row: _log_epoch(logger, row),
        )
    except InvalidPredictionError as error:
        failed = json.loads((output / "report.json").read_text())
        record.update(
            status="failed",
            validation_metrics=None,
            failure=str(error),
            report_sha256=sha256_file(output / "report.json"),
        )
        logger.log_artifact(output / "report.json")
        logger.finish("FAILED")
        return record, None, failed["history"]
    except BaseException:
        logger.finish("FAILED")
        raise
    row = trained["history"][trained["best_epoch"] - 1]
    record.update(
        status=trained["status"],
        validation_metrics=row["validation_metrics"],
        best_epoch=trained["best_epoch"],
        completed_epochs=trained["completed_epochs"],
        optimizer_steps=trained["optimizer_steps"],
        elapsed_seconds=trained["elapsed_seconds"],
        checkpoint=trained["checkpoint_manifest"]["best"],
        report_sha256=sha256_file(output / "report.json"),
    )
    logger.log_artifact(output / trained["checkpoint_manifest"]["best"]["path"])
    logger.log_artifact(output / "report.json")
    logger.finish()
    prediction = None
    if run_id.startswith("seed_"):
        with torch.no_grad():
            pieces = [
                model(
                    validation.parameters[
                        start : start + config["reporting"]["inference_batch_size"]
                    ].to(settings.device)
                ).cpu()
                for start in range(
                    0, len(validation.parameters), config["reporting"]["inference_batch_size"]
                )
            ]
            prediction = torch.cat(pieces) * validation.target_std + validation.target_mean
        write_arrays(
            output / "validation_predictions.npz",
            {
                "iv": prediction.numpy(),
                "metadata": np.array(
                    json.dumps(
                        {
                            "seed": seed,
                            "dataset_hash": identity["dataset_hash"],
                            "checkpoint": record["checkpoint"],
                        },
                        sort_keys=True,
                    )
                ),
            },
        )
        record["prediction_sha256"] = sha256_file(output / "validation_predictions.npz")
    return record, prediction, trained["history"]


def tuning_gate(
    records: list[dict[str, Any]], policy: dict[str, Any], threshold: float
) -> dict[str, Any]:
    """Apply the frozen M0 stop threshold only after the complete tuning budget."""
    if records and all(record["status"] == "failed" for record in records):
        # Validate every declared draw; only the all-failed selection error is expected.
        try:
            select_trial(records, policy)
        except ValueError as error:
            if str(error) != "every declared trial failed; no configuration can be selected":
                raise
        return {"passed": False, "reason": "all_declared_trials_failed", "selected": None}
    selected = select_trial(records, policy)
    passed = selected["validation_metrics"]["mean_absolute_error"] <= threshold
    return {
        "passed": passed,
        "reason": None if passed else "selected_tuning_MAE_exceeds_stop_threshold",
        "selected": selected,
    }


def _identity(
    root: Path,
    config_path: Path,
    config: dict[str, Any],
    frozen: dict[str, Any],
    data: dict[str, Any],
    previous: dict[str, Any] | None,
) -> dict[str, Any]:
    sources = [
        "scripts/train_m0.py",
        "src/nop/models/mlp.py",
        "src/nop/train/engine.py",
        "src/nop/train/experiments.py",
        "src/nop/train/losses.py",
        "src/nop/train/logging.py",
        "src/nop/eval/metrics.py",
        "src/nop/eval/validation_report.py",
        "src/nop/reproducibility.py",
    ]
    return {
        "scope": "synthetic_heston_training_and_validation_only",
        "model_id": "M0",
        "code_commit": previous["identity"]["code_commit"] if previous else current_commit(root),
        "config": config,
        "config_hashes": {
            "run": sha256_file(config_path),
            "model": sha256_file(root / config["model_config"]),
        },
        "frozen_files": frozen["files"],
        "dataset_hash": data["dataset_hash"],
        "normalization_sha256": data["normalization_sha256"],
        "source_hashes": {name: sha256_file(root / name) for name in sources},
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "torch_threads": torch.get_num_threads(),
            "device": config["device"],
            "dtype": "float32",
            "packages": {
                name: version(name) for name in ("numpy", "torch", "PyYAML", "mlflow-skinny")
            },
        },
    }


def run(root: Path, config_path: Path, *, tune_only: bool = False) -> dict[str, Any]:
    """Finish declared runs sequentially under one lock; never open research holdouts."""
    frozen = verify_frozen_files(root)
    config = yaml.safe_load(config_path.read_bytes())
    if (
        config["training_policy"] != "config/training.yaml"
        or config["evaluation_protocol"] != "config/eval_protocol.yaml"
        or config["model_config"] != "config/m0.yaml"
    ):
        raise ValueError(
            "M0 must use the verified training/evaluation policies and fixed model config"
        )
    policy = yaml.safe_load((root / config["training_policy"]).read_bytes())
    protocol = yaml.safe_load((root / config["evaluation_protocol"]).read_bytes())
    model_config = yaml.safe_load((root / config["model_config"]).read_bytes())
    torch.set_num_threads(config["torch_threads"])
    output = root / config["output"]
    with run_lock(output / ".run.lock"):
        ledger_path = output / "experiment.json"
        previous = json.loads(ledger_path.read_text()) if ledger_path.exists() else None
        train, validation, data = load_development_data(root, root / config["dataset"])
        identity = _identity(root, config_path, config, frozen, data, previous)
        if previous is not None and previous["identity"] != identity:
            raise ValueError(
                "M0 experiment provenance changed; retain existing runs and investigate"
            )
        ledger: dict[str, Any] = {
            "identity": identity,
            "draws": paired_trial_draws(policy),
            "trials": [],
            "seeds": [],
            "status": "tuning",
        }
        write_json(ledger_path, ledger)
        histories: dict[str, list[dict[str, Any]]] = {}
        for draw in ledger["draws"]:
            record, _, history = run_one(
                root,
                config,
                model_config,
                policy,
                identity,
                train,
                validation,
                draw,
                policy["tuning"]["training_seed"],
                draw["trial_id"],
            )
            ledger["trials"].append(record)
            histories[draw["trial_id"]] = history
            write_json(ledger_path, ledger)
            print(
                f"{draw['trial_id']}: {record['status']}; "
                f"validation={record['validation_metrics']}",
                flush=True,
            )
        threshold = protocol["stop_rules"]["M0_validation_MAE_vol_points_max"]
        gate = tuning_gate(ledger["trials"], policy, threshold)
        ledger["tuning_gate"] = gate
        if not gate["passed"]:
            ledger["status"] = "stopped"
            write_json(ledger_path, ledger)
            stop_report = root / config["report"]
            stop_report.parent.mkdir(parents=True, exist_ok=True)
            write_json(stop_report, ledger)
            return ledger
        ledger["status"] = "tuned"
        write_json(ledger_path, ledger)
        if tune_only:
            return ledger
        selected = gate["selected"]
        selected_config = {
            "model_id": "M0",
            "model_config": config["model_config"],
            "model_config_sha256": identity["config_hashes"]["model"],
            "training_policy": config["training_policy"],
            "selected_trial_id": selected["trial_id"],
            "learning_rate": selected["learning_rate"],
            "weight_decay": selected["weight_decay"],
            "seeds": policy["training"]["seeds"],
            "dataset_hash": data["dataset_hash"],
            "selection": "complete_twelve_trial_validation_only_budget_no_retuning",
        }
        selected_path = root / config["selected_config"]
        selected_bytes = yaml.safe_dump(selected_config, sort_keys=True).encode()
        if selected_path.exists() and selected_path.read_bytes() != selected_bytes:
            raise ValueError("selected M0 config already differs; do not overwrite or retune")
        if not selected_path.exists():
            selected_path.write_bytes(selected_bytes)
        ledger["status"] = "headline_seeds"
        write_json(ledger_path, ledger)
        predictions: dict[int, torch.Tensor] = {}
        for seed in policy["training"]["seeds"]:
            draw = {key: selected[key] for key in ("trial_id", "learning_rate", "weight_decay")}
            name = f"seed_{seed}"
            record, prediction, history = run_one(
                root, config, model_config, policy, identity, train, validation, draw, seed, name
            )
            ledger["seeds"].append(record)
            histories[name] = history
            if prediction is not None:
                predictions[seed] = prediction
            write_json(ledger_path, ledger)
            print(
                f"{name}: {record['status']}; validation={record['validation_metrics']}", flush=True
            )
        accuracy = validation_summary(
            predictions,
            validation.targets * validation.target_std + validation.target_mean,
            validation.valid,
            tuple(policy["training"]["seeds"]),
        )
        mean_mae = accuracy["aggregate"]["overall"]["mean_absolute_error"]["mean"]
        passed = accuracy["complete"] and mean_mae is not None and mean_mae <= threshold
        ledger.update(
            status="complete" if passed else "stopped",
            passed=passed,
            stop_threshold_vol_points=threshold,
            accuracy=accuracy,
            report_scope="validation_model_selection_data_not_independent_test",
        )
        grid = data["grid"]
        figures = write_validation_figures(
            accuracy,
            histories,
            [row[0] for row in grid["maturity"]],
            grid["z"][0],
            root / config["figures"],
            config["reporting"]["figure_dpi"],
        )
        ledger["figures"] = {str(path.relative_to(root)): sha256_file(path) for path in figures}
        write_json(ledger_path, ledger)
        report_path = root / config["report"]
        report_path.parent.mkdir(parents=True, exist_ok=True)
        write_json(report_path, ledger)
        return ledger


def main(argv: list[str] | None = None) -> int:
    """Return nonzero on the preregistered stop rule, with preserved evidence."""
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=root / "config/m0_run.yaml")
    parser.add_argument("--tune-only", action="store_true")
    args = parser.parse_args(argv)
    result = run(root, args.config, tune_only=args.tune_only)
    print(
        json.dumps(
            {
                "status": result["status"],
                "trials": len(result["trials"]),
                "seeds": len(result["seeds"]),
            },
            indent=2,
        )
    )
    return 2 if result["status"] == "stopped" else 0


if __name__ == "__main__":
    raise SystemExit(main())
