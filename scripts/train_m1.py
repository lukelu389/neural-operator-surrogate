"""Run M1's fixed paired tuning budget and three-seed validation comparison."""

import argparse
import json
import math
import platform
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from nop.data.dataset import current_commit, sha256_file, write_arrays, write_json
from nop.eval.validation_report import validation_summary, write_validation_figures
from nop.models.deeponet import DeepONet
from nop.protocol import verify_frozen_files
from nop.reproducibility import seed_everything
from nop.train.coordinates import CoordinateNormalization
from nop.train.engine import InvalidPredictionError, TrainingData, fit
from nop.train.experiments import (
    build_train_settings,
    load_development_data,
    paired_trial_draws,
    run_lock,
    select_trial,
)
from nop.train.logging import LocalRunLogger


def _normalization(
    model_config: dict[str, Any], train: TrainingData, grid: dict[str, Any]
) -> CoordinateNormalization:
    """Build coordinate moments solely from the verified training archive."""
    return CoordinateNormalization(
        z_axis=torch.tensor(grid["z"][0], dtype=torch.float32),
        T_axis=torch.tensor([row[0] for row in grid["maturity"]], dtype=torch.float32),
        mean=train.target_mean,
        std=train.target_std,
        k_scale=model_config["output"]["normalization_k_scale"],
    )


def _model(model_config: dict[str, Any], train: TrainingData, grid: dict[str, Any]) -> DeepONet:
    if not math.isclose(
        model_config["dot_product_scale"],
        model_config["rank"] ** -0.5,
        rel_tol=0.0,
        abs_tol=1e-15,
    ):
        raise ValueError("M1 dot-product scale must equal inverse square root of its rank")
    normalization = _normalization(model_config, train, grid)
    model = DeepONet(
        tuple(model_config["hidden_widths"]),
        model_config["rank"],
        (
            model_config["trunk_coordinate_scales"]["k"],
            model_config["trunk_coordinate_scales"]["maturity_years"],
        ),
        normalization,
        torch.tensor(grid["k"], dtype=torch.float32),
        torch.tensor(grid["maturity"], dtype=torch.float32),
    )
    count = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    if count != model_config["parameter_count"]:
        raise ValueError("M1 parameter count disagrees with its pretrial configuration")
    return model


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
    grid: dict[str, Any],
    draw: dict[str, Any],
    seed: int,
    run_id: str,
) -> tuple[dict[str, Any], torch.Tensor | None, list[dict[str, Any]]]:
    """Run/reuse one immutable trial or headline seed; preserve failed draws."""
    seed_everything(seed)
    model = _model(model_config, train, grid)
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
        "parameter_count": model_config["parameter_count"],
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
            prediction = torch.cat(
                [
                    model(
                        validation.parameters[
                            start : start + config["reporting"]["inference_batch_size"]
                        ].to(settings.device)
                    ).cpu()
                    for start in range(
                        0,
                        len(validation.parameters),
                        config["reporting"]["inference_batch_size"],
                    )
                ]
            )
            prediction = prediction * validation.target_std + validation.target_mean
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


def _identity(
    root: Path,
    config_path: Path,
    config: dict[str, Any],
    model_config: dict[str, Any],
    frozen: dict[str, Any],
    data: dict[str, Any],
    previous: dict[str, Any] | None,
) -> dict[str, Any]:
    sources = (
        "scripts/train_m1.py",
        "src/nop/models/deeponet.py",
        "src/nop/data/dataset.py",
        "src/nop/data/normalize.py",
        "src/nop/protocol.py",
        "src/nop/train/coordinates.py",
        "src/nop/train/engine.py",
        "src/nop/train/experiments.py",
        "src/nop/train/losses.py",
        "src/nop/train/logging.py",
        "src/nop/eval/metrics.py",
        "src/nop/eval/validation_report.py",
        "src/nop/reproducibility.py",
    )
    return {
        "scope": "synthetic_heston_training_and_validation_only",
        "model_id": "M1",
        "code_commit": previous["identity"]["code_commit"] if previous else current_commit(root),
        "config": config,
        "model_config": model_config,
        "config_hashes": {
            "run": sha256_file(config_path),
            "model": sha256_file(root / config["model_config"]),
            "m0_model": sha256_file(root / "config/m0.yaml"),
            "m0_selected": sha256_file(root / "config/m0_selected.yaml"),
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
    """Complete all twelve paired trials and three selected-config seeds."""
    frozen = verify_frozen_files(root)
    config = yaml.safe_load(config_path.read_bytes())
    if (
        config["training_policy"] != "config/training.yaml"
        or config["evaluation_protocol"] != "config/eval_protocol.yaml"
        or config["model_config"] != "config/m1.yaml"
    ):
        raise ValueError("M1 must use its fixed architecture and shared frozen policies")
    policy = yaml.safe_load((root / config["training_policy"]).read_bytes())
    model_config = yaml.safe_load((root / config["model_config"]).read_bytes())
    m0_config = yaml.safe_load((root / "config/m0.yaml").read_bytes())
    m0_selected = yaml.safe_load((root / "config/m0_selected.yaml").read_bytes())
    if model_config["model_id"] != "M1" or model_config["architecture"] != (
        "deeponet_branch_parameter_trunk_coordinate_dot_product"
    ):
        raise ValueError("M1 architecture configuration changed")
    m0_count = m0_config["trainable_parameters"]
    if m0_selected["model_config_sha256"] != sha256_file(root / "config/m0.yaml"):
        raise ValueError("M0 architecture differs from its selected Gate 2 configuration")
    if (
        abs(model_config["parameter_count"] / m0_count - 1.0)
        > policy["max_relative_parameter_difference_from_M0"]
    ):
        raise ValueError("M1 parameter count is outside the shared M0 comparison budget")
    torch.set_num_threads(config["torch_threads"])
    output = root / config["output"]
    with run_lock(output / ".run.lock"):
        ledger_path = output / "experiment.json"
        previous = json.loads(ledger_path.read_text()) if ledger_path.exists() else None
        train, validation, data = load_development_data(root, root / config["dataset"])
        identity = _identity(root, config_path, config, model_config, frozen, data, previous)
        if previous is not None and previous["identity"] != identity:
            raise ValueError("M1 experiment provenance changed; preserve runs and investigate")
        grid = data["grid"]
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
                grid,
                draw,
                policy["tuning"]["training_seed"],
                draw["trial_id"],
            )
            ledger["trials"].append(record)
            histories[draw["trial_id"]] = history
            write_json(ledger_path, ledger)
            print(f"{draw['trial_id']}: {record['status']}", flush=True)

        if all(record["status"] == "failed" for record in ledger["trials"]):
            ledger.update(status="stopped", stop_reason="all_declared_trials_failed")
            write_json(ledger_path, ledger)
            write_json(root / config["report"], ledger)
            return ledger
        selected = select_trial(ledger["trials"], policy)
        ledger["selected_trial"] = selected
        ledger["status"] = "tuned"
        write_json(ledger_path, ledger)
        if tune_only:
            return ledger

        selected_config = {
            "model_id": "M1",
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
            raise ValueError("selected M1 config differs; preserve it and do not retune")
        if not selected_path.exists():
            selected_path.write_bytes(selected_bytes)

        predictions: dict[int, torch.Tensor] = {}
        for seed in policy["training"]["seeds"]:
            draw = {key: selected[key] for key in ("trial_id", "learning_rate", "weight_decay")}
            name = f"seed_{seed}"
            record, prediction, history = run_one(
                root,
                config,
                model_config,
                policy,
                identity,
                train,
                validation,
                grid,
                draw,
                seed,
                name,
            )
            ledger["seeds"].append(record)
            histories[name] = history
            if prediction is not None:
                predictions[seed] = prediction
            write_json(ledger_path, ledger)
            print(f"{name}: {record['status']}", flush=True)

        accuracy = validation_summary(
            predictions,
            validation.targets * validation.target_std + validation.target_mean,
            validation.valid,
            tuple(policy["training"]["seeds"]),
        )
        ledger.update(
            status="complete" if accuracy["complete"] else "incomplete",
            accuracy=accuracy,
            report_scope="validation_model_selection_data_not_independent_test",
        )
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
        write_json(root / config["report"], ledger)
        return ledger


def main(argv: list[str] | None = None) -> int:
    """Run M1 or stop without holdout access if every declared trial failed."""
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=root / "config/m1_run.yaml")
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
    return 0 if result["status"] == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
