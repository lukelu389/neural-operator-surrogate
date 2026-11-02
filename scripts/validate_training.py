"""Run the frozen 100-training-sample reconstruction check; never access holdouts."""

import argparse
import hashlib
import json
import platform
from importlib.metadata import version
from pathlib import Path
from typing import Any

import torch
import yaml
from torch import nn

from nop.data.dataset import current_commit, load_split, sha256_file, write_json
from nop.data.normalize import fit_train_normalization
from nop.protocol import verify_frozen_files
from nop.reproducibility import seed_everything
from nop.train.engine import TrainingData, TrainSettings, fit
from nop.train.logging import LocalRunLogger
from nop.train.losses import masked_mse


def linear_check(config: dict[str, Any]) -> dict[str, Any]:
    """Check descent on a fixed convex fixture with a declared stable SGD step."""
    x = torch.tensor(config["design"], dtype=torch.float64)
    expected = x @ torch.tensor(config["coefficient"], dtype=torch.float64)
    coefficient = torch.zeros((x.shape[1], expected.shape[1]), dtype=x.dtype, requires_grad=True)
    optimizer = torch.optim.SGD([coefficient], lr=config["learning_rate"])
    losses = [float(((x @ coefficient - expected).square().mean()).detach())]
    for _ in range(config["steps"]):
        optimizer.zero_grad()
        loss = (x @ coefficient - expected).square().mean()
        loss.backward()  # type: ignore[no-untyped-call]  # Missing PyTorch annotation.
        optimizer.step()
        losses.append(float(((x @ coefficient - expected).square().mean()).detach()))
    return {
        "passed": all(after < before for before, after in zip(losses, losses[1:], strict=False)),
        "losses": losses,
        "scope": "deterministic_linear_fixture_not_a_neural_training_monotonicity_claim",
    }


def sanity_model(widths: list[int], grid_shape: tuple[int, ...]) -> nn.Module:
    """Build the declared temporary fixture, not the later M0 architecture."""
    layers: list[nn.Module] = []
    features = 5
    for width in widths:
        layers.extend((nn.Linear(features, width), nn.SiLU()))
        features = width
    head = nn.Linear(features, int(torch.tensor(grid_shape).prod()))
    nn.init.zeros_(head.weight)
    nn.init.zeros_(head.bias)
    layers.extend((head, nn.Unflatten(1, grid_shape)))
    return nn.Sequential(*layers)


def run(root: Path, config_path: Path) -> dict[str, Any]:
    """Verify G1, fit only the training prefix, and cache a fully identified check."""
    frozen = verify_frozen_files(root)
    config = yaml.safe_load(config_path.read_bytes())
    if config["frozen_policy"] != "config/training.yaml":
        raise ValueError("sanity must use the verified frozen training policy")
    policy = yaml.safe_load((root / config["frozen_policy"]).read_bytes())
    required = policy["sanity"]
    if (
        config["seed"] != required["seed"]
        or config["training_prefix_count"] != required["training_prefix_count"]
        or config["training"]["max_optimizer_steps"] != required["max_optimizer_steps"]
        or config["assessment"]["standardized_masked_MSE_max"]
        != required["standardized_masked_MSE_max"]
    ):
        raise ValueError("sanity settings disagree with the frozen check")
    dataset_path = root / config["dataset"]
    manifest = json.loads((dataset_path / "manifest.json").read_text())
    hashes = [
        chunk["sha256"]
        for split in ("train", "validation")
        for chunk in manifest["splits"][split]["chunks"]
    ] + [manifest["normalization_sha256"], manifest["grid_sha256"]]
    combined_hash = hashlib.sha256(json.dumps(hashes).encode()).hexdigest()
    if (
        manifest["dataset_hash"] != frozen["evidence"]["dataset"]["dataset_hash"]
        or combined_hash != manifest["dataset_hash"]
    ):
        raise ValueError("sanity requires the verified Gate 1 dataset identity")
    torch.set_num_threads(config["training"]["torch_threads"])
    seed_everything(config["seed"])
    # Existing loader verifies every training chunk. Nothing requests validation or holdouts.
    loaded = load_split(dataset_path, "train")
    count = config["training_prefix_count"]
    parameters, iv, valid = (
        loaded["parameters"][:count].clone(),
        loaded["iv"][:count].clone(),
        (loaded["status"][:count] == 0).clone(),
    )
    del loaded
    normalization = fit_train_normalization(parameters, iv, valid)
    data = TrainingData(
        parameters=normalization.transform_parameters(parameters).float(),
        targets=normalization.transform_iv(iv).float(),
        valid=valid,
        target_mean=normalization.iv_mean.float(),
        target_std=normalization.iv_std.float(),
    )
    model = sanity_model(config["model"]["hidden_widths"], tuple(iv.shape[1:]))
    settings_values = dict(config["training"])
    settings_values.pop("dtype")
    settings_values.pop("torch_threads")
    settings_values["betas"] = tuple(settings_values["betas"])
    settings = TrainSettings(seed=config["seed"], **settings_values)
    sources = sorted((root / "src/nop/train").glob("*.py")) + [
        root / "src/nop/eval/metrics.py",
        Path(__file__).resolve(),
    ]
    output, report_path = root / config["output"], root / config["report"]
    identity_path = output / "mlflow_identity.json"
    # Preserve the run's original commit across later commits when source hashes match.
    code_commit = (
        json.loads(identity_path.read_text())["code_commit"]
        if identity_path.exists()
        else current_commit(root)
    )
    identity = {
        "scope": config["scope"],
        "code_commit": code_commit,
        "seed": config["seed"],
        "config": config,
        "config_hash": sha256_file(config_path),
        "frozen_files": frozen["files"],
        "dataset_hash": manifest["dataset_hash"],
        "source_hashes": {str(path.relative_to(root)): sha256_file(path) for path in sources},
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "packages": {
                name: version(name) for name in ("torch", "numpy", "PyYAML", "mlflow-skinny")
            },
            "device": settings.device,
            "dtype": "float32",
            "torch_threads": torch.get_num_threads(),
        },
    }
    cache_key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    previous = None
    if report_path.exists():
        previous = json.loads(report_path.read_text())
        if previous["cache_key"] != cache_key:
            raise ValueError("sanity cache provenance changed; retain report and investigate")
    logger = LocalRunLogger(
        output_dir=output,
        identity=identity,
        experiment_name=config["experiment_name"],
        tracking_dir=root / config["tracking_directory"],
    )

    def log_epoch(row: dict[str, Any]) -> None:
        metrics = {
            key: float(row[key])
            for key in ("train_loss", "learning_rate", "optimizer_steps", "skipped_empty_batches")
        }
        metrics.update(
            {
                f"reconstruction_{key}": float(value)
                for key, value in row["validation_metrics"].items()
                if value is not None
            }
        )
        logger.log_epoch(row["epoch"], metrics)

    try:
        linear = linear_check(config["linear_check"])
        if not linear["passed"]:
            raise RuntimeError("linear gradient/loss sanity check failed")
        training = fit(model, data, data, settings, output, identity, logger=log_epoch)
        model.eval()
        with torch.no_grad():
            mse = float(masked_mse(model(data.parameters), data.targets, data.valid))
        result = {
            "passed": mse <= config["assessment"]["standardized_masked_MSE_max"],
            "cache_key": cache_key,
            "metadata": identity,
            "code_commit": code_commit,
            "scope": config["assessment"]["inputs_and_assessment_rows"],
            "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
            "standardized_masked_MSE": mse,
            "valid_quotes": int(valid.sum()),
            "attempted_quotes": valid.numel(),
            "linear_check": linear,
            "training": training,
            "mlflow_run_id": logger.run_id,
        }
        if previous is not None and result != previous:
            raise ValueError("cached sanity assessment changed; retain prior evidence")
        report_path.parent.mkdir(parents=True, exist_ok=True)
        write_json(report_path, result)
        logger.log_artifact(output / training["checkpoint_manifest"]["best"]["path"])
        logger.log_artifact(report_path)
        logger.finish(status="FINISHED" if result["passed"] else "FAILED")
        return result
    except BaseException:
        logger.finish(status="FAILED")
        raise


def main(argv: list[str] | None = None) -> int:
    """Return a failure status if the frozen reconstruction threshold is missed."""
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=root / "config/training_sanity.yaml")
    args = parser.parse_args(argv)
    result = run(root, args.config)
    print(
        json.dumps(
            {
                key: result[key]
                for key in ("passed", "standardized_masked_MSE", "valid_quotes", "parameter_count")
            },
            indent=2,
        )
    )
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
