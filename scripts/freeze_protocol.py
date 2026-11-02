"""Freeze verified Gate 1 preregistration; never read or generate test/OOD data."""

import argparse
import hashlib
import json
import math
import platform
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from nop.data.dataset import current_commit, provenance, sha256_file
from nop.data.labels import LabelStatus
from nop.pricing.iv import IVStatus
from nop.protocol import (
    GATE1_FILES,
    GATE1_REQUIREMENTS,
    checked_repo_file,
    read_json_object,
    verify_frozen_files,
)


def verify_report(path: Path, metadata: dict[str, Any]) -> dict[str, str]:
    """Reject failed, missing or stale numerical evidence without rerunning it."""
    report = read_json_object(path)
    cache_key = hashlib.sha256(json.dumps(metadata, sort_keys=True).encode()).hexdigest()
    if (
        report.get("passed") is not True
        or report.get("metadata") != metadata
        or report.get("cache_key") != cache_key
    ):
        raise ValueError(f"numerical validation failed or is stale: {path}")
    return {"sha256": sha256_file(path), "cache_key": cache_key}


def verify_dataset(
    root: Path,
    output: Path,
    config: dict[str, Any],
    metadata: dict[str, Any],
    threshold: float,
) -> dict[str, Any]:
    """Stream hash-verified train/validation masks and enforce actual coverage."""
    relative_output = output.relative_to(root).as_posix()

    def artifact(name: str) -> Path:
        return checked_repo_file(root, f"{relative_output}/{name}")

    manifest_path = artifact("manifest.json")
    manifest = read_json_object(manifest_path)
    design = config["data_design"]
    if (
        manifest.get("complete") is not True
        or manifest.get("scope") != "train_validation_only"
        or set(manifest.get("splits", {})) != {"train", "validation"}
        or manifest.get("metadata") != metadata
        or manifest.get("seed") != design["id_stream_seed"]
    ):
        raise ValueError("dataset is incomplete, has forbidden splits, or has stale provenance")
    grid = design["main_grid"]
    maturities = len(grid["maturity_year_fractions"])
    strikes = grid["z_linspace"]["count"]
    if not math.isfinite(threshold) or not 0 <= threshold < 1:
        raise ValueError("invalid reference-failure threshold")
    hashes: list[str] = []
    coverage: dict[str, Any] = {}
    known_statuses = [int(value) for value in IVStatus] + [int(value) for value in LabelStatus]
    for split in ("train", "validation"):
        record = manifest["splits"][split]
        recipe = design["id_splits"][split]
        if record["count"] != recipe["count"] or record["offset"] != recipe["offset"]:
            raise ValueError(f"dataset split count/offset differs from protocol: {split}")
        expected_start = 0
        failed_by_maturity = np.zeros(maturities, dtype=np.int64)
        for chunk in record["chunks"]:
            count = chunk["count"]
            if (
                type(count) is not int
                or count <= 0
                or chunk["start"] != expected_start
                or chunk["file"] != f"{expected_start:08d}.npz"
            ):
                raise ValueError(f"noncontiguous or unsafe dataset chunk: {split}")
            path = artifact(f"{split}/{chunk['file']}")
            sidecar = read_json_object(artifact(f"{split}/{expected_start:08d}.json"))
            if sidecar != {**chunk, "metadata": metadata}:
                raise ValueError(f"chunk sidecar differs from manifest: {path}")
            digest = sha256_file(path)
            if digest != chunk["sha256"]:
                raise ValueError(f"dataset chunk hash mismatch: {path}")
            hashes.append(digest)
            with np.load(path, allow_pickle=False) as arrays:
                if json.loads(str(arrays["metadata"].item())) != {
                    "seed": design["id_stream_seed"],
                    **metadata,
                }:
                    raise ValueError(f"embedded chunk provenance mismatch: {path}")
                status, iv, parameters = arrays["status"], arrays["iv"], arrays["parameters"]
                if (
                    status.shape != (count, maturities, strikes)
                    or iv.shape != status.shape
                    or parameters.shape != (count, len(design["parameter_order"]))
                    or not np.isfinite(parameters).all()
                    or not np.issubdtype(status.dtype, np.integer)
                    or not np.isin(status, known_statuses).all()
                ):
                    raise ValueError(f"invalid dataset shapes or status codes: {path}")
                failed = status != IVStatus.SUCCESS
                if (
                    not np.isfinite(iv[~failed]).all()
                    or (iv[~failed] <= 0).any()
                    or not np.isnan(iv[failed]).all()
                ):
                    raise ValueError(f"IV values disagree with explicit masks: {path}")
                failed_by_maturity += failed.sum(axis=(0, 2))
            expected_start += count
        if expected_start != recipe["count"] or expected_start <= 0:
            raise ValueError(f"dataset does not cover configured split: {split}")
        attempted = expected_start * maturities * strikes
        failed_count = int(failed_by_maturity.sum())
        fractions = failed_by_maturity / (expected_start * strikes)
        if failed_count / attempted > threshold or np.any(fractions > threshold):
            raise ValueError(f"STOP: {split} reference-failure fraction exceeds {threshold}")
        coverage[split] = {
            "surfaces": expected_start,
            "attempted": attempted,
            "failed": failed_count,
            "failure_fraction": failed_count / attempted,
            "failure_fraction_by_maturity": fractions.tolist(),
        }
        summary = record["summary"]
        for key in ("attempted", "failed", "failure_fraction", "failure_fraction_by_maturity"):
            if summary.get(key) != coverage[split][key]:
                raise ValueError(f"dataset stored coverage disagrees with verified masks: {split}")
        if summary.get("coverage_stop_triggered") is not False:
            raise ValueError(f"dataset coverage stop flag is not clear: {split}")
    for filename, field in (
        ("normalization.npz", "normalization_sha256"),
        ("grid.npz", "grid_sha256"),
    ):
        path = artifact(filename)
        digest = sha256_file(path)
        if digest != manifest[field]:
            raise ValueError(f"dataset auxiliary hash mismatch: {filename}")
        with np.load(path, allow_pickle=False) as arrays:
            if json.loads(str(arrays["metadata"].item())) != {
                "seed": design["id_stream_seed"],
                "fit_split": "train",
                **metadata,
            }:
                raise ValueError(f"dataset auxiliary provenance mismatch: {filename}")
        hashes.append(digest)
    dataset_hash = hashlib.sha256(json.dumps(hashes).encode()).hexdigest()
    if (
        manifest["dataset_hash"] != dataset_hash
        or manifest.get("coverage_stop_triggered") is not False
    ):
        raise ValueError("dataset combined hash mismatch or uncleared coverage stop flag")
    return {
        "manifest_sha256": sha256_file(manifest_path),
        "dataset_hash": dataset_hash,
        "threshold": threshold,
        "coverage": coverage,
    }


def verify_numerical_reports(root: Path, config: dict[str, Any]) -> dict[str, Any]:
    """Check current fixed/adaptive pricing and diagnostic study identities."""
    from validate_adaptive_labels import validation_metadata as adaptive_metadata
    from validate_diagnostics import validation_metadata as diagnostics_metadata

    numerics = checked_repo_file(root, config["validation"]["numerics_config"])
    sources = [
        root / "scripts/validate_data_numerics.py",
        root / "scripts/validate_iv.py",
        root / "src/nop/pricing/iv.py",
        root / "src/nop/pricing/bs.py",
        root / "src/nop/pricing/heston_cos.py",
    ]
    fixed_metadata = {
        "config_hash": sha256_file(numerics),
        "source_hash": hashlib.sha256(b"".join(path.read_bytes() for path in sources)).hexdigest(),
        "seed": None,
        "sampling": "none_deterministic_numerical_fixtures",
        "python": platform.python_version(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "pyyaml": yaml.__version__,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "torch_threads": torch.get_num_threads(),
        "dtype": "float64",
        "device": "cpu",
    }
    data_bytes = checked_repo_file(root, "config/data.yaml").read_bytes()
    diagnostics_bytes = checked_repo_file(root, "config/diagnostics.yaml").read_bytes()
    diagnostic_config = yaml.safe_load(diagnostics_bytes)
    reports = {
        "fixed_pricer": verify_report(
            checked_repo_file(root, config["validation"]["numerics_report"]), fixed_metadata
        ),
    }
    # Match make_data.py: fixed validation inherits the caller's thread setting,
    # then adaptive validation applies the data configuration's thread count.
    torch.set_num_threads(config["storage"]["torch_threads"])
    reports["adaptive_pricer"] = verify_report(
        checked_repo_file(root, "reports/tables/adaptive_labels.json"),
        adaptive_metadata(root, data_bytes, numerics.read_bytes()),
    )
    torch.set_num_threads(diagnostic_config["validation"]["torch_threads"])
    reports["diagnostics"] = verify_report(
        checked_repo_file(root, "reports/tables/diagnostics_validation.json"),
        diagnostics_metadata(root, diagnostics_bytes),
    )
    return reports


def validate_preregistration(root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Require resolved preregistration and consistent train-only access rules."""
    from make_data import validate_design

    configs = {
        name: yaml.safe_load(checked_repo_file(root, f"config/{name}.yaml").read_bytes())
        for name in (
            "eval_protocol",
            "data",
            "data_numerics",
            "training",
            "diagnostics",
            "calibration",
        )
    }
    protocol, data = configs["eval_protocol"], configs["data"]
    if protocol.get("status") != "frozen" or protocol.get("protocol_version") != 1:
        raise ValueError("protocol must be resolved and marked frozen before creating its hashes")
    required = protocol.get("required_before_G1_freeze")
    checklist = protocol.get("freeze_checklist")
    if (
        not isinstance(required, list)
        or set(required) != set(GATE1_REQUIREMENTS)
        or len(set(required)) != len(required)
        or not isinstance(checklist, dict)
        or set(checklist) != set(required)
        or any(value != "resolved" for value in checklist.values())
    ):
        raise ValueError("all eight pre-freeze checklist requirements must be resolved")
    validate_design(data, protocol)
    normalization = protocol["query_generalization"]["off_grid_target_normalization"]
    if (
        protocol["holdout_access"]["earliest_gate"] != "T3.4"
        or protocol["holdout_access"]["sole_reader_and_materializer"] != "scripts/final_eval.py"
        or not isinstance(normalization, dict)
        or normalization.get("implementation") != "src/nop/train/coordinates.py"
        or normalization.get("source")
        != "valid_training_quotes_only_on_coordinates_available_to_each_run"
        or normalization.get("method") != "bilinear_mean_and_log_standard_deviation_in_z_and_T"
        or normalization.get("outside_training_support")
        != "independently_clamp_z_and_T_for_moments_only"
    ):
        raise ValueError("unresolved off-grid normalization or invalid holdout access policy")
    for name in ("training", "diagnostics", "calibration"):
        if not isinstance(configs[name], dict) or configs[name].get("version") != 1:
            raise ValueError(f"missing or invalid preregistration: {name}")
        if protocol.get(f"{name}_config") != f"config/{name}.yaml":
            raise ValueError(f"protocol must reference its {name} preregistration")
    _validate_policies(configs)
    return protocol, data


def _positive(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        and value > 0
    )


def _validate_policies(configs: dict[str, Any]) -> None:
    """Check concrete budgets/partitions instead of trusting checklist labels alone."""
    protocol, data = configs["eval_protocol"], configs["data"]
    training, calibration, diagnostics = (
        configs[name] for name in ("training", "calibration", "diagnostics")
    )
    fairness = protocol["fairness"]
    run, tuning = training["training"], training["tuning"]
    if (
        training["model_ids"] != fairness["model_ids"]
        or training["nominal_parameters"] != fairness["nominal_parameters"]
        or training["max_relative_parameter_difference_from_M0"]
        != fairness["max_relative_parameter_difference_from_M0"]
        or run["optimizer"] != fairness["optimizer_family"]
        or run["seeds"] != fairness["final_training_seeds"]
        or tuning["trials_per_model"] != fairness["tuning_trials_per_architecture"]
        or tuning["training_seed"] != fairness["tuning_seed"]
        or tuning["selection_metric"] != fairness["selection_metric"]
        or tuning["paired_identical_draws_across_models"] is not True
        or not all(
            _positive(run[key]) for key in ("batch_size", "max_epochs", "gradient_clip_global_norm")
        )
        or not 0 < run["warmup_epochs"] < run["max_epochs"]
        or not _positive(run["early_stopping"]["patience_epochs"])
        or training["vega_ablation"]["seeds"] != fairness["final_training_seeds"]
        or not training["headline_loss"]["formula"]
        or not training["vega_ablation"]["raw_weight"]
        or not training["vega_ablation"]["normalization"]
    ):
        raise ValueError("training policy has unresolved budgets or disagrees with shared fairness")
    for name in ("learning_rate", "weight_decay"):
        limits = tuning[name]
        if not _positive(limits["low"]) or not limits["low"] < limits["high"] < math.inf:
            raise ValueError(f"unresolved tuning range: {name}")
    design = data["data_design"]
    validation_count = design["id_splits"]["validation"]["count"]
    indices = calibration["surface_indices"]
    selection = calibration["C2"]["validation_selection_indices"]
    if not (
        0 <= indices["start"] < indices["stop_exclusive"] <= validation_count
        and indices["step"] > 0
        and indices["stop_exclusive"]
        <= selection["start"]
        < selection["stop_exclusive"]
        <= validation_count
        and selection["step"] > 0
    ):
        raise ValueError("calibration surface and C2 selection indices must be valid and disjoint")
    partition = calibration["quote_partition"]
    fit, held_out = partition["fit_z_indices"], partition["held_out_z_indices"]
    if (
        not fit
        or not held_out
        or set(fit) & set(held_out)
        or sorted(fit + held_out) != list(range(design["main_grid"]["z_linspace"]["count"]))
    ):
        raise ValueError("calibration quote partitions must be disjoint and cover the grid")
    conditions = calibration["conditions"]
    solver, inverse = calibration["shared_solver"], calibration["C2"]
    if (
        calibration["split"] != "validation"
        or [pair["model_seed"] for pair in conditions["seed_pairs"]]
        != fairness["final_training_seeds"]
        or [pair["experiment_seed"] for pair in conditions["seed_pairs"]]
        != protocol["calibration"]["paired_experiment_seeds"]
        or conditions["noise_std_vol_points"] != protocol["calibration"]["noise_std_vol_points"]
        or conditions["missing_quote_fractions"]
        != protocol["calibration"]["missing_quote_fractions"]
        or solver["starts_per_surface"] != protocol["calibration"]["C0_C1_starts_per_surface"]
        or not all(
            _positive(solver[key])
            for key in (
                "max_residual_evaluations_per_start",
                "max_jacobian_evaluations_per_start",
                "ftol",
                "xtol",
                "gtol",
            )
        )
        or inverse["train_count"] != design["id_splits"]["train"]["count"]
        or inverse["seeds"] != fairness["final_training_seeds"]
        or not all(_positive(inverse[key]) for key in ("max_epochs", "batch_size", "learning_rate"))
        or inverse["tuning_trials"] != 0
    ):
        raise ValueError("calibration seed pairing, budgets or data scope are unresolved")
    greek = diagnostics["greek_sensitivities"]
    numerics = configs["data_numerics"]
    if (
        diagnostics["parameter_order"] != design["parameter_order"]
        or diagnostics["parameter_box"] != design["id_box"]
        or greek["iv"] != data["labels"]["iv"]
        or greek["cos"]
        != {
            "terms": numerics["cos"]["reference_terms"],
            "width": numerics["cos"]["reference_width"],
        }
        or not all(_positive(value) for value in greek["parameter_steps"].values())
        or set(greek["parameter_steps"]) != set(design["parameter_order"])
        or set(greek["coordinate_steps"]) != {"k", "T"}
        or not all(_positive(value) for value in greek["coordinate_steps"].values())
        or not all(
            _positive(diagnostics["arbitrage"][key]["absolute_tolerance"])
            for key in ("butterfly", "calendar")
        )
    ):
        raise ValueError(
            "diagnostic steps, tolerances or validated reference settings are unresolved"
        )


def freeze(root: Path, output: Path | None = None, *, verify_only: bool = False) -> dict[str, Any]:
    """Create one immutable G1 manifest, or verify the existing one without writing."""
    root = root.resolve()
    manifest_path = root / "frozen/manifest.json"
    if manifest_path.exists() or manifest_path.is_symlink() or verify_only:
        return verify_frozen_files(root)
    if (root / "frozen").is_symlink():
        raise ValueError("frozen directory must not be a symlink")
    protocol, config = validate_preregistration(root)
    original_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(config["storage"]["torch_threads"])
        metadata = provenance(root, root / "config/data.yaml")
        metadata["protocol_hash"] = sha256_file(root / "config/eval_protocol.yaml")
        output = output or root / config["storage"]["output"]
        if not output.is_absolute():
            output = root / output
        dataset = verify_dataset(
            root,
            output,
            config,
            metadata,
            protocol["stop_rules"]["invalid_reference_fraction_max_before_freeze"],
        )
        torch.set_num_threads(original_threads)
        numerical_reports = verify_numerical_reports(root, config)
    finally:
        torch.set_num_threads(original_threads)
    manifest = {
        "version": 1,
        "gate": "G1",
        "protocol_version": 1,
        "model_freeze_complete": False,
        "code_commit_at_freeze": current_commit(root),
        "files": {name: sha256_file(checked_repo_file(root, name)) for name in GATE1_FILES},
        "evidence": {"dataset": dataset, "numerical_reports": numerical_reports},
        "holdouts": "sealed; model/checkpoint freezing at T3.4 is still required",
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation also prevents accidental replacement after a concurrent freeze.
    with manifest_path.open("x") as stream:
        stream.write(json.dumps(manifest, sort_keys=True, indent=2, allow_nan=False) + "\n")
    return verify_frozen_files(root)


def main(argv: list[str] | None = None) -> int:
    """Return a clear refusal without creating a manifest when any gate fails."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--verify", action="store_true", help="Check hashes; never create a manifest"
    )
    parser.add_argument("--output", type=Path, help="Train/validation artifact directory")
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parents[1]
    try:
        manifest = freeze(root, args.output, verify_only=args.verify)
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f"Gate 1 freeze refused: {error}", file=sys.stderr)
        return 2
    print(f"Gate 1 protocol verified: {len(manifest['files'])} files; holdouts remain sealed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
