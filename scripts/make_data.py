"""Generate validated, cached training/validation surfaces; never holdouts."""

import argparse
import json
from pathlib import Path
from typing import Any

import torch
import yaml

from nop.data.dataset import current_commit, generate_dataset, provenance, sha256_file


def validate_design(config: dict[str, Any], protocol: dict[str, Any]) -> None:
    """Fail before generation if sampling or access policy differs from the draft."""
    if config["version"] != 1 or config["scope"] != "synthetic_heston_train_validation_only":
        raise ValueError("unsupported data configuration")
    if config["data_design"] != protocol["data_design"]:
        raise ValueError("data design must exactly match the evaluation protocol")
    if config["units"] != {
        key: protocol["units"][key] for key in ("spot", "rate", "dividend_yield")
    }:
        raise ValueError("pricing units must match the evaluation protocol")
    if protocol["holdout_access"]["gate1_materialization"] != "train_and_validation_only":
        raise ValueError("unexpected holdout access policy")
    if (
        config["validation"]["max_invalid_fraction"]
        != protocol["stop_rules"]["invalid_reference_fraction_max_before_freeze"]
    ):
        raise ValueError("coverage threshold must match the protocol")
    if config["normalization"] != {
        "fit_split": "train",
        "variance_ddof": 0,
        "invalid_iv": "NaN",
        "zero_variance": "error",
    }:
        raise ValueError("unsupported normalization policy")
    windows = config["data_design"]["id_splits"]
    train, validation, test = (windows[key] for key in ("train", "validation", "test"))
    if not (
        train["offset"] == 0
        and train["count"] == validation["offset"]
        and validation["offset"] + validation["count"] == test["offset"]
    ):
        raise ValueError("train/validation windows must end before the test window")
    threads = config["storage"]["torch_threads"]
    if isinstance(threads, bool) or not isinstance(threads, int) or threads < 1:
        raise ValueError("torch_threads must be a positive integer")


def main() -> int:
    """Validate numerics, then create or verify permitted split artifacts."""
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[1]
    parser.add_argument("--config", type=Path, default=root / "config/data.yaml")
    parser.add_argument("--output", type=Path, help="Override the configured artifact directory")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_bytes())
    protocol = yaml.safe_load((root / "config/eval_protocol.yaml").read_bytes())
    validate_design(config, protocol)
    # Recompute validation cache identities; never trust a stale passed flag.
    from validate_adaptive_labels import main as validate_adaptive
    from validate_data_numerics import main as validate_numerics

    if (
        validate_numerics(
            [
                "--config",
                str(root / config["validation"]["numerics_config"]),
                "--output",
                str(root / config["validation"]["numerics_report"]),
            ]
        )
        != 0
    ):
        raise RuntimeError("fixed-resolution numerical validation failed; no labels generated")
    if (
        validate_adaptive(
            [
                "--config",
                str(args.config),
                "--numerics-config",
                str(root / config["validation"]["numerics_config"]),
            ]
        )
        != 0
    ):
        raise RuntimeError("adaptive-label validation failed; no labels generated")
    torch.set_num_threads(config["storage"]["torch_threads"])
    metadata = provenance(root, args.config)
    metadata["protocol_hash"] = sha256_file(root / "config/eval_protocol.yaml")
    output = args.output if args.output is not None else root / config["storage"]["output"]
    manifest = generate_dataset(config, output, metadata, code_commit=current_commit(root))
    print(
        json.dumps(
            {
                "dataset_hash": manifest["dataset_hash"],
                "coverage_stop_triggered": manifest["coverage_stop_triggered"],
            },
            indent=2,
        )
    )
    if manifest["coverage_stop_triggered"]:
        print(
            "STOP: reference failures exceed the declared threshold. "
            "Preserve these results and revisit numerics/grid before freeze."
        )
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
