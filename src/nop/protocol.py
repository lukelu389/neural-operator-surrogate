"""Fail-closed verification of the Gate 1 preregistration, never holdout access."""

import json
import re
from pathlib import Path
from typing import Any, NoReturn

from nop.data.dataset import sha256_file

GATE1_FILES = (
    "config/eval_protocol.yaml",
    "config/data.yaml",
    "config/data_numerics.yaml",
    "config/training.yaml",
    "config/diagnostics.yaml",
    "config/calibration.yaml",
    "src/nop/train/coordinates.py",
    "src/nop/eval/interpolation.py",
)

GATE1_REQUIREMENTS = (
    "validated_COS_truncation_terms_precision_and_IV_tolerances_in_data_yaml",
    "failure_mask_stop_rule_checked_on_train_and_validation_only",
    "data_yaml_agrees_with_split_recipes_ranges_grids_seeds_and_counts_above",
    "shared_tuning_search_ranges_training_cap_warmup_clipping_early_stopping_and_loss_ablation",
    "train_only_off_grid_normalization_and_M0_adapter_validated_without_test_or_OOD_access",
    "Greek_finite_difference_steps_and_arbitrage_numerical_tolerances",
    "calibration_surface_ids_quote_partitions_training_and_experiment_seed_pairing_solver_caps_and_C2_training_budget",
    "resolve_any_requested_change_to_lazy_holdouts_or_validation_only_later_diagnostics",
)


def checked_repo_file(root: Path, name: str) -> Path:
    """Require a canonical relative regular file without any symlink component."""
    path = Path(name)
    if (
        not name
        or path.is_absolute()
        or path.as_posix() != name
        or any(part in (".", "..") for part in path.parts)
    ):
        raise ValueError(f"unsafe repository path: {name!r}")
    candidate = root.resolve()
    for component in path.parts:
        candidate = candidate / component
        if candidate.is_symlink():
            raise ValueError(f"symlink is not permitted in frozen evidence: {name}")
    if not candidate.is_file():
        raise ValueError(f"required file is missing: {name}")
    return candidate


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def read_json_object(path: Path) -> dict[str, Any]:
    """Read strict JSON evidence, rejecting duplicate keys and non-object roots."""

    def reject_constant(value: str) -> NoReturn:
        raise ValueError(f"nonfinite JSON constant: {value}")

    result = json.loads(
        path.read_text(), object_pairs_hook=_unique_object, parse_constant=reject_constant
    )
    if not isinstance(result, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return result


def verify_frozen_files(root: Path, manifest_path: Path | None = None) -> dict[str, Any]:
    """Verify every listed hash and mandatory G1 file; this does not authorize G3.

    The manifest is a local integrity record, not a cryptographic signature.
    Model/checkpoint freezing and permission to access holdouts remain separate
    requirements for T3.4, which this Gate 1 verifier cannot satisfy.
    """
    root = root.resolve()
    manifest_path = manifest_path or root / "frozen/manifest.json"
    if not manifest_path.is_absolute():
        manifest_path = root / manifest_path
    try:
        relative = manifest_path.relative_to(root).as_posix()
    except ValueError as error:
        raise ValueError("frozen manifest must be inside the repository") from error
    manifest = read_json_object(checked_repo_file(root, relative))
    if (
        type(manifest.get("version")) is not int
        or manifest["version"] != 1
        or manifest.get("gate") != "G1"
        or type(manifest.get("protocol_version")) is not int
        or manifest.get("protocol_version") != 1
        or manifest.get("model_freeze_complete") is not False
    ):
        raise ValueError("expected a Gate 1 v1 manifest; model freeze is not complete")
    files = manifest.get("files")
    if not isinstance(files, dict) or not set(GATE1_FILES).issubset(files):
        raise ValueError("frozen manifest is missing required Gate 1 file hashes")
    for name, digest in files.items():
        if not isinstance(digest, str) or re.fullmatch("[0-9a-f]{64}", digest) is None:
            raise ValueError(f"invalid SHA-256 for {name}")
        path = checked_repo_file(root, name)
        if sha256_file(path) != digest:
            raise ValueError(
                f"frozen hash mismatch: {name}; stop and record any approved deviation"
            )
    return manifest
