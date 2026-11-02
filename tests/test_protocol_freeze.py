"""Freeze gates use synthetic artifacts only; no research holdouts are opened."""

import copy
import hashlib
import json
import runpy
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml

from nop.data.dataset import sha256_file, write_arrays, write_json
from nop.protocol import GATE1_FILES, verify_frozen_files

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def script(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    return runpy.run_path(str(ROOT / "scripts/freeze_protocol.py"))


@pytest.fixture
def repository(tmp_path: Path) -> Path:
    for name in GATE1_FILES:
        destination = tmp_path / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, destination)
    return tmp_path


def _manifest(root: Path) -> Path:
    path = root / "frozen/manifest.json"
    path.parent.mkdir(exist_ok=True)
    write_json(
        path,
        {
            "version": 1,
            "gate": "G1",
            "protocol_version": 1,
            "model_freeze_complete": False,
            "files": {name: sha256_file(root / name) for name in GATE1_FILES},
        },
    )
    return path


def test_verified_gate1_manifest_does_not_authorize_model_freeze(repository: Path) -> None:
    path = _manifest(repository)
    before = path.read_bytes()
    result = verify_frozen_files(repository)
    assert result["gate"] == "G1" and result["model_freeze_complete"] is False
    assert set(result["files"]) == set(GATE1_FILES)
    assert path.read_bytes() == before


@pytest.mark.parametrize(
    "failure", ["tampered", "missing_file", "missing_hash", "empty", "bad_hash", "gate3"]
)
def test_hash_verifier_fails_closed(repository: Path, failure: str) -> None:
    path = _manifest(repository)
    manifest = json.loads(path.read_text())
    name = GATE1_FILES[0]
    if failure == "tampered":
        (repository / name).write_text("changed")
    elif failure == "missing_file":
        (repository / name).unlink()
    elif failure == "missing_hash":
        del manifest["files"][name]
    elif failure == "empty":
        manifest["files"] = {}
    elif failure == "bad_hash":
        manifest["files"][name] = "not-a-sha256"
    else:
        manifest["gate"] = "G3"
        manifest["model_freeze_complete"] = True
    write_json(path, manifest)
    with pytest.raises(ValueError):
        verify_frozen_files(repository)


@pytest.mark.parametrize(
    "name",
    [
        "../outside",
        "/etc/passwd",
        "config/../config/data.yaml",
        "config//data.yaml",
        "./config/data.yaml",
    ],
)
def test_manifest_rejects_noncanonical_paths(repository: Path, name: str) -> None:
    path = _manifest(repository)
    manifest = json.loads(path.read_text())
    manifest["files"][name] = "0" * 64
    write_json(path, manifest)
    with pytest.raises(ValueError, match="unsafe repository path"):
        verify_frozen_files(repository)


@pytest.mark.parametrize("component", ["file", "parent", "manifest"])
def test_hash_verifier_rejects_symlinks(repository: Path, component: str) -> None:
    path = _manifest(repository)
    if component == "file":
        original = repository / GATE1_FILES[0]
    elif component == "parent":
        original = repository / "config"
    else:
        original = path
    replacement = original.with_name(original.name + ".original")
    original.rename(replacement)
    original.symlink_to(replacement, target_is_directory=replacement.is_dir())
    with pytest.raises(ValueError, match="symlink"):
        verify_frozen_files(repository)


@pytest.mark.parametrize("content", ["{}", "[]", '{"version":1,"version":1}', '{"value":NaN}'])
def test_invalid_manifest_is_rejected(repository: Path, content: str) -> None:
    path = _manifest(repository)
    path.write_text(content)
    with pytest.raises(ValueError):
        verify_frozen_files(repository)


def test_completed_preregistration_is_structurally_consistent(
    repository: Path, script: dict[str, Any]
) -> None:
    protocol, data = script["validate_preregistration"](repository)
    assert protocol["data_design"] == data["data_design"]


@pytest.mark.parametrize(
    "failure",
    ["draft", "checklist", "normalization", "budget", "partition", "seed_pair", "diagnostic"],
)
def test_checklist_labels_cannot_bypass_concrete_preregistration_gates(
    repository: Path, script: dict[str, Any], failure: str
) -> None:
    name = {
        "budget": "training",
        "partition": "calibration",
        "seed_pair": "calibration",
        "diagnostic": "diagnostics",
    }.get(failure, "eval_protocol")
    path = repository / f"config/{name}.yaml"
    config = yaml.safe_load(path.read_text())
    if failure == "draft":
        config["status"] = "draft"
    elif failure == "checklist":
        del config["freeze_checklist"][config["required_before_G1_freeze"][0]]
    elif failure == "normalization":
        config["query_generalization"]["off_grid_target_normalization"] = "unresolved"
    elif failure == "budget":
        config["tuning"]["trials_per_model"] += 1
    elif failure == "partition":
        config["quote_partition"]["held_out_z_indices"].append(0)
    elif failure == "seed_pair":
        config["conditions"]["seed_pairs"][0]["model_seed"] = 100
    else:
        config["greek_sensitivities"]["coordinate_steps"]["T"] = 0
    path.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError):
        script["validate_preregistration"](repository)
    assert not (repository / "frozen").exists()


def _dataset(root: Path, failures: int = 0) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    config = yaml.safe_load((ROOT / "config/data.yaml").read_text())
    design = config["data_design"]
    design["id_splits"] = {
        "train": {"offset": 0, "count": 100},
        "validation": {"offset": 100, "count": 100},
        "test": {"offset": 200, "count": 100},
    }
    design["main_grid"]["maturity_year_fractions"] = [[1, 12], [1, 1]]
    design["main_grid"]["z_linspace"]["count"] = 2
    output = root / "synthetic"
    metadata = {"config_hash": "fixture", "protocol_hash": "fixture", "source_hashes": {}}
    manifest: dict[str, Any] = {
        "complete": True,
        "scope": "train_validation_only",
        "seed": design["id_stream_seed"],
        "metadata": metadata,
        "splits": {},
        "coverage_stop_triggered": False,
    }
    hashes = []
    for split, offset in (("train", 0), ("validation", 100)):
        directory = output / split
        directory.mkdir(parents=True)
        path = directory / "00000000.npz"
        status = np.zeros((100, 2, 2), dtype=np.int64)
        status[:failures, 0, 0] = 1
        iv = np.full(status.shape, 0.2)
        iv[status != 0] = np.nan
        write_arrays(
            path,
            {
                "parameters": np.ones((100, 5)),
                "iv": iv,
                "status": status,
                "metadata": np.array(json.dumps({"seed": design["id_stream_seed"], **metadata})),
            },
        )
        digest = sha256_file(path)
        hashes.append(digest)
        chunk = {"file": path.name, "start": 0, "count": 100, "sha256": digest}
        write_json(path.with_suffix(".json"), {**chunk, "metadata": metadata})
        failed = status != 0
        manifest["splits"][split] = {
            "count": 100,
            "offset": offset,
            "chunks": [chunk],
            "summary": {
                "attempted": 400,
                "failed": failures,
                "failure_fraction": failures / 400,
                "failure_fraction_by_maturity": failed.mean(axis=(0, 2)).tolist(),
                "coverage_stop_triggered": False,
            },
        }
    for name, key in (("normalization.npz", "normalization_sha256"), ("grid.npz", "grid_sha256")):
        write_arrays(
            output / name,
            {
                "metadata": np.array(
                    json.dumps({"seed": design["id_stream_seed"], "fit_split": "train", **metadata})
                ),
            },
        )
        manifest[key] = sha256_file(output / name)
        hashes.append(manifest[key])
    manifest["dataset_hash"] = hashlib.sha256(json.dumps(hashes).encode()).hexdigest()
    write_json(output / "manifest.json", manifest)
    return output, config, metadata


def test_coverage_is_recomputed_from_verified_masks(tmp_path: Path, script: dict[str, Any]) -> None:
    output, config, metadata = _dataset(tmp_path, failures=2)
    result = script["verify_dataset"](tmp_path, output, config, metadata, 0.03)
    assert result["coverage"]["train"]["attempted"] == 400
    assert result["coverage"]["train"]["failure_fraction"] == 0.005
    assert result["coverage"]["validation"]["failure_fraction_by_maturity"] == [0.01, 0.0]
    assert set(result["coverage"]) == {"train", "validation"}


def test_each_maturity_stop_rule_applies_even_when_overall_coverage_passes(
    tmp_path: Path, script: dict[str, Any]
) -> None:
    output, config, metadata = _dataset(tmp_path, failures=8)
    with pytest.raises(ValueError, match="STOP: train reference-failure"):
        script["verify_dataset"](tmp_path, output, config, metadata, 0.03)


@pytest.mark.parametrize(
    "failure", ["stale", "incomplete", "count", "gap", "hash", "summary", "holdout", "auxiliary"]
)
def test_dataset_gate_rejects_incomplete_stale_or_tampered_artifacts(
    tmp_path: Path, script: dict[str, Any], failure: str
) -> None:
    output, config, metadata = _dataset(tmp_path)
    path = output / "manifest.json"
    manifest = json.loads(path.read_text())
    if failure == "stale":
        manifest["metadata"]["protocol_hash"] = "old-protocol"
    elif failure == "incomplete":
        manifest["complete"] = False
    elif failure == "count":
        manifest["splits"]["train"]["count"] -= 1
    elif failure == "gap":
        manifest["splits"]["train"]["chunks"][0]["start"] = 1
    elif failure == "hash":
        with (output / "train/00000000.npz").open("ab") as stream:
            stream.write(b"tampered")
    elif failure == "summary":
        manifest["splits"]["train"]["summary"]["attempted"] = 399
    elif failure == "holdout":
        manifest["splits"]["test"] = {"file": "never-open-this"}
    else:
        (output / "normalization.npz").unlink()
    write_json(path, manifest)
    with pytest.raises(ValueError):
        script["verify_dataset"](tmp_path, output, config, metadata, 0.03)


@pytest.mark.parametrize("failure", ["failed", "stale", "cache_key"])
def test_numerical_report_must_be_passed_and_current(
    tmp_path: Path, script: dict[str, Any], failure: str
) -> None:
    metadata = {"source_hash": "current", "config_hash": "current"}
    report = {
        "passed": True,
        "metadata": copy.deepcopy(metadata),
        "cache_key": hashlib.sha256(json.dumps(metadata, sort_keys=True).encode()).hexdigest(),
    }
    path = tmp_path / "report.json"
    write_json(path, report)
    assert script["verify_report"](path, metadata)["sha256"] == sha256_file(path)
    if failure == "failed":
        report["passed"] = False
    elif failure == "stale":
        report["metadata"]["source_hash"] = "old"
    else:
        report["cache_key"] = "old"
    write_json(path, report)
    with pytest.raises(ValueError, match="failed or is stale"):
        script["verify_report"](path, metadata)


def test_freeze_creates_once_then_only_verifies(
    repository: Path, script: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    output, config, metadata = _dataset(repository)
    protocol = {"stop_rules": {"invalid_reference_fraction_max_before_freeze": 0.03}}
    # Synthetic data already exercises the real hash/mask verifier above; stub
    # external provenance/numerical studies to exercise the publication lifecycle.
    namespace = script["freeze"].__globals__
    monkeypatch.setitem(namespace, "validate_preregistration", lambda _root: (protocol, config))
    monkeypatch.setitem(namespace, "provenance", lambda *_args: dict(metadata))
    monkeypatch.setitem(namespace, "verify_dataset", lambda *_args: {"dataset_hash": "fixture"})
    monkeypatch.setitem(namespace, "verify_numerical_reports", lambda *_args: {"passed": True})
    monkeypatch.setitem(namespace, "current_commit", lambda _root: "fixture-commit")
    script["freeze"](repository, output)
    path = repository / "frozen/manifest.json"
    before = path.read_bytes()

    def forbidden(*_args: Any) -> None:
        raise AssertionError("an existing freeze must only verify hashes")

    monkeypatch.setitem(namespace, "validate_preregistration", forbidden)
    script["freeze"](repository, output)
    assert path.read_bytes() == before
    (repository / GATE1_FILES[0]).write_text("tampered")
    with pytest.raises(ValueError, match="hash mismatch"):
        script["freeze"](repository, output)
    assert path.read_bytes() == before


def test_verify_only_never_creates_manifest(repository: Path, script: dict[str, Any]) -> None:
    with pytest.raises(ValueError, match="required file is missing"):
        script["freeze"](repository, verify_only=True)
    assert not (repository / "frozen").exists()
