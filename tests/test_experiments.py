"""Frozen-budget helpers tested with invented data, never research split reads."""

import copy
import hashlib
import json
import math
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
import yaml

import nop.train.experiments as experiments
from nop.data.dataset import sha256_file, write_arrays, write_json
from nop.data.normalize import fit_train_normalization

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def policy() -> dict[str, Any]:
    result: dict[str, Any] = yaml.safe_load((ROOT / "config/training.yaml").read_text())
    return result


def _records(policy: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            **draw,
            "config_id": draw["trial_id"],
            "status": "completed",
            "parameter_count": 1_000_000,
            "validation_metrics": {
                "mean_absolute_error": 1.0,
                "p99_absolute_error": 2.0,
                "prediction_failure_count": 0,
            },
        }
        for draw in experiments.paired_trial_draws(policy)
    ]


def test_paired_draws_have_the_frozen_budget_stream_order_and_bounds(
    policy: dict[str, Any],
) -> None:
    draws = experiments.paired_trial_draws(policy)
    assert len(draws) == 12
    assert [draw["trial_id"] for draw in draws] == [f"trial_{index:02d}" for index in range(12)]
    generator = np.random.Generator(np.random.PCG64(101))
    for draw in draws:
        for name, low, high in (
            ("learning_rate", 1.0e-4, 3.0e-3),
            ("weight_decay", 1.0e-6, 1.0e-2),
        ):
            expected = float(np.exp(generator.uniform(math.log(low), math.log(high))))
            assert draw[name] == expected
            assert low <= draw[name] <= high
    for model in policy["model_ids"]:
        architecture_policy = copy.deepcopy(policy)
        architecture_policy["active_model"] = model
        assert experiments.paired_trial_draws(architecture_policy) == draws


def test_build_settings_uses_full_frozen_cap_and_all_declared_seeds(
    policy: dict[str, Any],
) -> None:
    draw = experiments.paired_trial_draws(policy)[0]
    for seed in [101, 11, 29, 47]:
        settings = experiments.build_train_settings(
            policy, draw["learning_rate"], draw["weight_decay"], seed, "cpu"
        )
        assert settings.seed == seed
        assert settings.learning_rate == draw["learning_rate"]
        assert settings.weight_decay == draw["weight_decay"]
        assert settings.max_epochs == 100
        assert settings.batch_size == 256
        assert settings.warmup_epochs == 5
        assert settings.patience_epochs == 10
        assert settings.gradient_clip_norm == 1.0
        assert settings.minimum_learning_rate_fraction == 0.01
        assert settings.minimum_improvement_vol_points == 0
        assert settings.betas == (0.9, 0.999)
        assert settings.epsilon == 1.0e-8
        assert settings.max_optimizer_steps is None


@pytest.mark.parametrize(
    ("learning_rate", "weight_decay", "seed"),
    [(1.0e-2, 1.0e-4, 101), (1.0e-3, 0.0, 101), (1.0e-3, 1.0e-4, 99)],
)
def test_outside_budget_settings_are_rejected(
    policy: dict[str, Any], learning_rate: float, weight_decay: float, seed: int
) -> None:
    with pytest.raises(ValueError, match="outside"):
        experiments.build_train_settings(policy, learning_rate, weight_decay, seed)


def test_trial_selection_applies_each_tie_break_in_order(policy: dict[str, Any]) -> None:
    records = _records(policy)
    assert experiments.select_trial(records[::-1], policy)["trial_id"] == "trial_00"
    records[8]["parameter_count"] -= 1
    assert experiments.select_trial(records, policy)["trial_id"] == "trial_08"
    records[5]["validation_metrics"]["p99_absolute_error"] = 1.9
    assert experiments.select_trial(records, policy)["trial_id"] == "trial_05"
    records[3]["validation_metrics"]["mean_absolute_error"] = 0.9
    assert experiments.select_trial(records, policy)["trial_id"] == "trial_03"


def test_failed_trials_consume_budget_and_never_compete(policy: dict[str, Any]) -> None:
    records = _records(policy)
    records[0].update(status="failed", validation_metrics=None)
    records[1]["status"] = "early_stopped"
    assert experiments.select_trial(records, policy)["trial_id"] == "trial_01"
    for record in records:
        record.update(status="failed", validation_metrics=None)
    with pytest.raises(ValueError, match="every declared trial failed"):
        experiments.select_trial(records, policy)


@pytest.mark.parametrize("change", ["partial", "extra", "duplicate", "running", "replacement"])
def test_partial_or_replaced_budget_cannot_select(policy: dict[str, Any], change: str) -> None:
    records = _records(policy)
    if change == "partial":
        records.pop()
    elif change == "extra":
        records.append({**records[0], "trial_id": "trial_12"})
    elif change == "duplicate":
        records[-1] = records[0]
    elif change == "running":
        records[0]["status"] = "running"
    else:
        records[0]["learning_rate"] = records[1]["learning_rate"]
    with pytest.raises(ValueError, match="budget|terminal|paired draw"):
        experiments.select_trial(records, policy)


@pytest.mark.parametrize("invalid", [None, float("nan"), float("inf"), -0.1, True])
def test_nonfinite_or_missing_successful_metrics_are_rejected(
    policy: dict[str, Any], invalid: Any
) -> None:
    records = _records(policy)
    records[0]["validation_metrics"]["mean_absolute_error"] = invalid
    with pytest.raises(ValueError, match="finite validation metrics"):
        experiments.select_trial(records, policy)


def _rehash_dataset(path: Path, manifest: dict[str, Any], frozen: dict[str, Any]) -> None:
    for name in ("normalization", "grid"):
        manifest[f"{name}_sha256"] = sha256_file(path / f"{name}.npz")
    hashes = [
        record["sha256"]
        for split in ("train", "validation")
        for record in manifest["splits"][split]["chunks"]
    ] + [manifest["normalization_sha256"], manifest["grid_sha256"]]
    manifest["dataset_hash"] = hashlib.sha256(json.dumps(hashes).encode()).hexdigest()
    write_json(path / "manifest.json", manifest)
    frozen["evidence"]["dataset"] = {
        "dataset_hash": manifest["dataset_hash"],
        "manifest_sha256": sha256_file(path / "manifest.json"),
    }


@pytest.fixture
def synthetic_dataset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Path, dict[str, Any], dict[str, Any], dict[str, torch.Tensor]]:
    path = tmp_path / "invented"
    path.mkdir()
    metadata = {"fixture": "invented_arrays_no_pricer_no_research_data"}
    manifest: dict[str, Any] = {
        "complete": True,
        "scope": "train_validation_only",
        "seed": 17,
        "metadata": metadata,
        "splits": {},
    }
    training: dict[str, torch.Tensor] = {}
    for split, count, shift in (("train", 4, 0.0), ("validation", 2, 0.6)):
        parameters = torch.arange(count * 5, dtype=torch.float64).reshape(count, 5) + shift
        iv = torch.arange(1, count + 1, dtype=torch.float64)[:, None, None] / 10 + shift
        iv = iv.expand(count, 8, 15).clone()
        status = torch.zeros_like(iv, dtype=torch.int64)
        iv[0, 0, 0], status[0, 0, 0] = float("nan"), 1
        values = {"parameters": parameters, "iv": iv, "status": status}
        values.update({name: torch.zeros_like(iv) for name in ("price", "vega", "discrepancy")})
        values["terms"] = torch.ones_like(status)
        (path / split).mkdir()
        archive = path / split / "00000000.npz"
        write_arrays(archive, {name: value.numpy() for name, value in values.items()})
        record = {
            "file": archive.name,
            "metadata": metadata,
            "start": 0,
            "count": count,
            "sha256": sha256_file(archive),
        }
        write_json(archive.with_suffix(".json"), record)
        manifest["splits"][split] = {"count": count, "chunks": [record]}
        if split == "train":
            training = values
    normalization = fit_train_normalization(
        training["parameters"], training["iv"], training["status"] == 0
    )
    archive_metadata = np.array(json.dumps({"seed": 17, "fit_split": "train", **metadata}))
    write_arrays(
        path / "normalization.npz",
        {name: value.numpy() for name, value in asdict(normalization).items()}
        | {"metadata": archive_metadata},
    )
    write_arrays(
        path / "grid.npz",
        {name: np.ones((8, 15)) for name in ("maturity", "z", "k", "strike")}
        | {"metadata": archive_metadata},
    )
    frozen: dict[str, Any] = {"evidence": {}}
    _rehash_dataset(path, manifest, frozen)
    monkeypatch.setattr(experiments, "verify_frozen_files", lambda root: frozen)
    return path, manifest, frozen, training


def test_data_loader_verifies_only_development_splits_and_reuses_train_moments(
    synthetic_dataset: tuple[Path, dict[str, Any], dict[str, Any], dict[str, torch.Tensor]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path, manifest, _, raw = synthetic_dataset
    opened: list[str] = []
    original = experiments.load_split

    def record_split(path: Path, split: str) -> dict[str, torch.Tensor]:
        assert split in ("train", "validation")
        opened.append(split)
        return original(path, split)

    monkeypatch.setattr(experiments, "load_split", record_split)
    train, validation, metadata = experiments.load_development_data(path.parent, path)
    assert opened == ["train", "validation"]
    assert train.targets.shape == (4, 8, 15)
    assert validation.targets.shape == (2, 8, 15)
    assert train.parameters.dtype == train.targets.dtype == torch.float32
    assert validation.parameters.dtype == validation.targets.dtype == torch.float32
    assert torch.equal(train.target_mean, validation.target_mean)
    assert torch.equal(train.target_std, validation.target_std)
    torch.testing.assert_close(train.parameters.mean(dim=0), torch.zeros(5), atol=1e-7, rtol=0)
    assert validation.targets[validation.valid].mean() > 4
    assert torch.isnan(train.targets[~train.valid]).all()
    reconstructed = train.targets * train.target_std + train.target_mean
    torch.testing.assert_close(reconstructed, raw["iv"].float(), equal_nan=True)
    assert metadata["dataset_hash"] == manifest["dataset_hash"]
    assert metadata["normalization_metadata"]["fit_split"] == "train"
    json.dumps(metadata, allow_nan=False)


@pytest.mark.parametrize("archive", ["normalization.npz", "grid.npz", "train/00000000.npz"])
def test_corrupt_archives_are_rejected(
    synthetic_dataset: tuple[Path, dict[str, Any], dict[str, Any], dict[str, torch.Tensor]],
    archive: str,
) -> None:
    path, _, _, _ = synthetic_dataset
    with (path / archive).open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="corrupt"):
        experiments.load_development_data(path.parent, path)


@pytest.mark.parametrize("change", ["false_combined_hash", "different_frozen_hash", "incomplete"])
def test_gate1_identity_must_pass_before_loading_splits(
    synthetic_dataset: tuple[Path, dict[str, Any], dict[str, Any], dict[str, torch.Tensor]],
    monkeypatch: pytest.MonkeyPatch,
    change: str,
) -> None:
    path, manifest, frozen, _ = synthetic_dataset
    if change == "false_combined_hash":
        manifest["splits"]["train"]["chunks"][0]["sha256"] = "invented"
    elif change == "different_frozen_hash":
        frozen["evidence"]["dataset"]["dataset_hash"] = "different"
    else:
        manifest["complete"] = False
    write_json(path / "manifest.json", manifest)

    def reject_load(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("invalid identity reached split loading")

    monkeypatch.setattr(experiments, "load_split", reject_load)
    with pytest.raises(ValueError, match="verified Gate 1 dataset identity"):
        experiments.load_development_data(path.parent, path)


@pytest.mark.parametrize("archive", ["normalization", "grid"])
def test_archive_provenance_cannot_claim_validation_fitted_moments(
    synthetic_dataset: tuple[Path, dict[str, Any], dict[str, Any], dict[str, torch.Tensor]],
    archive: str,
) -> None:
    path, manifest, frozen, _ = synthetic_dataset
    with np.load(path / f"{archive}.npz", allow_pickle=False) as saved:
        arrays = dict(saved)
    metadata = json.loads(arrays["metadata"].item())
    metadata["fit_split"] = "validation"
    arrays["metadata"] = np.array(json.dumps(metadata))
    write_arrays(path / f"{archive}.npz", arrays)
    _rehash_dataset(path, manifest, frozen)
    with pytest.raises(ValueError, match="metadata disagrees"):
        experiments.load_development_data(path.parent, path)


def test_informational_manifest_commit_can_change_without_invalidating_data(
    synthetic_dataset: tuple[Path, dict[str, Any], dict[str, Any], dict[str, torch.Tensor]],
) -> None:
    path, manifest, frozen, _ = synthetic_dataset
    manifest["code_commit"] = "later_source_identical_commit"
    write_json(path / "manifest.json", manifest)
    _, _, metadata = experiments.load_development_data(path.parent, path)
    assert metadata["dataset_manifest_sha256"] != frozen["evidence"]["dataset"]["manifest_sha256"]
    assert metadata["dataset_hash"] == frozen["evidence"]["dataset"]["dataset_hash"]


def test_run_lock_rejects_overlap_and_releases_after_exception(tmp_path: Path) -> None:
    path = tmp_path / "experiment.lock"
    with pytest.raises(ValueError, match="interrupted"):
        with experiments.run_lock(path):
            with pytest.raises(RuntimeError, match="another experiment process"):
                with experiments.run_lock(path):
                    pytest.fail("overlapping writer acquired the lock")
            raise ValueError("interrupted")
    with experiments.run_lock(path):
        assert path.is_file()


def test_os_releases_run_lock_when_holder_is_killed(tmp_path: Path) -> None:
    path = tmp_path / "crashed.lock"
    code = (
        "import fcntl, sys; "
        "stream = open(sys.argv[1], 'a+'); "
        "fcntl.flock(stream.fileno(), fcntl.LOCK_EX); "
        "print('locked', flush=True); "
        "sys.stdin.read()"
    )
    process = subprocess.Popen(
        [sys.executable, "-c", code, str(path)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        assert process.stdout is not None
        assert process.stdout.readline().strip() == "locked"
        with pytest.raises(RuntimeError, match="another experiment process"):
            with experiments.run_lock(path):
                pytest.fail("overlapping writer acquired the lock")
        process.kill()
        process.wait(timeout=10)
        with experiments.run_lock(path):
            pass
    finally:
        if process.poll() is None:
            process.kill()
        process.communicate(timeout=10)
