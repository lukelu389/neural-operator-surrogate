"""Seven-surface synthetic fixtures only; never run the production data command."""

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
import yaml

import nop.data.dataset as dataset


def _small_config() -> dict[str, Any]:
    root = Path(__file__).resolve().parents[1]
    config: dict[str, Any] = yaml.safe_load((root / "config/data.yaml").read_text())
    design = config["data_design"]
    design["id_stream_seed"] = 777
    design["id_box"] = {
        "v0": [0.04, 0.08],
        "kappa": [1.0, 2.0],
        "theta": [0.04, 0.08],
        "sigma": [0.1, 0.2],
        "rho": [-0.7, -0.3],
    }
    design["id_splits"] = {
        "train": {"offset": 0, "count": 4},
        "validation": {"offset": 4, "count": 3},
        "test": {"offset": 999, "count": 2},
    }
    # Inert synthetic sentinels, with no production holdout recipes retained.
    design["ood_sets"] = {"synthetic_inert_recipe": {"seed": 888, "count": 2}}
    design["main_grid"]["maturity_year_fractions"] = [[1, 2], [1, 1]]
    design["main_grid"]["z_linspace"] = {"start": -0.5, "stop": 0.5, "count": 3}
    config["storage"].update(chunk_size=2, pricing_batch_size=2, workers=1, torch_threads=1)
    return config


def _metadata(config: dict[str, Any]) -> dict[str, Any]:
    serialized = json.dumps(config, sort_keys=True).encode()
    return {
        "config_hash": hashlib.sha256(serialized).hexdigest(),
        "fixture": "seven_synthetic_surfaces_no_research_split",
    }


def _generate(config: dict[str, Any], output: Path) -> dict[str, Any]:
    return dataset.generate_dataset(config, output, _metadata(config), code_commit="test-fixture")


def test_cold_generations_have_identical_hashes_and_only_requested_slices(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _small_config()
    sampled: list[tuple[int, int]] = []
    original = dataset.sample_id_parameters

    def record_slice(
        count: int, offset: int, seed: int, bounds: tuple[tuple[float, float], ...]
    ) -> torch.Tensor:
        sampled.append((count, offset))
        return original(count, offset, seed, bounds)

    monkeypatch.setattr(dataset, "sample_id_parameters", record_slice)
    first_output, second_output = tmp_path / "first", tmp_path / "second"
    first = _generate(config, first_output)
    second = _generate(config, second_output)
    assert first["complete"] and second["complete"]
    assert first["dataset_hash"] == second["dataset_hash"]
    assert first["normalization_sha256"] == second["normalization_sha256"]
    assert first["grid_sha256"] == second["grid_sha256"]
    assert sampled == [(2, 0), (2, 2), (2, 4), (1, 6)] * 2
    assert set(first["splits"]) == {"train", "validation"}
    assert not (first_output / "test").exists()
    assert not (first_output / "synthetic_inert_recipe").exists()
    for split, count in (("train", 4), ("validation", 3)):
        chunks = first["splits"][split]["chunks"]
        assert [chunk["sha256"] for chunk in chunks] == [
            chunk["sha256"] for chunk in second["splits"][split]["chunks"]
        ]
        for chunk in chunks:
            relative = Path(split) / chunk["file"]
            assert (first_output / relative).read_bytes() == (second_output / relative).read_bytes()
        loaded = dataset.load_split(first_output, split)
        assert loaded["parameters"].shape == (count, 5)
        assert loaded["iv"].shape == (count, 2, 3)
        assert loaded["parameters"].dtype == torch.float64
        assert loaded["status"].dtype == torch.int64
        assert (loaded["status"] == 0).all()
        assert first["splits"][split]["summary"]["attempted"] == count * 6
        assert first["splits"][split]["summary"]["failed"] == 0


def test_verified_cache_does_not_reprice(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config = _small_config()
    first = _generate(config, tmp_path)

    def must_not_price(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("verified cache unexpectedly repriced a surface")

    monkeypatch.setattr(dataset, "label_surfaces", must_not_price)
    second = _generate(config, tmp_path)
    assert second["dataset_hash"] == first["dataset_hash"]
    assert second["splits"] == first["splits"]


def test_parallel_workers_preserve_cold_artifact_bytes(tmp_path: Path) -> None:
    serial = _small_config()
    parallel = copy.deepcopy(serial)
    parallel["storage"]["workers"] = 2
    # Hold metadata fixed to isolate execution strategy from the config-file hash.
    metadata = _metadata(serial)
    first = dataset.generate_dataset(serial, tmp_path / "serial", metadata, code_commit="fixture")
    second = dataset.generate_dataset(
        parallel, tmp_path / "parallel", metadata, code_commit="fixture"
    )
    assert first["dataset_hash"] == second["dataset_hash"]
    for split in ("train", "validation"):
        assert first["splits"][split]["chunks"] == second["splits"][split]["chunks"]


def test_chunk_corruption_is_rejected_by_loader_and_resume(tmp_path: Path) -> None:
    config = _small_config()
    manifest = _generate(config, tmp_path)
    chunk = tmp_path / "train" / manifest["splits"]["train"]["chunks"][0]["file"]
    with chunk.open("ab") as stream:
        stream.write(b"corrupt-fixture")
    with pytest.raises(ValueError, match="corrupt cached artifact"):
        dataset.load_split(tmp_path, "train")
    with pytest.raises(ValueError, match="corrupt cached artifact"):
        _generate(config, tmp_path)


def test_resume_rejects_changed_provenance(tmp_path: Path) -> None:
    config = _small_config()
    _generate(config, tmp_path)
    changed = _metadata(config) | {"config_hash": "different-fixture"}
    with pytest.raises(ValueError, match="provenance mismatch"):
        dataset.generate_dataset(config, tmp_path, changed, code_commit="test-fixture")


@pytest.mark.parametrize("malformation", ["missing", "duplicate", "path_escape", "hash"])
def test_loader_rejects_malformed_manifest_chunks(tmp_path: Path, malformation: str) -> None:
    manifest = _generate(_small_config(), tmp_path)
    chunks = manifest["splits"]["train"]["chunks"]
    if malformation == "missing":
        chunks.pop()
    elif malformation == "duplicate":
        chunks[1] = copy.deepcopy(chunks[0])
    elif malformation == "path_escape":
        chunks[0]["file"] = "../validation/00000000.npz"
    else:
        chunks[0]["sha256"] = "0" * 64
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        dataset.load_split(tmp_path, "train")


def test_resume_rejects_sidecar_filename_escape(tmp_path: Path) -> None:
    config = _small_config()
    manifest = _generate(config, tmp_path)
    chunk = tmp_path / "train" / manifest["splits"]["train"]["chunks"][0]["file"]
    sidecar = chunk.with_suffix(".json")
    record = json.loads(sidecar.read_text())
    record["file"] = "../validation/00000000.npz"
    sidecar.write_text(json.dumps(record))
    with pytest.raises(ValueError):
        _generate(config, tmp_path)


@pytest.mark.parametrize("overlap", ["train_validation", "train_test", "validation_test"])
def test_overlapping_sobol_windows_are_rejected_before_sampling(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, overlap: str
) -> None:
    config = _small_config()
    splits = config["data_design"]["id_splits"]
    if overlap == "train_validation":
        splits["validation"]["offset"] = 3
    elif overlap == "train_test":
        splits["train"]["offset"] = 999
    else:
        splits["validation"]["offset"] = 998

    def must_not_sample(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("overlapping fixture windows reached parameter sampling")

    monkeypatch.setattr(dataset, "sample_id_parameters", must_not_sample)
    with pytest.raises(ValueError):
        _generate(config, tmp_path)


@pytest.mark.parametrize(
    "split", ["test", "wider_parameters", "short_maturities", "parameter_corners", "../test"]
)
def test_loader_refuses_holdouts_before_filesystem_access(tmp_path: Path, split: str) -> None:
    with pytest.raises(ValueError, match="only train and validation"):
        dataset.load_split(tmp_path / "does-not-exist", split)


def test_validation_changes_do_not_change_fitted_training_moments(tmp_path: Path) -> None:
    config = _small_config()
    changed = copy.deepcopy(config)
    changed["data_design"]["id_splits"]["validation"]["offset"] = 7
    first, second = tmp_path / "first", tmp_path / "changed-validation"
    _generate(config, first)
    _generate(changed, second)
    validation_before = dataset.load_split(first, "validation")
    validation_after = dataset.load_split(second, "validation")
    assert not torch.equal(validation_before["parameters"], validation_after["parameters"])
    assert not torch.equal(validation_before["iv"], validation_after["iv"])
    with (
        np.load(first / "normalization.npz", allow_pickle=False) as before,
        np.load(second / "normalization.npz", allow_pickle=False) as after,
    ):
        for key in ("parameter_mean", "parameter_std", "iv_mean", "iv_std", "iv_count"):
            np.testing.assert_array_equal(before[key], after[key])
        train = dataset.load_split(first, "train")
        np.testing.assert_allclose(before["iv_mean"], train["iv"].numpy().mean(axis=0))
        np.testing.assert_allclose(before["iv_std"], train["iv"].numpy().std(axis=0, ddof=0))
        np.testing.assert_array_equal(before["iv_count"], np.full((2, 3), 4, dtype=np.int64))
