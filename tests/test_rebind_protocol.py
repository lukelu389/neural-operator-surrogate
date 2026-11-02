"""Metadata migration uses seven synthetic surfaces, never production/holdout data."""

import copy
import hashlib
import json
import runpy
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
import yaml

from nop.data.dataset import generate_dataset, provenance, sha256_file, write_arrays, write_json

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = runpy.run_path(str(ROOT / "scripts/rebind_data_protocol.py"))
REBIND = SCRIPT["rebind_protocol"]


@pytest.fixture(scope="module")
def synthetic_source(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("protocol-migration")
    for name in ("src/nop/reproducibility.py", "scripts/make_data.py"):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# synthetic source provenance fixture\n")
    config = yaml.safe_load((ROOT / "config/data.yaml").read_text())
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
        "test": {"offset": 7, "count": 2},
    }
    design["ood_sets"] = {"synthetic_inert_recipe": {"seed": 888, "count": 2}}
    design["main_grid"]["maturity_year_fractions"] = [[1, 2], [1, 1]]
    design["main_grid"]["z_linspace"] = {"start": -0.5, "stop": 0.5, "count": 3}
    config["storage"].update(chunk_size=2, pricing_batch_size=2, workers=1, torch_threads=1)
    (root / "config.yaml").write_text(yaml.safe_dump(config))
    protocol = {
        "data_design": design,
        "units": {**config["units"], "implied_volatility": "decimal"},
        "holdout_access": {"gate1_materialization": "train_and_validation_only"},
        "status": "draft",
    }
    (root / "prior.yaml").write_text(yaml.safe_dump(protocol))
    protocol["status"] = "ready_for_freeze"
    (root / "protocol.yaml").write_text(yaml.safe_dump(protocol))
    torch.set_num_threads(1)
    metadata = provenance(root, root / "config.yaml") | {
        "protocol_hash": sha256_file(root / "prior.yaml")
    }
    generate_dataset(config, root / "source", metadata, code_commit="historical-fixture-commit")
    return root


@pytest.fixture
def migration_args(tmp_path: Path, synthetic_source: Path) -> dict[str, Path]:
    root = tmp_path / "project"
    shutil.copytree(synthetic_source, root)
    return {
        "root": root,
        "source": root / "source",
        "output": root / "output",
        "prior_protocol": root / "prior.yaml",
        "protocol": root / "protocol.yaml",
        "config_path": root / "config.yaml",
    }


def _snapshot(directory: Path) -> dict[str, bytes]:
    return {
        str(path.relative_to(directory)): path.read_bytes()
        for path in directory.rglob("*")
        if path.is_file()
    }


def _refresh_hashes(directory: Path) -> None:
    manifest = json.loads((directory / "manifest.json").read_text())
    hashes = []
    for split in ("train", "validation"):
        for chunk in manifest["splits"][split]["chunks"]:
            path = directory / split / chunk["file"]
            chunk["sha256"] = sha256_file(path)
            sidecar = json.loads(path.with_suffix(".json").read_text())
            sidecar["sha256"] = chunk["sha256"]
            write_json(path.with_suffix(".json"), sidecar)
            hashes.append(chunk["sha256"])
    for name in ("normalization", "grid"):
        manifest[f"{name}_sha256"] = sha256_file(directory / f"{name}.npz")
        hashes.append(manifest[f"{name}_sha256"])
    manifest["dataset_hash"] = hashlib.sha256(json.dumps(hashes).encode()).hexdigest()
    write_json(directory / "manifest.json", manifest)


def _change_archive(path: Path, *, metadata: bool = False) -> None:
    with np.load(path, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    if metadata:
        value = json.loads(str(arrays["metadata"]))
        value["protocol_hash"] = "0" * 64
        arrays["metadata"] = np.array(json.dumps(value))
    else:
        arrays["iv"][0, 0, 0] += 0.001
    write_arrays(path, arrays)


def test_migration_preserves_all_payloads_and_source_bytes(migration_args: dict[str, Path]) -> None:
    before = _snapshot(migration_args["source"])
    proof = REBIND(**migration_args)
    assert _snapshot(migration_args["source"]) == before
    output = migration_args["output"]
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["code_commit"] == "historical-fixture-commit"
    assert manifest["dataset_hash"] != proof["source_dataset_hash"]
    assert proof["output_dataset_hash"] == manifest["dataset_hash"]
    assert proof["prior_protocol_hash"] == sha256_file(migration_args["prior_protocol"])
    assert manifest["metadata"]["protocol_hash"] == sha256_file(migration_args["protocol"])
    assert set(manifest["splits"]) == {"train", "validation"}
    for relative, fields in proof["unchanged_arrays"].items():
        with (
            np.load(migration_args["source"] / relative, allow_pickle=False) as old,
            np.load(output / relative, allow_pickle=False) as new,
        ):
            old_meta, new_meta = (json.loads(str(item["metadata"])) for item in (old, new))
            assert new_meta == {**old_meta, "protocol_hash": proof["protocol_hash"]}
            for name in fields:
                assert old[name].dtype == new[name].dtype
                assert old[name].shape == new[name].shape
                assert old[name].tobytes() == new[name].tobytes()
    migrated_bytes = _snapshot(output)
    assert REBIND(**migration_args) == proof
    assert _snapshot(output) == migrated_bytes


def test_distinct_nan_payloads_survive_byte_for_byte(migration_args: dict[str, Path]) -> None:
    path = migration_args["source"] / "train/00000000.npz"
    with np.load(path, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    bits = arrays["iv"].view(np.uint64)
    bits[0, 0, 0] = 0x7FF8000000000001
    bits[0, 0, 1] = 0x7FF80000000000FF
    write_arrays(path, arrays)
    _refresh_hashes(migration_args["source"])
    REBIND(**migration_args)
    with np.load(migration_args["output"] / "train/00000000.npz", allow_pickle=False) as arrays:
        assert np.isnan(arrays["iv"][0, 0, :2]).all()
        assert arrays["iv"].view(np.uint64)[0, 0, :2].tolist() == [
            0x7FF8000000000001,
            0x7FF80000000000FF,
        ]


def test_proof_survives_regular_dataset_replay(
    migration_args: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    proof = REBIND(**migration_args)
    output = migration_args["output"]
    proof_bytes = (output / "protocol_migration.json").read_bytes()
    config = yaml.safe_load(migration_args["config_path"].read_text())
    metadata = json.loads((output / "manifest.json").read_text())["metadata"]

    def no_repricing(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("metadata migration must not reprice labels")

    monkeypatch.setattr("nop.data.dataset.label_surfaces", no_repricing)
    replay = generate_dataset(config, output, metadata, code_commit="later-fixture-commit")
    assert replay["dataset_hash"] == proof["output_dataset_hash"]
    assert (output / "protocol_migration.json").read_bytes() == proof_bytes
    assert REBIND(**migration_args) == proof


@pytest.mark.parametrize("field", ["data_design", "units", "holdout_access"])
def test_protocol_semantic_drift_is_rejected(migration_args: dict[str, Path], field: str) -> None:
    protocol = yaml.safe_load(migration_args["protocol"].read_text())
    if field == "data_design":
        protocol[field]["id_stream_seed"] += 1
    elif field == "units":
        protocol[field]["implied_volatility"] = "percent"
    else:
        protocol[field]["gate1_materialization"] = "include_test"
    migration_args["protocol"].write_text(yaml.safe_dump(protocol))
    with pytest.raises(ValueError, match="cannot change|holdout access"):
        REBIND(**migration_args)
    assert not migration_args["output"].exists()


@pytest.mark.parametrize("field", ["config", "source", "environment", "old_protocol"])
def test_provenance_drift_is_rejected(migration_args: dict[str, Path], field: str) -> None:
    if field == "config":
        with migration_args["config_path"].open("a") as stream:
            stream.write("# config byte drift\n")
    elif field == "source":
        (migration_args["root"] / "scripts/make_data.py").write_text("# changed source\n")
    elif field == "old_protocol":
        with migration_args["prior_protocol"].open("a") as stream:
            stream.write("# wrong old protocol bytes\n")
    else:
        manifest_path = migration_args["source"] / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["metadata"]["numpy"] = "fictional"
        write_json(manifest_path, manifest)
    with pytest.raises(ValueError, match="provenance"):
        REBIND(**migration_args)
    assert not migration_args["output"].exists()


@pytest.mark.parametrize(
    "kind",
    [
        "incomplete",
        "dataset_hash",
        "sidecar",
        "archive",
        "chunk_order",
        "chunk_path",
        "extra_split",
        "extra_file",
        "symlink",
    ],
)
def test_source_corruption_is_rejected(migration_args: dict[str, Path], kind: str) -> None:
    source = migration_args["source"]
    path = source / "manifest.json"
    manifest = json.loads(path.read_text())
    if kind == "incomplete":
        manifest["complete"] = False
    elif kind == "dataset_hash":
        manifest["dataset_hash"] = "0" * 64
    elif kind == "sidecar":
        sidecar = source / "train/00000000.json"
        value = json.loads(sidecar.read_text())
        value["metadata"]["protocol_hash"] = "0" * 64
        write_json(sidecar, value)
    elif kind == "archive":
        _change_archive(source / "train/00000000.npz")
    elif kind == "chunk_order":
        manifest["splits"]["train"]["chunks"].reverse()
    elif kind == "chunk_path":
        manifest["splits"]["train"]["chunks"][0]["file"] = "../test/00000000.npz"
    elif kind == "extra_split":
        manifest["splits"]["test"] = copy.deepcopy(manifest["splits"]["validation"])
    elif kind == "extra_file":
        (source / "test").mkdir()
    else:
        chunk = source / "train/00000000.npz"
        original = chunk.with_suffix(".backup")
        chunk.rename(original)
        chunk.symlink_to(original)
    write_json(path, manifest)
    with pytest.raises(ValueError):
        REBIND(**migration_args)
    assert not migration_args["output"].exists()


@pytest.mark.parametrize("relative", ["train/00000000.npz", "normalization.npz", "grid.npz"])
def test_embedded_protocol_tampering_is_rejected_even_with_refreshed_hashes(
    migration_args: dict[str, Path], relative: str
) -> None:
    _change_archive(migration_args["source"] / relative, metadata=True)
    _refresh_hashes(migration_args["source"])
    with pytest.raises(ValueError, match="embedded metadata"):
        REBIND(**migration_args)


@pytest.mark.parametrize("kind", ["empty", "proof", "array", "summary"])
def test_existing_stale_or_partial_output_is_never_overwritten(
    migration_args: dict[str, Path], kind: str
) -> None:
    output = migration_args["output"]
    if kind == "empty":
        output.mkdir()
    else:
        REBIND(**migration_args)
        if kind == "proof":
            (output / "protocol_migration.json").write_text("{}")
        elif kind == "array":
            _change_archive(output / "train/00000000.npz")
            _refresh_hashes(output)
        else:
            path = output / "manifest.json"
            manifest = json.loads(path.read_text())
            manifest["splits"]["train"]["summary"]["attempted"] += 1
            write_json(path, manifest)
    before = _snapshot(output)
    with pytest.raises(ValueError):
        REBIND(**migration_args)
    assert _snapshot(output) == before


def test_freeze_refusal_precedes_source_access(migration_args: dict[str, Path]) -> None:
    (migration_args["root"] / "frozen").mkdir()
    (migration_args["root"] / "frozen/fixture.json").write_text("{}")
    shutil.rmtree(migration_args["source"])
    with pytest.raises(ValueError, match="forbidden after"):
        REBIND(**migration_args)


@pytest.mark.parametrize("relationship", ["same", "inside", "parent"])
def test_output_cannot_replace_or_nest_source(
    migration_args: dict[str, Path], relationship: str
) -> None:
    source = migration_args["source"]
    migration_args["output"] = {"same": source, "inside": source / "new", "parent": source.parent}[
        relationship
    ]
    before = _snapshot(source)
    with pytest.raises(ValueError, match="nonoverlapping"):
        REBIND(**migration_args)
    assert _snapshot(source) == before


def test_interrupted_copy_never_publishes_partial_output(
    migration_args: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    original = REBIND.__globals__["_copy_archive"]
    calls = 0

    def interrupt(source: Path, destination: Path, protocol_hash: str) -> None:
        nonlocal calls
        original(source, destination, protocol_hash)
        calls += 1
        if calls == 2:
            raise RuntimeError("synthetic interruption")

    monkeypatch.setitem(REBIND.__globals__, "_copy_archive", interrupt)
    before = _snapshot(migration_args["source"])
    with pytest.raises(RuntimeError, match="synthetic interruption"):
        REBIND(**migration_args)
    assert not migration_args["output"].exists()
    assert not list(migration_args["output"].parent.glob(".output.rebind-*"))
    assert _snapshot(migration_args["source"]) == before
