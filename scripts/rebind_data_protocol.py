"""Rebind verified train/validation metadata before freeze; never regenerate labels.

The output must be new or a fully verified prior result of this exact migration.
Interrupted work lives only in an unpublished temporary sibling directory. Never
resume a partial output: retain it for inspection and choose a new output path.
"""

import argparse
import copy
import hashlib
import io
import json
import tempfile
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from nop.data.dataset import provenance, sha256_file, write_json

SPLITS = ("train", "validation")
CHUNK_FIELDS = {"parameters", "iv", "status", "price", "vega", "discrepancy", "terms"}
AUXILIARY_FIELDS = {
    "normalization": {"parameter_mean", "parameter_std", "iv_mean", "iv_std", "iv_count"},
    "grid": {"maturity", "z", "k", "strike"},
}


def _unfrozen(root: Path) -> None:
    if any((root / "frozen").glob("*.json")):
        raise ValueError("protocol migration is forbidden after a frozen JSON manifest exists")


def _file(path: Path) -> None:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"missing, nonregular, or symlinked artifact: {path}")


def _json(path: Path) -> dict[str, Any]:
    _file(path)
    value: Any = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _archive(path: Path, metadata: dict[str, Any], fields: set[str]) -> dict[str, Any]:
    _file(path)
    payloads = {}
    with zipfile.ZipFile(path) as archive, np.load(path, allow_pickle=False) as arrays:
        names = archive.namelist()
        if len(names) != len(set(names)) or set(names) != {
            name + ".npy" for name in fields | {"metadata"}
        }:
            raise ValueError(f"unexpected or duplicate archive members: {path}")
        embedded = arrays["metadata"]
        if (
            embedded.shape != ()
            or embedded.dtype.kind != "U"
            or json.loads(str(embedded)) != metadata
        ):
            raise ValueError(f"embedded metadata mismatch: {path}")
        for name in sorted(fields):
            array = arrays[name]
            if array.dtype.hasobject:
                raise ValueError(f"object arrays are forbidden: {path}:{name}")
            payloads[name] = {
                "dtype": array.dtype.str,
                "shape": list(array.shape),
                "raw_sha256": hashlib.sha256(array.tobytes(order="C")).hexdigest(),
                "npy_sha256": hashlib.sha256(archive.read(name + ".npy")).hexdigest(),
            }
    return payloads


def _dataset_hash(manifest: dict[str, Any]) -> str:
    hashes = [
        chunk["sha256"] for split in SPLITS for chunk in manifest["splits"][split]["chunks"]
    ] + [manifest["normalization_sha256"], manifest["grid_sha256"]]
    return hashlib.sha256(json.dumps(hashes).encode()).hexdigest()


def _verify_dataset(
    directory: Path, config: dict[str, Any], metadata: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    if directory.is_symlink() or not directory.is_dir():
        raise ValueError(f"dataset must be a regular directory: {directory}")
    allowed = {"manifest.json", "normalization.npz", "grid.npz", "protocol_migration.json", *SPLITS}
    # Inspect names only; never descend into an unexpected/holdout directory.
    if {path.name for path in directory.iterdir()} - allowed:
        raise ValueError("unexpected dataset entry; only train/validation artifacts are permitted")
    manifest = _json(directory / "manifest.json")
    if (
        manifest.get("version") != 1
        or manifest.get("complete") is not True
        or manifest.get("scope") != "train_validation_only"
        or manifest.get("metadata") != metadata
        or manifest.get("seed") != config["data_design"]["id_stream_seed"]
        or set(manifest.get("splits", {})) != set(SPLITS)
    ):
        raise ValueError("incomplete dataset, unexpected scope, or provenance mismatch")
    payloads: dict[str, Any] = {}
    seed = config["data_design"]["id_stream_seed"]
    chunk_size = config["storage"]["chunk_size"]
    for split in SPLITS:
        split_dir = directory / split
        if split_dir.is_symlink() or not split_dir.is_dir():
            raise ValueError("split must be a regular directory")
        recipe = config["data_design"]["id_splits"][split]
        record = manifest["splits"][split]
        if record["count"] != recipe["count"] or record["offset"] != recipe["offset"]:
            raise ValueError("manifest split does not match the data configuration")
        expected_starts = list(range(0, recipe["count"], chunk_size))
        if len(record["chunks"]) != len(expected_starts):
            raise ValueError("noncanonical or incomplete chunk list")
        expected_files: set[str] = set()
        for start, chunk in zip(expected_starts, record["chunks"], strict=True):
            count = min(chunk_size, recipe["count"] - start)
            if (
                type(chunk["start"]) is not int
                or type(chunk["count"]) is not int
                or chunk["start"] != start
                or chunk["count"] != count
                or chunk["file"] != f"{start:08d}.npz"
            ):
                raise ValueError("noncanonical or noncontiguous manifest chunk")
            path = split_dir / chunk["file"]
            expected_files.update((path.name, path.with_suffix(".json").name))
            sidecar = _json(path.with_suffix(".json"))
            if sidecar != {**chunk, "metadata": metadata}:
                raise ValueError("sidecar and manifest disagree")
            _file(path)
            if sha256_file(path) != chunk["sha256"]:
                raise ValueError(f"archive hash mismatch: {path}")
            payloads[str(path.relative_to(directory))] = _archive(
                path, {"seed": seed, **metadata}, CHUNK_FIELDS
            )
        if {path.name for path in split_dir.iterdir()} != expected_files:
            raise ValueError("unexpected or missing split artifacts")
    for name, fields in AUXILIARY_FIELDS.items():
        path = directory / f"{name}.npz"
        _file(path)
        if sha256_file(path) != manifest[f"{name}_sha256"]:
            raise ValueError(f"archive hash mismatch: {path}")
        payloads[path.name] = _archive(
            path, {"seed": seed, "fit_split": "train", **metadata}, fields
        )
    if _dataset_hash(manifest) != manifest.get("dataset_hash"):
        raise ValueError("combined dataset hash mismatch")
    return manifest, payloads


def _copy_archive(source: Path, destination: Path, protocol_hash: str) -> None:
    with zipfile.ZipFile(source) as old, zipfile.ZipFile(destination, "w") as new:
        for entry in old.infolist():
            content = old.read(entry.filename)
            if entry.filename == "metadata.npy":
                metadata = json.loads(str(np.load(io.BytesIO(content), allow_pickle=False)))
                metadata["protocol_hash"] = protocol_hash
                buffer = io.BytesIO()
                np.save(buffer, np.array(json.dumps(metadata, sort_keys=True)), allow_pickle=False)
                content = buffer.getvalue()
            new.writestr(entry, content)


def _verify_unchanged_arrays(source: Path, output: Path, payloads: dict[str, Any]) -> None:
    for relative, fields in payloads.items():
        with (
            np.load(source / relative, allow_pickle=False) as before,
            np.load(output / relative, allow_pickle=False) as after,
        ):
            for name in fields:
                left, right = before[name], after[name]
                if (
                    left.dtype != right.dtype
                    or left.shape != right.shape
                    or left.tobytes(order="C") != right.tobytes(order="C")
                ):
                    raise ValueError(f"nonmetadata array changed: {relative}:{name}")


def _proof(
    source: Path,
    before: dict[str, Any],
    after: dict[str, Any],
    payloads: dict[str, Any],
) -> dict[str, Any]:
    return {
        "version": 1,
        "operation": "prefreeze_protocol_metadata_only",
        "scope": "train_validation_only",
        "source_manifest_sha256": sha256_file(source / "manifest.json"),
        "source_dataset_hash": before["dataset_hash"],
        "output_dataset_hash": after["dataset_hash"],
        "prior_protocol_hash": before["metadata"]["protocol_hash"],
        "protocol_hash": after["metadata"]["protocol_hash"],
        "config_hash": before["metadata"]["config_hash"],
        "source_code_commit": before["code_commit"],
        "verification": "all nonmetadata array dtypes, shapes, raw bytes and NPY bytes identical",
        "unchanged_arrays": payloads,
    }


def rebind_protocol(
    *,
    root: Path,
    source: Path,
    output: Path,
    prior_protocol: Path,
    protocol: Path,
    config_path: Path,
) -> dict[str, Any]:
    """Publish a verified metadata-only copy, refusing freezes and provenance drift."""
    _unfrozen(root)
    if source.is_symlink() or output.is_symlink():
        raise ValueError("source and output must not be symlinks")
    source, output = source.resolve(), output.resolve()
    if source == output or source in output.parents or output in source.parents:
        raise ValueError("migration requires a separate, nonoverlapping output directory")
    config = yaml.safe_load(config_path.read_bytes())
    old_protocol = yaml.safe_load(prior_protocol.read_bytes())
    new_protocol = yaml.safe_load(protocol.read_bytes())
    for key in ("data_design", "units"):
        if old_protocol[key] != new_protocol[key]:
            raise ValueError(f"metadata-only migration cannot change protocol {key}")
    if (
        config["data_design"] != old_protocol["data_design"]
        or config["units"]
        != {key: old_protocol["units"][key] for key in ("spot", "rate", "dividend_yield")}
        or any(
            item["holdout_access"]["gate1_materialization"] != "train_and_validation_only"
            for item in (old_protocol, new_protocol)
        )
    ):
        raise ValueError("config/protocol design, units, or holdout access mismatch")
    old_hash, new_hash = sha256_file(prior_protocol), sha256_file(protocol)
    if old_hash == new_hash:
        raise ValueError("protocol hashes are identical; no migration is needed")
    torch.set_num_threads(config["storage"]["torch_threads"])
    old_metadata = provenance(root, config_path) | {"protocol_hash": old_hash}
    new_metadata = {**old_metadata, "protocol_hash": new_hash}
    before, payloads = _verify_dataset(source, config, old_metadata)
    expected = copy.deepcopy(before)
    expected["metadata"] = new_metadata

    def verify_result(directory: Path) -> dict[str, Any]:
        after, migrated_payloads = _verify_dataset(directory, config, new_metadata)
        if migrated_payloads != payloads:
            raise ValueError("nonmetadata array payload changed")
        _verify_unchanged_arrays(source, directory, payloads)
        comparable = copy.deepcopy(after)
        # make_data records its current commit on replay; content identity is the
        # unchanged source hashes. The proof retains the original historical SHA.
        comparable["code_commit"] = before["code_commit"]
        for split in SPLITS:
            for chunk in comparable["splits"][split]["chunks"]:
                original = before["splits"][split]["chunks"][
                    chunk["start"] // config["storage"]["chunk_size"]
                ]
                chunk["sha256"] = original["sha256"]
        for key in ("dataset_hash", "grid_sha256", "normalization_sha256"):
            comparable[key] = before[key]
        if comparable != expected:
            raise ValueError("manifest changed beyond protocol/hash metadata")
        proof = _proof(source, before, after, payloads)
        if _json(directory / "protocol_migration.json") != proof:
            raise ValueError("migration proof mismatch")
        return proof

    if output.exists():
        return verify_result(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{output.name}.rebind-", dir=output.parent
    ) as temporary:
        stage = Path(temporary)
        after = copy.deepcopy(expected)
        for split in SPLITS:
            (stage / split).mkdir()
            for chunk in after["splits"][split]["chunks"]:
                relative = Path(split) / chunk["file"]
                _copy_archive(source / relative, stage / relative, new_hash)
                chunk["sha256"] = sha256_file(stage / relative)
                write_json(
                    (stage / relative).with_suffix(".json"), {**chunk, "metadata": new_metadata}
                )
        for name in AUXILIARY_FIELDS:
            relative = Path(f"{name}.npz")
            _copy_archive(source / relative, stage / relative, new_hash)
            after[f"{name}_sha256"] = sha256_file(stage / relative)
        after["dataset_hash"] = _dataset_hash(after)
        write_json(stage / "manifest.json", after)
        proof = _proof(source, before, after, payloads)
        write_json(stage / "protocol_migration.json", proof)
        verify_result(stage)
        _unfrozen(root)
        if output.exists():
            raise ValueError("output appeared during migration; refusing to replace it")
        stage.rename(output)
    return proof


def main(argv: list[str] | None = None) -> int:
    """Run the explicit pre-freeze migration and print its verified identities."""
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "output", "prior-protocol", "protocol", "config"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    args = parser.parse_args(argv)
    result = rebind_protocol(
        root=Path(__file__).resolve().parents[1],
        source=args.source,
        output=args.output,
        prior_protocol=args.prior_protocol,
        protocol=args.protocol,
        config_path=args.config,
    )
    print(
        json.dumps(
            {key: result[key] for key in ("source_dataset_hash", "output_dataset_hash")}, indent=2
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
