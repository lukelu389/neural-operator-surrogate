"""Deterministic, resumable train/validation artifacts; no holdout materializer."""

import hashlib
import io
import json
import platform
import subprocess
import zipfile
from concurrent.futures import ProcessPoolExecutor
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from nop.data.grids import QuoteGrid, build_grid
from nop.data.labels import LabelSettings, LabelStatus, label_surfaces
from nop.data.normalize import fit_train_normalization
from nop.data.sampling import sample_id_parameters
from nop.pricing.iv import IVSettings, IVStatus
from nop.reproducibility import seed_everything


def sha256_file(path: Path) -> str:
    """Hash bytes without loading an entire artifact into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    """Atomically publish canonical, finite JSON."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def write_arrays(path: Path, arrays: dict[str, np.ndarray[Any, Any]]) -> None:
    """Use fixed ZIP metadata so repeated array artifacts have identical bytes."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_STORED) as archive:
        for name, array in sorted(arrays.items()):
            buffer = io.BytesIO()
            np.save(buffer, array, allow_pickle=False)
            entry = zipfile.ZipInfo(name + ".npy", date_time=(1980, 1, 1, 0, 0, 0))
            entry.compress_type = zipfile.ZIP_STORED
            entry.external_attr = 0o600 << 16
            archive.writestr(entry, buffer.getvalue())
    temporary.replace(path)


def provenance(root: Path, config_path: Path) -> dict[str, Any]:
    """Record source/config/environment identities independently of commit timing."""
    sources = (
        sorted((root / "src/nop/data").glob("*.py"))
        + sorted((root / "src/nop/pricing").glob("*.py"))
        + [root / "src/nop/reproducibility.py", root / "scripts/make_data.py"]
    )
    source_hashes = {str(path.relative_to(root)): sha256_file(path) for path in sources}
    return {
        "config_hash": sha256_file(config_path),
        "source_hashes": source_hashes,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "pyyaml": yaml.__version__,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "device": "cpu",
        "dtype": "float64",
        "torch_default_dtype": str(torch.get_default_dtype()),
        "torch_threads": torch.get_num_threads(),
    }


def _load_chunk(path: Path, metadata: dict[str, Any], start: int, count: int) -> dict[str, Any]:
    sidecar = path.with_suffix(".json")
    if not sidecar.exists() or not path.exists():
        raise FileNotFoundError(path)
    record: dict[str, Any] = json.loads(sidecar.read_text())
    if (
        record["metadata"] != metadata
        or record["start"] != start
        or record["count"] != count
        or record["file"] != path.name
    ):
        raise ValueError(f"cache provenance mismatch: {path}; choose a new output directory")
    if sha256_file(path) != record["sha256"]:
        raise ValueError(f"corrupt cached artifact: {path}; restore or remove the affected chunk")
    return record


def _status_summary(status: np.ndarray[Any, Any]) -> dict[str, Any]:
    names = {int(value): value.name for value in IVStatus} | {
        int(value): value.name for value in LabelStatus
    }
    codes, counts = np.unique(status, return_counts=True)
    failed = status != IVStatus.SUCCESS
    return {
        "attempted": int(status.size),
        "failed": int(failed.sum()),
        "failure_fraction": float(failed.mean()),
        "status_counts": {
            names[int(code)]: int(count) for code, count in zip(codes, counts, strict=True)
        },
        "failure_fraction_by_maturity": failed.mean(axis=(0, 2)).tolist(),
        "failure_fraction_by_z": failed.mean(axis=(0, 1)).tolist(),
        "failure_fraction_by_grid_point": failed.mean(axis=0).tolist(),
    }


def load_split(output: Path, split: str) -> dict[str, torch.Tensor]:
    """Load and verify a completed permitted split for training or validation."""
    if split not in ("train", "validation"):
        raise ValueError("only train and validation are available before final evaluation")
    manifest = json.loads((output / "manifest.json").read_text())
    if not manifest["complete"]:
        raise ValueError("dataset generation is incomplete")
    fields = ("parameters", "iv", "status", "price", "vega", "discrepancy", "terms")
    parts: dict[str, list[torch.Tensor]] = {key: [] for key in fields}
    split_record = manifest["splits"][split]
    expected_start = 0
    for chunk in split_record["chunks"]:
        if (
            chunk["start"] != expected_start
            or chunk["count"] <= 0
            or chunk["file"] != f"{expected_start:08d}.npz"
        ):
            raise ValueError("invalid or noncontiguous manifest chunk")
        expected_start += chunk["count"]
    if expected_start != split_record["count"]:
        raise ValueError("manifest chunk count does not match split count")
    for chunk in split_record["chunks"]:
        path = output / split / chunk["file"]
        record = _load_chunk(path, manifest["metadata"], chunk["start"], chunk["count"])
        if record["sha256"] != chunk["sha256"]:
            raise ValueError("manifest and chunk hashes differ")
        with np.load(path, allow_pickle=False) as arrays:
            for key in fields:
                parts[key].append(torch.from_numpy(arrays[key]))
    return {key: torch.cat(value) for key, value in parts.items()}


ChunkJob = tuple[dict[str, Any], dict[str, Any], Path, int, int, int]


def _label_inputs(
    config: dict[str, Any],
) -> tuple[QuoteGrid, LabelSettings, tuple[tuple[float, float], ...]]:
    design = config["data_design"]
    grid_config = design["main_grid"]
    maturity_values = tuple(a / b for a, b in grid_config["maturity_year_fractions"])
    z_config = grid_config["z_linspace"]
    z_values = tuple(
        torch.linspace(
            z_config["start"], z_config["stop"], z_config["count"], dtype=torch.float64
        ).tolist()
    )
    grid = build_grid(maturity_values, z_values, grid_config["k_scale"], **config["units"])
    label_config = dict(config["labels"])
    label_config.pop("option_type")
    label_config["term_levels"] = tuple(label_config["term_levels"])
    label_config["iv"] = IVSettings(**label_config["iv"])
    settings = LabelSettings(**label_config)
    bounds = tuple(
        (float(design["id_box"][name][0]), float(design["id_box"][name][1]))
        for name in design["parameter_order"]
    )
    return grid, settings, bounds


def _materialize_chunk(job: ChunkJob) -> dict[str, Any]:
    config, metadata, directory, offset, start, count = job
    # Rebuild small tensors inside each worker; avoid Torch shared-memory IPC.
    grid, settings, bounds = _label_inputs(config)
    path = directory / f"{start:08d}.npz"
    if path.exists() and path.with_suffix(".json").exists():
        return _load_chunk(path, metadata, start, count)
    seed = config["data_design"]["id_stream_seed"]
    seed_everything(seed)
    torch.set_num_threads(config["storage"]["torch_threads"])
    batch_size = config["storage"]["pricing_batch_size"]
    parameters = sample_id_parameters(count, offset + start, seed, bounds)
    buffers: dict[str, list[torch.Tensor]] = {
        key: [] for key in ("price", "iv", "vega", "discrepancy", "status", "terms")
    }
    for batch_start in range(0, count, batch_size):
        labels = label_surfaces(
            parameters[batch_start : batch_start + batch_size],
            grid.strike,
            grid.maturity[:, 0],
            **config["units"],
            settings=settings,
        )
        for key in buffers:
            buffers[key].append(getattr(labels, key))
    arrays = {key: torch.cat(value).numpy() for key, value in buffers.items()}
    arrays["parameters"] = parameters.numpy()
    arrays["metadata"] = np.array(json.dumps({"seed": seed, **metadata}, sort_keys=True))
    write_arrays(path, arrays)
    record = {
        "file": path.name,
        "start": start,
        "count": count,
        "sha256": sha256_file(path),
        "metadata": metadata,
    }
    write_json(path.with_suffix(".json"), record)
    return record


def generate_dataset(
    config: dict[str, Any], output: Path, metadata: dict[str, Any], *, code_commit: str
) -> dict[str, Any]:
    """Generate only train/validation, resuming verified immutable chunks.

    The CLI verifies numerical validation and protocol agreement first. This
    lower-level routine also supports small synthetic configurations in tests.
    Only the requested Sobol windows are drawn; test/OOD recipes remain inert.
    """
    design, units, storage = config["data_design"], config["units"], config["storage"]
    if design["impose_feller_condition"] or design["sampler"] != "scrambled_sobol":
        raise ValueError("this pipeline requires unfiltered scrambled Sobol sampling")
    if config["labels"]["option_type"] != "put" or config["normalization"]["fit_split"] != "train":
        raise ValueError("put IV labels and train-only normalization required")
    if torch.get_default_dtype() != torch.float32:
        raise ValueError(
            "keep torch default float32 for the recorded Sobol stream; draws use float64"
        )
    chunk_size, batch_size = storage["chunk_size"], storage["pricing_batch_size"]
    if any(
        isinstance(n, bool) or not isinstance(n, int) or n < 1 for n in (chunk_size, batch_size)
    ):
        raise ValueError("positive integer chunk/batch sizes required")
    seed = design["id_stream_seed"]
    seed_everything(seed)
    grid, _, _ = _label_inputs(config)
    if design["parameter_order"] != ["v0", "kappa", "theta", "sigma", "rho"]:
        raise ValueError("unexpected Heston parameter order")
    windows = []
    for split in ("train", "validation", "test"):
        recipe = design["id_splits"][split]
        start, count = recipe["offset"], recipe["count"]
        if (
            any(isinstance(v, bool) or not isinstance(v, int) for v in (start, count))
            or start < 0
            or count < 2
            or start + count > 2**30
        ):
            raise ValueError("invalid Sobol split window")
        windows.append((start, start + count))
    for index, (start, end) in enumerate(windows):
        if any(
            start < other_end and other_start < end
            for other_start, other_end in windows[index + 1 :]
        ):
            raise ValueError("overlapping Sobol split windows")
    threads = storage["torch_threads"]
    if isinstance(threads, bool) or not isinstance(threads, int) or threads < 1:
        raise ValueError("torch_threads must be a positive integer")
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text())["metadata"] != metadata:
        raise ValueError("dataset provenance mismatch; choose a new output directory")
    manifest: dict[str, Any] = {
        "version": 1,
        "complete": False,
        "metadata": metadata,
        "code_commit": code_commit,
        "seed": seed,
        "scope": "train_validation_only",
        "splits": {},
        "holdouts": "recipes only in config; never generated or read by this command",
    }
    for split in ("train", "validation"):
        recipe = design["id_splits"][split]
        total, offset = recipe["count"], recipe["offset"]
        if total < 2 or offset < 0:
            raise ValueError("each split needs at least two surfaces and a nonnegative offset")
        directory = output / split
        directory.mkdir(exist_ok=True)
        records = []
        jobs = [
            (
                config,
                metadata,
                directory,
                offset,
                start,
                min(chunk_size, total - start),
            )
            for start in range(0, total, chunk_size)
        ]
        workers = storage.get("workers", 1)
        if isinstance(workers, bool) or not isinstance(workers, int) or workers < 1:
            raise ValueError("workers must be a positive integer")
        context = ProcessPoolExecutor(max_workers=workers) if workers > 1 else nullcontext()
        with context as executor:
            results = (
                executor.map(_materialize_chunk, jobs)
                if executor
                else map(_materialize_chunk, jobs)
            )
            for record in results:
                records.append({key: value for key, value in record.items() if key != "metadata"})
                manifest["splits"][split] = {"count": total, "offset": offset, "chunks": records}
                write_json(manifest_path, manifest)
                print(
                    f"{split}: {record['start'] + record['count']}/{total} surfaces verified",
                    flush=True,
                )

    # Each complete split is read once for summaries; only train enters fitting.
    train_parameters, train_iv, train_status = [], [], []
    for split in ("train", "validation"):
        statuses, discrepancies, iv_discrepancies, term_counts = [], [], [], []
        for record in manifest["splits"][split]["chunks"]:
            with np.load(output / split / record["file"], allow_pickle=False) as arrays:
                status_array = arrays["status"]
                statuses.append(status_array)
                valid = status_array == IVStatus.SUCCESS
                scale = torch.maximum(
                    units["spot"] * torch.exp(-units["dividend_yield"] * grid.maturity),
                    grid.strike * torch.exp(-units["rate"] * grid.maturity),
                ).numpy()
                discrepancies.append((arrays["discrepancy"] / scale)[valid])
                iv_discrepancies.append(arrays["discrepancy"][valid] / arrays["vega"][valid])
                term_counts.extend(arrays["terms"].tolist())
                if split == "train":
                    train_parameters.append(torch.from_numpy(arrays["parameters"]))
                    train_iv.append(torch.from_numpy(arrays["iv"]))
                    train_status.append(torch.from_numpy(status_array))
        summary = _status_summary(np.concatenate(statuses))
        for name, values in (
            ("normalized_resolution_discrepancy", discrepancies),
            ("iv_discrepancy_estimate", iv_discrepancies),
        ):
            values_array = np.concatenate(values)
            summary[name] = (
                {
                    "count": len(values_array),
                    "p99": float(np.quantile(values_array, 0.99)),
                    "max": float(values_array.max()),
                }
                if len(values_array)
                else {"count": 0, "p99": None, "max": None}
            )
        counts, frequencies = np.unique(term_counts, return_counts=True)
        summary["finest_terms_counts"] = {
            str(n): int(k) for n, k in zip(counts, frequencies, strict=True)
        }
        limit = config["validation"]["max_invalid_fraction"]
        summary["coverage_stop_triggered"] = summary["failure_fraction"] > limit or any(
            value > limit for value in summary["failure_fraction_by_maturity"]
        )
        manifest["splits"][split]["summary"] = summary
    normalization = fit_train_normalization(
        torch.cat(train_parameters),
        torch.cat(train_iv),
        torch.cat(train_status) == IVStatus.SUCCESS,
    )
    normalization_arrays = {key: value.numpy() for key, value in asdict(normalization).items()}
    normalization_arrays["metadata"] = np.array(
        json.dumps({"seed": seed, "fit_split": "train", **metadata}, sort_keys=True)
    )
    write_arrays(output / "normalization.npz", normalization_arrays)
    write_arrays(
        output / "grid.npz",
        {key: value.numpy() for key, value in asdict(grid).items()}
        | {"metadata": normalization_arrays["metadata"]},
    )
    manifest["normalization_sha256"] = sha256_file(output / "normalization.npz")
    manifest["grid_sha256"] = sha256_file(output / "grid.npz")
    hashes = [
        chunk["sha256"]
        for split in ("train", "validation")
        for chunk in manifest["splits"][split]["chunks"]
    ] + [manifest["normalization_sha256"], manifest["grid_sha256"]]
    manifest["dataset_hash"] = hashlib.sha256(json.dumps(hashes).encode()).hexdigest()
    manifest["coverage_stop_triggered"] = any(
        manifest["splits"][split]["summary"]["coverage_stop_triggered"]
        for split in ("train", "validation")
    )
    manifest["complete"] = True
    write_json(manifest_path, manifest)
    return manifest


def current_commit(root: Path) -> str:
    """Record Git provenance without using it as a content-cache identity."""
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=True
    ).stdout.strip()
