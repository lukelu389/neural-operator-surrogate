"""M1 orchestration uses the paired frozen trial budget on synthetic fixtures."""

import json
import runpy
import shutil
from pathlib import Path
from typing import Any

import pytest
import torch
import yaml

from nop.train.engine import TrainingData
from nop.train.experiments import paired_trial_draws

ROOT = Path(__file__).resolve().parents[1]


def test_m1_runs_all_paired_trials_before_the_three_headline_seeds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("MPLCONFIGDIR", str(tmp_path / "matplotlib"))
    command = runpy.run_path(str(ROOT / "scripts/train_m1.py"))
    (tmp_path / "config").mkdir()
    for name in (
        "training.yaml",
        "eval_protocol.yaml",
        "m0.yaml",
        "m0_selected.yaml",
        "m1.yaml",
        "m1_run.yaml",
    ):
        shutil.copyfile(ROOT / "config" / name, tmp_path / "config" / name)
    config_path = tmp_path / "config/m1_run.yaml"
    config = yaml.safe_load(config_path.read_text())
    policy = yaml.safe_load((tmp_path / "config/training.yaml").read_text())
    config.update(
        output="runs/m1_synthetic",
        report="reports/tables/m1_synthetic.json",
        selected_config="config/m1_selected_synthetic.yaml",
        figures="reports/figures/m1_synthetic",
    )
    config_path.write_text(yaml.safe_dump(config, sort_keys=True))
    (tmp_path / "reports/tables").mkdir(parents=True)

    shape = (8, 15)
    targets = torch.zeros(3, *shape)
    valid = torch.ones_like(targets, dtype=torch.bool)
    data = TrainingData(
        parameters=torch.zeros(3, 5),
        targets=targets,
        valid=valid,
        target_mean=torch.full(shape, 0.2),
        target_std=torch.full(shape, 0.05),
    )
    grid: dict[str, Any] = {
        "z": [torch.linspace(-2.5, 2.5, shape[1]).tolist()] * shape[0],
        "maturity": [
            ([1 / 12, 2 / 12, 3 / 12, 6 / 12, 9 / 12, 1.0, 1.5, 2.0][i],) * shape[1]
            for i in range(shape[0])
        ],
    }
    grid["k"] = [
        [z * 0.2 * maturity**0.5 for z in grid["z"][i]]
        for i, maturity in enumerate([1 / 12, 2 / 12, 3 / 12, 6 / 12, 9 / 12, 1.0, 1.5, 2.0])
    ]
    calls: list[tuple[str, int, dict[str, Any]]] = []

    def run_one(
        root: Path,
        run_config: dict[str, Any],
        model_config: dict[str, Any],
        shared_policy: dict[str, Any],
        identity: dict[str, Any],
        train: TrainingData,
        validation: TrainingData,
        run_grid: dict[str, Any],
        draw: dict[str, Any],
        seed: int,
        run_id: str,
    ) -> tuple[dict[str, Any], torch.Tensor | None, list[dict[str, Any]]]:
        calls.append((run_id, seed, draw.copy()))
        index = int(draw["trial_id"].removeprefix("trial_"))
        mae = 0.05 + index * 0.001 if run_id.startswith("trial_") else 0.1
        record = {
            **draw,
            "config_id": draw["trial_id"],
            "run_id": run_id,
            "seed": seed,
            "status": "completed",
            "parameter_count": model_config["parameter_count"],
            "validation_metrics": {
                "mean_absolute_error": mae,
                "p99_absolute_error": mae,
                "prediction_failure_count": 0,
            },
            "mlflow_run_id": run_id,
        }
        prediction = None if run_id.startswith("trial_") else torch.full((3, *shape), 0.2)
        return record, prediction, [{"epoch": 1, "train_loss": 1.0}]

    globals_ = command["run"].__globals__
    monkeypatch.setitem(globals_, "verify_frozen_files", lambda root: {"files": {}})
    monkeypatch.setitem(
        globals_,
        "load_development_data",
        lambda root, path: (
            data,
            data,
            {"grid": grid, "dataset_hash": "synthetic", "normalization_sha256": "norm"},
        ),
    )
    monkeypatch.setitem(
        globals_,
        "_identity",
        lambda *args: {
            "code_commit": "synthetic",
            "dataset_hash": "synthetic",
            "config_hashes": {"model": "synthetic_model_hash"},
        },
    )
    monkeypatch.setitem(globals_, "run_one", run_one)
    monkeypatch.setitem(
        globals_, "validation_summary", lambda *args: {"complete": True, "split": "validation"}
    )
    monkeypatch.setitem(globals_, "write_validation_figures", lambda *args: [])
    result = command["run"](tmp_path, config_path)

    assert result["status"] == "complete"
    assert [call[0] for call in calls[:12]] == [f"trial_{index:02d}" for index in range(12)]
    assert [call[0] for call in calls[12:]] == ["seed_11", "seed_29", "seed_47"]
    assert [call[2] for call in calls[:12]] == paired_trial_draws(policy)
    selected = result["selected_trial"]
    assert selected["trial_id"] == "trial_00"
    assert all(call[2]["learning_rate"] == selected["learning_rate"] for call in calls[12:])
    assert all(call[2]["weight_decay"] == selected["weight_decay"] for call in calls[12:])
    selected_config = yaml.safe_load((tmp_path / config["selected_config"]).read_text())
    assert selected_config["selected_trial_id"] == selected["trial_id"]
    assert json.loads((tmp_path / config["report"]).read_text())["status"] == "complete"
