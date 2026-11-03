"""M0 orchestration on synthetic arrays; no production training or holdout access."""

import json
import runpy
import shutil
from pathlib import Path
from typing import Any

import pytest
import torch
import yaml

from nop.train.engine import TrainingData
from nop.train.experiments import paired_trial_draws, run_lock

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def command() -> dict[str, Any]:
    return runpy.run_path(str(ROOT / "scripts/train_m0.py"))


@pytest.fixture
def experiment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, command: dict[str, Any]
) -> dict[str, Any]:
    for directory in ("src", "scripts"):
        shutil.copytree(ROOT / directory, tmp_path / directory)
    (tmp_path / "config").mkdir()
    for name in ("training.yaml", "eval_protocol.yaml", "m0.yaml", "m0_run.yaml"):
        shutil.copyfile(ROOT / "config" / name, tmp_path / "config" / name)
    config_path = tmp_path / "config/m0_run.yaml"
    config = yaml.safe_load(config_path.read_text())
    policy = yaml.safe_load((tmp_path / "config/training.yaml").read_text())
    targets = torch.zeros(3, 2, 2)
    valid = torch.ones_like(targets, dtype=torch.bool)
    valid[-1, -1, -1] = False
    targets[~valid] = torch.nan
    data = TrainingData(
        parameters=torch.zeros(3, 5),
        targets=targets,
        valid=valid,
        target_mean=torch.full((2, 2), 0.25),
        target_std=torch.full((2, 2), 0.05),
    )
    reference = data.targets * data.target_std + data.target_mean
    calls: list[tuple[str, int, dict[str, Any]]] = []
    figure_calls: list[dict[str, Any]] = []
    state: dict[str, Any] = {
        "root": tmp_path,
        "config_path": config_path,
        "config": config,
        "policy": policy,
        "data": data,
        "reference": reference,
        "calls": calls,
        "figure_calls": figure_calls,
        "tuning_mode": "success",
        "failed_seed": None,
        "seed_errors": {11: 0.05, 29: 0.10, 47: 0.15},
    }

    def run_one(
        root: Path,
        run_config: dict[str, Any],
        model_config: dict[str, Any],
        shared_policy: dict[str, Any],
        identity: dict[str, Any],
        train: TrainingData,
        validation: TrainingData,
        draw: dict[str, Any],
        seed: int,
        run_id: str,
    ) -> tuple[dict[str, Any], torch.Tensor | None, list[dict[str, Any]]]:
        assert root == tmp_path
        assert run_config == config
        assert shared_policy == policy
        assert train is data and validation is data
        assert identity["dataset_hash"] == "synthetic_dataset"
        calls.append((run_id, seed, draw.copy()))
        is_tuning = run_id.startswith("trial_")
        failed = (is_tuning and state["tuning_mode"] == "all_failed") or (
            not is_tuning and seed == state["failed_seed"]
        )
        if is_tuning:
            index = int(draw["trial_id"].removeprefix("trial_"))
            mae = (0.21 if state["tuning_mode"] == "above_threshold" else 0.05) + abs(
                index - 7
            ) * 0.001
        else:
            mae = state["seed_errors"][seed]
        metrics = {
            "mean_absolute_error": mae,
            "p99_absolute_error": mae,
            "prediction_failure_count": 0,
        }
        record = {
            **draw,
            "run_id": run_id,
            "seed": seed,
            "status": "failed" if failed else "completed",
            "parameter_count": model_config["trainable_parameters"],
            "validation_metrics": None if failed else metrics,
        }
        prediction = reference.double() + mae / 100.0 if not is_tuning and not failed else None
        history = [] if failed else [{"epoch": 1, "train_loss": 0.5, "validation_metrics": metrics}]
        return record, prediction, history

    def figures(
        accuracy: dict[str, Any],
        histories: dict[str, Any],
        maturity: list[float],
        z: list[float],
        output: Path,
        dpi: int,
    ) -> list[Path]:
        assert accuracy["split"] == "validation"
        assert maturity == [0.5, 1.0] and z == [-1.0, 1.0]
        figure_calls.append({"accuracy": accuracy, "histories": histories})
        return []

    globals_ = command["run"].__globals__
    monkeypatch.setitem(globals_, "verify_frozen_files", lambda root: {"files": {}})
    monkeypatch.setitem(globals_, "current_commit", lambda root: "synthetic_commit")
    monkeypatch.setitem(
        globals_,
        "load_development_data",
        lambda root, path: (
            data,
            data,
            {
                "dataset_hash": "synthetic_dataset",
                "normalization_sha256": "synthetic_normalization",
                "grid": {"maturity": [[0.5, 0.5], [1.0, 1.0]], "z": [[-1.0, 1.0]] * 2},
            },
        ),
    )
    monkeypatch.setitem(globals_, "run_one", run_one)
    monkeypatch.setitem(globals_, "write_validation_figures", figures)
    return state


def _run(command: dict[str, Any], experiment: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
    return command["run"](experiment["root"], experiment["config_path"], **kwargs)


def _forbid_run(*args: object, **kwargs: object) -> None:
    raise AssertionError("guard must reject before any trial or seed run")


def test_complete_budget_precedes_three_selected_config_seeds_and_reports_sample_std(
    command: dict[str, Any], experiment: dict[str, Any]
) -> None:
    result = _run(command, experiment)
    calls = experiment["calls"]
    draws = paired_trial_draws(experiment["policy"])
    assert calls[:12] == [(draw["trial_id"], 101, draw) for draw in draws]
    assert [call[:2] for call in calls[12:]] == [("seed_11", 11), ("seed_29", 29), ("seed_47", 47)]
    assert all(call[2] == draws[7] for call in calls[12:])
    assert result["status"] == "complete" and result["passed"]
    assert result["tuning_gate"]["selected"]["trial_id"] == "trial_07"
    accuracy = result["accuracy"]
    assert accuracy["reference_coverage"]["attempted"] == 12
    assert accuracy["reference_coverage"]["reference_failure_count"] == 1
    assert accuracy["aggregate"]["overall"]["mean_absolute_error"] == pytest.approx(
        {"mean": 0.1, "std": 0.05}
    )
    assert accuracy["required_seeds"] == [11, 29, 47]
    assert len(experiment["figure_calls"][0]["histories"]) == 15
    selected = yaml.safe_load(
        (experiment["root"] / experiment["config"]["selected_config"]).read_text()
    )
    assert selected["selected_trial_id"] == "trial_07"
    assert selected["learning_rate"] == draws[7]["learning_rate"]
    report = experiment["root"] / experiment["config"]["report"]
    assert json.loads(report.read_text()) == result


@pytest.mark.parametrize("mode", ["all_failed", "above_threshold"])
def test_tuning_stop_preserves_complete_budget_without_seed_runs(
    command: dict[str, Any], experiment: dict[str, Any], mode: str
) -> None:
    experiment["tuning_mode"] = mode
    result = _run(command, experiment)
    assert len(experiment["calls"]) == 12
    assert all(call[1] == 101 for call in experiment["calls"])
    assert result["status"] == "stopped" and not result["tuning_gate"]["passed"]
    assert result["seeds"] == []
    assert experiment["figure_calls"] == []
    assert not (experiment["root"] / experiment["config"]["selected_config"]).exists()
    assert json.loads((experiment["root"] / experiment["config"]["report"]).read_text()) == result


def test_tune_only_stops_after_exact_budget_without_spending_seed_runs(
    command: dict[str, Any], experiment: dict[str, Any]
) -> None:
    result = _run(command, experiment, tune_only=True)
    assert result["status"] == "tuned" and result["tuning_gate"]["passed"]
    assert len(experiment["calls"]) == 12 and result["seeds"] == []
    assert experiment["figure_calls"] == []


def test_failed_seed_does_not_get_replaced_or_complete_case_averaged(
    command: dict[str, Any], experiment: dict[str, Any]
) -> None:
    experiment["failed_seed"] = 29
    result = _run(command, experiment)
    assert len(experiment["calls"]) == 15
    assert result["status"] == "stopped" and not result["passed"]
    assert [seed["seed"] for seed in result["seeds"]] == [11, 29, 47]
    assert result["accuracy"]["missing_seeds"] == [29]
    assert result["accuracy"]["aggregate"]["overall"]["mean_absolute_error"] == {
        "mean": None,
        "std": None,
    }


def test_three_seed_stop_uses_mean_instead_of_best_seed(
    command: dict[str, Any], experiment: dict[str, Any]
) -> None:
    experiment["seed_errors"] = {11: 0.01, 29: 0.3, 47: 0.4}
    result = _run(command, experiment)
    assert result["tuning_gate"]["passed"]
    assert result["accuracy"]["complete"]
    assert result["status"] == "stopped" and not result["passed"]
    assert result["accuracy"]["aggregate"]["overall"]["mean_absolute_error"][
        "mean"
    ] == pytest.approx(0.71 / 3)


@pytest.mark.parametrize("source", ["scripts/train_m0.py", "src/nop/eval/validation_report.py"])
def test_changed_experiment_provenance_preserves_ledger_before_any_runs(
    monkeypatch: pytest.MonkeyPatch,
    command: dict[str, Any],
    experiment: dict[str, Any],
    source: str,
) -> None:
    _run(command, experiment, tune_only=True)
    ledger = experiment["root"] / experiment["config"]["output"] / "experiment.json"
    before = ledger.read_bytes()
    path = experiment["root"] / source
    path.write_bytes(path.read_bytes() + b"\n# Synthetic provenance change.\n")
    monkeypatch.setitem(command["run"].__globals__, "run_one", _forbid_run)
    with pytest.raises(ValueError, match="provenance changed"):
        _run(command, experiment)
    assert ledger.read_bytes() == before


def test_existing_selected_config_is_never_overwritten_or_used_for_new_seeds(
    command: dict[str, Any], experiment: dict[str, Any]
) -> None:
    selected = experiment["root"] / experiment["config"]["selected_config"]
    selected.write_text("selected_trial_id: prior_different_trial\n")
    before = selected.read_bytes()
    with pytest.raises(ValueError, match="do not overwrite or retune"):
        _run(command, experiment)
    assert selected.read_bytes() == before
    assert len(experiment["calls"]) == 12


def test_concurrent_writer_rejected_before_data_access_or_ledger_change(
    monkeypatch: pytest.MonkeyPatch, command: dict[str, Any], experiment: dict[str, Any]
) -> None:
    output = experiment["root"] / experiment["config"]["output"]
    output.mkdir(parents=True)
    ledger = output / "experiment.json"
    ledger.write_text("prior ledger preserved\n")
    monkeypatch.setitem(command["run"].__globals__, "load_development_data", _forbid_run)
    with run_lock(output / ".run.lock"), pytest.raises(RuntimeError, match="another experiment"):
        _run(command, experiment)
    assert ledger.read_text() == "prior ledger preserved\n"
    assert experiment["calls"] == []


def test_run_one_seeds_before_model_construction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, command: dict[str, Any]
) -> None:
    order: list[tuple[str, Any]] = []

    class Constructed(Exception):
        pass

    def construct(*args: object) -> None:
        order.append(("model", None))
        raise Constructed

    globals_ = command["run_one"].__globals__
    monkeypatch.setitem(globals_, "seed_everything", lambda seed: order.append(("seed", seed)))
    monkeypatch.setitem(globals_, "MLP", construct)
    data = TrainingData(
        torch.zeros(1, 5),
        torch.zeros(1, 1, 1),
        torch.ones(1, 1, 1, dtype=torch.bool),
        torch.ones(1, 1),
        torch.ones(1, 1),
    )
    with pytest.raises(Constructed):
        command["run_one"](
            tmp_path, {}, {"hidden_widths": [2]}, {}, {}, data, data, {}, 29, "seed_29"
        )
    assert order == [("seed", 29), ("model", None)]


@pytest.mark.parametrize(("status", "exit_code"), [("complete", 0), ("tuned", 0), ("stopped", 2)])
def test_command_reports_stop_as_nonzero_exit(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    command: dict[str, Any],
    status: str,
    exit_code: int,
) -> None:
    monkeypatch.setitem(
        command["main"].__globals__,
        "run",
        lambda root, config, *, tune_only: {"status": status, "trials": [None] * 12, "seeds": []},
    )
    assert command["main"](["--tune-only"]) == exit_code
    assert json.loads(capsys.readouterr().out) == {"status": status, "trials": 12, "seeds": 0}
