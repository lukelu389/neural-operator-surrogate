"""Small deterministic fixtures: no production labels or tuning trials."""

import json
import random
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from torch import nn

import nop.train.engine as engine
from nop.train.engine import (
    InvalidPredictionError,
    TrainingData,
    TrainSettings,
    fit,
    learning_rate_factor,
)


def _data() -> TrainingData:
    parameters = torch.linspace(-1, 1, 7).reshape(-1, 1)
    targets = torch.cat((parameters * 0.3, parameters * -0.2), dim=1)
    valid = torch.ones_like(targets, dtype=torch.bool)
    valid[1, 0] = False
    valid[3] = False
    targets[~valid] = torch.nan
    return TrainingData(parameters, targets, valid, torch.ones(2), torch.full((2,), 0.1))


def _settings(**kwargs: Any) -> TrainSettings:
    return replace(
        TrainSettings(learning_rate=0.02, weight_decay=0.001, seed=19),
        max_epochs=6,
        warmup_epochs=1,
        batch_size=3,
        patience_epochs=6,
        **kwargs,
    )


def _model() -> nn.Module:
    torch.manual_seed(12)
    return nn.Sequential(nn.Linear(1, 5), nn.Tanh(), nn.Dropout(0.1), nn.Linear(5, 2))


class _AllRngModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = _model()

    def forward(self, parameters: torch.Tensor) -> torch.Tensor:
        noise = 0.01 * (random.random() + float(np.random.random()))
        return self.layers(parameters) + noise


def _state(output: Path, which: str = "last") -> dict[str, Any]:
    manifest = json.loads((output / "checkpoint.json").read_text())
    return torch.load(output / manifest[which]["path"], weights_only=True)


def _assert_equal(left: Any, right: Any) -> None:
    if isinstance(left, torch.Tensor):
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            _assert_equal(left[key], right[key])
    elif isinstance(left, list | tuple):
        assert len(left) == len(right)
        for item, other in zip(left, right, strict=True):
            _assert_equal(item, other)
    else:
        assert left == right


def test_schedule_warmup_cosine_and_endpoint() -> None:
    factors = [learning_rate_factor(i, 6, 2, 0.01) for i in range(6)]
    assert factors[:2] == [0.5, 1.0]
    assert factors[-1] == pytest.approx(0.01)
    assert all(a > b for a, b in zip(factors[1:-1], factors[2:], strict=True))
    assert learning_rate_factor(0, 4, 0, 0.01) == 1.0
    assert learning_rate_factor(3, 4, 0, 0.01) == pytest.approx(0.01)
    assert learning_rate_factor(40, 4, 0, 0.01) == pytest.approx(0.01)
    assert learning_rate_factor(0, 1, 0, 0.01) == 1.0
    assert learning_rate_factor(2, 3, 3, 0.01) == 1.0


@pytest.mark.parametrize("arguments", [(-1, 4, 1, 0.1), (0, 0, 0, 0.1), (0, 4, 5, 0.1)])
def test_invalid_schedule(arguments: tuple[int, int, int, float]) -> None:
    with pytest.raises(ValueError):
        learning_rate_factor(*arguments)


def test_fit_masks_partial_batches_and_cached_replay(tmp_path: Path) -> None:
    data, settings, model = _data(), _settings(), _model()
    records: list[dict[str, Any]] = []
    report = fit(model, data, data, settings, tmp_path, {"fixture": "tiny"}, records.append)
    assert report["completed_epochs"] == 6
    assert len(records) == 6
    assert all(record["train_valid_quote_count"] == int(data.valid.sum()) for record in records)
    assert all(
        record["validation_metrics"]["valid_count"] == int(data.valid.sum()) for record in records
    )
    assert all(torch.isfinite(value).all() for value in model.parameters())
    assert report["history"] == records
    assert len(list(tmp_path.glob("epoch-*.pt"))) <= 2
    final_weights = {key: value.clone() for key, value in model.state_dict().items()}
    replay_model = _model()
    replay_records: list[dict[str, Any]] = []
    replay = fit(
        replay_model, data, data, settings, tmp_path, {"fixture": "tiny"}, replay_records.append
    )
    assert replay == report
    assert len(records) == 6
    assert replay_records == records
    _assert_equal(final_weights, replay_model.state_dict())
    assert not replay_model.training
    for key in ("optimizer", "scheduler", "rng", "identity", "config_hash"):
        assert key in _state(tmp_path, "best")


def test_empty_batches_skip_optimizer_and_schedule(tmp_path: Path) -> None:
    data = _data()
    settings = replace(_settings(), batch_size=1, max_epochs=2, warmup_epochs=0)
    report = fit(_model(), data, data, settings, tmp_path, {})
    assert report["optimizer_steps"] == 12  # Six nonempty rows, twice.
    assert all(row["skipped_empty_batches"] == 1 for row in report["history"])
    state = _state(tmp_path)
    assert state["scheduler"]["last_epoch"] == 12
    assert int(next(iter(state["optimizer"]["state"].values()))["step"]) == 12


def test_quote_weighted_epoch_loss_and_validation_metrics(tmp_path: Path) -> None:
    # No parameter movement: a tiny positive learning rate rounds away in float32.
    model = nn.Linear(1, 2)
    nn.init.zeros_(model.weight)
    nn.init.zeros_(model.bias)
    parameters = torch.zeros(3, 1)
    targets = torch.tensor([[1.0, 2.0], [3.0, torch.nan], [5.0, 6.0]])
    valid = torch.isfinite(targets)
    weights = torch.tensor([[2.0, 1.0], [3.0, torch.nan], [0.5, 1.0]])
    data = TrainingData(parameters, targets, valid, torch.ones(2), torch.full((2,), 0.1), weights)
    settings = replace(
        _settings(), max_epochs=1, warmup_epochs=0, batch_size=2, learning_rate=1e-40
    )
    report = fit(model, data, data, settings, tmp_path, {})
    row = report["history"][0]
    assert row["train_valid_quote_count"] == 5
    assert row["train_loss"] == pytest.approx(
        float((weights[valid] * targets[valid].square()).mean())
    )
    assert row["validation_metrics"]["mean_absolute_error"] == pytest.approx(34.0, abs=1e-5)


def test_best_epoch_uses_validation_and_restores_earliest_tie(tmp_path: Path) -> None:
    model = nn.Linear(1, 1, bias=False)
    nn.init.zeros_(model.weight)
    # Zero inputs keep all predictions fixed, so every validation score ties.
    data = TrainingData(
        torch.zeros(3, 1),
        torch.ones(3, 1),
        torch.ones(3, 1, dtype=torch.bool),
        torch.ones(1),
        torch.ones(1),
    )
    settings = replace(_settings(), patience_epochs=2)
    report = fit(model, data, data, settings, tmp_path, {})
    assert report["status"] == "early_stopped"
    assert report["best_epoch"] == 1
    assert report["completed_epochs"] == 3
    _assert_equal(model.state_dict(), _state(tmp_path, "best")["model"])
    assert _state(tmp_path, "best")["completed_epochs"] == 1


def test_best_validation_weights_restored_after_worsening(tmp_path: Path) -> None:
    model = nn.Linear(1, 1)
    nn.init.zeros_(model.weight)
    nn.init.zeros_(model.bias)
    train = TrainingData(
        torch.zeros(3, 1),
        torch.ones(3, 1),
        torch.ones(3, 1, dtype=torch.bool),
        torch.ones(1),
        torch.full((1,), 0.1),
    )
    validation = replace(train, targets=-train.targets)
    report = fit(model, train, validation, replace(_settings(), patience_epochs=2), tmp_path, {})
    assert report["best_epoch"] == 1
    assert report["completed_epochs"] == 3
    _assert_equal(model.state_dict(), _state(tmp_path, "best")["model"])
    assert not torch.equal(model.bias, _state(tmp_path)["model"]["bias"])


def test_epoch_boundary_resume_exact_weights_optimizer_schedule_rng(tmp_path: Path) -> None:
    data, settings = _data(), _settings()
    full_model = _AllRngModel()
    full = fit(full_model, data, data, settings, tmp_path / "full", {})

    def interrupt(record: dict[str, Any]) -> None:
        # Logging's random draws must never affect training, including interruption.
        random.random()
        np.random.random()
        torch.rand(5)
        if record["epoch"] == 2:
            raise RuntimeError("simulated interruption")

    with pytest.raises(RuntimeError, match="simulated interruption"):
        fit(_AllRngModel(), data, data, settings, tmp_path / "resumed", {}, interrupt)
    random.random()
    np.random.random()
    torch.rand(19)
    resumed_model = _AllRngModel()
    records: list[dict[str, Any]] = []
    resumed = fit(resumed_model, data, data, settings, tmp_path / "resumed", {}, records.append)
    assert records == full["history"]  # Replay also repairs missing prior logger callbacks.
    assert resumed["history"] == full["history"]
    assert resumed["best_epoch"] == full["best_epoch"]
    _assert_equal(full_model.state_dict(), resumed_model.state_dict())
    for key in ("optimizer", "scheduler", "rng", "model"):
        _assert_equal(_state(tmp_path / "full")[key], _state(tmp_path / "resumed")[key])


def test_interrupted_epoch_replays_from_previous_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data, settings = _data(), _settings()
    complete = fit(_model(), data, data, settings, tmp_path / "full", {})
    original = engine._validate

    def interrupt(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("interrupted before epoch commit")

    monkeypatch.setattr(engine, "_validate", interrupt)
    with pytest.raises(RuntimeError, match="before epoch commit"):
        fit(_model(), data, data, settings, tmp_path / "replay", {})
    assert _state(tmp_path / "replay")["completed_epochs"] == 0
    monkeypatch.setattr(engine, "_validate", original)
    replay = fit(_model(), data, data, settings, tmp_path / "replay", {})
    assert replay["history"] == complete["history"]


@pytest.mark.parametrize("change", ["metadata", "settings", "data", "initial_weights"])
def test_stale_cache_rejected(tmp_path: Path, change: str) -> None:
    data, settings = _data(), _settings()
    fit(_model(), data, data, settings, tmp_path, {})
    model, metadata = _model(), {}
    if change == "metadata":
        metadata = {"different": True}
    elif change == "settings":
        settings = replace(settings, learning_rate=0.01)
    elif change == "data":
        data = replace(data, parameters=data.parameters + 0.1)
    elif change == "initial_weights":
        with torch.no_grad():
            next(model.parameters()).add_(0.1)
    with pytest.raises(ValueError, match="stale training cache"):
        fit(model, data, data, settings, tmp_path, metadata)


@pytest.mark.parametrize("which", ["best", "last"])
def test_corrupt_checkpoint_cannot_return_cached_report(tmp_path: Path, which: str) -> None:
    data, settings = _data(), _settings()
    fit(_model(), data, data, settings, tmp_path, {})
    manifest = json.loads((tmp_path / "checkpoint.json").read_text())
    with (tmp_path / manifest[which]["path"]).open("ab") as stream:
        stream.write(b"corruption")
    with pytest.raises(ValueError, match="hash mismatch"):
        fit(_model(), data, data, settings, tmp_path, {})


def test_corrupt_report_checkpoint_identity_rejected(tmp_path: Path) -> None:
    data, settings = _data(), _settings()
    fit(_model(), data, data, settings, tmp_path, {})
    path = tmp_path / "report.json"
    report = json.loads(path.read_text())
    report["checkpoint_sha256"] = "incorrect hash"
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="report does not match"):
        fit(_model(), data, data, settings, tmp_path, {})


@pytest.mark.parametrize("bias,message", [(-100.0, "nonpositive"), (1e30, "nonfinite loss")])
def test_invalid_prediction_fails_and_caches_failure(
    tmp_path: Path, bias: float, message: str
) -> None:
    data, settings = _data(), _settings()

    def invalid_model() -> nn.Module:
        model = _model()
        with torch.no_grad():
            model[-1].bias.fill_(bias)
        return model

    with pytest.raises(InvalidPredictionError, match=message):
        fit(invalid_model(), data, data, settings, tmp_path, {})
    assert json.loads((tmp_path / "report.json").read_text())["status"] == "failed"
    with pytest.raises(InvalidPredictionError, match=message):
        fit(invalid_model(), data, data, settings, tmp_path, {})
    assert _state(tmp_path)["optimizer_steps"] == 0


def test_optimizer_step_cap_and_native_grid(tmp_path: Path) -> None:
    data = _data()
    native = replace(
        data,
        targets=data.targets.unsqueeze(-1),
        valid=data.valid.unsqueeze(-1),
        target_mean=data.target_mean.unsqueeze(-1),
        target_std=data.target_std.unsqueeze(-1),
    )
    model = nn.Sequential(_model(), nn.Unflatten(1, (2, 1)))
    report = fit(model, native, native, replace(_settings(), max_optimizer_steps=4), tmp_path, {})
    assert report["optimizer_steps"] == 4
    assert report["status"] == "step_limit"
    assert report["completed_epochs"] == 2


@pytest.mark.parametrize(
    "name,value",
    [
        ("learning_rate", 0),
        ("weight_decay", -1),
        ("seed", True),
        ("batch_size", 0),
        ("warmup_epochs", 101),
        ("minimum_learning_rate_fraction", 0),
        ("epsilon", float("nan")),
        ("betas", (0.9, 1.0)),
        ("max_optimizer_steps", 0),
    ],
)
def test_settings_reject_invalid_values(name: str, value: Any) -> None:
    with pytest.raises(ValueError):
        replace(_settings(), **{name: value})


def test_data_validation_and_training_moments_required(tmp_path: Path) -> None:
    data = _data()
    with pytest.raises(ValueError, match="float32"):
        replace(data, parameters=data.parameters.double())
    with pytest.raises(ValueError, match="at least one"):
        replace(data, valid=torch.zeros_like(data.valid))
    with pytest.raises(ValueError, match="finite positive"):
        replace(data, targets=torch.full_like(data.targets, torch.nan))
    with pytest.raises(ValueError, match="reuse training"):
        fit(
            _model(),
            data,
            replace(data, target_mean=data.target_mean + 1),
            _settings(),
            tmp_path,
            {},
        )
