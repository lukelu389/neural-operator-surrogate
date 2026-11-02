"""T2.1 contract and synthetic fixtures only; no production training or holdout reads."""

import hashlib
import json
import runpy
from pathlib import Path
from typing import Any

import pytest
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def sanity() -> dict[str, Any]:
    return runpy.run_path(str(ROOT / "scripts/validate_training.py"))


@pytest.fixture
def config() -> dict[str, Any]:
    result: dict[str, Any] = yaml.safe_load((ROOT / "config/training_sanity.yaml").read_text())
    return result


def _repository(tmp_path: Path, config: dict[str, Any]) -> Path:
    policy_path = tmp_path / "config/training.yaml"
    policy_path.parent.mkdir()
    policy_path.write_bytes((ROOT / "config/training.yaml").read_bytes())
    config_path = tmp_path / "config/training_sanity.yaml"
    config_path.write_text(yaml.safe_dump(config))
    return config_path


def _forbid_data_or_training(*args: object, **kwargs: object) -> None:
    raise AssertionError("sanity guard must fail before dataset access or training")


@pytest.mark.parametrize(
    ("section", "key", "replacement"),
    [
        (None, "seed", 102),
        (None, "training_prefix_count", 99),
        ("training", "max_optimizer_steps", 5001),
        ("assessment", "standardized_masked_MSE_max", 1.0e-3),
    ],
)
def test_frozen_sanity_settings_cannot_be_changed_before_data_access(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    sanity: dict[str, Any],
    config: dict[str, Any],
    section: str | None,
    key: str,
    replacement: int | float,
) -> None:
    (config if section is None else config[section])[key] = replacement
    config_path = _repository(tmp_path, config)
    globals_ = sanity["run"].__globals__
    monkeypatch.setitem(globals_, "verify_frozen_files", lambda root: {})
    monkeypatch.setitem(globals_, "load_split", _forbid_data_or_training)
    monkeypatch.setitem(globals_, "fit", _forbid_data_or_training)
    with pytest.raises(ValueError, match="disagree with the frozen check"):
        sanity["run"](tmp_path, config_path)


def test_failed_freeze_verification_stops_before_config_or_data_access(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, sanity: dict[str, Any]
) -> None:
    def reject_freeze(root: Path) -> None:
        raise ValueError("frozen file hash mismatch")

    globals_ = sanity["run"].__globals__
    monkeypatch.setitem(globals_, "verify_frozen_files", reject_freeze)
    monkeypatch.setitem(globals_, "load_split", _forbid_data_or_training)
    monkeypatch.setitem(globals_, "fit", _forbid_data_or_training)
    with pytest.raises(ValueError, match="frozen file hash mismatch"):
        sanity["run"](tmp_path, tmp_path / "absent.yaml")


def test_sanity_config_cannot_redirect_its_policy_to_unfrozen_settings(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    sanity: dict[str, Any],
    config: dict[str, Any],
) -> None:
    config["frozen_policy"] = "config/unfrozen.yaml"
    config["seed"] = 102
    config_path = _repository(tmp_path, config)
    policy = yaml.safe_load((tmp_path / "config/training.yaml").read_text())
    policy["sanity"]["seed"] = 102
    (tmp_path / config["frozen_policy"]).write_text(yaml.safe_dump(policy))
    globals_ = sanity["run"].__globals__
    monkeypatch.setitem(globals_, "verify_frozen_files", lambda root: {})
    monkeypatch.setitem(globals_, "load_split", _forbid_data_or_training)
    monkeypatch.setitem(globals_, "fit", _forbid_data_or_training)
    with pytest.raises(ValueError, match="verified frozen training policy"):
        sanity["run"](tmp_path, config_path)


@pytest.mark.parametrize("failure", ["different_dataset", "forged_claim"])
def test_dataset_identity_must_match_gate1_before_loading_training_rows(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    sanity: dict[str, Any],
    config: dict[str, Any],
    failure: str,
) -> None:
    config_path = _repository(tmp_path, config)
    dataset_path = tmp_path / config["dataset"]
    dataset_path.mkdir(parents=True)
    frozen_hash = hashlib.sha256(json.dumps(["train", "validation", "norm", "grid"]).encode())
    changed_hash = hashlib.sha256(json.dumps(["changed", "validation", "norm", "grid"]).encode())
    manifest = {
        "dataset_hash": changed_hash.hexdigest()
        if failure == "different_dataset"
        else frozen_hash.hexdigest(),
        "splits": {
            "train": {"chunks": [{"sha256": "changed"}]},
            "validation": {"chunks": [{"sha256": "validation"}]},
        },
        "normalization_sha256": "norm",
        "grid_sha256": "grid",
    }
    (dataset_path / "manifest.json").write_text(json.dumps(manifest))
    globals_ = sanity["run"].__globals__
    monkeypatch.setitem(
        globals_,
        "verify_frozen_files",
        lambda root: {"evidence": {"dataset": {"dataset_hash": frozen_hash.hexdigest()}}},
    )
    monkeypatch.setitem(globals_, "load_split", _forbid_data_or_training)
    monkeypatch.setitem(globals_, "fit", _forbid_data_or_training)
    with pytest.raises(ValueError, match="verified Gate 1 dataset identity"):
        sanity["run"](tmp_path, config_path)


def test_declared_linear_fixture_is_deterministic_and_decreases_at_each_step(
    sanity: dict[str, Any], config: dict[str, Any]
) -> None:
    first = sanity["linear_check"](config["linear_check"])
    second = sanity["linear_check"](config["linear_check"])
    assert first == second
    assert first["passed"]
    assert len(first["losses"]) == config["linear_check"]["steps"] + 1
    assert all(
        after < before for before, after in zip(first["losses"], first["losses"][1:], strict=False)
    )
    assert first["losses"][-1] < 1.0e-4 * first["losses"][0]


@pytest.mark.parametrize("learning_rate", [0.0, 10.0])
def test_linear_fixture_detects_no_descent_or_an_unstable_step(
    sanity: dict[str, Any], config: dict[str, Any], learning_rate: float
) -> None:
    config["linear_check"]["learning_rate"] = learning_rate
    assert not sanity["linear_check"](config["linear_check"])["passed"]


@pytest.mark.parametrize("grid_shape", [(2, 3), (8, 15)])
def test_sanity_model_preserves_native_surface_shape_and_starts_at_training_mean(
    sanity: dict[str, Any], grid_shape: tuple[int, ...]
) -> None:
    with torch.random.fork_rng():
        model = sanity["sanity_model"]([8, 8], grid_shape)
    prediction = model(torch.ones(4, 5))
    assert prediction.shape == (4, *grid_shape)
    assert prediction.dtype == torch.float32
    assert prediction.requires_grad
    assert torch.count_nonzero(prediction) == 0  # Zero standardized IV gives training mean.


@pytest.mark.parametrize(("passed", "expected_exit"), [(False, 1), (True, 0)])
def test_command_exit_status_tracks_the_assessed_threshold(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    sanity: dict[str, Any],
    passed: bool,
    expected_exit: int,
) -> None:
    result = {
        "passed": passed,
        "standardized_masked_MSE": 0.0 if passed else 1.0,
        "valid_quotes": 12000,
        "parameter_count": 100,
    }
    monkeypatch.setitem(sanity["main"].__globals__, "run", lambda root, path: result)
    assert sanity["main"](["--config", "synthetic-unused.yaml"]) == expected_exit
    assert json.loads(capsys.readouterr().out) == result
