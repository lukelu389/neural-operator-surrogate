import json
import math
from pathlib import Path
from typing import Any

import matplotlib.image as mpimg
import numpy as np
import pytest
import torch

from nop.eval.validation_report import validation_summary, write_validation_figures

SEEDS = (11, 29, 47)
ERROR_METRICS = (
    "mean_absolute_error",
    "root_mean_squared_error",
    "p99_absolute_error",
    "max_absolute_error",
)


def _fixture() -> tuple[dict[int, torch.Tensor], torch.Tensor, torch.Tensor]:
    reference = torch.full((4, 2, 2), 0.3, dtype=torch.float64)
    valid = torch.ones_like(reference, dtype=torch.bool)
    valid[0, 0, 0] = False
    reference[~valid] = math.nan
    errors = torch.arange(1, 17, dtype=torch.float64).reshape_as(reference) * 0.001
    predictions = {
        seed: reference + scale * errors for seed, scale in zip(SEEDS, (1, 2, 4), strict=True)
    }
    return predictions, reference, valid


def _independent_metrics(errors: np.ndarray[Any, Any]) -> dict[str, float]:
    absolute = np.abs(errors)
    return {
        "mean_absolute_error": float(absolute.mean()),
        "root_mean_squared_error": float(np.sqrt(np.mean(absolute**2))),
        "p99_absolute_error": float(np.quantile(absolute, 0.99, method="linear")),
        "max_absolute_error": float(absolute.max()),
    }


def test_independent_per_seed_metrics_and_sample_aggregation_without_pooled_tails() -> None:
    predictions, reference, valid = _fixture()
    result = validation_summary(predictions, reference, valid, SEEDS)
    expected = [
        _independent_metrics(((predictions[seed] - reference)[valid] * 100).numpy())
        for seed in SEEDS
    ]
    assert result["complete"] and result["missing_seeds"] == []
    assert result["required_seeds"] == list(SEEDS)
    assert result["split"] == "validation" and result["units"] == "vol_points"
    assert result["grid_shape"] == [2, 2]
    for seed, metrics in zip(SEEDS, expected, strict=True):
        actual = result["per_seed"][str(seed)]["overall"]
        assert actual["attempted"] == 16
        assert actual["valid_count"] == 15
        assert actual["reference_failure_count"] == 1
        assert actual["prediction_failure_count"] == 0
        for name in ERROR_METRICS:
            assert actual[name] == pytest.approx(metrics[name])
    for name in ERROR_METRICS:
        values = [metrics[name] for metrics in expected]
        aggregate = result["aggregate"]["overall"][name]
        assert aggregate["mean"] == pytest.approx(np.mean(values))
        assert aggregate["std"] == pytest.approx(np.std(values, ddof=1))
    pooled = np.concatenate(
        [((prediction - reference)[valid] * 100).numpy() for prediction in predictions.values()]
    )
    assert result["aggregate"]["overall"]["p99_absolute_error"]["mean"] != pytest.approx(
        np.quantile(pooled, 0.99)
    )
    assert result["aggregate"]["overall"]["max_absolute_error"]["mean"] != pytest.approx(
        pooled.max()
    )
    assert result["aggregate"]["overall"]["valid_count"] == {"mean": 15.0, "std": 0.0}
    json.dumps(result, allow_nan=False)


def test_maturity_moneyness_and_cell_strata_keep_their_own_masks() -> None:
    predictions, reference, valid = _fixture()
    result = validation_summary(predictions, reference, valid, SEEDS)
    for t in range(2):
        for z in range(2):
            strata = [
                ("per_maturity", t, (slice(None), t, slice(None))),
                ("per_moneyness", z, (slice(None), slice(None), z)),
                ("per_cell", (t, z), (slice(None), t, z)),
            ]
            for name, index, slices in strata:
                seed_values = []
                for seed in SEEDS:
                    error = ((predictions[seed] - reference)[slices][valid[slices]] * 100).numpy()
                    expected = _independent_metrics(error)
                    seed_stratum = result["per_seed"][str(seed)][name]
                    actual = (
                        seed_stratum[index[0]][index[1]]
                        if isinstance(index, tuple)
                        else seed_stratum[index]
                    )
                    assert actual["valid_count"] == int(valid[slices].sum())
                    for metric in ERROR_METRICS:
                        assert actual[metric] == pytest.approx(expected[metric])
                    seed_values.append(expected)
                aggregates = result["aggregate"][name]
                aggregate = (
                    aggregates[index[0]][index[1]]
                    if isinstance(index, tuple)
                    else aggregates[index]
                )
                for metric in ERROR_METRICS:
                    values = [value[metric] for value in seed_values]
                    assert aggregate[metric]["mean"] == pytest.approx(np.mean(values))
                    assert aggregate[metric]["std"] == pytest.approx(np.std(values, ddof=1))


def test_missing_seed_does_not_become_complete_case_average() -> None:
    predictions, reference, valid = _fixture()
    del predictions[29]
    result = validation_summary(predictions, reference, valid, SEEDS)
    assert result["missing_seeds"] == [29] and not result["complete"]
    assert result["per_seed"]["29"]["status"] == "missing"
    assert result["per_seed"]["29"]["overall"] is None
    assert result["per_seed"]["11"]["overall"]["valid_count"] == 15
    assert result["reference_coverage"]["reference_failure_count"] == 1
    assert all(
        value == {"mean": None, "std": None} for value in result["aggregate"]["overall"].values()
    )
    assert result["aggregate"]["per_cell"][1][1]["mean_absolute_error"] == {
        "mean": None,
        "std": None,
    }
    json.dumps(result, allow_nan=False)


def test_invalid_prediction_propagates_only_to_affected_strata_and_keeps_failure_counts() -> None:
    predictions, reference, valid = _fixture()
    predictions[29][1, 0, 1] = math.nan
    result = validation_summary(predictions, reference, valid, SEEDS)
    assert not result["complete"]
    failed = result["per_seed"]["29"]
    assert failed["status"] == "invalid_predictions"
    assert failed["overall"]["prediction_failure_count"] == 1
    assert failed["overall"]["valid_count"] == 15
    for name in ERROR_METRICS:
        assert result["aggregate"]["overall"][name] == {"mean": None, "std": None}
        assert result["aggregate"]["per_cell"][0][1][name] == {"mean": None, "std": None}
        assert result["aggregate"]["per_cell"][1][1][name]["mean"] is not None
    assert result["aggregate"]["overall"]["prediction_failure_count"]["mean"] == pytest.approx(
        1 / 3
    )
    assert result["aggregate"]["overall"]["prediction_failure_count"]["std"] == pytest.approx(
        math.sqrt(1 / 3)
    )
    assert result["aggregate"]["per_maturity"][1]["mean_absolute_error"]["mean"] is not None
    assert result["aggregate"]["per_moneyness"][0]["mean_absolute_error"]["mean"] is not None


def test_masked_prediction_failure_is_ignored_but_reference_count_retained() -> None:
    predictions, reference, valid = _fixture()
    predictions[11][~valid] = -1
    result = validation_summary(predictions, reference, valid, SEEDS)
    assert result["complete"]
    assert result["per_seed"]["11"]["overall"]["prediction_failure_count"] == 0
    assert result["per_seed"]["11"]["overall"]["reference_failure_count"] == 1


def test_empty_cell_has_undefined_errors_and_no_prediction_failures() -> None:
    predictions, reference, valid = _fixture()
    valid[:, 1, 1] = False
    reference[:, 1, 1] = math.nan
    result = validation_summary(predictions, reference, valid, SEEDS)
    assert result["complete"]
    cell = result["aggregate"]["per_cell"][1][1]
    assert cell["valid_count"] == {"mean": 0.0, "std": 0.0}
    assert cell["prediction_failure_count"] == {"mean": 0.0, "std": 0.0}
    assert all(cell[name] == {"mean": None, "std": None} for name in ERROR_METRICS)


def test_all_missing_seeds_and_all_masked_reference_remain_json_safe() -> None:
    reference = torch.full((2, 1, 1), math.nan)
    result = validation_summary({}, reference, torch.zeros_like(reference, dtype=torch.bool), SEEDS)
    assert not result["complete"]
    assert result["missing_seeds"] == list(SEEDS)
    assert result["reference_coverage"]["valid_count"] == 0
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("seeds", [(11, 29), (11, 11, 47), (11, True, 47)])
def test_invalid_seed_declarations_rejected(seeds: tuple[int, ...]) -> None:
    predictions, reference, valid = _fixture()
    with pytest.raises(ValueError, match="at least three distinct integer"):
        validation_summary(predictions, reference, valid, seeds)


def test_bad_reference_and_undeclared_prediction_contracts_rejected() -> None:
    predictions, reference, valid = _fixture()
    with pytest.raises(ValueError, match="undeclared seed"):
        validation_summary({**predictions, 99: reference}, reference, valid, SEEDS)
    with pytest.raises(ValueError, match=r"shape \[N,T,Z\]"):
        validation_summary({}, reference[0], valid[0], SEEDS)
    with pytest.raises(ValueError, match="share shape and device"):
        validation_summary({}, reference, valid[0], SEEDS)
    with pytest.raises(TypeError, match="Boolean"):
        validation_summary({}, reference, valid.float(), SEEDS)
    with pytest.raises(ValueError, match="finite and positive"):
        validation_summary({}, reference, torch.ones_like(valid), SEEDS)


def test_validation_figures_are_written_and_can_be_read(tmp_path: Path) -> None:
    predictions, reference, valid = _fixture()
    report = validation_summary(predictions, reference, valid, SEEDS)
    rows = [
        {
            "epoch": epoch,
            "train_loss": 1 / epoch,
            "validation_metrics": {"mean_absolute_error": 0.3 / epoch},
        }
        for epoch in range(1, 4)
    ]
    histories = {f"trial_{trial:02d}": rows for trial in range(12)} | {
        f"seed_{seed}": rows for seed in SEEDS
    }
    paths = write_validation_figures(report, histories, [0.25, 1.0], [-1.0, 1.0], tmp_path, 90)
    assert [path.name for path in paths] == ["learning_curves.png", "validation_error_heatmaps.png"]
    for path in paths:
        assert path.is_file() and path.stat().st_size > 1000
        pixels = mpimg.imread(path)
        assert pixels.ndim == 3 and min(pixels.shape[:2]) > 300
        assert pixels[:, :, :3].std() > 0.05
    assert (
        write_validation_figures(report, histories, [0.25, 1.0], [-1.0, 1.0], tmp_path, 90) == paths
    )


def test_missing_seed_figures_render_undefined_heatmaps_and_empty_histories(tmp_path: Path) -> None:
    _, reference, valid = _fixture()
    report = validation_summary({}, reference, valid, SEEDS)
    paths = write_validation_figures(report, {}, [0.25, 1.0], [-1.0, 1.0], tmp_path, 90)
    assert all(path.stat().st_size > 1000 for path in paths)


def test_figure_coordinates_and_split_checked(tmp_path: Path) -> None:
    predictions, reference, valid = _fixture()
    report = validation_summary(predictions, reference, valid, SEEDS)
    with pytest.raises(ValueError, match="grid_shape"):
        write_validation_figures(report, {}, [0.25], [-1.0, 1.0], tmp_path, 90)
    with pytest.raises(ValueError, match="positive maturities"):
        write_validation_figures(report, {}, [-1.0, 1.0], [-1.0, 1.0], tmp_path, 90)
    with pytest.raises(ValueError, match="positive integer"):
        write_validation_figures(report, {}, [0.25, 1.0], [-1.0, 1.0], tmp_path, 0)
    report["split"] = "test"
    with pytest.raises(ValueError, match="validation report"):
        write_validation_figures(report, {}, [0.25, 1.0], [-1.0, 1.0], tmp_path, 90)
