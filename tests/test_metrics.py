import math

import pytest
import torch

from nop.eval.metrics import iv_metrics

ERROR_METRICS = (
    "mean_absolute_error",
    "root_mean_squared_error",
    "p99_absolute_error",
    "max_absolute_error",
)


def test_decimal_iv_errors_are_vol_points_with_linear_tail_quantile() -> None:
    reference = torch.full((4,), 0.2, dtype=torch.float64)
    prediction = torch.tensor([0.2, 0.21, 0.18, 1.2], dtype=torch.float64)
    result = iv_metrics(prediction, reference, torch.ones(4, dtype=torch.bool))
    assert result["mean_absolute_error"] == pytest.approx(25.75)
    assert result["root_mean_squared_error"] == pytest.approx(math.sqrt(2501.25))
    assert result["p99_absolute_error"] == pytest.approx(97.06)
    assert result["max_absolute_error"] == pytest.approx(100.0)
    assert result["attempted"] == result["valid_count"] == 4
    assert result["reference_failure_count"] == result["prediction_failure_count"] == 0
    assert result["reference_failure_rate"] == result["prediction_failure_rate"] == 0.0


def test_mask_is_shared_and_quote_weighting_does_not_average_surface_averages() -> None:
    reference = torch.tensor([[0.2, math.nan, math.nan], [0.2, 0.2, 0.2]], dtype=torch.float64)
    prediction = torch.tensor([[0.3, math.inf, -1.0], [0.2, 0.2, 0.2]], dtype=torch.float64)
    result = iv_metrics(prediction, reference, torch.isfinite(reference))
    assert result["mean_absolute_error"] == pytest.approx(2.5)
    assert result["root_mean_squared_error"] == pytest.approx(5.0)
    assert result["attempted"] == 6
    assert result["valid_count"] == 4
    assert result["reference_failure_count"] == 2
    assert result["reference_failure_rate"] == pytest.approx(1 / 3)
    assert result["prediction_failure_count"] == result["prediction_failure_rate"] == 0


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf, 0.0, -0.1])
def test_invalid_predictions_make_accuracy_undefined_without_shrinking_mask(value: float) -> None:
    reference = torch.tensor([0.2, 0.3, math.nan])
    prediction = torch.tensor([0.2, value, math.nan])
    valid = torch.tensor([True, True, False])
    result = iv_metrics(prediction, reference, valid)
    assert result["valid_count"] == 2
    assert result["reference_failure_count"] == 1
    assert result["prediction_failure_count"] == 1
    assert result["prediction_failure_rate"] == 0.5
    assert all(result[key] is None for key in ERROR_METRICS)
    assert valid.tolist() == [True, True, False]


def test_prediction_failures_include_every_nonfinite_or_nonpositive_valid_quote() -> None:
    prediction = torch.tensor([math.nan, math.inf, 0.0, -1.0, 0.2])
    result = iv_metrics(prediction, torch.full((5,), 0.2), torch.ones(5, dtype=torch.bool))
    assert result["prediction_failure_count"] == 4
    assert result["prediction_failure_rate"] == 0.8


@pytest.mark.parametrize("shape", [(0,), (2, 3)])
def test_empty_strata_report_zero_counts_and_undefined_errors(shape: tuple[int, ...]) -> None:
    reference = torch.full(shape, math.nan)
    result = iv_metrics(reference, reference, torch.zeros(shape, dtype=torch.bool))
    assert result["attempted"] == reference.numel()
    assert result["valid_count"] == result["prediction_failure_count"] == 0
    assert result["prediction_failure_rate"] is None
    assert result["reference_failure_rate"] == (1.0 if reference.numel() else None)
    assert all(result[key] is None for key in ERROR_METRICS)


def test_one_quote_and_perfect_prediction_have_defined_tail_errors() -> None:
    reference = torch.tensor([0.3], dtype=torch.float64)
    result = iv_metrics(reference, reference, torch.ones(1, dtype=torch.bool))
    assert all(result[key] == 0.0 for key in ERROR_METRICS)


@pytest.mark.parametrize("value", [math.nan, math.inf, 0.0, -0.1])
def test_reference_mask_cannot_mark_bad_iv_as_valid(value: float) -> None:
    with pytest.raises(ValueError, match="reference IV must be finite and positive"):
        iv_metrics(torch.tensor([0.2]), torch.tensor([value]), torch.ones(1, dtype=torch.bool))


def test_shape_and_dtype_contract_rejects_accidental_broadcasting() -> None:
    values = torch.ones(2, 3)
    valid = torch.ones_like(values, dtype=torch.bool)
    with pytest.raises(ValueError, match="same shape"):
        iv_metrics(values[0], values, valid)
    with pytest.raises(ValueError, match="same shape"):
        iv_metrics(values, values, valid[0])
    with pytest.raises(TypeError, match="Boolean"):
        iv_metrics(values, values, valid.long())
    with pytest.raises(TypeError, match="float32 or float64"):
        iv_metrics(values.long(), values, valid)


def test_float32_large_errors_reduce_in_float64_without_spurious_square_overflow() -> None:
    prediction = torch.tensor([1e20], dtype=torch.float32, requires_grad=True)
    reference = torch.ones(1, dtype=torch.float64)
    result = iv_metrics(prediction, reference, torch.ones(1, dtype=torch.bool))
    expected = 100.0 * (prediction.item() - 1.0)
    for key in ERROR_METRICS:
        assert result[key] == pytest.approx(expected)
    assert prediction.grad is None


def test_unrepresentable_metric_is_undefined_instead_of_serializing_infinity() -> None:
    prediction = torch.tensor([1e308], dtype=torch.float64)
    reference = torch.ones(1, dtype=torch.float64)
    result = iv_metrics(prediction, reference, torch.ones(1, dtype=torch.bool))
    assert result["prediction_failure_count"] == 0  # Input is finite and positive.
    assert all(result[key] is None for key in ERROR_METRICS)
