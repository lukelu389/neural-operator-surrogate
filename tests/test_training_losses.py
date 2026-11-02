import math

import pytest
import torch

from nop.train.losses import masked_mse, vega_weights


def test_masked_mse_uses_quote_count_and_excludes_nan_before_arithmetic() -> None:
    prediction = torch.tensor([1.0, math.nan, 3.0], dtype=torch.float64, requires_grad=True)
    target = torch.tensor([0.0, math.nan, 1.0], dtype=torch.float64, requires_grad=True)
    valid = torch.tensor([True, False, True])
    loss = masked_mse(prediction, target, valid)
    assert loss.item() == 2.5
    loss.backward()
    torch.testing.assert_close(prediction.grad, torch.tensor([1.0, 0.0, 2.0]).double())
    torch.testing.assert_close(target.grad, torch.tensor([-1.0, 0.0, -2.0]).double())


def test_weighted_loss_divides_by_valid_count_and_masked_weights_have_safe_gradients() -> None:
    prediction = torch.tensor([1.0, 2.0, math.inf], requires_grad=True)
    target = torch.tensor([0.0, 0.0, math.nan])
    weights = torch.tensor([2.0, 4.0, math.nan], requires_grad=True)
    loss = masked_mse(prediction, target, torch.tensor([True, True, False]), weights)
    assert loss.item() == 9.0  # (2 * 1**2 + 4 * 2**2) / 2, not / 6.
    loss.backward()
    torch.testing.assert_close(prediction.grad, torch.tensor([2.0, 8.0, 0.0]))
    torch.testing.assert_close(weights.grad, torch.tensor([0.5, 2.0, 0.0]))


def test_vega_weighted_standardized_loss_matches_linearized_price_error() -> None:
    vega = torch.tensor([[[1.0, 2.0]], [[3.0, math.nan]]], dtype=torch.float64)
    std = torch.tensor([[0.1, 0.2]], dtype=torch.float64)
    valid = torch.isfinite(vega)
    reference = torch.tensor([[[0.2, 0.3]], [[0.4, math.nan]]], dtype=torch.float64)
    prediction = torch.tensor([[[0.21, 0.28]], [[0.43, math.nan]]], dtype=torch.float64)
    weights = vega_weights(vega, std, valid)
    raw_mean = torch.tensor([0.1**2, 0.4**2, 0.3**2], dtype=torch.float64).mean()
    actual = masked_mse(prediction / std, reference / std, valid, weights)
    price_residual = vega[valid] * (prediction[valid] - reference[valid])
    torch.testing.assert_close(actual, price_residual.square().mean() / raw_mean)
    assert weights[~valid].item() == 0.0
    assert weights[valid].mean().item() == pytest.approx(1.0)

    # Reuse fixed training weights in each batch; their means need not be one.
    first = masked_mse(prediction[:1] / std, reference[:1] / std, valid[:1], weights[:1])
    second = masked_mse(prediction[1:] / std, reference[1:] / std, valid[1:], weights[1:])
    torch.testing.assert_close((2 * first + second) / 3, actual)
    assert weights[:1][valid[:1]].mean().item() != pytest.approx(1.0)


def test_weight_fit_detaches_statistics_and_ignores_masked_nan_standard_deviation() -> None:
    vega = torch.tensor([1.0, math.nan, 2.0], requires_grad=True)
    std = torch.tensor([0.2, math.nan, 0.4], requires_grad=True)
    valid = torch.tensor([True, False, True])
    weights = vega_weights(vega, std, valid)
    assert not weights.requires_grad
    torch.testing.assert_close(weights, torch.tensor([2 / 17, 0, 32 / 17]))


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_nonfinite_valid_prediction_fails_without_resampling(value: float) -> None:
    with pytest.raises(FloatingPointError, match="predictions must be finite"):
        masked_mse(torch.tensor([value]), torch.zeros(1), torch.ones(1, dtype=torch.bool))


def test_nonfinite_valid_target_and_overflow_fail_explicitly() -> None:
    with pytest.raises(ValueError, match="targets must be finite"):
        masked_mse(torch.ones(1), torch.tensor([math.nan]), torch.ones(1, dtype=torch.bool))
    with pytest.raises(FloatingPointError, match="nonfinite loss"):
        masked_mse(torch.tensor([1e30]), torch.zeros(1), torch.ones(1, dtype=torch.bool))


@pytest.mark.parametrize("value", [math.nan, math.inf, -1.0])
def test_invalid_valid_quote_weight_is_rejected(value: float) -> None:
    with pytest.raises(ValueError, match="finite and nonnegative"):
        masked_mse(
            torch.ones(1), torch.zeros(1), torch.ones(1, dtype=torch.bool), torch.tensor([value])
        )


@pytest.mark.parametrize("shape", [(0,), (2, 3)])
def test_empty_loss_and_weight_fit_are_undefined(shape: tuple[int, ...]) -> None:
    values = torch.full(shape, math.nan)
    valid = torch.zeros(shape, dtype=torch.bool)
    with pytest.raises(ValueError, match="at least one valid quote"):
        masked_mse(values, values, valid)
    with pytest.raises(ValueError, match="at least one valid training quote"):
        vega_weights(values, torch.tensor(0.1), valid)


@pytest.mark.parametrize("name", ["vega", "target_std"])
@pytest.mark.parametrize("value", [math.nan, math.inf, 0.0, -1.0])
def test_weight_fit_rejects_nonpositive_or_nonfinite_valid_inputs(name: str, value: float) -> None:
    arguments = {"vega": torch.ones(1), "target_std": torch.ones(1)}
    arguments[name] = torch.tensor([value])
    with pytest.raises(ValueError, match="finite and positive"):
        vega_weights(**arguments, valid=torch.ones(1, dtype=torch.bool))


@pytest.mark.parametrize("value", [1e-30, 1e30])
def test_weight_underflow_and_overflow_fail_without_clipping(value: float) -> None:
    with pytest.raises(FloatingPointError, match="finite positive training mean"):
        vega_weights(torch.tensor([value]), torch.ones(1), torch.ones(1, dtype=torch.bool))


def test_loss_rejects_broadcasting_and_implicit_masks() -> None:
    values = torch.ones(2, 3)
    valid = torch.ones_like(values, dtype=torch.bool)
    with pytest.raises(ValueError, match="same shape"):
        masked_mse(values, values[0], valid)
    with pytest.raises(ValueError, match="same shape"):
        masked_mse(values, values, valid[0])
    with pytest.raises(ValueError, match="same shape"):
        masked_mse(values, values, valid, torch.ones(3))
    with pytest.raises(TypeError, match="Boolean"):
        masked_mse(values, values, valid.long())
    with pytest.raises(TypeError, match="float32 or float64"):
        masked_mse(values.long(), values, valid)


def test_weight_fit_rejects_bad_shape_or_dtype() -> None:
    values = torch.ones(2, 3)
    valid = torch.ones_like(values, dtype=torch.bool)
    with pytest.raises(ValueError, match="broadcast"):
        vega_weights(values, torch.ones(4), valid)
    with pytest.raises(ValueError, match="same shape"):
        vega_weights(values, torch.ones(3), valid[0])
    with pytest.raises(TypeError, match="Boolean"):
        vega_weights(values, torch.ones(3), valid.long())
    with pytest.raises(TypeError, match="float32 or float64"):
        vega_weights(values, torch.ones(3).long(), valid)
