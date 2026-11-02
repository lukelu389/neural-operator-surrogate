import math

import numpy as np
import pytest
import torch

from nop.data.normalize import fit_train_normalization


def _training_arrays() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    parameters = torch.arange(20, dtype=torch.float64).reshape(4, 5)
    iv = torch.tensor(
        [
            [[0.1, 0.2], [0.3, 0.4]],
            [[0.3, math.nan], [0.5, 0.8]],
            [[0.5, 0.6], [math.nan, 1.2]],
            [[0.7, 0.8], [0.9, math.nan]],
        ],
        dtype=torch.float64,
    )
    return parameters, iv, torch.isfinite(iv)


def test_population_moments_use_only_valid_quotes_at_each_grid_point() -> None:
    parameters, iv, valid = _training_arrays()
    normalization = fit_train_normalization(parameters, iv, valid)
    torch.testing.assert_close(normalization.parameter_mean, parameters.mean(dim=0))
    torch.testing.assert_close(
        normalization.parameter_std, torch.full((5,), math.sqrt(31.25), dtype=torch.float64)
    )
    torch.testing.assert_close(
        normalization.iv_mean, torch.from_numpy(np.nanmean(iv.numpy(), axis=0))
    )
    torch.testing.assert_close(
        normalization.iv_std, torch.from_numpy(np.nanstd(iv.numpy(), axis=0))
    )
    assert normalization.iv_count.tolist() == [[4, 3], [3, 3]]
    assert normalization.iv_count.dtype == torch.int64
    transformed = normalization.transform_iv(iv)
    assert torch.equal(torch.isnan(transformed), ~valid)
    torch.testing.assert_close(torch.nanmean(transformed, dim=0), torch.zeros((2, 2)).double())
    torch.testing.assert_close(
        torch.nanmean(transformed.square(), dim=0), torch.ones((2, 2)).double()
    )
    torch.testing.assert_close(normalization.inverse_iv(transformed), iv, equal_nan=True)


def test_validation_transform_cannot_refit_training_statistics() -> None:
    parameters, iv, valid = _training_arrays()
    normalization = fit_train_normalization(parameters, iv, valid)
    means_before = normalization.iv_mean.clone()
    validation_iv = iv + 10.0
    validation_parameters = parameters + 100.0
    expected = 10.0 / normalization.iv_std
    torch.testing.assert_close(
        torch.nanmean(normalization.transform_iv(validation_iv), dim=0), expected
    )
    transformed_parameters = normalization.transform_parameters(validation_parameters)
    torch.testing.assert_close(
        transformed_parameters.mean(dim=0), 100.0 / normalization.parameter_std
    )
    assert torch.equal(normalization.iv_mean, means_before)
    # A distinct fit on a nested training subset has its own moments.
    subset = fit_train_normalization(parameters[:3], iv[:3], valid[:3])
    assert not torch.equal(subset.iv_mean, normalization.iv_mean)
    assert subset.iv_count.tolist() == [[3, 2], [2, 3]]


def test_fitting_detaches_moments_and_transforms_preserve_gradients_and_dtype() -> None:
    parameters, iv, valid = _training_arrays()
    normalization = fit_train_normalization(parameters.requires_grad_(), iv.requires_grad_(), valid)
    assert not normalization.parameter_mean.requires_grad
    assert not normalization.parameter_std.requires_grad
    assert not normalization.iv_mean.requires_grad
    assert not normalization.iv_std.requires_grad
    predicted = torch.ones((3, 2, 2), dtype=torch.float32, requires_grad=True)
    inverse = normalization.inverse_iv(predicted)
    assert inverse.dtype == torch.float32
    inverse.sum().backward()
    torch.testing.assert_close(predicted.grad, normalization.iv_std.float().expand_as(predicted))
    vector = parameters[0].detach().float().requires_grad_()
    normalized = normalization.transform_parameters(vector)
    assert normalized.shape == (5,)
    normalized.sum().backward()
    torch.testing.assert_close(vector.grad, 1 / normalization.parameter_std.float())


def test_fitting_does_not_mutate_source_arrays() -> None:
    parameters, iv, valid = _training_arrays()
    originals = tuple(value.clone() for value in (parameters, iv, valid))
    fit_train_normalization(parameters, iv, valid)
    for actual, expected in zip((parameters, iv, valid), originals, strict=True):
        torch.testing.assert_close(actual, expected, equal_nan=True)


@pytest.mark.parametrize("value", [math.nan, math.inf, 0.0, -0.1])
def test_valid_iv_requires_finite_positive_values(value: float) -> None:
    parameters, iv, valid = _training_arrays()
    iv[0, 0, 0] = value
    with pytest.raises(ValueError, match="finite and positive"):
        fit_train_normalization(parameters, iv, valid)


@pytest.mark.parametrize("value", [0.0, 1.0, math.inf])
def test_invalid_iv_must_be_nan(value: float) -> None:
    parameters, iv, valid = _training_arrays()
    iv[1, 0, 1] = value
    with pytest.raises(ValueError, match="must be NaN"):
        fit_train_normalization(parameters, iv, valid)


def test_missing_grid_point_cannot_borrow_other_points_or_validation_quotes() -> None:
    parameters, iv, valid = _training_arrays()
    iv[:, 0, 0] = math.nan
    valid[:, 0, 0] = False
    with pytest.raises(ValueError, match="every grid point"):
        fit_train_normalization(parameters, iv, valid)


@pytest.mark.parametrize("constant", ["parameter", "iv"])
def test_zero_training_variance_is_rejected(constant: str) -> None:
    parameters, iv, valid = _training_arrays()
    if constant == "parameter":
        parameters[:, 2] = 1.0
    else:
        iv[:, 0, 0] = 0.2
    with pytest.raises(ValueError, match="nonzero training variance"):
        fit_train_normalization(parameters, iv, valid)


def test_shape_dtype_and_nonfinite_parameters_are_rejected() -> None:
    parameters, iv, valid = _training_arrays()
    with pytest.raises(ValueError, match="same N"):
        fit_train_normalization(parameters[:3], iv, valid)
    with pytest.raises(ValueError, match="same shape"):
        fit_train_normalization(parameters, iv, valid[:, :, :1])
    with pytest.raises(TypeError, match="Boolean"):
        fit_train_normalization(parameters, iv, valid.long())
    with pytest.raises(TypeError, match="float32 or float64"):
        fit_train_normalization(parameters.long(), iv, valid)
    with pytest.raises(ValueError, match="share dtype"):
        fit_train_normalization(parameters.float(), iv, valid)
    parameters[0, 0] = math.nan
    with pytest.raises(ValueError, match="parameters must be finite"):
        fit_train_normalization(parameters, iv, valid)


def test_transforms_preserve_mask_and_reject_infinity_or_incorrect_grid() -> None:
    parameters, iv, valid = _training_arrays()
    normalization = fit_train_normalization(parameters, iv, valid)
    standardized = torch.tensor([[math.nan, -100.0], [0.0, 1.0]], dtype=torch.float64)
    inverted = normalization.inverse_iv(standardized)
    assert torch.equal(torch.isnan(inverted), torch.isnan(standardized))
    assert inverted[0, 1] < 0  # Prediction validity belongs to downstream metrics, not clipping.
    with pytest.raises(ValueError, match="grid shape"):
        normalization.transform_iv(torch.ones((2, 3), dtype=torch.float64))
    with pytest.raises(ValueError, match="infinities"):
        normalization.inverse_iv(torch.full((2, 2), math.inf, dtype=torch.float64))
