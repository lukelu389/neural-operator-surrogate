import math
from dataclasses import replace

import pytest
import torch

from nop.data.normalize import fit_train_normalization
from nop.train.coordinates import CoordinateNormalization


def _normalization() -> CoordinateNormalization:
    z = torch.tensor([-2.0, 0.0, 2.0], dtype=torch.float64)
    T = torch.tensor([0.25, 1.0, 2.0], dtype=torch.float64)
    mean = 0.3 + 0.01 * z[None, :] + 0.02 * T[:, None]
    log_std = -3.0 + 0.1 * z[None, :] - 0.2 * T[:, None]
    return CoordinateNormalization(z, T, mean, log_std.exp(), k_scale=0.2)


def test_native_grid_recovers_supplied_training_moments() -> None:
    norm = _normalization()
    T = norm.T_axis[:, None]
    k = norm.k_scale * T.sqrt() * norm.z_axis[None, :]
    mean, std = norm.at(k, T)
    torch.testing.assert_close(mean, norm.mean)
    torch.testing.assert_close(std, norm.std)


def test_affine_mean_and_log_std_are_recovered_at_broadcast_queries() -> None:
    norm = _normalization()
    T = torch.tensor([[0.5], [1.5]], dtype=torch.float64)
    z = torch.tensor([[-1.5, -0.25, 1.0]], dtype=torch.float64)
    k = norm.k_scale * T.sqrt() * z
    mean, std = norm.at(k, T)
    assert mean.shape == (2, 3)
    torch.testing.assert_close(mean, 0.3 + 0.01 * z + 0.02 * T)
    torch.testing.assert_close(std, (-3.0 + 0.1 * z - 0.2 * T).exp())
    assert (std > 0).all()


def test_log_std_interpolation_uses_geometric_instead_of_arithmetic_mean() -> None:
    norm = CoordinateNormalization(
        torch.tensor([-1.0, 1.0], dtype=torch.float64),
        torch.tensor([1.0], dtype=torch.float64),
        torch.tensor([[0.2, 0.4]], dtype=torch.float64),
        torch.tensor([[0.01, 1.0]], dtype=torch.float64),
        k_scale=0.2,
    )
    mean, std = norm.at(torch.tensor(0.0).double(), torch.tensor(1.0).double())
    assert mean.item() == pytest.approx(0.3)
    assert std.item() == pytest.approx(0.1)


def test_clamp_uses_actual_query_z_then_independent_axis_endpoints() -> None:
    norm = _normalization()
    T = torch.tensor([0.01, 0.01, 5.0, 5.0], dtype=torch.float64)
    z = torch.tensor([-4.0, 1.0, -1.0, 4.0], dtype=torch.float64)
    mean, std = norm.at(norm.k_scale * T.sqrt() * z, T)
    bounded_z = z.clamp(norm.z_axis[0], norm.z_axis[-1])
    bounded_T = T.clamp(norm.T_axis[0], norm.T_axis[-1])
    torch.testing.assert_close(mean, 0.3 + 0.01 * bounded_z + 0.02 * bounded_T)
    torch.testing.assert_close(std, (-3.0 + 0.1 * bounded_z - 0.2 * bounded_T).exp())


def test_singleton_axes_produce_constant_moments_and_zero_coordinate_gradients() -> None:
    norm = CoordinateNormalization(
        torch.tensor([0.0]).double(),
        torch.tensor([1.0]).double(),
        torch.tensor([[0.3]]).double(),
        torch.tensor([[0.1]]).double(),
        k_scale=0.2,
    )
    k = torch.tensor([-1.0, 1.0], dtype=torch.float64, requires_grad=True)
    T = torch.tensor([0.5, 2.0], dtype=torch.float64, requires_grad=True)
    mean, std = norm.at(k, T)
    torch.testing.assert_close(mean, torch.full_like(k, 0.3))
    torch.testing.assert_close(std, torch.full_like(k, 0.1))
    (mean + std).sum().backward()
    torch.testing.assert_close(k.grad, torch.zeros_like(k))
    torch.testing.assert_close(T.grad, torch.zeros_like(T))


def test_withheld_labels_never_enter_subset_training_moments() -> None:
    # Synthetic labels only: the odd strikes are withheld, including their masks.
    parameters = torch.arange(20, dtype=torch.float64).reshape(4, 5)
    z = torch.arange(-2.0, 3.0, dtype=torch.float64)
    T = torch.tensor([0.25, 1.0], dtype=torch.float64)
    iv = (
        0.3
        + torch.arange(4, dtype=torch.float64)[:, None, None] * 0.01
        + 0.01 * z[None, None, :]
        + 0.02 * T[None, :, None]
    )

    def fit_subset(labels: torch.Tensor) -> CoordinateNormalization:
        subset = labels[:, :, ::2]
        moments = fit_train_normalization(parameters, subset, torch.isfinite(subset))
        return CoordinateNormalization(z[::2], T, moments.iv_mean, moments.iv_std, k_scale=0.2)

    before = fit_subset(iv)
    iv[:, :, 1::2] = math.nan
    after = fit_subset(iv)
    k = 0.2 * T[:, None].sqrt() * z[None, 1::2]
    for original, modified in zip(before.at(k, T[:, None]), after.at(k, T[:, None]), strict=True):
        torch.testing.assert_close(original, modified, rtol=0.0, atol=0.0)
    mean, _ = after.at(k, T[:, None])
    torch.testing.assert_close(mean, 0.315 + 0.01 * z[None, 1::2] + 0.02 * T[:, None])


def test_roundtrip_preserves_nan_masks_and_does_not_clip_negative_predictions() -> None:
    norm = _normalization()
    k = torch.tensor([0.0, 0.1, -0.1], dtype=torch.float64)
    T = torch.tensor(0.5, dtype=torch.float64)
    iv = torch.tensor([[0.1, math.nan, 0.3], [0.4, 0.5, 0.6]], dtype=torch.float64)
    transformed = norm.transform(iv, k, T)
    torch.testing.assert_close(norm.inverse(transformed, k, T), iv, equal_nan=True)
    negative = norm.inverse(torch.full_like(k, -100.0), k, T)
    assert (negative < 0).all()


def test_query_and_prediction_derivatives_match_analytic_interpolation() -> None:
    norm = _normalization()
    k = torch.tensor(0.05, dtype=torch.float64, requires_grad=True)
    T = torch.tensor(0.6, dtype=torch.float64, requires_grad=True)
    value = torch.tensor(-0.4, dtype=torch.float64, requires_grad=True)
    result = norm.inverse(value, k, T)
    z = k / (norm.k_scale * T.sqrt())
    std = (-3.0 + 0.1 * z - 0.2 * T).exp()
    dz_dk = 1 / (norm.k_scale * T.sqrt())
    dz_dT = -z / (2 * T)
    dk, dT, dv = torch.autograd.grad(result, (k, T, value), create_graph=True)
    torch.testing.assert_close(dk, (0.01 + value * std * 0.1) * dz_dk)
    torch.testing.assert_close(dT, 0.02 + 0.01 * dz_dT + value * std * (0.1 * dz_dT - 0.2))
    torch.testing.assert_close(dv, std)
    assert torch.autograd.gradcheck(norm.inverse, (value, k, T))
    assert torch.autograd.gradgradcheck(norm.inverse, (value, k, T))


def test_moments_are_detached_snapshots_and_cast_to_query_dtype() -> None:
    original = _normalization()
    source_mean = original.mean.clone().requires_grad_()
    norm = replace(original, mean=source_mean)
    assert not norm.mean.requires_grad
    with torch.no_grad():
        source_mean.add_(10.0)
    k = torch.tensor(0.0, dtype=torch.float32)
    T = torch.tensor(0.5, dtype=torch.float32)
    mean, std = norm.at(k, T)
    assert mean.dtype == std.dtype == torch.float32
    assert mean.item() == pytest.approx(0.31)


@pytest.mark.parametrize("field", ["z_axis", "T_axis", "mean", "std"])
def test_nonfinite_or_nonfloat_statistics_are_rejected(field: str) -> None:
    norm = _normalization()
    with pytest.raises(ValueError, match="must be finite"):
        replace(norm, **{field: torch.full_like(getattr(norm, field), math.nan)})
    with pytest.raises(TypeError, match="float32 or float64"):
        replace(norm, **{field: getattr(norm, field).long()})


@pytest.mark.parametrize("field", ["z_axis", "T_axis"])
def test_empty_multidimensional_or_unordered_axes_are_rejected(field: str) -> None:
    norm = _normalization()
    for value in (torch.empty(0).double(), torch.ones(2, 2).double()):
        with pytest.raises(ValueError, match="nonempty one-dimensional"):
            replace(norm, **{field: value})
    for value in (torch.tensor([1.0, 1.0]).double(), torch.tensor([2.0, 1.0]).double()):
        with pytest.raises(ValueError, match="strictly increasing"):
            replace(norm, **{field: value})


def test_moment_shapes_variance_and_maturity_are_validated() -> None:
    norm = _normalization()
    with pytest.raises(ValueError, match="mean must have shape"):
        replace(norm, mean=norm.mean[None])
    for std in (norm.std[:, :1], torch.zeros_like(norm.std), -norm.std):
        with pytest.raises(ValueError, match="same shape as mean and be positive"):
            replace(norm, std=std)
    with pytest.raises(ValueError, match="T_axis must be positive"):
        replace(norm, T_axis=torch.tensor([0.0, 1.0, 2.0]).double())
    with pytest.raises(ValueError, match="share dtype and device"):
        replace(norm, mean=norm.mean.float())


@pytest.mark.parametrize("scale", [0.0, -0.2, math.inf, math.nan])
def test_invalid_scale_is_rejected(scale: float) -> None:
    with pytest.raises(ValueError, match="k_scale must be finite and positive"):
        replace(_normalization(), k_scale=scale)


@pytest.mark.parametrize("k,T", [(math.nan, 1.0), (math.inf, 1.0), (0.0, 0.0), (0.0, -1.0)])
def test_invalid_queries_are_rejected(k: float, T: float) -> None:
    with pytest.raises(ValueError, match="finite with positive T"):
        _normalization().at(torch.tensor(k).double(), torch.tensor(T).double())


def test_infinite_predictions_and_incompatible_dtypes_are_rejected() -> None:
    norm = _normalization()
    k, T = torch.tensor(0.0).double(), torch.tensor(1.0).double()
    for operation in (norm.transform, norm.inverse):
        with pytest.raises(ValueError, match="not infinities"):
            operation(torch.tensor(math.inf).double(), k, T)
        with pytest.raises(ValueError, match="share dtype and device"):
            operation(torch.tensor(1.0).float(), k, T)
    with pytest.raises(ValueError, match="share dtype and device"):
        norm.at(k.float(), T)
    with pytest.raises(TypeError, match="float32 or float64"):
        norm.at(k.long(), T)
