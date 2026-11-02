"""Independent analytic fixtures for the predeclared M0 query adapter."""

import pytest
import torch

from nop.eval.interpolation import InterpolationStatus, interpolate_m0_iv

DTYPE = torch.float64


def _grid() -> tuple[torch.Tensor, torch.Tensor]:
    maturity = torch.tensor([1 / 12, 1 / 6, 0.5], dtype=DTYPE)
    z = torch.linspace(-2.5, 2.5, 15, dtype=DTYPE)
    return maturity, z[None, :] * 0.2 * maturity[:, None].sqrt()


def test_constant_vol_on_withheld_strikes_with_generic_leading_axes() -> None:
    maturities, k = _grid()
    vol = torch.tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype=DTYPE)
    native_iv = vol[:, :, None, None].expand(2, 3, 3, 8)
    result = interpolate_m0_iv(
        native_iv,
        native_maturities=maturities,
        native_k=k[:, ::2],
        query_maturity=maturities[:, None],
        query_k=k[:, 1::2],
    )
    assert result.iv.shape == (2, 3, 3, 7)
    assert result.valid.all()
    assert (result.status == InterpolationStatus.SUCCESS).all()
    torch.testing.assert_close(result.iv, vol[:, :, None, None].expand_as(result.iv))


def test_affine_total_variance_uses_fixed_k_across_scaled_native_grids() -> None:
    maturities, k = _grid()
    w = 0.01 + 0.03 * maturities[:, None] + 0.02 * k
    query_t = torch.tensor([1 / 52, 0.12, 0.3], dtype=DTYPE)
    query_k = torch.tensor([-0.04, 0.02, 0.03], dtype=DTYPE)
    result = interpolate_m0_iv(
        torch.sqrt(w / maturities[:, None]),
        native_maturities=maturities,
        native_k=k,
        query_maturity=query_t,
        query_k=query_k,
    )
    expected = torch.sqrt((0.01 + 0.03 * query_t + 0.02 * query_k) / query_t)
    assert result.valid.all()
    torch.testing.assert_close(result.iv, expected, atol=2e-15, rtol=2e-14)


def test_native_nodes_reproduce_arbitrary_positive_iv() -> None:
    maturities, k = _grid()
    native_iv = 0.2 + 0.1 * torch.cos(15 * k) + maturities[:, None] * 0.02
    result = interpolate_m0_iv(
        native_iv,
        native_maturities=maturities,
        native_k=k,
        query_maturity=maturities[:, None],
        query_k=k,
    )
    assert result.valid.all()
    torch.testing.assert_close(result.iv, native_iv, atol=1e-15, rtol=1e-15)


def test_known_natural_cubic_curve_and_endpoint_second_derivatives() -> None:
    maturities = torch.tensor([1.0, 2.0], dtype=DTYPE)
    k = torch.tensor([0.0, 1.0, 2.0], dtype=DTYPE).expand(2, -1)
    w = torch.tensor([1.0, 2.0, 1.0], dtype=DTYPE).expand(2, -1) * 0.04
    query_k = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0], dtype=DTYPE)
    query_k.requires_grad_()
    result = interpolate_m0_iv(
        torch.sqrt(w / maturities[:, None]),
        native_maturities=maturities,
        native_k=k,
        query_maturity=torch.tensor(1.0, dtype=DTYPE),
        query_k=query_k,
    )
    reflected_x = torch.minimum(query_k, 2 - query_k)
    expected_w = 0.04 * (1 + 1.5 * reflected_x - 0.5 * reflected_x.pow(3))
    assert result.valid.all()
    actual_w = result.iv.square()
    torch.testing.assert_close(actual_w, expected_w, atol=2e-16, rtol=2e-14)
    derivative = torch.autograd.grad(actual_w.sum(), query_k, create_graph=True)[0]
    second = torch.autograd.grad(derivative.sum(), query_k)[0]
    torch.testing.assert_close(second[[0, -1]], torch.zeros(2, dtype=DTYPE), atol=1e-15, rtol=0)


def test_short_maturity_constant_vol_and_scalar_query() -> None:
    maturities, k = _grid()
    result = interpolate_m0_iv(
        torch.full_like(k, 0.2),
        native_maturities=maturities,
        native_k=k,
        query_maturity=torch.tensor(1 / 52, dtype=DTYPE),
        query_k=torch.tensor(0.01, dtype=DTYPE),
    )
    assert result.iv.shape == ()
    assert result.valid
    torch.testing.assert_close(result.iv, torch.tensor(0.2, dtype=DTYPE))


def test_short_maturity_requires_intersection_of_both_strike_domains() -> None:
    maturities = torch.tensor([1.0, 2.0], dtype=DTYPE)
    k = torch.tensor([[-0.1, 0.0, 0.1], [-0.05, 0.05, 0.15]], dtype=DTYPE)
    result = interpolate_m0_iv(
        torch.full_like(k, 0.2),
        native_maturities=maturities,
        native_k=k,
        query_maturity=torch.tensor(0.5, dtype=DTYPE),
        query_k=torch.tensor([-0.075, -0.05, 0.1, 0.125], dtype=DTYPE),
    )
    assert result.valid.tolist() == [False, True, True, False]
    assert torch.isnan(result.iv[[0, 3]]).all()
    assert (result.status[[0, 3]] == InterpolationStatus.OUTSIDE_STRIKE_SUPPORT).all()


def test_no_strike_or_long_maturity_extrapolation() -> None:
    maturities, k = _grid()
    result = interpolate_m0_iv(
        torch.full_like(k, 0.2),
        native_maturities=maturities,
        native_k=k,
        query_maturity=torch.tensor([1 / 12, 0.5, 0.51], dtype=DTYPE),
        query_k=torch.tensor([k[0, 0] - 1e-6, k[-1, -1] + 1e-6, 0.0], dtype=DTYPE),
    )
    assert not result.valid.any()
    assert torch.isnan(result.iv).all()
    assert result.status.tolist() == [
        InterpolationStatus.OUTSIDE_STRIKE_SUPPORT,
        InterpolationStatus.OUTSIDE_STRIKE_SUPPORT,
        InterpolationStatus.OUTSIDE_MATURITY_SUPPORT,
    ]


@pytest.mark.parametrize("query_t", [0.5, 0.875])
def test_nonpositive_short_extrapolated_variance_fails(query_t: float) -> None:
    maturities = torch.tensor([1.0, 1.75], dtype=DTYPE)
    k = torch.tensor([-0.1, 0.1], dtype=DTYPE).expand(2, -1)
    # Exactly representable native IV gives w(T)=8*T-7, zero at T=0.875.
    iv = torch.tensor([1.0, 2.0], dtype=DTYPE)[:, None].expand_as(k)
    result = interpolate_m0_iv(
        iv,
        native_maturities=maturities,
        native_k=k,
        query_maturity=torch.tensor(query_t, dtype=DTYPE),
        query_k=torch.tensor(0.0, dtype=DTYPE),
    )
    assert not result.valid
    assert torch.isnan(result.iv)
    assert result.status == InterpolationStatus.NONPOSITIVE_VARIANCE


@pytest.mark.parametrize("query_t", [1.0, 1.5])
def test_positive_knots_can_have_nonpositive_cubic_variance(query_t: float) -> None:
    maturities = torch.tensor([1.0, 2.0], dtype=DTYPE)
    k = torch.arange(4, dtype=DTYPE).expand(2, -1)
    w = torch.tensor([[0.01, 0.001, 0.001, 0.01], [0.04, 0.04, 0.04, 0.04]], dtype=DTYPE)
    result = interpolate_m0_iv(
        torch.sqrt(w / maturities[:, None]),
        native_maturities=maturities,
        native_k=k,
        query_maturity=torch.tensor(query_t, dtype=DTYPE),
        query_k=torch.tensor(1.5, dtype=DTYPE),
    )
    assert not result.valid
    assert result.status == InterpolationStatus.NONPOSITIVE_VARIANCE
    assert torch.isnan(result.iv)


@pytest.mark.parametrize("bad_iv", [0.0, -0.2, float("nan"), float("inf"), 1e308])
def test_invalid_reference_quote_invalidates_its_whole_row_only(bad_iv: float) -> None:
    maturities, k = _grid()
    iv = torch.full((2, *k.shape), 0.2, dtype=DTYPE)
    iv[0, 1, -1] = bad_iv
    result = interpolate_m0_iv(
        iv,
        native_maturities=maturities,
        native_k=k,
        query_maturity=torch.tensor([1 / 12, 1 / 6, 0.5, 0.12, 1 / 52], dtype=DTYPE),
        query_k=torch.tensor(0.0, dtype=DTYPE),
    )
    assert result.valid[0].tolist() == [True, False, True, False, False]
    assert result.valid[1].all()
    assert torch.isnan(result.iv[0, [1, 3, 4]]).all()
    assert (result.status[0, [1, 3, 4]] == InterpolationStatus.INVALID_NATIVE_ROW).all()


def test_explicit_missing_quote_mask_is_not_silently_dropped() -> None:
    maturities, k = _grid()
    mask = torch.ones_like(k, dtype=torch.bool)
    mask[0, -1] = False
    result = interpolate_m0_iv(
        torch.full_like(k, 0.2),
        native_maturities=maturities,
        native_k=k,
        native_valid=mask,
        query_maturity=maturities,
        query_k=torch.tensor(0.0, dtype=DTYPE),
    )
    assert result.valid.tolist() == [False, True, True]
    assert result.status[0] == InterpolationStatus.INVALID_NATIVE_ROW
    assert torch.isnan(result.iv[0])


@pytest.mark.parametrize("bad_t", [0.0, -1.0, float("nan"), float("inf")])
def test_invalid_query_maturity_is_explicit_failure(bad_t: float) -> None:
    maturities, k = _grid()
    result = interpolate_m0_iv(
        torch.full_like(k, 0.2),
        native_maturities=maturities,
        native_k=k,
        query_maturity=torch.tensor(bad_t, dtype=DTYPE),
        query_k=torch.tensor(0.0, dtype=DTYPE),
    )
    assert not result.valid
    assert torch.isnan(result.iv)
    assert result.status == InterpolationStatus.INVALID_QUERY


def test_nonfinite_query_k_is_explicit_failure() -> None:
    maturities, k = _grid()
    result = interpolate_m0_iv(
        torch.full_like(k, 0.2),
        native_maturities=maturities,
        native_k=k,
        query_maturity=maturities[0],
        query_k=torch.tensor([float("nan"), float("inf")], dtype=DTYPE),
    )
    assert not result.valid.any()
    assert torch.isnan(result.iv).all()
    assert (result.status == InterpolationStatus.INVALID_QUERY).all()


def test_float32_and_native_prediction_gradient() -> None:
    maturities, k = _grid()
    iv = torch.full_like(k, 0.2, dtype=torch.float32, requires_grad=True)
    result = interpolate_m0_iv(
        iv,
        native_maturities=maturities,
        native_k=k,
        query_maturity=torch.tensor(0.12),
        query_k=torch.tensor(0.01),
    )
    assert result.valid
    assert result.iv.dtype == torch.float32
    result.iv.backward()
    assert iv.grad is not None
    assert torch.isfinite(iv.grad).all()
    assert iv.grad.abs().sum() > 0


@pytest.mark.parametrize("malformation", ["maturity_order", "strike_order", "shape", "mask"])
def test_malformed_native_grid_or_mask_rejected(malformation: str) -> None:
    maturities, k = _grid()
    iv = torch.full_like(k, 0.2)
    mask = torch.ones_like(k, dtype=torch.bool)
    if malformation == "maturity_order":
        maturities = maturities.flip(0)
    elif malformation == "strike_order":
        k = k.flip(1)
    elif malformation == "shape":
        k = k[:, :-1]
    else:
        mask = mask.float()
    with pytest.raises(ValueError):
        interpolate_m0_iv(
            iv,
            native_maturities=maturities,
            native_k=k,
            native_valid=mask,
            query_maturity=torch.tensor(0.1),
            query_k=torch.tensor(0.0),
        )
