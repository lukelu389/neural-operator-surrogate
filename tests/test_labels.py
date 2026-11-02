"""Numerical surface fixtures for converged IV labels and retained failures."""

from dataclasses import replace

import pytest
import torch

from nop.data.labels import LabelSettings, LabelStatus, label_surfaces
from nop.pricing.bs import black_scholes_price
from nop.pricing.heston_grid import heston_cos_grid_price
from nop.pricing.iv import IVSettings, IVStatus


@pytest.fixture
def settings() -> LabelSettings:
    return LabelSettings(
        term_levels=(128, 256, 512, 1024, 2048),
        width=20.0,
        reference_width=28.0,
        price_rtol=1e-8,
        iv_error_tolerance=1e-6,
        iv=IVSettings(
            price_atol=0.0,
            price_rtol=1e-10,
            iv_atol=1e-10,
            min_normalized_vega=1e-4,
            volatility_upper=5.0,
            max_iterations=100,
        ),
    )


@pytest.fixture
def grid() -> tuple[torch.Tensor, torch.Tensor]:
    maturities = torch.tensor([0.25, 1.0], dtype=torch.float64)
    strikes = torch.tensor([[0.95, 1.0, 1.05], [0.9, 1.0, 1.1]], dtype=torch.float64)
    return strikes, maturities


@pytest.mark.parametrize("v0", [0.04, 0.09])
def test_deterministic_variance_recovers_known_bs_surface(
    settings: LabelSettings, grid: tuple[torch.Tensor, torch.Tensor], v0: float
) -> None:
    strikes, maturities = grid
    parameters = torch.tensor([[v0, 1.3, 0.04, 0.0, -0.7]], dtype=torch.float64)
    actual = label_surfaces(parameters, strikes, maturities, 1.0, -0.02, 0.03, settings=settings)
    variance = 0.04 + (v0 - 0.04) * -torch.expm1(-1.3 * maturities) / (1.3 * maturities)
    expected_iv = variance.sqrt()[:, None].expand_as(strikes)
    expected_price = black_scholes_price(
        1.0, strikes, maturities[:, None], expected_iv, -0.02, 0.03, option_type="put"
    )
    assert torch.all(actual.valid)
    torch.testing.assert_close(actual.iv[0], expected_iv, rtol=0, atol=2e-10)
    torch.testing.assert_close(actual.price[0], expected_price, rtol=1e-11, atol=1e-12)
    assert torch.all(actual.discrepancy >= 0)


def test_refined_labels_match_a_finer_fixed_reference(
    settings: LabelSettings, grid: tuple[torch.Tensor, torch.Tensor]
) -> None:
    strikes, maturities = grid
    parameters = torch.tensor(
        [[0.04, 1.5, 0.04, 0.3, -0.7], [0.09, 0.7, 0.02, 0.8, -0.95]], dtype=torch.float64
    )
    actual = label_surfaces(parameters, strikes, maturities, 1.0, 0.02, 0.0, settings=settings)
    expected = heston_cos_grid_price(
        parameters,
        strikes,
        maturities,
        1.0,
        0.02,
        0.0,
        n_terms=8192,
        truncation_width=36.0,
    )
    assert torch.all(actual.valid)
    torch.testing.assert_close(actual.price, expected, rtol=0, atol=1e-8)
    repriced = black_scholes_price(
        1.0, strikes, maturities[:, None], actual.iv, 0.02, 0.0, option_type="put"
    )
    torch.testing.assert_close(repriced, actual.price, rtol=0, atol=2e-10)
    assert torch.all(actual.discrepancy <= settings.iv_error_tolerance * actual.vega)


def test_underresolved_cap_retains_failed_quotes_with_nan_iv(
    settings: LabelSettings, grid: tuple[torch.Tensor, torch.Tensor]
) -> None:
    strikes, maturities = grid
    parameters = torch.tensor([[0.04, 1.5, 0.04, 0.3, -0.7]], dtype=torch.float64)
    insufficient = replace(settings, term_levels=(2,), width=1.0, reference_width=2.0)
    actual = label_surfaces(parameters, strikes, maturities, 1.0, 0.02, 0.0, settings=insufficient)
    assert actual.price.shape == actual.iv.shape == actual.status.shape == (1, 2, 3)
    assert torch.all(actual.status == LabelStatus.COS_NOT_CONVERGED)
    assert torch.all(torch.isnan(actual.iv))
    assert torch.all(torch.isfinite(actual.price))
    assert actual.terms.tolist() == [4]


def test_nonfinite_cos_failure_does_not_contaminate_a_regular_surface(
    settings: LabelSettings, grid: tuple[torch.Tensor, torch.Tensor]
) -> None:
    strikes, maturities = grid
    parameters = torch.tensor(
        [[0.04, 1.5, 0.04, 1e200, -0.7], [0.04, 1.5, 0.04, 0.3, -0.7]], dtype=torch.float64
    )
    actual = label_surfaces(parameters, strikes, maturities, 1.0, 0.02, 0.0, settings=settings)
    assert torch.all(actual.status[0] == LabelStatus.COS_FAILURE)
    assert torch.all(torch.isnan(actual.iv[0]))
    assert torch.all(actual.valid[1])
    assert torch.all(torch.isfinite(actual.iv[1]))


def test_expiry_and_zero_variance_keep_iv_failure_reasons(settings: LabelSettings) -> None:
    parameters = torch.tensor([[0.0, 1.5, 0.0, 0.5, -0.7]], dtype=torch.float64)
    maturities = torch.tensor([0.0, 1.0], dtype=torch.float64)
    strikes = torch.tensor([[0.9, 1.0, 1.1]] * 2, dtype=torch.float64)
    actual = label_surfaces(parameters, strikes, maturities, 1.0, 0.0, 0.0, settings=settings)
    assert torch.all(actual.status[0, 0] == IVStatus.EXPIRED)
    assert torch.all(actual.status[0, 1] == IVStatus.NEAR_LOWER_BOUND)
    assert not torch.any(actual.valid)
    assert torch.all(torch.isnan(actual.iv))
    torch.testing.assert_close(actual.price[0], torch.clamp(strikes - 1.0, min=0), rtol=0, atol=0)


def test_low_vega_failure_survives_price_convergence(
    settings: LabelSettings, grid: tuple[torch.Tensor, torch.Tensor]
) -> None:
    strikes, maturities = grid
    parameters = torch.tensor([[0.04, 1.5, 0.04, 0.0, -0.7]], dtype=torch.float64)
    high_floor = replace(settings, iv=replace(settings.iv, min_normalized_vega=1.0))
    actual = label_surfaces(parameters, strikes, maturities, 1.0, 0.02, 0.0, settings=high_floor)
    assert torch.all(actual.status == IVStatus.LOW_VEGA)
    assert torch.all(torch.isnan(actual.iv))
    assert torch.all(torch.isfinite(actual.vega))


def test_results_are_independent_of_surface_batch_composition(
    settings: LabelSettings, grid: tuple[torch.Tensor, torch.Tensor]
) -> None:
    strikes, maturities = grid
    parameters = torch.tensor(
        [
            [0.04, 1.5, 0.04, 0.0, -0.7],
            [0.09, 0.7, 0.02, 0.8, -0.95],
            [0.0, 1.5, 0.0, 0.5, -0.7],
        ],
        dtype=torch.float64,
    )
    together = label_surfaces(parameters, strikes, maturities, 1.0, 0.0, 0.0, settings=settings)
    separate = [
        label_surfaces(row[None, :], strikes, maturities, 1.0, 0.0, 0.0, settings=settings)
        for row in parameters
    ]
    for field in ("price", "iv", "vega", "discrepancy", "status", "terms"):
        expected = torch.cat([getattr(result, field) for result in separate])
        torch.testing.assert_close(
            getattr(together, field), expected, rtol=0, atol=0, equal_nan=True
        )
    assert torch.all(torch.isnan(together.iv[~together.valid]))


def test_reference_labels_reject_float32_parameters(
    settings: LabelSettings, grid: tuple[torch.Tensor, torch.Tensor]
) -> None:
    strikes, maturities = grid
    parameters = torch.tensor([[0.04, 1.5, 0.04, 0.3, -0.7]], dtype=torch.float32)
    with pytest.raises(ValueError, match="CPU float64"):
        label_surfaces(parameters, strikes, maturities, 1.0, 0.02, 0.0, settings=settings)
