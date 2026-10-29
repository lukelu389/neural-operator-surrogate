"""Independent numerical checks for the Heston characteristic function and COS."""

import cmath
import math
from typing import Literal

import pytest
import torch
from torch import Tensor

from nop.pricing.bs import black_scholes_price
from nop.pricing.heston_cos import (
    heston_characteristic_function,
    heston_cos_price,
    heston_cumulants,
)

OptionType = Literal["call", "put"]


def riccati_reference(frequency: float, maturity: float, rho: float = -0.8) -> complex:
    """Solve the affine Riccati ODE independently of the closed-form CF."""
    v0, kappa, theta, sigma, rate, q = 0.09, 0.7, 0.07, 0.9, -0.01, 0.025
    steps = 4096
    step = maturity / steps
    c, d = 0j, 0j
    iu = 1j * frequency

    def rhs(current_d: complex) -> tuple[complex, complex]:
        return (
            iu * (rate - q) + kappa * theta * current_d,
            0.5 * sigma**2 * current_d**2
            + (rho * sigma * iu - kappa) * current_d
            - 0.5 * (frequency**2 + iu),
        )

    for _ in range(steps):
        c1, d1 = rhs(d)
        c2, d2 = rhs(d + step * d1 / 2)
        c3, d3 = rhs(d + step * d2 / 2)
        c4, d4 = rhs(d + step * d3)
        c += step * (c1 + 2 * c2 + 2 * c3 + c4) / 6
        d += step * (d1 + 2 * d2 + 2 * d3 + d4) / 6
    return cmath.exp(c + v0 * d)


@pytest.mark.parametrize("frequency", [0.25, 2.0, 12.0])
@pytest.mark.parametrize("maturity", [0.05, 1.7, 10.0])
@pytest.mark.parametrize("rho", [-0.8, 0.8])
def test_characteristic_function_matches_independent_riccati_ode(
    frequency: float, maturity: float, rho: float
) -> None:
    actual = heston_characteristic_function(
        frequency, maturity, 0.09, 0.7, 0.07, 0.9, rho, -0.01, 0.025
    )
    assert actual.dtype == torch.complex128
    assert actual.item() == pytest.approx(
        riccati_reference(frequency, maturity, rho), rel=2e-9, abs=2e-10
    )


@pytest.mark.parametrize("maturity", [0.0, 0.05, 2.0, 30.0])
def test_characteristic_function_normalization_conjugacy_and_modulus(maturity: float) -> None:
    frequency = torch.arange(-1000, 1001, dtype=torch.float64) / 20
    values = heston_characteristic_function(frequency, maturity, 0.08, 0.2, 0.07, 1.0, -0.95, 0.02)
    assert torch.all(torch.isfinite(values))
    torch.testing.assert_close(values[1000], torch.ones((), dtype=torch.complex128), rtol=0, atol=0)
    torch.testing.assert_close(values, values.flip(0).conj(), rtol=2e-13, atol=2e-13)
    assert torch.all(values.abs() <= 1 + 1e-13)
    # This fixed-frequency mesh catches branch discontinuities at long maturities.
    assert torch.max(torch.abs(torch.diff(values))).item() < 0.1
    if maturity == 0:
        torch.testing.assert_close(values, torch.ones_like(values), rtol=0, atol=0)


@pytest.mark.parametrize("maturity", [0.02, 0.8, 5.0])
def test_cumulants_match_expected_log_return_and_cf_curvature(maturity: float) -> None:
    v0, kappa, theta, sigma, rho, rate, q = 0.09, 0.7, 0.04, 0.8, -0.7, -0.02, 0.03
    c1, c2 = heston_cumulants(maturity, v0, kappa, theta, sigma, rho, rate, q)
    integrated_variance = theta * maturity + (v0 - theta) * -math.expm1(-kappa * maturity) / kappa
    assert c1.item() == pytest.approx((rate - q) * maturity - integrated_variance / 2, abs=1e-13)

    def curvature(step: float) -> float:
        frequencies = torch.tensor([-step, 0.0, step], dtype=torch.float64)
        log_cf = torch.log(
            heston_characteristic_function(
                frequencies, maturity, v0, kappa, theta, sigma, rho, rate, q
            )
        )
        return (-(log_cf[0] - 2 * log_cf[1] + log_cf[2]) / step**2).real.item()

    # Richardson extrapolation removes the leading finite-difference error.
    extrapolated = (4 * curvature(0.001) - curvature(0.002)) / 3
    assert c2.item() >= 0
    assert c2.item() == pytest.approx(extrapolated, rel=2e-7, abs=2e-9)


@pytest.mark.parametrize("option_type", ["call", "put"])
@pytest.mark.parametrize("v0", [0.01, 0.09])
def test_zero_vol_of_vol_matches_bs_with_integrated_deterministic_variance(
    option_type: OptionType, v0: float
) -> None:
    maturity = torch.tensor([0.05, 0.5, 2.0], dtype=torch.float64)
    strike = torch.tensor([85.0, 100.0, 115.0], dtype=torch.float64)
    kappa, theta = 1.3, 0.04
    variance = theta + (v0 - theta) * -torch.expm1(-kappa * maturity) / (kappa * maturity)
    actual = heston_cos_price(
        100.0,
        strike,
        maturity,
        v0,
        kappa,
        theta,
        0.0,
        -0.7,
        -0.02,
        0.03,
        n_terms=512,
        truncation_width=12.0,
        option_type=option_type,
    )
    expected = black_scholes_price(
        100.0, strike, maturity, torch.sqrt(variance), -0.02, 0.03, option_type=option_type
    )
    torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-10)


@pytest.mark.parametrize("option_type", ["call", "put"])
def test_small_positive_vol_of_vol_converges_to_bs(option_type: OptionType) -> None:
    strike = torch.tensor([85.0, 100.0, 115.0], dtype=torch.float64)
    expected = black_scholes_price(100.0, strike, 1.0, 0.2, 0.02, 0.01, option_type=option_type)
    actual = heston_cos_price(
        100.0,
        strike,
        1.0,
        0.04,
        1.5,
        0.04,
        1e-7,
        -0.7,
        0.02,
        0.01,
        n_terms=1024,
        truncation_width=12.0,
        option_type=option_type,
    )
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("option_type", ["call", "put"])
def test_mixed_expiry_and_zero_variance_quotes(option_type: OptionType) -> None:
    spot = torch.tensor([120.0, 80.0, 100.0], dtype=torch.float64)
    maturity = torch.tensor([0.0, 0.0, 0.8], dtype=torch.float64)
    actual = heston_cos_price(
        spot,
        100.0,
        maturity,
        0.0,
        1.5,
        0.0,
        0.5,
        -0.7,
        -0.02,
        0.03,
        n_terms=512,
        truncation_width=12.0,
        option_type=option_type,
    )
    expected = black_scholes_price(spot, 100.0, maturity, 0.0, -0.02, 0.03, option_type=option_type)
    torch.testing.assert_close(actual, expected, rtol=1e-13, atol=1e-13)


def test_batched_put_call_parity_with_dividends_and_negative_rates() -> None:
    strike = torch.tensor([[85.0, 100.0, 115.0]], dtype=torch.float64)
    maturity = torch.tensor([[0.2], [1.0], [2.0]], dtype=torch.float64)
    v0 = torch.tensor([[0.03], [0.04], [0.09]], dtype=torch.float64)
    args = (100.0, strike, maturity, v0, 1.5, 0.04, 0.3, -0.7, -0.02, 0.03)
    call = heston_cos_price(*args, n_terms=1024, truncation_width=14.0)
    put = heston_cos_price(*args, n_terms=1024, truncation_width=14.0, option_type="put")
    assert call.shape == (3, 3)
    a = 100 * torch.exp(-0.03 * maturity)
    b = strike * torch.exp(0.02 * maturity)
    torch.testing.assert_close(call - put, a - b, rtol=1e-12, atol=1e-12)
    assert torch.all(call >= torch.clamp(a - b, min=0) - 1e-10)
    assert torch.all(put >= torch.clamp(b - a, min=0) - 1e-10)
    assert torch.all(call <= a + 1e-10)
    assert torch.all(put <= b + 1e-10)


def test_price_converges_in_term_count_and_truncation_width() -> None:
    strikes = torch.tensor([80.0, 100.0, 120.0], dtype=torch.float64)
    args = (100.0, strikes, 1.0, 0.04, 1.5, 0.04, 0.3, -0.7, 0.02)
    reference = heston_cos_price(*args, n_terms=4096, truncation_width=24.0)
    width_errors = [
        (heston_cos_price(*args, n_terms=512, truncation_width=width) - reference).abs().max()
        for width in (8.0, 12.0, 16.0)
    ]
    term_errors = [
        (heston_cos_price(*args, n_terms=terms, truncation_width=16.0) - reference).abs().max()
        for terms in (64, 128, 256)
    ]
    assert width_errors[0] > width_errors[1] > width_errors[2]
    assert term_errors[0] > term_errors[1] > term_errors[2]
    assert width_errors[-1] < 1e-8
    assert term_errors[-1] < 1e-8


# Independent fixtures: QuantLib v1.38 testCosHestonEngine, cross-checked there
# against analytic Heston integration, not generated by this implementation.
# https://github.com/lballabio/QuantLib/blob/v1.38/test-suite/hestonmodel.cpp#L1698-L1736
@pytest.mark.parametrize(
    ("option_type", "strike", "expected"),
    [
        ("call", 120.0, 9.364410588426075),
        ("call", 250.0, 0.01036797658132471),
        ("put", 80.0, 5.319092971836708),
        ("put", 10.0, 0.01032681906278383),
    ],
)
def test_independent_heston_price_fixtures(
    option_type: OptionType, strike: float, expected: float
) -> None:
    actual = heston_cos_price(
        100.0,
        strike,
        1.0,
        0.1,
        4.0,
        0.22,
        1.8,
        -0.75,
        0.15,
        0.07,
        n_terms=600,
        truncation_width=25.0,
        option_type=option_type,
    )
    assert actual.item() == pytest.approx(expected, abs=1e-10, rel=0)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_price_broadcasts_like_separate_quotes_and_preserves_dtype(dtype: torch.dtype) -> None:
    maturity = torch.tensor([[0.3], [1.2]], dtype=dtype)
    strike = torch.tensor([90.0, 100.0, 110.0], dtype=dtype)
    actual = heston_cos_price(
        100.0,
        strike,
        maturity,
        0.04,
        1.5,
        0.04,
        0.3,
        -0.7,
        n_terms=512,
        truncation_width=12.0,
    )
    assert actual.shape == (2, 3)
    assert actual.dtype == dtype
    assert actual.device == maturity.device
    for row in range(2):
        for column in range(3):
            expected = heston_cos_price(
                100.0,
                strike[column],
                maturity[row, 0],
                0.04,
                1.5,
                0.04,
                0.3,
                -0.7,
                n_terms=512,
                truncation_width=12.0,
            )
            torch.testing.assert_close(actual[row, column], expected)


def test_price_autograd_matches_finite_differences_for_all_inputs() -> None:
    inputs = tuple(
        torch.tensor(value, dtype=torch.float64, requires_grad=True)
        for value in [100.0, 105.0, 0.8, 0.04, 1.5, 0.04, 0.3, -0.7, 0.02, 0.01]
    )

    def price(*args: Tensor) -> Tensor:
        return heston_cos_price(*args, n_terms=512, truncation_width=14.0)

    assert torch.autograd.gradcheck(price, inputs, atol=2e-5, rtol=2e-4)
    assert torch.autograd.gradgradcheck(price, inputs, atol=2e-4, rtol=2e-3)


@pytest.mark.parametrize(
    ("index", "value"),
    [
        (0, 0.0),
        (1, 0.0),
        (2, -1.0),
        (3, -0.01),
        (4, 0.0),
        (5, -0.01),
        (6, -0.1),
        (7, -1.01),
        (7, 1.01),
    ],
)
def test_price_rejects_invalid_domains(index: int, value: float) -> None:
    args = [100.0, 100.0, 1.0, 0.04, 1.5, 0.04, 0.3, -0.7]
    args[index] = value
    with pytest.raises(ValueError):
        heston_cos_price(*args, n_terms=128, truncation_width=12.0)


@pytest.mark.parametrize("index", range(10))
@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_price_rejects_nonfinite_inputs(index: int, value: float) -> None:
    args = [100.0, 100.0, 1.0, 0.04, 1.5, 0.04, 0.3, -0.7, 0.02, 0.01]
    args[index] = value
    with pytest.raises(ValueError):
        heston_cos_price(*args, n_terms=128, truncation_width=12.0)


@pytest.mark.parametrize("dtype", [torch.int64, torch.float16, torch.bfloat16, torch.complex64])
def test_price_rejects_unsupported_dtype(dtype: torch.dtype) -> None:
    with pytest.raises(TypeError):
        heston_cos_price(
            torch.tensor(100, dtype=dtype),
            100.0,
            1.0,
            0.04,
            1.5,
            0.04,
            0.3,
            -0.7,
            n_terms=128,
            truncation_width=12.0,
        )


def test_price_rejects_mixed_tensor_dtypes() -> None:
    with pytest.raises(ValueError):
        heston_cos_price(
            torch.tensor(100.0, dtype=torch.float32),
            torch.tensor(100.0, dtype=torch.float64),
            1.0,
            0.04,
            1.5,
            0.04,
            0.3,
            -0.7,
            n_terms=128,
            truncation_width=12.0,
        )


def test_price_rejects_unknown_option_type() -> None:
    with pytest.raises(ValueError):
        heston_cos_price(
            100.0,
            100.0,
            1.0,
            0.04,
            1.5,
            0.04,
            0.3,
            -0.7,
            n_terms=128,
            truncation_width=12.0,
            option_type="straddle",  # type: ignore[arg-type]
        )


@pytest.mark.parametrize("width", [0.0, -1.0, math.nan, math.inf])
def test_price_rejects_invalid_truncation_width(width: float) -> None:
    with pytest.raises(ValueError):
        heston_cos_price(
            100.0,
            100.0,
            1.0,
            0.04,
            1.5,
            0.04,
            0.3,
            -0.7,
            n_terms=128,
            truncation_width=width,
        )


@pytest.mark.parametrize("n_terms", [0, 1, -1, True, 2.5])
def test_price_rejects_invalid_term_count(n_terms: int) -> None:
    with pytest.raises(ValueError):
        heston_cos_price(
            100.0, 100.0, 1.0, 0.04, 1.5, 0.04, 0.3, -0.7, n_terms=n_terms, truncation_width=12.0
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA hardware unavailable")
def test_cuda_matches_cpu_float64() -> None:
    strike = torch.tensor([90.0, 100.0, 110.0], dtype=torch.float64)
    kwargs = {"n_terms": 1024, "truncation_width": 20.0}
    cpu = heston_cos_price(100.0, strike, 1.0, 0.04, 1.5, 0.04, 0.3, -0.7, **kwargs)
    gpu = heston_cos_price(100.0, strike.cuda(), 1.0, 0.04, 1.5, 0.04, 0.3, -0.7, **kwargs)
    torch.testing.assert_close(gpu.cpu(), cpu, rtol=1e-10, atol=1e-10)
