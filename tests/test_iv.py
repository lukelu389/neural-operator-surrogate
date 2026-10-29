"""Recovery, identifiability, and per-quote failures for implied volatility."""

import math
from dataclasses import replace
from typing import Literal

import pytest
import torch

from nop.pricing.bs import black_scholes_greeks, black_scholes_price
from nop.pricing.iv import IVSettings, IVStatus, implied_volatility

OptionType = Literal["call", "put"]


@pytest.fixture
def settings() -> IVSettings:
    return IVSettings(
        price_atol=0.0,
        price_rtol=1e-10,
        iv_atol=1e-10,
        min_normalized_vega=1e-4,
        volatility_upper=5.0,
        max_iterations=100,
    )


@pytest.mark.parametrize(
    ("option_type", "price"),
    [("call", 10.450583572185565), ("put", 5.573526022256971)],
)
def test_recovers_iv_from_known_prices(
    settings: IVSettings, option_type: OptionType, price: float
) -> None:
    result = implied_volatility(
        price, 100.0, 100.0, 1.0, 0.05, settings=settings, option_type=option_type
    )
    assert result.status.item() == IVStatus.SUCCESS
    assert result.volatility.item() == pytest.approx(0.2, abs=1e-10)
    assert result.volatility.dtype == torch.float64
    assert result.volatility.device.type == "cpu"
    assert result.volatility.shape == ()
    assert result.valid.item()
    assert not result.failed.item()


@pytest.mark.parametrize("option_type", ["call", "put"])
def test_broadcast_roundtrip_with_dividends_and_negative_rates(
    settings: IVSettings, option_type: OptionType
) -> None:
    spot = torch.tensor([[80.0], [100.0], [120.0]], dtype=torch.float64)
    strike = torch.tensor([90.0, 100.0, 110.0], dtype=torch.float64)
    maturity = torch.tensor([[0.1], [0.8], [2.0]], dtype=torch.float64)
    volatility = torch.tensor([0.18, 0.3, 0.65], dtype=torch.float64)
    rate, q = -0.025, 0.04
    price = black_scholes_price(
        spot, strike, maturity, volatility, rate, q, option_type=option_type
    )
    result = implied_volatility(
        price, spot, strike, maturity, rate, q, settings=settings, option_type=option_type
    )
    assert result.volatility.shape == (3, 3)
    assert result.status.shape == result.iterations.shape == result.volatility.shape
    assert result.price_residual.shape == result.vega.shape == result.volatility.shape
    assert torch.all(result.valid)
    assert not torch.any(result.failed)
    torch.testing.assert_close(result.volatility, volatility.expand(3, 3), rtol=0, atol=1e-10)
    repriced = black_scholes_price(
        spot, strike, maturity, result.volatility, rate, q, option_type=option_type
    )
    scale = torch.maximum(spot * torch.exp(-q * maturity), strike * torch.exp(-rate * maturity))
    assert torch.all(torch.abs(repriced - price) <= settings.price_rtol * scale)
    expected_vega = black_scholes_greeks(
        spot, strike, maturity, result.volatility, rate, q, option_type=option_type
    ).vega
    torch.testing.assert_close(result.vega, expected_vega, rtol=1e-12, atol=1e-12)


def test_bad_rows_leave_neighboring_quotes_unchanged(settings: IVSettings) -> None:
    price = black_scholes_price(100.0, 100.0, 1.0, 0.2).repeat(9)
    spot = torch.full((9,), 100.0, dtype=torch.float64)
    maturity = torch.ones(9, dtype=torch.float64)
    price[1] = math.nan
    spot[2] = -100.0
    maturity[3] = 0.0
    price[4] = -1.0
    price[5] = 0.0
    price[6] = 100.0
    price[7] = black_scholes_price(100.0, 100.0, 1.0, 3.0)
    limited = replace(settings, volatility_upper=2.0)
    result = implied_volatility(price, spot, 100.0, maturity, settings=limited)
    expected = torch.tensor(
        [
            IVStatus.SUCCESS,
            IVStatus.NONFINITE_INPUT,
            IVStatus.INVALID_DOMAIN,
            IVStatus.EXPIRED,
            IVStatus.PRICE_OUT_OF_BOUNDS,
            IVStatus.NEAR_LOWER_BOUND,
            IVStatus.NEAR_UPPER_BOUND,
            IVStatus.UNBRACKETED,
            IVStatus.SUCCESS,
        ],
        dtype=result.status.dtype,
    )
    torch.testing.assert_close(result.status, expected)
    torch.testing.assert_close(result.valid, expected == IVStatus.SUCCESS)
    torch.testing.assert_close(result.failed, ~result.valid)
    assert torch.all(torch.isnan(result.volatility[result.failed]))
    reference = implied_volatility(price[0], 100.0, 100.0, 1.0, settings=limited)
    torch.testing.assert_close(result.volatility[[0, 8]], reference.volatility.expand(2))
    torch.testing.assert_close(result.iterations[[0, 8]], reference.iterations.expand(2))


@pytest.mark.parametrize("option_type", ["call", "put"])
def test_prices_near_bounds_are_masked_without_clipping(
    settings: IVSettings, option_type: OptionType
) -> None:
    spot, strike, maturity, rate, q = 110.0, 100.0, 0.7, -0.02, 0.04
    discounted_spot = spot * math.exp(-q * maturity)
    discounted_strike = strike * math.exp(-rate * maturity)
    lower = max(
        discounted_spot - discounted_strike
        if option_type == "call"
        else discounted_strike - discounted_spot,
        0.0,
    )
    upper = discounted_spot if option_type == "call" else discounted_strike
    tolerance = settings.price_rtol * max(discounted_spot, discounted_strike)
    quotes = torch.tensor(
        [
            lower - 2 * tolerance,
            lower - tolerance / 2,
            lower,
            lower + tolerance / 2,
            upper - tolerance / 2,
            upper,
            upper + tolerance / 2,
            upper + 2 * tolerance,
        ],
        dtype=torch.float64,
    )
    result = implied_volatility(
        quotes, spot, strike, maturity, rate, q, settings=settings, option_type=option_type
    )
    assert result.status.tolist() == [
        IVStatus.PRICE_OUT_OF_BOUNDS,
        IVStatus.PRICE_OUT_OF_BOUNDS,  # Strict bounds apply even within tolerance.
        IVStatus.NEAR_LOWER_BOUND,
        IVStatus.NEAR_LOWER_BOUND,
        IVStatus.NEAR_UPPER_BOUND,
        IVStatus.NEAR_UPPER_BOUND,
        IVStatus.PRICE_OUT_OF_BOUNDS,
        IVStatus.PRICE_OUT_OF_BOUNDS,
    ]
    assert torch.all(result.failed)
    assert torch.all(torch.isnan(result.volatility))


def test_expired_option_has_no_identifiable_iv(settings: IVSettings) -> None:
    result = implied_volatility(20.0, 120.0, 100.0, 0.0, settings=settings)
    assert result.status.item() == IVStatus.EXPIRED
    assert result.failed.item()
    assert torch.isnan(result.volatility)


def test_bracket_cap_failure_is_explicit(settings: IVSettings) -> None:
    price = black_scholes_price(100.0, 100.0, 1.0, 0.8)
    result = implied_volatility(
        price, 100.0, 100.0, 1.0, settings=replace(settings, volatility_upper=0.5)
    )
    assert result.status.item() == IVStatus.UNBRACKETED
    assert result.failed.item()
    assert torch.isnan(result.volatility)


def test_low_vega_rejection_uses_normalized_vega(settings: IVSettings) -> None:
    price = black_scholes_price(100.0, 100.0, 1.0, 0.2)
    result = implied_volatility(
        price, 100.0, 100.0, 1.0, settings=replace(settings, min_normalized_vega=0.5)
    )
    assert result.status.item() == IVStatus.LOW_VEGA
    assert result.vega.item() > 0.5  # Raw currency vega is much larger than the floor.
    assert result.vega.item() / 100 < 0.5
    assert result.failed.item()
    assert torch.isnan(result.volatility)


def test_iteration_exhaustion_does_not_return_a_best_guess(settings: IVSettings) -> None:
    price = black_scholes_price(100.0, 100.0, 1.0, 0.37)
    result = implied_volatility(
        price, 100.0, 100.0, 1.0, settings=replace(settings, max_iterations=1)
    )
    assert result.status.item() == IVStatus.NO_CONVERGENCE
    assert result.failed.item()
    assert torch.isnan(result.volatility)


def test_exact_midpoint_price_still_requires_a_small_iv_interval(settings: IVSettings) -> None:
    price = black_scholes_price(100.0, 100.0, 1.0, settings.volatility_upper / 2)
    result = implied_volatility(
        price, 100.0, 100.0, 1.0, settings=replace(settings, max_iterations=1)
    )
    assert result.status.item() == IVStatus.NO_CONVERGENCE
    assert torch.isnan(result.volatility)


def test_small_iv_interval_alone_does_not_satisfy_the_price_tolerance(settings: IVSettings) -> None:
    price = black_scholes_price(100.0, 100.0, 1.0, 0.37)
    result = implied_volatility(
        price, 100.0, 100.0, 1.0, settings=replace(settings, max_iterations=1, iv_atol=10.0)
    )
    assert result.status.item() == IVStatus.NO_CONVERGENCE
    assert torch.isnan(result.volatility)


def test_relative_tolerance_and_vega_floor_are_currency_scale_invariant(
    settings: IVSettings,
) -> None:
    scale = torch.tensor([1e-6, 1.0, 1e6], dtype=torch.float64)
    spot, strike, maturity, volatility, rate, q = scale, 1.1 * scale, 0.8, 0.32, -0.02, 0.03
    price = black_scholes_price(spot, strike, maturity, volatility, rate, q)
    result = implied_volatility(price, spot, strike, maturity, rate, q, settings=settings)
    assert torch.all(result.valid)
    torch.testing.assert_close(
        result.volatility, torch.full_like(scale, volatility), rtol=0, atol=1e-10
    )
    torch.testing.assert_close(result.vega / scale, (result.vega / scale)[1].expand(3))


def test_float32_recovers_iv_with_precision_appropriate_tolerances(settings: IVSettings) -> None:
    spot = torch.tensor([90.0, 100.0, 110.0], dtype=torch.float32)
    price = black_scholes_price(spot, 100.0, 0.8, 0.3, 0.02, 0.01)
    coarse = replace(settings, price_rtol=2e-6, iv_atol=1e-5)
    result = implied_volatility(price, spot, 100.0, 0.8, 0.02, 0.01, settings=coarse)
    assert (
        result.volatility.dtype == result.vega.dtype == result.price_residual.dtype == torch.float32
    )
    assert result.volatility.device == spot.device
    assert torch.all(result.valid)
    torch.testing.assert_close(result.volatility, torch.full_like(spot, 0.3), rtol=0, atol=1e-5)


def test_solver_outputs_are_detached(settings: IVSettings) -> None:
    spot = torch.tensor(100.0, dtype=torch.float64, requires_grad=True)
    price = black_scholes_price(spot, 100.0, 1.0, 0.2)
    result = implied_volatility(price, spot, 100.0, 1.0, settings=settings)
    assert result.valid.item()
    for value in [result.volatility, result.price_residual, result.vega]:
        assert not value.requires_grad
        assert value.grad_fn is None


@pytest.mark.parametrize("index", range(6))
@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_nonfinite_quote_inputs_produce_masks(
    settings: IVSettings, index: int, value: float
) -> None:
    args = [8.0, 100.0, 100.0, 1.0, 0.02, 0.01]
    args[index] = value
    result = implied_volatility(*args, settings=settings)
    assert result.status.item() == IVStatus.NONFINITE_INPUT
    assert torch.isnan(result.volatility)


@pytest.mark.parametrize(("index", "value"), [(1, 0.0), (1, -1.0), (2, 0.0), (2, -1.0), (3, -1.0)])
def test_invalid_quote_domains_produce_masks(
    settings: IVSettings, index: int, value: float
) -> None:
    args = [8.0, 100.0, 100.0, 1.0]
    args[index] = value
    result = implied_volatility(*args, settings=settings)
    assert result.status.item() == IVStatus.INVALID_DOMAIN
    assert torch.isnan(result.volatility)


@pytest.mark.parametrize("dtype", [torch.int64, torch.float16, torch.bfloat16, torch.complex64])
def test_unsupported_tensor_dtype_is_a_structural_error(
    settings: IVSettings, dtype: torch.dtype
) -> None:
    with pytest.raises(TypeError):
        implied_volatility(torch.tensor(8, dtype=dtype), 100.0, 100.0, 1.0, settings=settings)


def test_mixed_tensor_dtypes_are_a_structural_error(settings: IVSettings) -> None:
    with pytest.raises(ValueError):
        implied_volatility(
            torch.tensor(8.0, dtype=torch.float32),
            torch.tensor(100.0, dtype=torch.float64),
            100.0,
            1.0,
            settings=settings,
        )


def test_unknown_option_type_is_a_structural_error(settings: IVSettings) -> None:
    with pytest.raises(ValueError):
        implied_volatility(8.0, 100.0, 100.0, 1.0, settings=settings, option_type="straddle")  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("price_atol", -1.0),
        ("price_rtol", -1.0),
        ("iv_atol", 0.0),
        ("iv_atol", -1.0),
        ("min_normalized_vega", -1.0),
        ("volatility_upper", 0.0),
        ("volatility_upper", -1.0),
        ("price_atol", math.nan),
        ("price_rtol", math.inf),
        ("iv_atol", math.nan),
        ("min_normalized_vega", math.inf),
        ("volatility_upper", math.inf),
        ("max_iterations", 0),
        ("max_iterations", -1),
    ],
)
def test_invalid_solver_settings_are_rejected(
    settings: IVSettings, field: str, value: float | int
) -> None:
    with pytest.raises(ValueError):
        invalid = replace(settings, **{field: value})
        implied_volatility(8.0, 100.0, 100.0, 1.0, settings=invalid)


@pytest.mark.parametrize("value", [True, 2.5])
def test_iteration_budget_requires_an_integer(settings: IVSettings, value: object) -> None:
    with pytest.raises(TypeError):
        invalid = replace(settings, max_iterations=value)
        implied_volatility(8.0, 100.0, 100.0, 1.0, settings=invalid)


def test_zero_price_tolerances_are_rejected(settings: IVSettings) -> None:
    with pytest.raises(ValueError):
        replace(settings, price_atol=0, price_rtol=0)


def test_empty_batch_preserves_shape(settings: IVSettings) -> None:
    result = implied_volatility(torch.empty((0, 3)), 1.0, 1.0, 1.0, settings=settings)
    assert result.volatility.shape == (0, 3)
    assert result.status.dtype == result.iterations.dtype == torch.int64


def test_discount_overflow_does_not_break_neighbors(settings: IVSettings) -> None:
    result = implied_volatility(
        torch.tensor([0.08, 0.08], dtype=torch.float64),
        1.0,
        1.0,
        1.0,
        torch.tensor([-1000.0, 0.0], dtype=torch.float64),
        settings=settings,
    )
    assert result.status.tolist() == [IVStatus.NUMERICAL_ERROR, IVStatus.SUCCESS]
    assert torch.isnan(result.volatility[0])


def test_nonfinite_greek_does_not_break_neighbors(settings: IVSettings) -> None:
    # Subnormal currency units make gamma overflow in the BS Greek API; its
    # per-batch exception must not discard the ordinary quote alongside it.
    spot = torch.tensor([1e-310, 1.0], dtype=torch.float64)
    price = black_scholes_price(spot, spot, 1.0, 0.2)
    result = implied_volatility(price, spot, spot, 1.0, settings=settings)
    assert result.status.tolist() == [IVStatus.NUMERICAL_ERROR, IVStatus.SUCCESS]
    assert torch.isnan(result.volatility[0])
    assert result.volatility[1].item() == pytest.approx(0.2, abs=1e-10)


@pytest.mark.parametrize("cap", [1e100, 1e-100])
def test_unrepresentable_float32_bracket_is_masked(settings: IVSettings, cap: float) -> None:
    result = implied_volatility(
        torch.tensor(0.08, dtype=torch.float32),
        1.0,
        1.0,
        1.0,
        settings=replace(settings, volatility_upper=cap),
    )
    assert result.status.item() == IVStatus.NUMERICAL_ERROR


def test_float32_stagnation_is_explicit(settings: IVSettings) -> None:
    price = black_scholes_price(torch.tensor(1.0, dtype=torch.float32), 1.0, 1.0, 0.2)
    result = implied_volatility(price, 1.0, 1.0, 1.0, settings=settings)
    assert result.status.item() == IVStatus.NO_CONVERGENCE
    assert result.iterations.item() < settings.max_iterations
    assert torch.isnan(result.volatility)


def test_exact_cap_can_be_recovered(settings: IVSettings) -> None:
    cap = 0.4
    price = black_scholes_price(1.0, 1.0, 1.0, cap)
    result = implied_volatility(
        price, 1.0, 1.0, 1.0, settings=replace(settings, volatility_upper=cap)
    )
    assert result.valid.item()
    assert result.volatility.item() == pytest.approx(cap, abs=settings.iv_atol)
    repriced = black_scholes_price(1.0, 1.0, 1.0, result.volatility)
    torch.testing.assert_close(result.price_residual, repriced - price, rtol=0, atol=0)
