"""Independent values, differential checks, and limits for Black--Scholes."""

import math
from typing import Literal

import pytest
import torch
from torch import Tensor

from nop.pricing.bs import black_scholes_greeks, black_scholes_price

OptionType = Literal["call", "put"]


@pytest.mark.parametrize(
    ("option_type", "expected"),
    [("call", 10.450583572185565), ("put", 5.573526022256971)],
)
def test_known_at_the_money_price(option_type: OptionType, expected: float) -> None:
    price = black_scholes_price(100.0, 100.0, 1.0, 0.2, 0.05, option_type=option_type)
    assert price.dtype == torch.float64
    assert price.device.type == "cpu"
    assert price.shape == ()
    assert price.item() == pytest.approx(expected, abs=2e-13)


# Haug fixtures independently reproduced in QuantLib v1.38:
# https://github.com/lballabio/QuantLib/blob/v1.38/test-suite/europeanoption.cpp
@pytest.mark.parametrize(
    ("option_type", "spot", "strike", "maturity", "volatility", "rate", "q", "expected"),
    [
        ("call", 60.0, 65.0, 0.25, 0.30, 0.08, 0.0, 2.1334),
        ("put", 100.0, 95.0, 0.50, 0.20, 0.10, 0.05, 2.4648),
    ],
)
def test_published_price_fixtures(
    option_type: OptionType,
    spot: float,
    strike: float,
    maturity: float,
    volatility: float,
    rate: float,
    q: float,
    expected: float,
) -> None:
    actual = black_scholes_price(
        spot, strike, maturity, volatility, rate, q, option_type=option_type
    )
    assert actual.item() == pytest.approx(expected, abs=1e-4)


@pytest.mark.parametrize(
    ("field", "option_type", "spot", "strike", "maturity", "volatility", "rate", "q", "expected"),
    [
        ("delta", "call", 105.0, 100.0, 0.5, 0.36, 0.10, 0.10, 0.5946),
        ("delta", "put", 105.0, 100.0, 0.5, 0.36, 0.10, 0.10, -0.3566),
        ("gamma", "call", 55.0, 60.0, 0.75, 0.30, 0.10, 0.0, 0.0278),
        ("vega", "call", 55.0, 60.0, 0.75, 0.30, 0.10, 0.0, 18.9358),
        ("theta", "put", 430.0, 405.0, 1 / 12, 0.20, 0.07, 0.05, -31.1924),
        ("rho", "call", 72.0, 75.0, 1.0, 0.19, 0.09, 0.0, 38.7325),
    ],
)
def test_published_greek_fixtures(
    field: str,
    option_type: OptionType,
    spot: float,
    strike: float,
    maturity: float,
    volatility: float,
    rate: float,
    q: float,
    expected: float,
) -> None:
    greeks = black_scholes_greeks(
        spot, strike, maturity, volatility, rate, q, option_type=option_type
    )
    assert getattr(greeks, field).item() == pytest.approx(expected, abs=1e-4)


def integrated_payoff(
    spot: float,
    strike: float,
    maturity: float,
    volatility: float,
    rate: float,
    q: float,
    option_type: OptionType,
) -> float:
    """Integrate the risk-neutral payoff, without the closed-form pricing formula."""
    width = volatility * math.sqrt(maturity)
    drift = (rate - q - volatility**2 / 2) * maturity
    boundary = (math.log(strike / spot) - drift) / width
    lower, upper = (boundary, 12.0) if option_type == "call" else (-12.0, boundary)
    intervals = 8192
    step = (upper - lower) / intervals

    def integrand(z: float) -> float:
        terminal_spot = spot * math.exp(drift + width * z)
        payoff = terminal_spot - strike if option_type == "call" else strike - terminal_spot
        return max(payoff, 0.0) * math.exp(-z * z / 2) / math.sqrt(2 * math.pi)

    weighted = integrand(lower) + integrand(upper)
    weighted += 4 * math.fsum(integrand(lower + i * step) for i in range(1, intervals, 2))
    weighted += 2 * math.fsum(integrand(lower + i * step) for i in range(2, intervals, 2))
    return math.exp(-rate * maturity) * step * weighted / 3


@pytest.mark.parametrize("option_type", ["call", "put"])
@pytest.mark.parametrize(
    ("spot", "strike", "maturity", "volatility", "rate", "q"),
    [(91.0, 103.0, 0.7, 0.31, -0.02, 0.06), (100.0, 400.0, 1.0, 0.2, 0.0, 0.0)],
)
def test_price_against_integrated_payoff_including_tails(
    option_type: OptionType,
    spot: float,
    strike: float,
    maturity: float,
    volatility: float,
    rate: float,
    q: float,
) -> None:
    # Reverse S and K for the tail put, so both option types exercise a tiny OTM price.
    if strike == 400.0 and option_type == "put":
        spot, strike = strike, spot
    expected = integrated_payoff(spot, strike, maturity, volatility, rate, q, option_type)
    actual = black_scholes_price(
        spot, strike, maturity, volatility, rate, q, option_type=option_type
    )
    assert actual.item() > 0.0
    assert actual.item() == pytest.approx(expected, rel=2e-9, abs=1e-24)


def test_parity_with_dividends_negative_rates_and_broadcasting() -> None:
    spot = torch.tensor([[80.0], [100.0], [130.0]], dtype=torch.float64)
    strike = torch.tensor([90.0, 100.0, 120.0], dtype=torch.float64)
    maturity = torch.tensor([0.01, 0.5, 2.0], dtype=torch.float64)
    rate, q = -0.03, 0.04
    call = black_scholes_price(spot, strike, maturity, 0.27, rate, q)
    put = black_scholes_price(spot, strike, maturity, 0.27, rate, q, option_type="put")
    assert call.shape == (3, 3)
    discounted_spot = spot * torch.exp(-q * maturity)
    discounted_strike = strike * torch.exp(-rate * maturity)
    torch.testing.assert_close(
        call - put, discounted_spot - discounted_strike, rtol=1e-13, atol=3e-14
    )
    assert torch.all(call >= torch.clamp(discounted_spot - discounted_strike, min=0))
    assert torch.all(put >= torch.clamp(discounted_strike - discounted_spot, min=0))
    assert torch.all(call <= discounted_spot)
    assert torch.all(put <= discounted_strike)


@pytest.mark.parametrize("option_type", ["call", "put"])
def test_all_greeks_against_central_finite_differences(option_type: OptionType) -> None:
    args = [103.0, 97.0, 0.8, 0.27, -0.015, 0.04]
    greeks = black_scholes_greeks(*args, option_type=option_type)

    def price(index: int, displacement: float) -> float:
        shifted = args.copy()
        shifted[index] += displacement
        return black_scholes_price(*shifted, option_type=option_type).item()

    for field, index, step, sign in [
        ("delta", 0, 1e-3, 1),
        ("vega", 3, 1e-5, 1),
        ("theta", 2, 1e-5, -1),
        ("rho", 4, 1e-5, 1),
    ]:
        derivative = sign * (price(index, step) - price(index, -step)) / (2 * step)
        assert getattr(greeks, field).item() == pytest.approx(derivative, rel=2e-8, abs=2e-8)
    step = 0.02
    gamma = (price(0, step) - 2 * price(0, 0.0) + price(0, -step)) / step**2
    assert greeks.gamma.item() == pytest.approx(gamma, rel=2e-7, abs=2e-9)


@pytest.mark.parametrize("option_type", ["call", "put"])
@pytest.mark.parametrize(
    "input_values",
    [[103.0, 97.0, 0.8, 0.27, -0.015, 0.04], [100.0, 100.0, 1.0, 0.2, 0.0, 0.0]],
    ids=["dividend_and_negative_rate", "forward_atm_branch_boundary"],
)
def test_autograd_matches_greeks_and_supports_second_derivatives(
    option_type: OptionType, input_values: list[float]
) -> None:
    inputs = tuple(
        torch.tensor(value, dtype=torch.float64, requires_grad=True) for value in input_values
    )

    def price(*args: Tensor) -> Tensor:
        return black_scholes_price(*args, option_type=option_type)

    assert torch.autograd.gradcheck(price, inputs)
    assert torch.autograd.gradgradcheck(price, inputs)
    derivatives = torch.autograd.grad(price(*inputs), inputs, create_graph=True)
    gamma = torch.autograd.grad(derivatives[0], inputs[0])[0]
    greeks = black_scholes_greeks(*inputs, option_type=option_type)
    for actual, expected in zip(
        [derivatives[0], gamma, derivatives[3], -derivatives[2], derivatives[4]],
        [greeks.delta, greeks.gamma, greeks.vega, greeks.theta, greeks.rho],
        strict=True,
    ):
        torch.testing.assert_close(actual, expected, rtol=2e-13, atol=2e-13)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("option_type", ["call", "put"])
def test_prices_and_greeks_broadcast_and_preserve_tensor_dtype(
    dtype: torch.dtype, option_type: OptionType
) -> None:
    spot = torch.tensor([[85.0], [115.0]], dtype=dtype)
    strike = torch.tensor([90.0, 100.0, 110.0], dtype=dtype)
    price = black_scholes_price(spot, strike, 0.5, 0.3, option_type=option_type)
    greeks = black_scholes_greeks(spot, strike, 0.5, 0.3, option_type=option_type)
    for value in [price, greeks.delta, greeks.gamma, greeks.vega, greeks.theta, greeks.rho]:
        assert value.dtype == dtype
        assert value.device == spot.device
        assert value.shape == (2, 3)
        assert torch.all(torch.isfinite(value))
    for row in range(2):
        for column in range(3):
            expected = black_scholes_price(
                spot[row, 0], strike[column], 0.5, 0.3, option_type=option_type
            )
            torch.testing.assert_close(price[row, column], expected)


@pytest.mark.parametrize("option_type", ["call", "put"])
def test_expiry_and_zero_volatility_share_a_batch_with_regular_quotes(
    option_type: OptionType,
) -> None:
    spot = torch.tensor([120.0, 80.0, 120.0, 80.0, 100.0], dtype=torch.float64)
    maturity = torch.tensor([0.0, 0.0, 0.5, 0.5, 1.0], dtype=torch.float64)
    volatility = torch.tensor([0.0, 0.2, 0.0, 0.0, 0.2], dtype=torch.float64)
    rate, q = -0.02, 0.03
    actual = black_scholes_price(
        spot, 100.0, maturity, volatility, rate, q, option_type=option_type
    )
    sign = 1 if option_type == "call" else -1
    expected = torch.clamp(
        sign * (spot[:4] * torch.exp(-q * maturity[:4]) - 100 * torch.exp(-rate * maturity[:4])),
        min=0,
    )
    torch.testing.assert_close(actual[:4], expected, rtol=1e-14, atol=1e-14)
    regular = black_scholes_price(100.0, 100.0, 1.0, 0.2, rate, q, option_type=option_type)
    torch.testing.assert_close(actual[4], regular)


@pytest.mark.parametrize("option_type", ["call", "put"])
def test_small_time_and_volatility_converge_to_deterministic_limits(
    option_type: OptionType,
) -> None:
    spot = torch.tensor([80.0, 100.0, 120.0], dtype=torch.float64)
    for maturity, volatility, zero_maturity, zero_volatility in [
        (1.0, 1e-10, 1.0, 0.0),
        (1e-16, 0.2, 0.0, 0.2),
    ]:
        actual = black_scholes_price(
            spot, 100.0, maturity, volatility, 0.02, option_type=option_type
        )
        limit = black_scholes_price(
            spot, 100.0, zero_maturity, zero_volatility, 0.02, option_type=option_type
        )
        torch.testing.assert_close(actual, limit, rtol=1e-8, atol=1e-6)


@pytest.mark.parametrize("function", [black_scholes_price, black_scholes_greeks])
@pytest.mark.parametrize(
    ("index", "value"),
    [(0, 0.0), (0, -1.0), (1, 0.0), (1, -1.0), (2, -1.0), (3, -0.1)],
)
def test_rejects_invalid_domains(function: object, index: int, value: float) -> None:
    args = [100.0, 100.0, 1.0, 0.2]
    args[index] = value
    with pytest.raises(ValueError):
        function(*args)  # type: ignore[operator]


@pytest.mark.parametrize("function", [black_scholes_price, black_scholes_greeks])
@pytest.mark.parametrize("index", range(6))
@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_rejects_nonfinite_inputs(function: object, index: int, value: float) -> None:
    args = [100.0, 100.0, 1.0, 0.2, 0.03, 0.01]
    args[index] = value
    with pytest.raises(ValueError):
        function(*args)  # type: ignore[operator]


@pytest.mark.parametrize("index", [2, 3])
def test_greeks_reject_degenerate_boundary_anywhere_in_batch(index: int) -> None:
    args = [torch.tensor([value, value], dtype=torch.float64) for value in [100.0, 100.0, 1.0, 0.2]]
    args[index][1] = 0.0
    with pytest.raises(ValueError):
        black_scholes_greeks(*args)


@pytest.mark.parametrize("function", [black_scholes_price, black_scholes_greeks])
def test_rejects_invalid_option_type(function: object) -> None:
    with pytest.raises(ValueError):
        function(100.0, 100.0, 1.0, 0.2, option_type="straddle")  # type: ignore[operator]


@pytest.mark.parametrize("function", [black_scholes_price, black_scholes_greeks])
@pytest.mark.parametrize("dtype", [torch.int64, torch.float16, torch.bfloat16, torch.complex64])
def test_rejects_unsupported_tensor_dtype(function: object, dtype: torch.dtype) -> None:
    with pytest.raises(TypeError):
        function(torch.tensor(100, dtype=dtype), 100.0, 1.0, 0.2)  # type: ignore[operator]


@pytest.mark.parametrize("function", [black_scholes_price, black_scholes_greeks])
def test_rejects_mixed_tensor_dtypes(function: object) -> None:
    with pytest.raises(ValueError):
        function(  # type: ignore[operator]
            torch.tensor(100.0, dtype=torch.float32),
            torch.tensor(100.0, dtype=torch.float64),
            1.0,
            0.2,
        )


@pytest.mark.parametrize(
    ("option_type", "strike", "volatility"),
    [
        ("put", 0.9999998211860657, 1.584893141171051e-7),
        ("call", 1.0000008344650269, 5.011872303839482e-7),
    ],
)
def test_float32_cancellation_raises_instead_of_returning_negative_price(
    option_type: OptionType, strike: float, volatility: float
) -> None:
    with pytest.raises(FloatingPointError, match="price"):
        black_scholes_price(
            torch.tensor(1.0, dtype=torch.float32),
            strike,
            1.0,
            volatility,
            option_type=option_type,
        )
    reference = black_scholes_price(1.0, strike, 1.0, volatility, option_type=option_type)
    assert reference.item() > 0.0
    assert math.isfinite(reference.item())


@pytest.mark.parametrize("function", [black_scholes_price, black_scholes_greeks])
def test_finite_inputs_that_overflow_raise_numerical_error(function: object) -> None:
    with pytest.raises(FloatingPointError, match="nonfinite"):
        function(100.0, 100.0, 10.0, 0.2, -1e308)  # type: ignore[operator]
