"""European Black–Scholes prices and analytic Greeks with continuous dividends.

Inputs broadcast using PyTorch rules. Scalar-only calls return CPU float64
tensors. Otherwise scalars adopt the tensor inputs' common dtype and device;
tensor inputs must share a float32 or float64 dtype and device. Tensor inputs
retain their autograd graph. Use float64 for numerical reference checks.

Rates, dividend yields, and volatility are decimal annual quantities; maturity
is in years. Prices are in the same currency units as spot and strike. Pricing
supports expiry and zero volatility, but derivatives at those boundaries are
not part of the autograd contract. The analytic Greek API rejects them.
"""

import math
from dataclasses import dataclass
from typing import Literal

import torch

OptionType = Literal["call", "put"]
ScalarOrTensor = float | torch.Tensor


@dataclass(frozen=True)
class BlackScholesGreeks:
    """Price sensitivities, with other inputs fixed.

    ``delta`` and ``gamma`` differentiate spot once and twice. ``vega`` and
    ``rho`` differentiate decimal volatility and the decimal risk-free rate
    (not a one-percentage-point change). ``theta`` is calendar-time decay,
    ``-dV/dT``, per year at fixed spot and strike, not per day.
    """

    delta: torch.Tensor
    gamma: torch.Tensor
    vega: torch.Tensor
    theta: torch.Tensor
    rho: torch.Tensor


def _prepare_inputs(
    spot: ScalarOrTensor,
    strike: ScalarOrTensor,
    maturity: ScalarOrTensor,
    volatility: ScalarOrTensor,
    rate: ScalarOrTensor,
    dividend_yield: ScalarOrTensor,
    option_type: OptionType,
) -> tuple[torch.Tensor, ...]:
    if option_type not in ("call", "put"):
        raise ValueError("option_type must be 'call' or 'put'")
    values = (spot, strike, maturity, volatility, rate, dividend_yield)
    tensors = [value for value in values if isinstance(value, torch.Tensor)]
    dtype = tensors[0].dtype if tensors else torch.float64
    device = tensors[0].device if tensors else torch.device("cpu")
    if any(tensor.dtype not in (torch.float32, torch.float64) for tensor in tensors):
        raise TypeError("tensor inputs must have float32 or float64 dtype")
    if any(tensor.dtype != dtype or tensor.device != device for tensor in tensors):
        raise ValueError("tensor inputs must share the same dtype and device")
    inputs = tuple(
        torch.broadcast_tensors(  # type: ignore[no-untyped-call]  # PyTorch lacks this annotation.
            *(torch.as_tensor(value, dtype=dtype, device=device) for value in values)
        )
    )
    if any(not torch.isfinite(value).all() for value in inputs):
        raise ValueError("all inputs must be finite")
    s, k, t, vol, _, _ = inputs
    if torch.any(s <= 0) or torch.any(k <= 0):
        raise ValueError("spot and strike must be positive")
    if torch.any(t < 0) or torch.any(vol < 0):
        raise ValueError("maturity and volatility must be nonnegative")
    return inputs


def _normal_cdf(value: torch.Tensor) -> torch.Tensor:
    # erfc keeps small lower-tail probabilities that 1 + erf would round to zero.
    return 0.5 * torch.erfc(-value / math.sqrt(2.0))


def _terms(
    spot: torch.Tensor,
    strike: torch.Tensor,
    maturity: torch.Tensor,
    volatility: torch.Tensor,
    rate: torch.Tensor,
    dividend_yield: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    discounted_spot = spot * torch.exp(-dividend_yield * maturity)
    discounted_strike = strike * torch.exp(-rate * maturity)
    # The price function selects exact boundary payoffs separately. Safe inputs
    # keep the unused regular branch from dividing by zero or poisoning autograd.
    safe_t = torch.where(maturity > 0, maturity, torch.ones_like(maturity))
    safe_vol = torch.where(volatility > 0, volatility, torch.ones_like(volatility))
    stddev = safe_vol * torch.sqrt(safe_t)
    log_forward_moneyness = torch.log(spot) - torch.log(strike) + (rate - dividend_yield) * maturity
    d1 = log_forward_moneyness / stddev + 0.5 * stddev
    d2 = d1 - stddev
    return discounted_spot, discounted_strike, stddev, d1, d2


def black_scholes_price(
    spot: ScalarOrTensor,
    strike: ScalarOrTensor,
    maturity: ScalarOrTensor,
    volatility: ScalarOrTensor,
    rate: ScalarOrTensor = 0.0,
    dividend_yield: ScalarOrTensor = 0.0,
    *,
    option_type: OptionType = "call",
) -> torch.Tensor:
    """Return broadcast European call or put prices.

    Spot/strike must be positive; maturity/volatility must be nonnegative;
    all inputs must be finite. Negative rates and dividend yields are supported.
    At expiry return intrinsic value. At zero volatility return the discounted
    deterministic payoff. Invalid inputs raise rather than being clipped.

    Compute the out-of-the-money leg with normal-tail probabilities and recover
    the in-the-money leg by parity. Extreme tails can still underflow at the
    chosen precision; this is not an arbitrary-precision pricing implementation.
    Nonfinite or negative computed prices raise FloatingPointError. Float32 can
    suffer cancellation at very small total volatility; use float64 references.
    """
    s, k, t, vol, r, q = _prepare_inputs(
        spot, strike, maturity, volatility, rate, dividend_yield, option_type
    )
    a, b, _, d1, d2 = _terms(s, k, t, vol, r, q)
    call = a * _normal_cdf(d1) - b * _normal_cdf(d2)
    put = b * _normal_cdf(-d2) - a * _normal_cdf(-d1)
    if option_type == "call":
        regular_price = torch.where(a <= b, call, put + (a - b))
        deterministic_price = torch.clamp_min(a - b, 0.0)
    else:
        regular_price = torch.where(a <= b, call + (b - a), put)
        deterministic_price = torch.clamp_min(b - a, 0.0)
    price = torch.where((t == 0) | (vol == 0), deterministic_price, regular_price)
    if not torch.isfinite(price).all() or torch.any(price < 0):
        raise FloatingPointError(
            "Black–Scholes price is nonfinite or negative at this precision; "
            "use float64 or less extreme inputs"
        )
    return price


def black_scholes_greeks(
    spot: ScalarOrTensor,
    strike: ScalarOrTensor,
    maturity: ScalarOrTensor,
    volatility: ScalarOrTensor,
    rate: ScalarOrTensor = 0.0,
    dividend_yield: ScalarOrTensor = 0.0,
    *,
    option_type: OptionType = "call",
) -> BlackScholesGreeks:
    """Return analytic delta, gamma, vega, annual theta, and rho.

    The pricing input/dtype contract applies, with strictly positive maturity
    and volatility required for every quote. Boundary Greeks can be singular
    or undefined, so a batch containing a boundary quote raises ValueError.
    Nonfinite computed Greeks raise FloatingPointError.
    See ``BlackScholesGreeks`` for units and derivative conventions.
    """
    s, k, t, vol, r, q = _prepare_inputs(
        spot, strike, maturity, volatility, rate, dividend_yield, option_type
    )
    if torch.any(t == 0) or torch.any(vol == 0):
        raise ValueError("Greeks require strictly positive maturity and volatility")
    a, b, stddev, d1, d2 = _terms(s, k, t, vol, r, q)
    sign = 1.0 if option_type == "call" else -1.0
    cdf1 = _normal_cdf(sign * d1)
    cdf2 = _normal_cdf(sign * d2)
    density = torch.exp(-0.5 * d1.square()) / math.sqrt(2.0 * math.pi)
    discount = torch.exp(-q * t)
    result = BlackScholesGreeks(
        delta=sign * discount * cdf1,
        gamma=discount * density / (s * stddev),
        vega=a * density * torch.sqrt(t),
        theta=-a * density * vol / (2.0 * torch.sqrt(t)) + sign * (q * a * cdf1 - r * b * cdf2),
        rho=sign * t * b * cdf2,
    )
    if any(
        not torch.isfinite(value).all()
        for value in (result.delta, result.gamma, result.vega, result.theta, result.rho)
    ):
        raise FloatingPointError(
            "Black–Scholes Greeks are nonfinite at this precision; "
            "use float64 or less extreme inputs"
        )
    return result
