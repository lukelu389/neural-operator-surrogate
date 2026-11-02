"""COS surface pricing with one characteristic-function evaluation per maturity.

This adapter shares the validated numeric primitives in ``heston_cos``. Their
internal interfaces are intentionally coupled: changes there must retain grid
versus quote-pricer regression coverage. Input/domain errors raise; numerical
failures are returned as NaN at the affected quotes, without clipping. This is
a reference-data path, with no autograd contract.
"""

import math

import torch

from nop.pricing.bs import OptionType
from nop.pricing.heston_cos import _characteristic_function, _cumulants


@torch.no_grad()
def heston_cos_grid_price(
    parameters: torch.Tensor,
    strikes: torch.Tensor,
    maturities: torch.Tensor,
    spot: float,
    rate: float,
    dividend_yield: float,
    *,
    n_terms: int,
    truncation_width: float,
    option_type: OptionType = "put",
) -> torch.Tensor:
    """Return [N,T,K] prices for parameters ordered (v0,kappa,theta,sigma,rho).

    ``parameters`` has shape [N,5], ``strikes`` [T,K], and ``maturities`` [T].
    All tensors must share a float32/float64 dtype and device. Spot/strikes and
    kappa are positive, T/v0/theta/sigma nonnegative, and rho lies in [-1,1].
    N>=2 cosine terms and finite positive truncation width are explicit choices.
    The parameter batch is caller-controlled; temporary storage scales as
    batch size * maturity count * term count, independently of strike count.
    """
    tensors = (parameters, strikes, maturities)
    if any(not isinstance(value, torch.Tensor) for value in tensors):
        raise TypeError("parameters, strikes, and maturities must be tensors")
    if any(value.dtype not in (torch.float32, torch.float64) for value in tensors):
        raise TypeError("tensor inputs must have float32 or float64 dtype")
    if any(
        value.dtype != parameters.dtype or value.device != parameters.device for value in tensors
    ):
        raise ValueError("tensor inputs must share the same dtype and device")
    if parameters.ndim != 2 or parameters.shape[1] != 5:
        raise ValueError("parameters must have shape [N,5]")
    if maturities.ndim != 1 or maturities.numel() == 0:
        raise ValueError("maturities must have nonempty shape [T]")
    if strikes.ndim != 2 or strikes.shape[0] != maturities.numel() or strikes.shape[1] == 0:
        raise ValueError("strikes must have nonempty shape [T,K]")
    if any(not torch.isfinite(value).all() for value in tensors) or not all(
        math.isfinite(value) for value in (spot, rate, dividend_yield)
    ):
        raise ValueError("all inputs must be finite")
    if spot <= 0 or torch.any(strikes <= 0):
        raise ValueError("spot and strikes must be positive")
    if isinstance(n_terms, bool) or not isinstance(n_terms, int) or n_terms < 2:
        raise ValueError("n_terms must be an integer >= 2")
    if not math.isfinite(truncation_width) or truncation_width <= 0:
        raise ValueError("truncation_width must be finite and positive")
    if option_type not in ("call", "put"):
        raise ValueError("option_type must be 'call' or 'put'")
    v0, kappa, theta, sigma, rho = (parameters[:, i, None] for i in range(5))
    t = maturities[None, :]
    if torch.any(kappa <= 0):
        raise ValueError("kappa must be positive")
    if any(torch.any(value < 0) for value in (t, v0, theta, sigma)):
        raise ValueError("maturities, v0, theta, and sigma must be nonnegative")
    if torch.any(rho.abs() > 1):
        raise ValueError("rho must be in [-1,1]")
    r = torch.as_tensor(rate, dtype=parameters.dtype, device=parameters.device)
    q = torch.as_tensor(dividend_yield, dtype=parameters.dtype, device=parameters.device)
    c1, c2 = _cumulants(t, v0, kappa, theta, sigma, rho, r, q)
    deterministic = (t == 0) | ((v0 == 0) & (theta == 0))
    valid_cumulants = torch.isfinite(c1) & torch.isfinite(c2) & (c2 >= 0)
    valid_cumulants &= deterministic | (c2 > 0)
    safe_variance = torch.where(valid_cumulants & ~deterministic, c2, torch.ones_like(c2))
    radius = truncation_width * torch.sqrt(safe_variance)
    lower_return = torch.where(valid_cumulants, c1, torch.zeros_like(c1)) - radius
    frequencies = (
        torch.arange(n_terms, dtype=parameters.dtype, device=parameters.device)
        * math.pi
        / (2 * radius[..., None])
    )
    cf = _characteristic_function(
        frequencies,
        *(value[..., None] for value in (t, v0, kappa, theta, sigma, rho, r, q)),
    )
    density = (cf * torch.exp(-1j * frequencies * lower_return[..., None])).real
    valid_density = torch.isfinite(cf).all(dim=-1) & torch.isfinite(density).all(dim=-1)
    weights = torch.ones(n_terms, dtype=parameters.dtype, device=parameters.device)
    weights[0] = 0.5
    weighted_density = density * weights
    safe_frequency = torch.where(frequencies == 0, torch.ones_like(frequencies), frequencies)
    denominator = 1 + frequencies.square()
    discounted_spot = spot * torch.exp(-q * t)
    log_spot = torch.log(torch.as_tensor(spot, dtype=parameters.dtype, device=parameters.device))
    prices = []
    # Sharing the density removes the expensive CF calculation from this loop.
    for strike in strikes.unbind(dim=1):
        a = lower_return + (log_spot - torch.log(strike)[None, :])
        b = a + 2 * radius
        upper = torch.minimum(b, torch.zeros_like(b))
        active = upper > a
        lower = torch.where(active, a, torch.zeros_like(a))[..., None]
        upper = torch.where(active, upper, torch.zeros_like(upper))[..., None]
        length = upper - lower
        angle = frequencies * length
        sin_angle = torch.sin(angle)
        psi = torch.where(frequencies == 0, length, sin_angle / safe_frequency)
        chi = (
            torch.exp(upper)
            * (-torch.expm1(-length) - 2 * torch.sin(angle / 2).square() + frequencies * sin_angle)
            / denominator
        )
        coefficients = (psi - chi) / radius[..., None]
        discounted_strike = strike[None, :] * torch.exp(-r * t)
        put = discounted_strike * torch.sum(weighted_density * coefficients, dim=-1)
        price = put if option_type == "put" else put + (discounted_spot - discounted_strike)
        sign = 1 if option_type == "call" else -1
        intrinsic = torch.clamp_min(sign * (discounted_spot - discounted_strike), 0.0)
        price = torch.where(deterministic, intrinsic, price)
        valid = (
            (deterministic | (valid_cumulants & valid_density))
            & torch.isfinite(price)
            & (price >= 0)
        )
        prices.append(torch.where(valid, price, torch.full_like(price, torch.nan)))
    return torch.stack(prices, dim=-1)
