"""Independent NumPy Heston Monte Carlo for numerical cross-checks.

Full-truncation Euler uses an auxiliary variance that may be negative; its
positive part enters both the drift and diffusion coefficients. Log-Euler stock
updates use the same left-endpoint variance. This is a biased time-discretized
scheme, not exact Heston simulation. See Lord, Koekkoek and van Dijk (2010):
https://doi.org/10.1080/14697680802392496.
"""

import math
from dataclasses import dataclass

import numpy as np

from nop.pricing.bs import OptionType


@dataclass(frozen=True)
class MonteCarloPrice:
    """Price and approximate normal sampling interval, in price units.

    Independent observations are antithetic *pair means*, not individual paths.
    The interval does not include time-discretization bias, and its normal
    approximation can be poor for small samples or rare positive payoffs.
    ``confidence_z`` is the multiplier, e.g. 1.959963984540054 for nominal 95%.
    """

    estimate: float
    standard_error: float
    confidence_low: float
    confidence_high: float
    n_pairs: int
    n_steps: int
    seed: int
    confidence_z: float


def heston_mc_price(
    spot: float,
    strike: float,
    maturity: float,
    v0: float,
    kappa: float,
    theta: float,
    sigma: float,
    rho: float,
    rate: float = 0.0,
    dividend_yield: float = 0.0,
    *,
    n_pairs: int,
    n_steps: int,
    seed: int,
    confidence_z: float = 1.959963984540054,
    option_type: OptionType = "call",
) -> MonteCarloPrice:
    """Estimate a European price with ``2 * n_pairs`` float64 CPU paths.

    The local NumPy generator leaves global random state untouched. Within each
    independent pair, negate both Brownian drivers for the second path; their
    correlation is ``rho``. The full-truncation scheme retains the untruncated
    auxiliary variance in its recurrence, including after negative proposals.
    No Feller restriction is imposed. Rates, yields, and variance inputs are
    annual decimal quantities, and maturity is in years.

    Require finite scalar inputs, positive spot/strike/kappa, nonnegative
    maturity/v0/theta/sigma, rho in [-1, 1], at least two pairs, and at least one
    step. At expiry return exact intrinsic value and zero sampling error.
    Nonfinite computed states or statistics raise rather than being clipped.
    """
    if option_type not in ("call", "put"):
        raise ValueError("option_type must be 'call' or 'put'")
    inputs = (spot, strike, maturity, v0, kappa, theta, sigma, rho, rate, dividend_yield)
    if any(not math.isfinite(value) for value in (*inputs, confidence_z)):
        raise ValueError("all inputs must be finite scalars")
    if spot <= 0 or strike <= 0 or kappa <= 0:
        raise ValueError("spot, strike, and kappa must be positive")
    if maturity < 0 or v0 < 0 or theta < 0 or sigma < 0:
        raise ValueError("maturity, v0, theta, and sigma must be nonnegative")
    if not -1 <= rho <= 1:
        raise ValueError("rho must be in [-1, 1]")
    if confidence_z <= 0:
        raise ValueError("confidence_z must be positive")
    for name, value, minimum in (
        ("n_pairs", n_pairs, 2),
        ("n_steps", n_steps, 1),
        ("seed", seed, 0),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")

    sign = 1.0 if option_type == "call" else -1.0
    if maturity == 0:
        estimate = max(sign * (spot - strike), 0.0)
        standard_error = 0.0
    else:
        generator = np.random.default_rng(seed)
        dt = maturity / n_steps
        variance = np.full((2, n_pairs), v0, dtype=np.float64)
        log_stock = np.full((2, n_pairs), math.log(spot), dtype=np.float64)
        independent_weight = math.sqrt((1.0 - rho) * (1.0 + rho))
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            for _ in range(n_steps):
                normals = generator.standard_normal((2, n_pairs))
                variance_normal = np.stack((normals[0], -normals[0]))
                stock_driver = rho * normals[0] + independent_weight * normals[1]
                stock_normal = np.stack((stock_driver, -stock_driver))
                positive_variance = np.maximum(variance, 0.0)
                diffusion_scale = np.sqrt(positive_variance * dt)
                log_stock += (rate - dividend_yield - 0.5 * positive_variance) * dt
                log_stock += diffusion_scale * stock_normal
                variance += kappa * (theta - positive_variance) * dt
                variance += sigma * diffusion_scale * variance_normal
            if not np.isfinite(log_stock).all() or not np.isfinite(variance).all():
                raise FloatingPointError("nonfinite Heston Monte Carlo state")
            payoff = np.maximum(sign * (np.exp(log_stock) - strike), 0.0)
            discounted_payoff = np.exp(-rate * maturity) * payoff
            pair_means = 0.5 * discounted_payoff[0] + 0.5 * discounted_payoff[1]
            estimate = float(np.mean(pair_means))
            standard_error = float(np.std(pair_means, ddof=1) / math.sqrt(n_pairs))
    half_width = confidence_z * standard_error
    low, high = estimate - half_width, estimate + half_width
    if not all(math.isfinite(value) for value in (estimate, standard_error, low, high)):
        raise FloatingPointError("nonfinite Heston Monte Carlo result")
    return MonteCarloPrice(
        estimate, standard_error, low, high, n_pairs, n_steps, seed, confidence_z
    )
