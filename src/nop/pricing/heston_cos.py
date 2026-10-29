"""Batched Heston European pricing with a stable characteristic function and COS.

The characteristic function is for log(S_T/S_0), including risk-neutral drift.
It uses the decaying-exponential Little Heston Trap representation, rearranged
to avoid division by sigma squared, including at sigma=0. COS truncation uses
the first two log-return cumulants: c1 +/- L sqrt(c2). Both N and L must be
chosen by convergence checks; no production accuracy is implied by this API.

Scalar-only inputs use CPU float64. Tensor inputs must share a float32/float64
dtype and device; scalar arguments adopt these. Operations broadcast and retain
autograd on regular inputs. Float64 is recommended for reference pricing.
"""

import math

import torch

from nop.pricing.bs import OptionType, ScalarOrTensor


def _broadcast(*values: ScalarOrTensor) -> tuple[torch.Tensor, ...]:
    tensors = [value for value in values if isinstance(value, torch.Tensor)]
    dtype = tensors[0].dtype if tensors else torch.float64
    device = tensors[0].device if tensors else torch.device("cpu")
    if any(value.dtype not in (torch.float32, torch.float64) for value in tensors):
        raise TypeError("tensor inputs must have float32 or float64 dtype")
    if any(value.dtype != dtype or value.device != device for value in tensors):
        raise ValueError("tensor inputs must share the same dtype and device")
    result = tuple(
        torch.broadcast_tensors(  # type: ignore[no-untyped-call]  # Missing PyTorch annotation.
            *(torch.as_tensor(value, dtype=dtype, device=device) for value in values)
        )
    )
    if any(not torch.isfinite(value).all() for value in result):
        raise ValueError("all inputs must be finite")
    return result


def _validate_model(
    maturity: torch.Tensor,
    v0: torch.Tensor,
    kappa: torch.Tensor,
    theta: torch.Tensor,
    sigma: torch.Tensor,
    rho: torch.Tensor,
) -> None:
    if torch.any(kappa <= 0):
        raise ValueError("kappa must be positive")
    if any(torch.any(value < 0) for value in (maturity, v0, theta, sigma)):
        raise ValueError("maturity, v0, theta, and sigma must be nonnegative")
    if torch.any(torch.abs(rho) > 1):
        raise ValueError("rho must be in [-1, 1]")


def _polynomial(x: torch.Tensor, coefficients: tuple[float, ...]) -> torch.Tensor:
    result = torch.full_like(x, coefficients[-1])
    for coefficient in reversed(coefficients[:-1]):
        result = result * x + coefficient
    return result


def _decay_ratio(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (1-exp(-x))/x and its complement without small-x cancellation."""
    small = x.abs() < 0.1
    safe_x = torch.where(small, torch.ones_like(x), x)
    series_x = torch.where(small, x, torch.zeros_like(x))
    ratio = -torch.expm1(-safe_x) / safe_x
    # Taylor remainder at |x|<0.1 is below float64 roundoff with 12 terms.
    coefficients = tuple((-1.0) ** n / math.factorial(n + 1) for n in range(12))
    complement = series_x * _polynomial(series_x, tuple(-value for value in coefficients[1:]))
    return (
        torch.where(small, 1 - complement, ratio),
        torch.where(small, complement, 1 - ratio),
    )


def _log_ratio_complement(z: torch.Tensor) -> torch.Tensor:
    """Return 1-log(1+z)/z, analytically continued through zero."""
    small = z.abs() < 0.001
    safe_z = torch.where(small, torch.ones_like(z), z)
    series_z = torch.where(small, z, torch.zeros_like(z))
    series = series_z * _polynomial(series_z, (0.5, -1 / 3, 0.25, -0.2, 1 / 6, -1 / 7))
    return torch.where(small, series, 1 - torch.log1p(safe_z) / safe_z)


def _characteristic_function(
    u: torch.Tensor,
    t: torch.Tensor,
    v0: torch.Tensor,
    kappa: torch.Tensor,
    theta: torch.Tensor,
    sigma: torch.Tensor,
    rho: torch.Tensor,
    rate: torch.Tensor,
    dividend_yield: torch.Tensor,
) -> torch.Tensor:
    iu = 1j * u
    w = u.square() + iu
    beta = kappa - rho * sigma * iu
    # Principal square root has positive real part for real u and kappa>0.
    d = torch.sqrt(beta.square() + sigma.square() * w)
    h = -w / (beta + d)  # (beta-d)/sigma**2, without cancellation as sigma -> 0.
    decay, decay_complement = _decay_ratio(d * t)
    q_over_d = t * decay
    z = 0.5 * sigma.square() * h * q_over_d
    c = kappa * theta * h * t * (decay_complement + decay * _log_ratio_complement(z))
    d_coefficient = -0.5 * w * q_over_d / (1 + z)
    return torch.exp(iu * (rate - dividend_yield) * t + c + d_coefficient * v0)


def heston_characteristic_function(
    frequency: ScalarOrTensor,
    maturity: ScalarOrTensor,
    v0: ScalarOrTensor,
    kappa: ScalarOrTensor,
    theta: ScalarOrTensor,
    sigma: ScalarOrTensor,
    rho: ScalarOrTensor,
    rate: ScalarOrTensor = 0.0,
    dividend_yield: ScalarOrTensor = 0.0,
) -> torch.Tensor:
    """Return E[exp(i*u*log(S_T/S_0))] at real frequencies u.

    Require kappa>0, T/v0/theta/sigma>=0 and rho in [-1,1]. Feller filtering is
    not imposed. The sigma=0 limit includes the full mean-reverting deterministic
    variance path, even when v0 differs from theta. Complex frequencies/damping
    moments are not supported by this interface.
    """
    u, t, v, k, th, sig, corr, r, q = _broadcast(
        frequency, maturity, v0, kappa, theta, sigma, rho, rate, dividend_yield
    )
    _validate_model(t, v, k, th, sig, corr)
    result = _characteristic_function(u, t, v, k, th, sig, corr, r, q)
    if not torch.isfinite(result).all():
        raise FloatingPointError("nonfinite Heston characteristic function")
    return result


def _cumulants(
    t: torch.Tensor,
    v0: torch.Tensor,
    kappa: torch.Tensor,
    theta: torch.Tensor,
    sigma: torch.Tensor,
    rho: torch.Tensor,
    rate: torch.Tensor,
    dividend_yield: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    x = kappa * t
    a, one_minus_a = _decay_ratio(x)
    integrated_variance = t * (v0 * a + theta * one_minus_a)
    small = x.abs() < 0.1
    safe_x = torch.where(small, torch.ones_like(x), x)
    e = torch.exp(-safe_x)
    aa = -torch.expm1(-safe_x) / safe_x
    b = -torch.expm1(-2 * safe_x) / (2 * safe_x)
    # Normalized integral kernels avoid division by tiny kappa and cancellation.
    # A_n=(-1)^n/(n+1)!, B_n=(-2)^n/(n+1)!, E_n=(-1)^n/n!.
    j1v, j1theta, j2v, j2theta = [], [], [], []
    for n in range(12):
        m = n + 1
        an, en = (-1.0) ** m / math.factorial(m + 1), (-1.0) ** m / math.factorial(m)
        j1v.append(an - en)
        j1theta.append(-2 * an + en)
        m = n + 2
        an = (-1.0) ** m / math.factorial(m + 1)
        bn = (-2.0) ** m / math.factorial(m + 1)
        en = (-1.0) ** m / math.factorial(m)
        j2v.append(2 * (bn - en))
        j2theta.append(-2 * an - bn + 2 * en)
    f1v = torch.where(small, _polynomial(x, tuple(j1v)), (aa - e) / safe_x)
    f1theta = torch.where(small, _polynomial(x, tuple(j1theta)), (1 - 2 * aa + e) / safe_x)
    f2v = torch.where(small, _polynomial(x, tuple(j2v)), 2 * (b - e) / safe_x.square())
    f2theta = torch.where(
        small, _polynomial(x, tuple(j2theta)), (1 - 2 * aa - b + 2 * e) / safe_x.square()
    )
    covariance = rho * sigma * t.square() * (v0 * f1v + theta * f1theta)
    variance_of_integral = sigma.square() * t.pow(3) * (v0 * f2v + theta * f2theta)
    c1 = (rate - dividend_yield) * t - 0.5 * integrated_variance
    c2 = integrated_variance + 0.25 * variance_of_integral - covariance
    return c1, c2


def heston_cumulants(
    maturity: ScalarOrTensor,
    v0: ScalarOrTensor,
    kappa: ScalarOrTensor,
    theta: ScalarOrTensor,
    sigma: ScalarOrTensor,
    rho: ScalarOrTensor,
    rate: ScalarOrTensor = 0.0,
    dividend_yield: ScalarOrTensor = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the mean and variance of log(S_T/S_0), not log(S_T/K).

    With I=integral(v dt) and M=integral(sqrt(v) dW_S), c2 is
    E[I] + Var(I)/4 - Cov(I,M). Closed-form integral kernels are evaluated with
    small-kappa*T series. Negative computed variance is an error, not abs-clipped.
    """
    values = _broadcast(maturity, v0, kappa, theta, sigma, rho, rate, dividend_yield)
    _validate_model(*values[:6])
    c1, c2 = _cumulants(*values)
    if not torch.isfinite(c1).all() or not torch.isfinite(c2).all() or torch.any(c2 < 0):
        raise FloatingPointError("invalid Heston cumulants at this precision")
    return c1, c2


def heston_cos_price(
    spot: ScalarOrTensor,
    strike: ScalarOrTensor,
    maturity: ScalarOrTensor,
    v0: ScalarOrTensor,
    kappa: ScalarOrTensor,
    theta: ScalarOrTensor,
    sigma: ScalarOrTensor,
    rho: ScalarOrTensor,
    rate: ScalarOrTensor = 0.0,
    dividend_yield: ScalarOrTensor = 0.0,
    *,
    n_terms: int,
    truncation_width: float,
    option_type: OptionType = "call",
) -> torch.Tensor:
    """Price European options with N cosine terms and c1 +/- L sqrt(c2).

    N>=2 and finite L>0 are explicit convergence settings. Evaluate the bounded
    put payoff and recover calls by parity to avoid exp(b) call coefficients.
    Numerical fixtures do not certify arbitrary N/L or every parameter/quote.
    Expiry and identically zero variance return deterministic payoffs. Nonfinite
    or negative computed prices raise rather than being silently clipped.
    """
    if isinstance(n_terms, bool) or not isinstance(n_terms, int) or n_terms < 2:
        raise ValueError("n_terms must be an integer >= 2")
    if not math.isfinite(truncation_width) or truncation_width <= 0:
        raise ValueError("truncation_width must be finite and positive")
    if option_type not in ("call", "put"):
        raise ValueError("option_type must be 'call' or 'put'")
    s, strike_t, t, v, k, th, sig, corr, r, q = _broadcast(
        spot, strike, maturity, v0, kappa, theta, sigma, rho, rate, dividend_yield
    )
    if torch.any(s <= 0) or torch.any(strike_t <= 0):
        raise ValueError("spot and strike must be positive")
    _validate_model(t, v, k, th, sig, corr)
    c1, c2 = _cumulants(t, v, k, th, sig, corr, r, q)
    if not torch.isfinite(c1).all() or not torch.isfinite(c2).all() or torch.any(c2 < 0):
        raise FloatingPointError("invalid Heston cumulants at this precision")
    deterministic = (t == 0) | ((v == 0) & (th == 0))
    safe_variance = torch.where(deterministic, torch.ones_like(c2), c2)
    radius = truncation_width * torch.sqrt(safe_variance)
    lower_return = c1 - radius
    log_moneyness = torch.log(s) - torch.log(strike_t)
    a = lower_return + log_moneyness
    b = a + 2 * radius
    frequencies = (
        torch.arange(n_terms, dtype=s.dtype, device=s.device) * math.pi / (2 * radius[..., None])
    )
    upper = torch.minimum(b, torch.zeros_like(b))
    active = upper > a
    lower = torch.where(active, a, torch.zeros_like(a))[..., None]
    upper = torch.where(active, upper, torch.zeros_like(upper))[..., None]
    length = upper - lower
    angle = frequencies * length
    # Handle n=0 explicitly: torch.sinc's second derivative at zero can be NaN.
    safe_frequency = torch.where(frequencies == 0, torch.ones_like(frequencies), frequencies)
    psi = torch.where(frequencies == 0, length, torch.sin(angle) / safe_frequency)
    chi = (
        torch.exp(upper)
        * (
            -torch.expm1(-length)
            - 2 * torch.sin(angle / 2).square()
            + frequencies * torch.sin(angle)
        )
        / (1 + frequencies.square())
    )
    coefficients = (psi - chi) / radius[..., None]
    cf = _characteristic_function(
        frequencies, *(value[..., None] for value in (t, v, k, th, sig, corr, r, q))
    )
    density = (cf * torch.exp(-1j * frequencies * lower_return[..., None])).real
    weights = torch.ones(n_terms, dtype=s.dtype, device=s.device)
    weights[0] = 0.5
    discounted_strike = strike_t * torch.exp(-r * t)
    discounted_spot = s * torch.exp(-q * t)
    put = discounted_strike * torch.sum(weights * density * coefficients, dim=-1)
    price = put if option_type == "put" else put + (discounted_spot - discounted_strike)
    sign = 1 if option_type == "call" else -1
    intrinsic = torch.clamp_min(sign * (discounted_spot - discounted_strike), 0.0)
    price = torch.where(deterministic, intrinsic, price)
    if not torch.isfinite(price).all() or torch.any(price < 0):
        raise FloatingPointError(
            "invalid COS price; check precision, term count, and truncation width"
        )
    return price
