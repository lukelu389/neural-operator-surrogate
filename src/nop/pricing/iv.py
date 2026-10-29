"""Batched, bounded Black–Scholes IV inversion with explicit failure reasons.

IV is annual decimal volatility. This is a detached numerical label solver,
not a differentiable pricing operation. Invalid quotes remain in the output;
their volatility is NaN and their status explains why they were rejected.
"""

import math
from dataclasses import dataclass
from enum import IntEnum

import torch

from nop.pricing.bs import (
    OptionType,
    ScalarOrTensor,
    black_scholes_greeks,
    black_scholes_price,
)


class IVStatus(IntEnum):
    """Mutually exclusive quote outcomes, in input/bounds/solver check order."""

    SUCCESS = 0
    NONFINITE_INPUT = 1
    INVALID_DOMAIN = 2
    EXPIRED = 3
    PRICE_OUT_OF_BOUNDS = 4
    NEAR_LOWER_BOUND = 5
    NEAR_UPPER_BOUND = 6
    UNBRACKETED = 7
    LOW_VEGA = 8
    NO_CONVERGENCE = 9
    NUMERICAL_ERROR = 10


@dataclass(frozen=True)
class IVSettings:
    """Explicit numerical policy; no settings are frozen production defaults.

    Price tolerance is ``price_atol + price_rtol * scale``, with
    ``scale = max(S exp(-qT), K exp(-rT))``. The same tolerance defines
    proximity to static price bounds. Vega is per decimal volatility unit;
    reject roots with ``vega / scale < min_normalized_vega``. This conditioning
    filter does not certify the accuracy of the input price.
    """

    price_atol: float
    price_rtol: float
    iv_atol: float
    min_normalized_vega: float
    volatility_upper: float
    max_iterations: int

    def __post_init__(self) -> None:
        values = (
            self.price_atol,
            self.price_rtol,
            self.iv_atol,
            self.min_normalized_vega,
            self.volatility_upper,
        )
        if any(not math.isfinite(value) for value in values):
            raise ValueError("IV settings must be finite")
        if self.price_atol < 0 or self.price_rtol < 0 or self.price_atol + self.price_rtol <= 0:
            raise ValueError("price tolerances must be nonnegative and at least one positive")
        if self.iv_atol <= 0 or self.volatility_upper <= 0 or self.min_normalized_vega < 0:
            raise ValueError(
                "IV tolerance and upper bound must be positive; vega floor nonnegative"
            )
        if isinstance(self.max_iterations, bool) or not isinstance(self.max_iterations, int):
            raise TypeError("max_iterations must be an integer")
        if self.max_iterations < 1:
            raise ValueError("max_iterations must be a positive integer")


@dataclass(frozen=True)
class IVResult:
    """Broadcast outputs; failed IVs are NaN, never clipped or replaced.

    ``price_residual`` is repriced minus input price at the latest candidate,
    or NaN before a candidate is evaluated. ``vega`` is computed for converged
    roots, including rejected low-vega roots; otherwise it is NaN. Iterations
    count bisection evaluations. Status and iterations have int64 dtype.
    """

    volatility: torch.Tensor
    status: torch.Tensor
    iterations: torch.Tensor
    price_residual: torch.Tensor
    vega: torch.Tensor

    @property
    def valid(self) -> torch.Tensor:
        return self.status == IVStatus.SUCCESS

    @property
    def failed(self) -> torch.Tensor:
        return ~self.valid


def _prepare_inputs(values: tuple[ScalarOrTensor, ...]) -> tuple[torch.Tensor, ...]:
    tensors = [value for value in values if isinstance(value, torch.Tensor)]
    dtype = tensors[0].dtype if tensors else torch.float64
    device = tensors[0].device if tensors else torch.device("cpu")
    if any(tensor.dtype not in (torch.float32, torch.float64) for tensor in tensors):
        raise TypeError("tensor inputs must have float32 or float64 dtype")
    if any(tensor.dtype != dtype or tensor.device != device for tensor in tensors):
        raise ValueError("tensor inputs must share the same dtype and device")
    return tuple(
        torch.broadcast_tensors(  # type: ignore[no-untyped-call]
            *(torch.as_tensor(value, dtype=dtype, device=device) for value in values)
        )
    )


def _evaluate(
    inputs: tuple[torch.Tensor, ...], option_type: OptionType, *, vega: bool = False
) -> torch.Tensor:
    """Reuse the reference BS implementation, isolating rare arithmetic failures."""

    def evaluate(batch: tuple[torch.Tensor, ...]) -> torch.Tensor:
        s, k, t, vol, r, q = batch
        if vega:
            return black_scholes_greeks(s, k, t, vol, r, q, option_type=option_type).vega
        return black_scholes_price(s, k, t, vol, r, q, option_type=option_type)

    output = torch.full_like(inputs[0], math.nan)
    # An upper bound or midpoint can overflow/underflow when cast to float32.
    admissible = torch.isfinite(inputs[3]) & (inputs[3] > 0)
    indices = torch.nonzero(admissible, as_tuple=True)[0]
    if indices.numel() == 0:
        return output
    batch = tuple(value[indices] for value in inputs)
    try:
        output[indices] = evaluate(batch)
    except FloatingPointError:
        # Usually one vectorized call suffices. A bad quote must not mask its
        # valid neighbors merely because the underlying pricer raises by batch.
        for index in indices:
            try:
                output[index] = evaluate(tuple(value[index] for value in inputs))
            except FloatingPointError:
                pass
    return output


@torch.no_grad()
def implied_volatility(
    price: ScalarOrTensor,
    spot: ScalarOrTensor,
    strike: ScalarOrTensor,
    maturity: ScalarOrTensor,
    rate: ScalarOrTensor = 0.0,
    dividend_yield: ScalarOrTensor = 0.0,
    *,
    settings: IVSettings,
    option_type: OptionType = "call",
) -> IVResult:
    """Invert prices by bisection on [0, settings.volatility_upper].

    Require both a small price residual and a small IV bracket. Inputs follow
    BS broadcasting/dtype/device conventions, but bad numeric quotes are
    masked individually. Structural errors (dtype/device/shape/option/settings)
    raise. Expiry, prices near either static bound, and low-vega roots are
    unidentified or poorly conditioned, so they do not produce usable IVs.
    Strict out-of-bound prices are rejected even within tolerance: no clipping.
    Float32 generally requires coarser caller-specified tolerances than float64.
    """
    if not isinstance(settings, IVSettings):
        raise TypeError("settings must be IVSettings")
    if option_type not in ("call", "put"):
        raise ValueError("option_type must be 'call' or 'put'")
    broadcast = _prepare_inputs((price, spot, strike, maturity, rate, dividend_yield))
    shape = broadcast[0].shape
    p, s, k, t, r, q = (value.reshape(-1) for value in broadcast)
    volatility = torch.full_like(p, math.nan)
    residual = torch.full_like(p, math.nan)
    vega = torch.full_like(p, math.nan)
    status = torch.full_like(p, IVStatus.NO_CONVERGENCE, dtype=torch.int64)
    iterations = torch.zeros_like(status)
    pending = torch.ones_like(p, dtype=torch.bool)

    def reject(mask: torch.Tensor, reason: IVStatus) -> None:
        rejected = pending & mask
        status[rejected] = reason
        pending[rejected] = False

    finite = torch.stack([torch.isfinite(value) for value in (p, s, k, t, r, q)]).all(dim=0)
    reject(~finite, IVStatus.NONFINITE_INPUT)
    reject((s <= 0) | (k <= 0) | (t < 0), IVStatus.INVALID_DOMAIN)
    reject(t == 0, IVStatus.EXPIRED)
    a, b = s * torch.exp(-q * t), k * torch.exp(-r * t)
    scale = torch.maximum(a, b)
    tolerance = settings.price_atol + settings.price_rtol * scale
    reject(
        ~torch.isfinite(a)
        | ~torch.isfinite(b)
        | (a <= 0)
        | (b <= 0)
        | ~torch.isfinite(tolerance)
        | (tolerance <= 0),
        IVStatus.NUMERICAL_ERROR,
    )
    lower = torch.clamp_min(a - b if option_type == "call" else b - a, 0.0)
    upper = a if option_type == "call" else b
    reject((p < lower) | (p > upper), IVStatus.PRICE_OUT_OF_BOUNDS)
    reject(p - lower <= tolerance, IVStatus.NEAR_LOWER_BOUND)
    reject(upper - p <= tolerance, IVStatus.NEAR_UPPER_BOUND)

    lo = torch.zeros_like(p)
    hi = (
        torch.as_tensor(settings.volatility_upper, dtype=p.dtype, device=p.device)
        .expand_as(p)
        .clone()
    )
    indices = torch.nonzero(pending, as_tuple=True)[0]
    cap_price = _evaluate(tuple(value[indices] for value in (s, k, t, hi, r, q)), option_type)
    status[indices[~torch.isfinite(cap_price)]] = IVStatus.NUMERICAL_ERROR
    status[indices[p[indices] > cap_price]] = IVStatus.UNBRACKETED
    pending[indices] = torch.isfinite(cap_price) & (p[indices] <= cap_price)
    roots = torch.full_like(p, math.nan)
    for iteration in range(1, settings.max_iterations + 1):
        indices = torch.nonzero(pending, as_tuple=True)[0]
        if indices.numel() == 0:
            break
        mid = lo[indices] + (hi[indices] - lo[indices]) * 0.5
        stagnated = (mid == lo[indices]) | (mid == hi[indices])
        computed = _evaluate(
            (s[indices], k[indices], t[indices], mid, r[indices], q[indices]), option_type
        )
        error = computed - p[indices]
        residual[indices] = error
        iterations[indices] = iteration
        numerical_error = ~torch.isfinite(computed)
        status[indices[numerical_error]] = IVStatus.NUMERICAL_ERROR
        lo[indices] = torch.where(error < 0, mid, lo[indices])
        hi[indices] = torch.where(error >= 0, mid, hi[indices])
        converged = (
            ~numerical_error
            & (error.abs() <= tolerance[indices])
            & (hi[indices] - lo[indices] <= settings.iv_atol)
        )
        roots[indices[converged]] = mid[converged]
        status[indices[converged]] = IVStatus.SUCCESS
        pending[indices] = ~(numerical_error | converged | stagnated)

    indices = torch.nonzero(status == IVStatus.SUCCESS, as_tuple=True)[0]
    root_vega = _evaluate(
        tuple(value[indices] for value in (s, k, t, roots, r, q)), option_type, vega=True
    )
    vega[indices] = root_vega
    status[indices[root_vega / scale[indices] < settings.min_normalized_vega]] = IVStatus.LOW_VEGA
    status[indices[~torch.isfinite(root_vega)]] = IVStatus.NUMERICAL_ERROR
    valid = status == IVStatus.SUCCESS
    volatility[valid] = roots[valid]
    return IVResult(
        *(value.reshape(shape) for value in (volatility, status, iterations, residual, vega))
    )
