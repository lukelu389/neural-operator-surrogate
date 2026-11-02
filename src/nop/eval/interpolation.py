"""Predeclared M0 adapter: natural cubic total variance in forward k."""

from dataclasses import dataclass
from enum import IntEnum

import torch


class InterpolationStatus(IntEnum):
    """Per-query prediction outcomes, separate from reference-label validity."""

    SUCCESS = 0
    INVALID_QUERY = 1
    OUTSIDE_MATURITY_SUPPORT = 2
    OUTSIDE_STRIKE_SUPPORT = 3
    INVALID_NATIVE_ROW = 4
    NONPOSITIVE_VARIANCE = 5
    NUMERICAL_FAILURE = 6


@dataclass(frozen=True)
class InterpolationResult:
    """Decimal IV, validity, and integer status; unsuccessful IVs are NaN."""

    iv: torch.Tensor
    valid: torch.Tensor
    status: torch.Tensor


def _spline_row(
    k: torch.Tensor,
    variance: torch.Tensor,
    native_valid: torch.Tensor,
    query_k: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Evaluate one native maturity for all leading-axis surfaces."""
    row_valid = native_valid.all(dim=-1)
    values = torch.where(row_valid[:, None], variance, 0.0)
    gaps = k.diff()
    second = torch.zeros_like(values)
    if k.numel() > 2:
        matrix = torch.diag(2 * (gaps[:-1] + gaps[1:]))
        if k.numel() > 3:
            matrix = matrix + torch.diag(gaps[1:-1], diagonal=1)
            matrix = matrix + torch.diag(gaps[1:-1], diagonal=-1)
        slopes = values.diff(dim=-1) / gaps
        second[:, 1:-1] = torch.linalg.solve(matrix, 6 * slopes.diff(dim=-1).T).T

    index = torch.searchsorted(k, query_k.contiguous(), right=True).clamp(1, k.numel() - 1)
    width = gaps[index - 1]
    left = (k[index] - query_k) / width
    right = (query_k - k[index - 1]) / width
    interpolated = left * values[:, index - 1] + right * values[:, index]
    interpolated = interpolated + width.square() / 6 * (
        (left.pow(3) - left) * second[:, index - 1] + (right.pow(3) - right) * second[:, index]
    )
    status = torch.zeros_like(interpolated, dtype=torch.int64)
    status[:, (query_k < k[0]) | (query_k > k[-1])] = InterpolationStatus.OUTSIDE_STRIKE_SUPPORT
    status[(status == 0) & ~row_valid[:, None]] = InterpolationStatus.INVALID_NATIVE_ROW
    status[(status == 0) & ~torch.isfinite(interpolated)] = InterpolationStatus.NUMERICAL_FAILURE
    status[(status == 0) & (interpolated <= 0)] = InterpolationStatus.NONPOSITIVE_VARIANCE
    return interpolated, status


def interpolate_m0_iv(
    native_iv: torch.Tensor,
    *,
    native_maturities: torch.Tensor,
    native_k: torch.Tensor,
    query_maturity: torch.Tensor,
    query_k: torch.Tensor,
    native_valid: torch.Tensor | None = None,
) -> InterpolationResult:
    """Map native decimal IV ``[..., T, K]`` onto broadcast query coordinates.

    ``native_maturities`` is [T], ``native_k`` is [T, K], and the result shape
    is ``native_iv.shape[:-2] + broadcast(query_maturity, query_k).shape``.
    Queries are shared across all leading-axis surfaces. Coordinates are
    converted to the IV tensor's dtype/device, which must be float32/float64.

    Interpolate w=T*IV**2 with a natural cubic spline in actual forward k at
    each native maturity. At fixed k, interpolate w linearly between maturities
    and extrapolate below the first maturity using the first two rows. Every
    required row must support that k; no strike or long-maturity extrapolation
    is allowed. An exact native maturity uses only that row. Two strike nodes
    give the natural spline's linear special case.

    Malformed grids raise ValueError. Invalid/nonpositive native IV, nonfinite
    native w, or a false optional native_valid entry invalidate the entire
    affected native row: a natural spline depends on all its knots. These are
    prediction failures, never permission to shrink a common reference mask.
    Nonpositive/nonfinite interpolated w fails explicitly without clipping.
    This adapter preserves autograd through valid arithmetic; no accelerator
    accuracy or extrapolation-accuracy guarantee is implied.
    """
    if native_iv.dtype not in (torch.float32, torch.float64) or native_iv.ndim < 2:
        raise ValueError("native_iv must be a float32/float64 tensor with shape [..., T, K]")
    maturity_count, strike_count = native_iv.shape[-2:]
    if maturity_count < 2 or strike_count < 2:
        raise ValueError("at least two native maturities and strike nodes are required")
    if native_maturities.shape != (maturity_count,) or native_k.shape != (
        maturity_count,
        strike_count,
    ):
        raise ValueError("native coordinate shapes must match the last two IV axes")
    maturities = native_maturities.to(native_iv).contiguous()
    strikes = native_k.to(native_iv).contiguous()
    if (
        not torch.isfinite(maturities).all()
        or (maturities <= 0).any()
        or not torch.isfinite(maturities.diff()).all()
        or (maturities.diff() <= 0).any()
        or not torch.isfinite(strikes).all()
        or not torch.isfinite(strikes.diff(dim=-1)).all()
        or (strikes.diff(dim=-1) <= 0).any()
    ):
        raise ValueError("native maturities must be positive and both axes finite/increasing")
    if native_valid is not None and (
        native_valid.shape != native_iv.shape or native_valid.dtype != torch.bool
    ):
        raise ValueError("native_valid must be a boolean tensor matching native_iv")

    query_t, query_x = torch.broadcast_tensors(  # type: ignore[no-untyped-call]
        query_maturity.to(native_iv), query_k.to(native_iv)
    )
    output_shape = native_iv.shape[:-2] + query_t.shape
    query_t, query_x = query_t.reshape(-1), query_x.reshape(-1)
    surfaces = native_iv.reshape(-1, maturity_count, strike_count)
    variance = maturities[:, None] * surfaces.square()
    good_native = torch.isfinite(surfaces) & (surfaces > 0)
    good_native = good_native & torch.isfinite(variance) & (variance > 0)
    if native_valid is not None:
        good_native = good_native & native_valid.to(device=native_iv.device).reshape_as(surfaces)

    good_query = torch.isfinite(query_t) & (query_t > 0) & torch.isfinite(query_x)
    query_status = torch.zeros_like(query_t, dtype=torch.int64)
    query_status[~good_query] = InterpolationStatus.INVALID_QUERY
    query_status[good_query & (query_t > maturities[-1])] = (
        InterpolationStatus.OUTSIDE_MATURITY_SUPPORT
    )
    active = query_status == InterpolationStatus.SUCCESS
    upper = torch.searchsorted(maturities, query_t.contiguous()).clamp_max(maturity_count - 1)
    exact = query_t == maturities[upper]
    lower = torch.where(exact, upper, (upper - 1).clamp_min(0))
    upper = torch.where(query_t < maturities[0], 1, upper)
    shape = (surfaces.shape[0], query_t.numel())
    low_w = torch.full(shape, torch.nan, dtype=native_iv.dtype, device=native_iv.device)
    high_w = torch.full_like(low_w, torch.nan)
    low_status = query_status.expand(shape).clone()
    high_status = low_status.clone()
    for row in range(maturity_count):
        selected = active & ((lower == row) | (upper == row))
        if not selected.any():
            continue
        evaluated, row_status = _spline_row(
            strikes[row], variance[:, row], good_native[:, row], query_x[selected]
        )
        for indices, target, statuses in (
            (lower, low_w, low_status),
            (upper, high_w, high_status),
        ):
            local = indices[selected] == row
            target[:, selected & (indices == row)] = evaluated[:, local]
            statuses[:, selected & (indices == row)] = row_status[:, local]

    status = torch.maximum(low_status, high_status)
    denominator = torch.where(lower == upper, 1.0, maturities[upper] - maturities[lower])
    weight = (query_t - maturities[lower]) / denominator
    w = low_w + weight * (high_w - low_w)
    status[(status == 0) & ~torch.isfinite(w)] = InterpolationStatus.NUMERICAL_FAILURE
    status[(status == 0) & (w <= 0)] = InterpolationStatus.NONPOSITIVE_VARIANCE
    # Safe placeholders prevent invalid square roots; failed outputs remain NaN.
    safe_w = torch.where(status == 0, w, 1.0)
    safe_t = torch.where(good_query, query_t, 1.0)
    iv = torch.sqrt(safe_w / safe_t)
    status[(status == 0) & (~torch.isfinite(iv) | (iv <= 0))] = (
        InterpolationStatus.NUMERICAL_FAILURE
    )
    valid = status == InterpolationStatus.SUCCESS
    return InterpolationResult(
        iv=torch.where(valid, iv, torch.nan).reshape(output_shape),
        valid=valid.reshape(output_shape),
        status=status.reshape(output_shape),
    )
