"""Masked standardized-IV objectives under the frozen training protocol."""

import torch


def _check_mask(value: torch.Tensor, valid: torch.Tensor) -> None:
    if valid.dtype != torch.bool:
        raise TypeError("valid must have Boolean dtype")
    if valid.shape != value.shape:
        raise ValueError("valid must have the same shape as the values")
    if valid.device != value.device:
        raise ValueError("valid and values must share device")


def _check_float(value: torch.Tensor, name: str) -> None:
    if value.dtype not in (torch.float32, torch.float64):
        raise TypeError(f"{name} must have float32 or float64 dtype")


def masked_mse(
    prediction: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor,
    weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Mean squared standardized-IV residual over reference-valid quotes.

    Every argument has the same shape and device; floating inputs may be
    float32 or float64. Optional fixed weights multiply squared residuals;
    the denominator remains the valid quote count, never the weight sum.
    Masked entries are indexed out before arithmetic, including masked NaNs.
    An empty mask raises ValueError so the caller can record and skip the
    batch without an optimizer or scheduler step. Nonfinite predictions on
    valid quotes fail explicitly; they never alter the reference mask.
    """
    _check_float(prediction, "prediction")
    _check_float(target, "target")
    _check_mask(prediction, valid)
    if target.shape != prediction.shape:
        raise ValueError("prediction and target must have the same shape")
    if target.device != prediction.device:
        raise ValueError("prediction and target must share device")
    if weights is not None:
        _check_float(weights, "weights")
        if weights.shape != prediction.shape:
            raise ValueError("weights and prediction must have the same shape")
        if weights.device != prediction.device:
            raise ValueError("weights and prediction must share device")
    if not valid.any():
        raise ValueError("masked_mse requires at least one valid quote; skip empty batches")

    predicted, reference = prediction[valid], target[valid]
    if not torch.isfinite(reference).all():
        raise ValueError("targets must be finite at valid quotes")
    if not torch.isfinite(predicted).all():
        raise FloatingPointError("predictions must be finite at valid quotes")
    squared = (predicted - reference).square()
    if weights is not None:
        selected_weights = weights[valid]
        if not torch.isfinite(selected_weights).all() or (selected_weights < 0).any():
            raise ValueError("weights must be finite and nonnegative at valid quotes")
        squared = squared * selected_weights
    loss = squared.mean()
    if not torch.isfinite(loss):
        raise FloatingPointError("masked MSE produced a nonfinite loss")
    return loss


@torch.no_grad()
def vega_weights(vega: torch.Tensor, target_std: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """Fit fixed vega-ablation weights on the complete training split only.

    The caller must supply all training quotes once, then reuse slices of the
    returned weights during training; never refit on batches or validation.
    ``target_std`` contains training IV standard deviations and broadcasts
    to ``vega`` (usually [T,Z] onto [N,T,Z]). Each raw weight is
    ``(vega * target_std)**2``, divided by its mean over valid training quotes.
    Valid inputs must be finite and positive. Masked weights are zero and
    cannot contaminate the fit, even when their inputs are NaN.
    """
    _check_float(vega, "vega")
    _check_float(target_std, "target_std")
    _check_mask(vega, valid)
    if target_std.device != vega.device:
        raise ValueError("vega and target_std must share device")
    try:
        standard_deviation = torch.broadcast_to(target_std, vega.shape)
    except RuntimeError as error:
        raise ValueError("target_std must broadcast to the vega shape") from error
    if not valid.any():
        raise ValueError("vega_weights requires at least one valid training quote")
    selected_vega, selected_std = vega[valid], standard_deviation[valid]
    if (
        not torch.isfinite(selected_vega).all()
        or (selected_vega <= 0).any()
        or not torch.isfinite(selected_std).all()
        or (selected_std <= 0).any()
    ):
        raise ValueError("vega and target_std must be finite and positive at valid quotes")
    raw_weights = (selected_vega * selected_std).square()
    scale = raw_weights.mean()
    if not torch.isfinite(raw_weights).all() or not torch.isfinite(scale) or scale <= 0:
        raise FloatingPointError("raw vega weights must have a finite positive training mean")
    normalized = raw_weights / scale
    if not torch.isfinite(normalized).all() or (normalized <= 0).any():
        raise FloatingPointError("normalized vega weights must be finite and positive")
    result = torch.zeros_like(vega, dtype=normalized.dtype)
    result[valid] = normalized
    return result
