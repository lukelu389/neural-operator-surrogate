"""Uniform common-mask IV accuracy and explicit failure accounting."""

import torch


@torch.no_grad()
def iv_metrics(
    prediction_iv: torch.Tensor, reference_iv: torch.Tensor, valid: torch.Tensor
) -> dict[str, int | float | None]:
    """Report decimal-IV errors in vol points (100 times decimal error).

    Inputs have identical shapes and devices, with an explicit Boolean
    reference mask. Floating inputs may be float32 or float64; reductions use
    CPU float64 and the 99th percentile uses linear interpolation. Reference
    values marked valid must be finite and positive. Predictions never change
    that mask: any nonfinite or nonpositive prediction at a valid reference
    quote makes all four accuracy metrics undefined (None).

    Reference failure rate divides by all attempted quotes; prediction failure
    rate divides by valid reference quotes. An empty denominator yields None,
    as do all error metrics when there are no valid references. Masked values
    are ignored without dropping their counts from reference coverage.
    """
    for name, value in (("prediction_iv", prediction_iv), ("reference_iv", reference_iv)):
        if value.dtype not in (torch.float32, torch.float64):
            raise TypeError(f"{name} must have float32 or float64 dtype")
    if valid.dtype != torch.bool:
        raise TypeError("valid must have Boolean dtype")
    if prediction_iv.shape != reference_iv.shape or valid.shape != reference_iv.shape:
        raise ValueError("prediction_iv, reference_iv, and valid must have the same shape")
    if prediction_iv.device != reference_iv.device or valid.device != reference_iv.device:
        raise ValueError("prediction_iv, reference_iv, and valid must share device")

    attempted = reference_iv.numel()
    valid_count = int(valid.sum().item())
    reference = reference_iv[valid].to(device="cpu", dtype=torch.float64)
    prediction = prediction_iv[valid].to(device="cpu", dtype=torch.float64)
    if not torch.isfinite(reference).all() or (reference <= 0).any():
        raise ValueError("reference IV must be finite and positive at valid quotes")
    prediction_failure_count = int((~torch.isfinite(prediction) | (prediction <= 0)).sum().item())
    result: dict[str, int | float | None] = {
        "attempted": attempted,
        "valid_count": valid_count,
        "reference_failure_count": attempted - valid_count,
        "reference_failure_rate": (attempted - valid_count) / attempted if attempted else None,
        "prediction_failure_count": prediction_failure_count,
        "prediction_failure_rate": prediction_failure_count / valid_count if valid_count else None,
        "mean_absolute_error": None,
        "root_mean_squared_error": None,
        "p99_absolute_error": None,
        "max_absolute_error": None,
    }
    if valid_count == 0 or prediction_failure_count:
        return result

    absolute_error = (100.0 * (prediction - reference)).abs()
    metrics = {
        "mean_absolute_error": absolute_error.mean(),
        "root_mean_squared_error": absolute_error.square().mean().sqrt(),
        "p99_absolute_error": torch.quantile(absolute_error, 0.99, interpolation="linear"),
        "max_absolute_error": absolute_error.max(),
    }
    for name, value in metrics.items():
        result[name] = float(value.item()) if torch.isfinite(value) else None
    return result
