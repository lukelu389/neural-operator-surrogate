"""Training-only population moments for parameters and masked IV surfaces."""

from dataclasses import dataclass

import torch


def _check_float_tensor(value: torch.Tensor, name: str) -> None:
    if value.dtype not in (torch.float32, torch.float64):
        raise TypeError(f"{name} must have float32 or float64 dtype")


@dataclass(frozen=True)
class TrainNormalization:
    """Training moments; IV statistics/counts have the surface's two grid axes.

    Transformations reuse these fixed statistics, casting them to the input's
    float32/float64 dtype and device. They support leading batch axes, preserve
    IV NaNs and autodiff, and never refit statistics or clip predictions.
    """

    parameter_mean: torch.Tensor
    parameter_std: torch.Tensor
    iv_mean: torch.Tensor
    iv_std: torch.Tensor
    iv_count: torch.Tensor

    def transform_parameters(self, parameters: torch.Tensor) -> torch.Tensor:
        """Standardize finite parameter vectors with final dimension five."""
        _check_float_tensor(parameters, "parameters")
        if parameters.ndim < 1 or parameters.shape[-1] != 5:
            raise ValueError("parameters must have final dimension five")
        if not torch.isfinite(parameters).all():
            raise ValueError("parameters must be finite")
        result = (parameters - self.parameter_mean.to(parameters)) / self.parameter_std.to(
            parameters
        )
        if not torch.isfinite(result).all():
            raise FloatingPointError("parameter normalization produced nonfinite values")
        return result

    def _check_iv(self, iv: torch.Tensor) -> None:
        _check_float_tensor(iv, "iv")
        if iv.ndim < 2 or iv.shape[-2:] != self.iv_mean.shape:
            raise ValueError("IV tensors must end in the fitted grid shape")
        if torch.isinf(iv).any():
            raise ValueError("IV tensors may contain NaN masks but not infinities")

    def transform_iv(self, iv: torch.Tensor) -> torch.Tensor:
        """Standardize decimal IVs, retaining NaN quotes at their original positions."""
        self._check_iv(iv)
        result = (iv - self.iv_mean.to(iv)) / self.iv_std.to(iv)
        if not torch.equal(torch.isfinite(result), torch.isfinite(iv)):
            raise FloatingPointError("IV normalization produced nonfinite values")
        return result

    def inverse_iv(self, standardized_iv: torch.Tensor) -> torch.Tensor:
        """Return decimal IV without positivity clipping; preserve NaNs and gradients."""
        self._check_iv(standardized_iv)
        result = standardized_iv * self.iv_std.to(standardized_iv) + self.iv_mean.to(
            standardized_iv
        )
        if not torch.equal(torch.isfinite(result), torch.isfinite(standardized_iv)):
            raise FloatingPointError("inverse IV normalization produced nonfinite values")
        return result


@torch.no_grad()
def fit_train_normalization(
    parameters: torch.Tensor, iv: torch.Tensor, valid: torch.Tensor
) -> TrainNormalization:
    """Fit population statistics using only the training rows supplied by the caller.

    Require parameters [N,5], decimal IV [N,M,Z], and a Boolean validity mask of
    the same shape. Valid IVs must be finite and positive; invalid IVs must be
    NaN. Every grid point needs nonzero valid population variance. No validation
    or holdout array is accepted separately or used to fill missing moments.
    """
    _check_float_tensor(parameters, "parameters")
    _check_float_tensor(iv, "iv")
    if parameters.ndim != 2 or parameters.shape[1] != 5 or parameters.shape[0] == 0:
        raise ValueError("parameters must have nonempty shape [N,5]")
    if iv.ndim != 3 or iv.shape[0] != parameters.shape[0] or min(iv.shape) == 0:
        raise ValueError("iv must have nonempty shape [N,M,Z] with the same N as parameters")
    if valid.dtype != torch.bool:
        raise TypeError("valid must have Boolean dtype")
    if valid.shape != iv.shape:
        raise ValueError("valid must have the same shape as iv")
    if parameters.dtype != iv.dtype or parameters.device != iv.device or valid.device != iv.device:
        raise ValueError("parameters and iv must share dtype/device; valid must share device")
    if not torch.isfinite(parameters).all():
        raise ValueError("parameters must be finite")
    if not torch.isfinite(iv[valid]).all() or (iv[valid] <= 0).any():
        raise ValueError("valid IVs must be finite and positive")
    if not torch.isnan(iv[~valid]).all():
        raise ValueError("invalid IVs must be NaN")
    iv_count = valid.sum(dim=0)
    if (iv_count == 0).any():
        raise ValueError("every grid point must have valid training quotes")
    parameter_mean = parameters.mean(dim=0)
    parameter_std = parameters.std(dim=0, unbiased=False)
    iv_mean = torch.where(valid, iv, 0.0).sum(dim=0) / iv_count
    centered = torch.where(valid, iv - iv_mean, 0.0)
    iv_std = torch.sqrt(centered.square().sum(dim=0) / iv_count)
    if not torch.isfinite(parameter_mean).all() or not torch.isfinite(iv_mean).all():
        raise FloatingPointError("normalization means must be finite")
    if not torch.isfinite(parameter_std).all() or not torch.isfinite(iv_std).all():
        raise FloatingPointError("normalization standard deviations must be finite")
    if (parameter_std <= 0).any() or (iv_std <= 0).any():
        raise ValueError("every parameter and grid point must have nonzero training variance")
    return TrainNormalization(parameter_mean, parameter_std, iv_mean, iv_std, iv_count)
