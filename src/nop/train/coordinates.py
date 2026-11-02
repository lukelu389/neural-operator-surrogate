"""Interpolate fixed training moments for coordinate-query IV models."""

import math
from dataclasses import dataclass

import torch


def _check_float(value: torch.Tensor, name: str) -> None:
    if value.dtype not in (torch.float32, torch.float64):
        raise TypeError(f"{name} must have float32 or float64 dtype")


def _bracket(
    axis: torch.Tensor, query: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return neighboring indices and weights, with constant endpoint extension."""
    if axis.numel() == 1:
        index = torch.zeros_like(query, dtype=torch.long)
        return index, index, query * 0.0
    bounded = query.clamp(min=axis[0], max=axis[-1])
    upper = torch.searchsorted(axis, bounded.contiguous(), right=True).clamp(1, axis.numel() - 1)
    lower = upper - 1
    weight = (bounded - axis[lower]) / (axis[upper] - axis[lower])
    return lower, upper, weight


@dataclass(frozen=True)
class CoordinateNormalization:
    """Training-only IV moments on a rectangular native ``(T, z)`` grid.

    Supply training means/stds of shape ``[len(T_axis), len(z_axis)]`` and the
    configured scale in ``z = k / (k_scale * sqrt(T))``. For withheld strikes,
    supply only the training-coordinate axes and moments (the even-z subset in
    protocol v1); held-out labels must never be used to construct these moments.
    This class accepts no labels and never fits statistics. Inputs are copied
    and detached so these moments remain fixed during model differentiation.

    Means and log standard deviations are bilinearly interpolated. Coordinates
    outside either supplied axis use that axis's endpoint moments; a singleton
    axis is constant. This clamps normalization coordinates only, and neither
    clips predictions nor determines the model's valid prediction domain.

    Query tensors broadcast and must share a float32/float64 dtype and device.
    Stored moments adopt the query dtype/device. Coordinate gradients exist
    inside interpolation cells; derivatives at knots/clamp boundaries are not
    part of the contract. IV NaNs and prediction gradients are preserved.
    """

    z_axis: torch.Tensor
    T_axis: torch.Tensor
    mean: torch.Tensor
    std: torch.Tensor
    k_scale: float

    def __post_init__(self) -> None:
        for name in ("z_axis", "T_axis", "mean", "std"):
            value = getattr(self, name)
            _check_float(value, name)
            if not torch.isfinite(value).all():
                raise ValueError(f"{name} must be finite")
        if any(
            value.dtype != self.mean.dtype or value.device != self.mean.device
            for value in (self.z_axis, self.T_axis, self.std)
        ):
            raise ValueError("axes and moments must share dtype and device")
        for name, axis in (("z_axis", self.z_axis), ("T_axis", self.T_axis)):
            if axis.ndim != 1 or axis.numel() == 0:
                raise ValueError(f"{name} must be a nonempty one-dimensional axis")
            if (axis[1:] <= axis[:-1]).any():
                raise ValueError(f"{name} must be strictly increasing")
        if (self.T_axis <= 0).any():
            raise ValueError("T_axis must be positive")
        if self.mean.shape != (self.T_axis.numel(), self.z_axis.numel()):
            raise ValueError("mean must have shape [len(T_axis), len(z_axis)]")
        if self.std.shape != self.mean.shape or (self.std <= 0).any():
            raise ValueError("std must have the same shape as mean and be positive")
        if not math.isfinite(self.k_scale) or self.k_scale <= 0:
            raise ValueError("k_scale must be finite and positive")
        for name in ("z_axis", "T_axis", "mean", "std"):
            object.__setattr__(self, name, getattr(self, name).detach().clone())

    def at(self, k: torch.Tensor, T: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return interpolated mean/std at finite log-moneyness and positive T."""
        _check_float(k, "k")
        _check_float(T, "T")
        if k.dtype != T.dtype or k.device != T.device:
            raise ValueError("query tensors must share dtype and device")
        if not torch.isfinite(k).all() or not torch.isfinite(T).all() or (T <= 0).any():
            raise ValueError("queries must be finite with positive T")
        k, T = torch.broadcast_tensors(k, T)  # type: ignore[no-untyped-call]
        z = k / (self.k_scale * torch.sqrt(T))
        if not torch.isfinite(z).all():
            raise FloatingPointError("query z coordinates must be finite")
        z0, z1, wz = _bracket(self.z_axis.to(k), z)
        t0, t1, wt = _bracket(self.T_axis.to(k), T)

        def interpolate(values: torch.Tensor) -> torch.Tensor:
            lower = torch.lerp(values[t0, z0], values[t0, z1], wz)
            upper = torch.lerp(values[t1, z0], values[t1, z1], wz)
            return torch.lerp(lower, upper, wt)

        mean = interpolate(self.mean.to(k))
        std = interpolate(self.std.log().to(k)).exp()
        if not torch.isfinite(mean).all() or not torch.isfinite(std).all() or (std <= 0).any():
            raise FloatingPointError("interpolated moments must be finite with positive std")
        return mean, std

    def _check_values(self, values: torch.Tensor, k: torch.Tensor) -> None:
        _check_float(values, "IV values")
        if values.dtype != k.dtype or values.device != k.device:
            raise ValueError("IV values and queries must share dtype and device")
        if torch.isinf(values).any():
            raise ValueError("IV values may contain NaN masks but not infinities")

    def transform(self, iv: torch.Tensor, k: torch.Tensor, T: torch.Tensor) -> torch.Tensor:
        """Standardize decimal IV at each query, retaining all NaN masks."""
        self._check_values(iv, k)
        mean, std = self.at(k, T)
        result = (iv - mean) / std
        if not torch.equal(torch.isfinite(result), torch.isfinite(iv).expand_as(result)):
            raise FloatingPointError("coordinate normalization produced nonfinite values")
        return result

    def inverse(self, values: torch.Tensor, k: torch.Tensor, T: torch.Tensor) -> torch.Tensor:
        """Return decimal IV without positivity clipping; retain NaNs and gradients."""
        self._check_values(values, k)
        mean, std = self.at(k, T)
        result = values * std + mean
        if not torch.equal(torch.isfinite(result), torch.isfinite(values).expand_as(result)):
            raise FloatingPointError("inverse coordinate normalization produced nonfinite values")
        return result
