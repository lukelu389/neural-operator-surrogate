"""Maturity-major option grids in forward log-moneyness coordinates."""

import math
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class QuoteGrid:
    """CPU float64 tensors of shape [number of maturities, number of z values]."""

    maturity: torch.Tensor
    z: torch.Tensor
    k: torch.Tensor
    strike: torch.Tensor


def build_grid(
    maturities: tuple[float, ...],
    z: tuple[float, ...],
    k_scale: float,
    spot: float,
    rate: float,
    dividend_yield: float,
) -> QuoteGrid:
    """Build k=z*k_scale*sqrt(T), K=S*exp((r-q)*T+k), with T in years.

    Both coordinate axes must be strictly increasing. The configured main grid
    is 8 by 15; accepting other sizes permits small numerical fixtures without
    embedding data configuration or any holdout-generation policy here.
    """
    for name, coordinates in (("maturities", maturities), ("z", z)):
        if not coordinates or any(not math.isfinite(value) for value in coordinates):
            raise ValueError(f"{name} must be nonempty and finite")
        if any(left >= right for left, right in zip(coordinates, coordinates[1:], strict=False)):
            raise ValueError(f"{name} must be strictly increasing")
    if maturities[0] <= 0:
        raise ValueError("maturities must be positive")
    if any(not math.isfinite(value) for value in (k_scale, spot, rate, dividend_yield)):
        raise ValueError("grid scalars must be finite")
    if k_scale <= 0 or spot <= 0:
        raise ValueError("k_scale and spot must be positive")
    maturity = torch.tensor(maturities, dtype=torch.float64)[:, None].expand(-1, len(z))
    scaled_z = torch.tensor(z, dtype=torch.float64)[None, :].expand(len(maturities), -1)
    k = scaled_z * k_scale * torch.sqrt(maturity)
    strike = spot * torch.exp((rate - dividend_yield) * maturity + k)
    if not torch.isfinite(k).all() or not torch.isfinite(strike).all() or (strike <= 0).any():
        raise FloatingPointError("grid arithmetic produced nonfinite or nonpositive outputs")
    return QuoteGrid(maturity=maturity, z=scaled_z, k=k, strike=strike)
