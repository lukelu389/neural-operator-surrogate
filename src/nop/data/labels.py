"""Convergence-checked synthetic IV labels, with all failed quotes retained."""

import math
from dataclasses import dataclass
from enum import IntEnum

import torch

from nop.pricing.heston_grid import heston_cos_grid_price
from nop.pricing.iv import IVSettings, IVStatus, implied_volatility


class LabelStatus(IntEnum):
    """Additional label failures; codes 0–10 retain the IVStatus meaning."""

    COS_FAILURE = 20
    COS_NOT_CONVERGED = 21


@dataclass(frozen=True)
class LabelSettings:
    """Numerical effort and paired-resolution checks, not certified error bounds."""

    term_levels: tuple[int, ...]
    width: float
    reference_width: float
    price_rtol: float
    iv_error_tolerance: float
    iv: IVSettings

    def __post_init__(self) -> None:
        if not self.term_levels or any(
            isinstance(n, bool) or not isinstance(n, int) or n < 2 for n in self.term_levels
        ):
            raise ValueError("term_levels must contain positive integer counts >= 2")
        if any(b != 2 * a for a, b in zip(self.term_levels, self.term_levels[1:], strict=False)):
            raise ValueError("term levels must double at each refinement")
        if (
            not all(
                math.isfinite(v) and v > 0
                for v in (
                    self.width,
                    self.reference_width,
                    self.price_rtol,
                    self.iv_error_tolerance,
                )
            )
            or self.reference_width <= self.width
        ):
            raise ValueError("positive finite tolerances and increasing widths required")


@dataclass(frozen=True)
class SurfaceLabels:
    """Prices/IV/vega/discrepancy/status [N,T,K], finest term count [N]."""

    price: torch.Tensor
    iv: torch.Tensor
    vega: torch.Tensor
    discrepancy: torch.Tensor
    status: torch.Tensor
    terms: torch.Tensor

    @property
    def valid(self) -> torch.Tensor:
        return self.status == IVStatus.SUCCESS


@torch.no_grad()
def label_surfaces(
    parameters: torch.Tensor,
    strikes: torch.Tensor,
    maturities: torch.Tensor,
    spot: float,
    rate: float,
    dividend_yield: float,
    *,
    settings: LabelSettings,
) -> SurfaceLabels:
    """Price puts at three resolutions and invert the finest-price candidate.

    At each N compare (N,L), (2N,L), and (2N,L_ref). Accept a surface once all
    quotes pass numerical comparison, including the stricter price tolerance
    for masked IVs. Otherwise double N, up to the explicit cap. Each successful
    IV must pass both normalized-price and price-discrepancy/vega checks. Failed
    quotes remain NaN IVs, with distinct source-pricing/convergence/IV reasons.
    These comparisons are diagnostics, not rigorous error bounds.
    """
    if parameters.dtype != torch.float64 or parameters.device.type != "cpu":
        raise ValueError("reference labels require CPU float64 parameters")
    shape = (len(parameters), len(maturities), strikes.shape[-1])
    price, iv, vega, discrepancy = (
        torch.full(shape, math.nan, dtype=torch.float64) for _ in range(4)
    )
    status = torch.full(shape, LabelStatus.COS_FAILURE, dtype=torch.int64)
    terms = torch.zeros(len(parameters), dtype=torch.int64)
    active = torch.arange(len(parameters))
    t = maturities[:, None]
    scale = torch.maximum(spot * torch.exp(-dividend_yield * t), strikes * torch.exp(-rate * t))
    for n_terms in settings.term_levels:
        if active.numel() == 0:
            break

        def evaluate(n: int, width: float, indices: torch.Tensor = active) -> torch.Tensor:
            return heston_cos_grid_price(
                parameters[indices],
                strikes,
                maturities,
                spot,
                rate,
                dividend_yield,
                n_terms=n,
                truncation_width=width,
                option_type="put",
            )

        coarse = evaluate(n_terms, settings.width)
        refined = evaluate(2 * n_terms, settings.width)
        reference = evaluate(2 * n_terms, settings.reference_width)
        difference = torch.maximum((refined - coarse).abs(), (reference - refined).abs())
        result = implied_volatility(
            reference,
            spot,
            strikes,
            t,
            rate,
            dividend_yield,
            settings=settings.iv,
            option_type="put",
        )
        comparable = torch.isfinite(difference)
        accurate = comparable & (difference <= settings.price_rtol * scale)
        accurate &= torch.where(
            result.valid,
            difference <= settings.iv_error_tolerance * result.vega,
            difference <= settings.iv.price_atol + settings.iv.price_rtol * scale,
        )
        final_status = result.status.clone()
        final_status[~accurate] = LabelStatus.COS_NOT_CONVERGED
        final_status[~torch.isfinite(reference)] = LabelStatus.COS_FAILURE
        accepted = final_status == IVStatus.SUCCESS
        price[active] = reference
        iv[active] = torch.where(accepted, result.volatility, math.nan)
        vega[active] = result.vega
        discrepancy[active] = difference
        status[active] = final_status
        terms[active] = 2 * n_terms
        active = active[~accurate.flatten(1).all(dim=1)]
    return SurfaceLabels(price, iv, vega, discrepancy, status, terms)
