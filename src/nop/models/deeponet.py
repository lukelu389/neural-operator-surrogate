"""M1: a DeepONet-style parameter branch and coordinate trunk."""

import math

import torch
from torch import nn
from torch.nn import functional as F

from nop.train.coordinates import CoordinateNormalization


class DeepONet(nn.Module):
    """Map standardized parameters and (k,T) queries to standardized positive IV.

    The branch embeds five Heston parameters; the trunk embeds each forward
    log-moneyness/maturity pair. Their scaled dot product is passed through
    softplus in decimal-IV space and standardized with training-only coordinate
    moments. `forward` evaluates the configured native grid for the shared
    training loop; `query` accepts arbitrary broadcastable coordinate tensors.
    """

    native_k: torch.Tensor
    native_maturity: torch.Tensor

    def __init__(
        self,
        hidden_widths: tuple[int, ...],
        rank: int,
        coordinate_scales: tuple[float, float],
        normalization: CoordinateNormalization,
        native_k: torch.Tensor,
        native_maturity: torch.Tensor,
    ) -> None:
        super().__init__()
        if not hidden_widths or any(
            isinstance(width, bool) or not isinstance(width, int) or width < 1
            for width in hidden_widths
        ):
            raise ValueError("hidden_widths must contain positive integers")
        if isinstance(rank, bool) or not isinstance(rank, int) or rank < 1:
            raise ValueError("rank must be a positive integer")
        if len(coordinate_scales) != 2 or any(
            not math.isfinite(value) or value <= 0 for value in coordinate_scales
        ):
            raise ValueError("coordinate scales must be finite and positive")
        if native_k.ndim != 2 or native_k.shape != native_maturity.shape:
            raise ValueError("native coordinates must have matching [T,Z] shapes")
        if native_k.dtype not in (torch.float32, torch.float64) or (
            native_maturity.dtype != native_k.dtype or native_maturity.device != native_k.device
        ):
            raise ValueError("native coordinates must share a float32/float64 dtype and device")
        if not torch.isfinite(native_k).all() or not torch.isfinite(native_maturity).all():
            raise ValueError("native coordinates must be finite")
        if (native_maturity <= 0).any():
            raise ValueError("native maturities must be positive")
        if normalization.mean.dtype != native_k.dtype or (
            normalization.mean.device != native_k.device
        ):
            raise ValueError("normalization and native coordinates must share dtype and device")
        if normalization.mean.shape != native_k.shape:
            raise ValueError("normalization moments must match the native grid")

        self.coordinate_normalization = normalization
        self.k_scale, self.maturity_scale = coordinate_scales
        self.register_buffer("normalization_z_axis", normalization.z_axis.detach().clone())
        self.register_buffer("normalization_T_axis", normalization.T_axis.detach().clone())
        self.register_buffer("target_mean", normalization.mean.detach().clone())
        self.register_buffer("target_std", normalization.std.detach().clone())
        self.register_buffer("native_k", native_k.detach().clone())
        self.register_buffer("native_maturity", native_maturity.detach().clone())
        self.hidden_widths = hidden_widths
        self.rank = rank
        dtype, device = native_k.dtype, native_k.device

        def tower(input_width: int) -> nn.Sequential:
            layers: list[nn.Module] = []
            current_width = input_width
            for width in hidden_widths:
                layers.extend(
                    [
                        nn.Linear(current_width, width, dtype=dtype, device=device),
                        nn.SiLU(),
                    ]
                )
                current_width = width
            layers.append(nn.Linear(current_width, rank, dtype=dtype, device=device))
            return nn.Sequential(*layers)

        self.branch = tower(5)
        self.trunk = tower(2)
        self.output_bias = nn.Parameter(torch.zeros((), dtype=dtype, device=device))
        self.dot_product_scale = rank**-0.5

    def query(self, parameters: torch.Tensor, k: torch.Tensor, T: torch.Tensor) -> torch.Tensor:
        """Return standardized IV for broadcastable arbitrary (k,T) queries."""
        if parameters.ndim != 2 or parameters.shape[1] != 5:
            raise ValueError("parameters must have shape [N,5]")
        if parameters.dtype not in (torch.float32, torch.float64):
            raise TypeError("parameters must have float32 or float64 dtype")
        if parameters.dtype != self.native_k.dtype or parameters.device != self.native_k.device:
            raise ValueError("parameters must share the model's dtype and device")
        if not torch.isfinite(parameters).all():
            raise ValueError("parameters must be finite")
        if k.dtype not in (torch.float32, torch.float64) or T.dtype not in (
            torch.float32,
            torch.float64,
        ):
            raise TypeError("query coordinates must have float32 or float64 dtype")
        if k.dtype != T.dtype or k.device != T.device:
            raise ValueError("query coordinates must share dtype and device")
        if k.dtype != parameters.dtype or k.device != parameters.device:
            raise ValueError("query coordinates and parameters must share dtype and device")
        k, T = torch.broadcast_tensors(k, T)  # type: ignore[no-untyped-call]
        if not torch.isfinite(k).all() or not torch.isfinite(T).all() or (T <= 0).any():
            raise ValueError("query coordinates must be finite with positive maturity")

        output_shape = k.shape
        k_flat, T_flat = k.reshape(-1), T.reshape(-1)
        coordinates = torch.stack((k_flat / self.k_scale, T_flat / self.maturity_scale), dim=-1)
        branch = self.branch(parameters)
        trunk = self.trunk(coordinates)
        score = torch.einsum("nr,qr->nq", branch, trunk) * self.dot_product_scale
        decimal_iv = F.softplus(score + self.output_bias)
        mean, std = self.coordinate_normalization.at(k_flat, T_flat)
        standardized = (decimal_iv - mean.unsqueeze(0)) / std.unsqueeze(0)
        return standardized.reshape(parameters.shape[0], *output_shape)

    def forward(self, parameters: torch.Tensor) -> torch.Tensor:
        """Evaluate standardized IV on the native [T,Z] training grid."""
        return self.query(parameters, self.native_k, self.native_maturity)
