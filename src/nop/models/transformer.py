"""M2: a bidirectional parameter/query-token Transformer."""

import math

import torch
from torch import nn
from torch.nn import functional as F

from nop.train.coordinates import CoordinateNormalization


class QueryTransformer(nn.Module):
    """Map Heston parameters and continuous query coordinates to standardized IV.

    The sequence contains five parameter-value tokens followed by all query
    tokens. Self-attention is bidirectional over this joint sequence, and only
    query-token outputs are read. Continuous coordinates use normalized raw
    values and fixed Fourier features at frequencies 1, 2, 4, and 8 cycles.
    The query set is permutation-equivariant; callers should submit the same
    coordinate set when comparing its outputs.
    """

    native_k: torch.Tensor
    native_maturity: torch.Tensor

    def __init__(
        self,
        model_width: int,
        feedforward_width: int,
        attention_heads: int,
        layer_count: int,
        coordinate_scales: tuple[float, float],
        fourier_frequencies: tuple[int, ...],
        normalization: CoordinateNormalization,
        native_k: torch.Tensor,
        native_maturity: torch.Tensor,
    ) -> None:
        super().__init__()
        if model_width < 1 or feedforward_width < 1 or layer_count < 1:
            raise ValueError("Transformer widths and layer count must be positive")
        if attention_heads < 1 or model_width % attention_heads:
            raise ValueError("model width must be divisible by positive attention head count")
        if len(coordinate_scales) != 2 or any(
            not math.isfinite(value) or value <= 0 for value in coordinate_scales
        ):
            raise ValueError("coordinate scales must be finite and positive")
        if not fourier_frequencies or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 1
            for value in fourier_frequencies
        ):
            raise ValueError("Fourier frequencies must be positive integers")
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
        self.fourier_frequencies = fourier_frequencies
        self.register_buffer("normalization_z_axis", normalization.z_axis.detach().clone())
        self.register_buffer("normalization_T_axis", normalization.T_axis.detach().clone())
        self.register_buffer("target_mean", normalization.mean.detach().clone())
        self.register_buffer("target_std", normalization.std.detach().clone())
        self.register_buffer("native_k", native_k.detach().clone())
        self.register_buffer("native_maturity", native_maturity.detach().clone())
        dtype, device = native_k.dtype, native_k.device
        self.parameter_projection = nn.Linear(1, model_width, dtype=dtype, device=device)
        self.parameter_identity = nn.Parameter(
            torch.empty(5, model_width, dtype=dtype, device=device)
        )
        nn.init.normal_(self.parameter_identity, std=0.02)
        coordinate_feature_count = 2 + 4 * len(fourier_frequencies)
        self.query_projection = nn.Linear(
            coordinate_feature_count, model_width, dtype=dtype, device=device
        )
        self.query_identity = nn.Parameter(torch.empty(model_width, dtype=dtype, device=device))
        nn.init.normal_(self.query_identity, std=0.02)
        self.layers = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=model_width,
                    nhead=attention_heads,
                    dim_feedforward=feedforward_width,
                    dropout=0.0,
                    activation=F.gelu,
                    batch_first=True,
                    norm_first=False,
                    dtype=dtype,
                    device=device,
                )
                for _ in range(layer_count)
            ]
        )
        self.output_norm = nn.LayerNorm(model_width, dtype=dtype, device=device)
        self.output = nn.Linear(model_width, 1, dtype=dtype, device=device)
        self.output_bias = nn.Parameter(torch.zeros((), dtype=dtype, device=device))

    def _coordinate_features(self, k: torch.Tensor, T: torch.Tensor) -> torch.Tensor:
        normalized = torch.stack((k / self.k_scale, T / self.maturity_scale), dim=-1)
        features = [normalized]
        for frequency in self.fourier_frequencies:
            phase = normalized * (2.0 * math.pi * frequency)
            features.extend((phase.sin(), phase.cos()))
        return torch.cat(features, dim=-1)

    def query(self, parameters: torch.Tensor, k: torch.Tensor, T: torch.Tensor) -> torch.Tensor:
        """Return standardized IV for broadcastable coordinates in one query set."""
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
        if k.ndim > 2:
            raise ValueError("query coordinate sets may have at most two dimensions")
        if not torch.isfinite(k).all() or not torch.isfinite(T).all() or (T <= 0).any():
            raise ValueError("query coordinates must be finite with positive maturity")

        output_shape = k.shape
        k_flat, T_flat = k.reshape(-1), T.reshape(-1)
        parameter_tokens = self.parameter_projection(parameters.unsqueeze(-1))
        parameter_tokens = parameter_tokens + self.parameter_identity.unsqueeze(0)
        query_tokens = self.query_projection(self._coordinate_features(k_flat, T_flat))
        query_tokens = query_tokens + self.query_identity
        query_tokens = query_tokens.unsqueeze(0).expand(parameters.shape[0], -1, -1)
        tokens = torch.cat((parameter_tokens, query_tokens), dim=1)
        for layer in self.layers:
            tokens = layer(tokens)
        query_hidden = self.output_norm(tokens[:, 5:, :])
        decimal_iv = F.softplus(self.output(query_hidden).squeeze(-1) + self.output_bias)
        mean, std = self.coordinate_normalization.at(k_flat, T_flat)
        standardized = (decimal_iv - mean.unsqueeze(0)) / std.unsqueeze(0)
        return standardized.reshape(parameters.shape[0], *output_shape)

    def forward(self, parameters: torch.Tensor) -> torch.Tensor:
        """Evaluate standardized IV on the configured native [T,Z] grid."""
        return self.query(parameters, self.native_k, self.native_maturity)
