"""M0: a parameter-to-native-grid MLP with positive decimal-IV outputs."""

import torch
from torch import nn
from torch.nn import functional as F


class MLP(nn.Module):
    """Map standardized Heston parameters ``[N,5]`` to standardized IV ``[N,T,Z]``.

    The caller supplies training-only target moments. A softplus head maps raw
    scores to decimal IV before standardization; it does not clip outputs or
    change reference masks. Floating-point underflow can still produce zero IV,
    which the shared training engine treats as an explicit failed prediction.
    Initial outputs equal the supplied training mean at every native grid point.
    """

    target_mean: torch.Tensor
    target_std: torch.Tensor

    def __init__(
        self,
        hidden_widths: tuple[int, ...],
        target_mean: torch.Tensor,
        target_std: torch.Tensor,
    ) -> None:
        super().__init__()
        if not hidden_widths or any(
            isinstance(width, bool) or not isinstance(width, int) or width < 1
            for width in hidden_widths
        ):
            raise ValueError("hidden_widths must contain positive integers")
        if target_mean.ndim != 2 or 0 in target_mean.shape:
            raise ValueError("target moments must have nonempty shape [T,Z]")
        if target_std.shape != target_mean.shape:
            raise ValueError("target mean and standard deviation must have the same shape")
        if target_mean.dtype not in (torch.float32, torch.float64):
            raise TypeError("target moments must use float32 or float64")
        if target_std.dtype != target_mean.dtype or target_std.device != target_mean.device:
            raise ValueError("target moments must share dtype and device")
        if target_mean.requires_grad or target_std.requires_grad:
            raise ValueError("target moments must not require gradients")
        if not all(
            torch.isfinite(value).all() and (value > 0).all() for value in (target_mean, target_std)
        ):
            raise ValueError("target moments must be finite and positive")
        self.register_buffer("target_mean", target_mean.detach().clone())
        self.register_buffer("target_std", target_std.detach().clone())

        layers: list[nn.Module] = []
        input_width = 5
        for width in hidden_widths:
            layers.extend(
                [
                    nn.Linear(
                        input_width, width, dtype=target_mean.dtype, device=target_mean.device
                    ),
                    nn.SiLU(),
                ]
            )
            input_width = width
        self.hidden = nn.Sequential(*layers)
        self.output = nn.Linear(
            input_width,
            target_mean.numel(),
            dtype=target_mean.dtype,
            device=target_mean.device,
        )
        with torch.no_grad():
            self.output.weight.zero_()
            # Stable inverse softplus: x + log(1 - exp(-x)), including small x.
            self.output.bias.copy_((target_mean + torch.log(-torch.expm1(-target_mean))).flatten())

    def forward(self, parameters: torch.Tensor) -> torch.Tensor:
        """Return standardized native-grid IV without changing quote validity."""
        if parameters.ndim != 2 or parameters.shape[1] != 5:
            raise ValueError("parameters must have shape [N,5]")
        if (
            parameters.dtype != self.target_mean.dtype
            or parameters.device != self.target_mean.device
        ):
            raise ValueError("parameters must share the model's dtype and device")
        if not torch.isfinite(parameters).all():
            raise ValueError("parameters must be finite")
        scores: torch.Tensor = self.output(self.hidden(parameters))
        iv = F.softplus(scores).reshape(parameters.shape[0], *self.target_mean.shape)
        return (iv - self.target_mean) / self.target_std
