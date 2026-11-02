"""Deterministic slices of the shared in-distribution parameter stream."""

import math

import torch


def sample_id_parameters(
    count: int,
    offset: int,
    seed: int,
    bounds: tuple[tuple[float, float], ...],
) -> torch.Tensor:
    """Return CPU float64 parameters in column order (v0, kappa, theta, sigma, rho).

    Use one scrambled Sobol stream, advancing directly to ``offset`` without
    drawing its prefix. Split offsets and bounds belong to the caller's data
    configuration; no Feller filtering or rejection/resampling is performed.
    Reproducibility requires the same PyTorch version and default floating dtype
    because PyTorch constructs the first Sobol point using that default dtype.
    """
    for name, value in (("count", count), ("offset", offset), ("seed", seed)):
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{name} must be an integer")
    if count < 0 or offset < 0:
        raise ValueError("count and offset must be nonnegative")
    if not 0 <= seed < 2**32:
        raise ValueError("seed must be in [0, 2**32)")
    if offset + count > 2**torch.quasirandom.SobolEngine.MAXBIT:
        raise ValueError("requested slice exceeds the Sobol stream capacity")
    if len(bounds) != 5 or any(len(pair) != 2 for pair in bounds):
        raise ValueError("bounds must contain five (lower, upper) pairs")
    if any(
        not math.isfinite(lower)
        or not math.isfinite(upper)
        or not math.isfinite(upper - lower)
        or lower >= upper
        for lower, upper in bounds
    ):
        raise ValueError("each bound must have finite, strictly increasing endpoints and width")
    if count == 0:
        return torch.empty((0, 5), dtype=torch.float64)
    engine = torch.quasirandom.SobolEngine(  # type: ignore[no-untyped-call]
        dimension=5, scramble=True, seed=seed
    )
    if offset:
        engine.fast_forward(offset)  # type: ignore[no-untyped-call]
    unit = engine.draw(count, dtype=torch.float64)
    limits = torch.tensor(bounds, dtype=torch.float64)
    return limits[:, 0] + unit * (limits[:, 1] - limits[:, 0])
