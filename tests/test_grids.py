import math

import pytest
import torch

from nop.data.grids import build_grid


def test_forward_moneyness_and_maturity_major_order() -> None:
    grid = build_grid((0.25, 1.0), (-1.0, 0.0, 1.0), 0.2, 100.0, 0.03, 0.01)
    expected_k = torch.tensor([[-0.1, 0.0, 0.1], [-0.2, 0.0, 0.2]], dtype=torch.float64)
    expected_strikes = torch.tensor(
        [
            [100 * math.exp(-0.095), 100 * math.exp(0.005), 100 * math.exp(0.105)],
            [100 * math.exp(-0.18), 100 * math.exp(0.02), 100 * math.exp(0.22)],
        ],
        dtype=torch.float64,
    )
    torch.testing.assert_close(grid.k, expected_k, rtol=0, atol=1e-16)
    torch.testing.assert_close(grid.strike, expected_strikes, rtol=1e-15, atol=0)
    assert grid.maturity.flatten().tolist() == [0.25] * 3 + [1.0] * 3
    assert grid.z.flatten().tolist() == [-1.0, 0.0, 1.0] * 2
    for value in (grid.maturity, grid.z, grid.k, grid.strike):
        assert value.shape == (2, 3)
        assert value.dtype == torch.float64
        assert value.device.type == "cpu"


def test_main_grid_shape_spacing_and_negative_carry() -> None:
    maturities = (1 / 12, 2 / 12, 3 / 12, 6 / 12, 9 / 12, 1.0, 1.5, 2.0)
    z = tuple(torch.linspace(-2.5, 2.5, 15, dtype=torch.float64).tolist())
    grid = build_grid(maturities, z, 0.2, 1.0, -0.02, 0.03)
    forward = torch.exp(-0.05 * grid.maturity)
    assert grid.strike.shape == (8, 15)
    assert (grid.strike.diff(dim=1) > 0).all()
    torch.testing.assert_close(torch.log(grid.strike / forward), grid.k, rtol=1e-14, atol=2e-16)
    torch.testing.assert_close(grid.strike[:, 7], forward[:, 7])


@pytest.mark.parametrize(
    "maturities,z",
    [
        ((), (0.0,)),
        ((1.0,), ()),
        ((0.0,), (0.0,)),
        ((-0.1,), (0.0,)),
        ((1.0, 1.0), (0.0,)),
        ((1.0, 0.5), (0.0,)),
        ((math.inf,), (0.0,)),
        ((1.0,), (math.nan,)),
        ((1.0,), (1.0, -1.0)),
    ],
)
def test_invalid_coordinates(maturities: tuple[float, ...], z: tuple[float, ...]) -> None:
    with pytest.raises(ValueError):
        build_grid(maturities, z, 0.2, 1.0, 0.02, 0.0)


@pytest.mark.parametrize(
    "scale,spot,rate,dividend",
    [(0.0, 1.0, 0.0, 0.0), (0.2, -1.0, 0.0, 0.0), (0.2, 1.0, math.nan, 0.0)],
)
def test_invalid_scalars(scale: float, spot: float, rate: float, dividend: float) -> None:
    with pytest.raises(ValueError):
        build_grid((1.0,), (0.0,), scale, spot, rate, dividend)


@pytest.mark.parametrize("rate", [-1000.0, 1000.0])
def test_unrepresentable_strikes_raise(rate: float) -> None:
    with pytest.raises(FloatingPointError):
        build_grid((1.0,), (0.0,), 0.2, 1.0, rate, 0.0)
