"""Grid adapter equivalence and isolation of invalid numerical quotes."""

import math
from typing import Literal

import pytest
import torch

from nop.pricing.bs import black_scholes_price
from nop.pricing.heston_cos import heston_cos_price
from nop.pricing.heston_grid import heston_cos_grid_price

OptionType = Literal["call", "put"]


@pytest.fixture
def grid() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    parameters = torch.tensor(
        [[0.04, 1.5, 0.04, 0.3, -0.7], [0.09, 0.7, 0.02, 0.8, -0.95], [0.01, 3.0, 0.15, 0.1, -0.2]],
        dtype=torch.float64,
    )
    maturities = torch.tensor([0.25, 1.2], dtype=torch.float64)
    strikes = torch.tensor([[90.0, 100.0, 110.0], [85.0, 105.0, 120.0]], dtype=torch.float64)
    return parameters, strikes, maturities


@pytest.mark.parametrize("option_type", ["call", "put"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_matches_validated_quote_pricer_with_batched_parameters(
    grid: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    option_type: OptionType,
    dtype: torch.dtype,
) -> None:
    parameters, strikes, maturities = (value.to(dtype=dtype) for value in grid)
    actual = heston_cos_grid_price(
        parameters,
        strikes,
        maturities,
        100.0,
        -0.02,
        0.03,
        n_terms=1024,
        truncation_width=20.0,
        option_type=option_type,
    )
    expected = heston_cos_price(
        100.0,
        strikes[None, :, :],
        maturities[None, :, None],
        *(parameters[:, i, None, None] for i in range(5)),
        -0.02,
        0.03,
        n_terms=1024,
        truncation_width=20.0,
        option_type=option_type,
    )
    assert actual.shape == (3, 2, 3)
    assert actual.dtype == dtype
    assert actual.device == parameters.device
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("option_type", ["call", "put"])
def test_expiry_and_deterministic_variance_limits(option_type: OptionType) -> None:
    parameters = torch.tensor(
        [[0.09, 1.3, 0.04, 0.0, -0.7], [0.0, 1.5, 0.0, 0.8, 0.2]], dtype=torch.float64
    )
    maturities = torch.tensor([0.0, 0.5, 2.0], dtype=torch.float64)
    strikes = torch.tensor([[90.0, 100.0, 110.0]] * 3, dtype=torch.float64)
    actual = heston_cos_grid_price(
        parameters,
        strikes,
        maturities,
        100.0,
        -0.02,
        0.03,
        n_terms=512,
        truncation_width=12.0,
        option_type=option_type,
    )
    variance = 0.04 + (0.09 - 0.04) * -torch.expm1(-1.3 * maturities[1:]) / (1.3 * maturities[1:])
    regular = black_scholes_price(
        100.0,
        strikes[1:],
        maturities[1:, None],
        variance.sqrt()[:, None],
        -0.02,
        0.03,
        option_type=option_type,
    )
    torch.testing.assert_close(actual[0, 1:], regular, rtol=1e-11, atol=1e-11)
    deterministic = black_scholes_price(
        100.0, strikes, maturities[:, None], 0.0, -0.02, 0.03, option_type=option_type
    )
    torch.testing.assert_close(actual[1], deterministic, rtol=0, atol=0)
    torch.testing.assert_close(actual[0, 0], deterministic[0], rtol=0, atol=0)


def test_negative_price_is_nan_without_invalidating_neighboring_strike() -> None:
    parameters = torch.tensor([[0.04, 1.5, 0.04, 0.3, -0.7]], dtype=torch.float64)
    strikes = torch.tensor([[50.0, 100.0]], dtype=torch.float64)
    maturities = torch.tensor([0.25], dtype=torch.float64)
    result = heston_cos_grid_price(
        parameters,
        strikes,
        maturities,
        100.0,
        0.02,
        0.0,
        n_terms=4,
        truncation_width=12.0,
    )
    with pytest.raises(FloatingPointError):
        heston_cos_price(
            100.0,
            50.0,
            0.25,
            *parameters[0],
            0.02,
            n_terms=4,
            truncation_width=12.0,
            option_type="put",
        )
    assert torch.isnan(result[0, 0, 0])
    expected = heston_cos_price(
        100.0,
        100.0,
        0.25,
        *parameters[0],
        0.02,
        n_terms=4,
        truncation_width=12.0,
        option_type="put",
    )
    torch.testing.assert_close(result[0, 0, 1], expected, rtol=0, atol=0)


def test_numerical_overflow_does_not_invalidate_neighboring_surface(
    grid: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> None:
    parameters, strikes, maturities = grid
    original = heston_cos_grid_price(
        parameters[1:],
        strikes,
        maturities,
        100.0,
        0.02,
        0.0,
        n_terms=512,
        truncation_width=20.0,
    )
    parameters[0, 3] = 1e200
    actual = heston_cos_grid_price(
        parameters,
        strikes,
        maturities,
        100.0,
        0.02,
        0.0,
        n_terms=512,
        truncation_width=20.0,
    )
    assert torch.all(torch.isnan(actual[0]))
    torch.testing.assert_close(actual[1:], original, rtol=0, atol=0)


@pytest.mark.parametrize(
    ("index", "value"),
    [(0, -0.1), (1, 0.0), (2, -0.1), (3, -0.1), (4, -1.1), (4, 1.1), (0, math.nan)],
)
def test_invalid_model_parameters_raise(
    grid: tuple[torch.Tensor, torch.Tensor, torch.Tensor], index: int, value: float
) -> None:
    parameters, strikes, maturities = grid
    parameters[0, index] = value
    with pytest.raises(ValueError):
        heston_cos_grid_price(
            parameters,
            strikes,
            maturities,
            100.0,
            0.02,
            0.0,
            n_terms=128,
            truncation_width=12.0,
        )


@pytest.mark.parametrize(
    "bad_input",
    ["parameter_shape", "strike_shape", "maturity_shape", "maturity_count", "empty_strikes"],
)
def test_invalid_shapes_raise(
    grid: tuple[torch.Tensor, torch.Tensor, torch.Tensor], bad_input: str
) -> None:
    parameters, strikes, maturities = grid
    if bad_input == "parameter_shape":
        parameters = parameters[:, :4]
    elif bad_input == "strike_shape":
        strikes = strikes[0]
    elif bad_input == "maturity_shape":
        maturities = maturities[:, None]
    elif bad_input == "maturity_count":
        maturities = maturities[:1]
    else:
        strikes = strikes[:, :0]
    with pytest.raises(ValueError):
        heston_cos_grid_price(
            parameters,
            strikes,
            maturities,
            100.0,
            0.02,
            0.0,
            n_terms=128,
            truncation_width=12.0,
        )


def test_empty_parameter_batch_returns_empty_surface_batch(
    grid: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> None:
    parameters, strikes, maturities = grid
    actual = heston_cos_grid_price(
        parameters[:0],
        strikes,
        maturities,
        100.0,
        0.02,
        0.0,
        n_terms=128,
        truncation_width=12.0,
    )
    assert actual.shape == (0, 2, 3)


@pytest.mark.parametrize("dtype", [torch.int64, torch.float16, torch.complex64])
def test_unsupported_tensor_dtypes_raise(
    grid: tuple[torch.Tensor, torch.Tensor, torch.Tensor], dtype: torch.dtype
) -> None:
    with pytest.raises(TypeError):
        heston_cos_grid_price(
            *(value.to(dtype=dtype) for value in grid),
            100.0,
            0.02,
            0.0,
            n_terms=128,
            truncation_width=12.0,
        )


def test_mixed_tensor_dtypes_raise(grid: tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> None:
    parameters, strikes, maturities = grid
    with pytest.raises(ValueError):
        heston_cos_grid_price(
            parameters.float(),
            strikes,
            maturities,
            100.0,
            0.02,
            0.0,
            n_terms=128,
            truncation_width=12.0,
        )
