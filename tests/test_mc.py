"""Independent numerical checks for the Heston Monte Carlo estimator."""

import math

import numpy as np
import pytest

from nop.pricing.bs import OptionType, black_scholes_price
from nop.pricing.mc import heston_mc_price


@pytest.mark.parametrize("option_type", ["call", "put"])
def test_constant_variance_black_scholes_limit(option_type: OptionType) -> None:
    result = heston_mc_price(
        100.0,
        105.0,
        0.75,
        0.04,
        2.0,
        0.04,
        0.0,
        -0.7,
        0.03,
        0.01,
        n_pairs=40_000,
        n_steps=8,
        seed=103,
        confidence_z=4.0,
        option_type=option_type,
    )
    exact = black_scholes_price(100.0, 105.0, 0.75, 0.2, 0.03, 0.01, option_type=option_type)
    assert result.confidence_low < exact.item() < result.confidence_high
    assert 0 < result.standard_error < 0.04


def test_discounted_stock_martingale_via_shared_path_parity() -> None:
    args = (100.0, 100.0, 0.5, 0.04, 1.5, 0.05, 0.5, -0.6, 0.02, 0.01)
    call = heston_mc_price(*args, n_pairs=30_000, n_steps=64, seed=307)
    put = heston_mc_price(*args, n_pairs=30_000, n_steps=64, seed=307, option_type="put")
    expected = 100.0 * math.exp(-0.01 * 0.5) - 100.0 * math.exp(-0.02 * 0.5)
    # Shared draws make call-put equal the discounted terminal-stock difference.
    # Sum of marginal SEs bounds its SE, without assuming the estimates independent.
    assert abs(call.estimate - put.estimate - expected) < 4 * (
        call.standard_error + put.standard_error
    )


def test_standard_error_uses_independent_pair_means() -> None:
    n_pairs, seed = 100, 97
    result = heston_mc_price(
        100.0,
        98.0,
        0.5,
        0.04,
        1.5,
        0.03,
        0.6,
        0.0,
        0.02,
        n_pairs=n_pairs,
        n_steps=1,
        seed=seed,
        confidence_z=2.5,
    )
    # A single stock step depends only on v0. Recover its known lognormal law
    # directly, independent of the Heston variance-update implementation.
    draws = np.random.default_rng(seed).standard_normal((2, n_pairs))[1]
    positive_stock = 100 * np.exp((0.02 - 0.04 / 2) * 0.5 + math.sqrt(0.04 * 0.5) * draws)
    negative_stock = 100 * np.exp((0.02 - 0.04 / 2) * 0.5 - math.sqrt(0.04 * 0.5) * draws)
    payoffs = np.exp(-0.02 * 0.5) * np.maximum(
        np.concatenate((positive_stock, negative_stock)) - 98,
        0,
    )
    pair_means = (payoffs[:n_pairs] + payoffs[n_pairs:]) / 2
    expected_se = np.std(pair_means, ddof=1) / math.sqrt(n_pairs)
    assert result.estimate == pytest.approx(np.mean(pair_means), abs=1e-12)
    assert result.standard_error == pytest.approx(expected_se, abs=1e-13)
    assert result.standard_error != pytest.approx(np.std(payoffs, ddof=1) / math.sqrt(2 * n_pairs))
    assert result.confidence_low == pytest.approx(result.estimate - 2.5 * expected_se)
    assert result.confidence_high == pytest.approx(result.estimate + 2.5 * expected_se)
    assert (result.n_pairs, result.n_steps, result.seed, result.confidence_z) == (100, 1, 97, 2.5)


def test_seed_is_reproducible_and_does_not_touch_global_rng() -> None:
    before = np.random.get_state()
    args = (100.0, 100.0, 1.0, 0.04, 1.5, 0.04, 0.4, -0.7)
    first = heston_mc_price(*args, n_pairs=128, n_steps=16, seed=12)
    second = heston_mc_price(*args, n_pairs=128, n_steps=16, seed=12)
    third = heston_mc_price(*args, n_pairs=128, n_steps=16, seed=13)
    after = np.random.get_state()
    assert first == second
    assert first.estimate != third.estimate
    assert before[0] == after[0]
    np.testing.assert_array_equal(before[1], after[1])
    assert before[2:] == after[2:]


@pytest.mark.parametrize("option_type", ["call", "put"])
@pytest.mark.parametrize("spot", [85.0, 100.0, 115.0])
def test_expiry_is_exact_with_zero_error(option_type: OptionType, spot: float) -> None:
    result = heston_mc_price(
        spot,
        100.0,
        0.0,
        0.04,
        1.5,
        0.04,
        0.4,
        -0.7,
        n_pairs=2,
        n_steps=1,
        seed=0,
        option_type=option_type,
    )
    expected = max((1 if option_type == "call" else -1) * (spot - 100), 0)
    assert result.estimate == result.confidence_low == result.confidence_high == expected
    assert result.standard_error == 0


@pytest.mark.parametrize("rho", [-1.0, 1.0])
def test_zero_variance_and_extreme_correlations(rho: float) -> None:
    result = heston_mc_price(
        110.0,
        100.0,
        0.5,
        0.0,
        1.5,
        0.0,
        0.4,
        rho,
        -0.02,
        0.01,
        n_pairs=16,
        n_steps=8,
        seed=0,
    )
    expected = 110 * math.exp(-0.01 * 0.5) - 100 * math.exp(0.02 * 0.5)
    assert result.estimate == pytest.approx(expected, abs=1e-12)
    assert result.standard_error == 0


@pytest.mark.parametrize(
    ("index", "value"),
    [
        (0, 0.0),
        (1, -1.0),
        (2, -0.1),
        (3, -0.01),
        (4, 0.0),
        (5, -0.01),
        (6, -0.01),
        (7, -1.01),
        (7, 1.01),
    ],
)
def test_rejects_invalid_domains(index: int, value: float) -> None:
    args = [100.0, 100.0, 1.0, 0.04, 1.5, 0.04, 0.4, -0.7]
    args[index] = value
    with pytest.raises(ValueError):
        heston_mc_price(*args, n_pairs=2, n_steps=1, seed=0)


@pytest.mark.parametrize("index", range(10))
@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_rejects_nonfinite_inputs(index: int, value: float) -> None:
    args = [100.0, 100.0, 1.0, 0.04, 1.5, 0.04, 0.4, -0.7, 0.02, 0.01]
    args[index] = value
    with pytest.raises(ValueError):
        heston_mc_price(*args, n_pairs=2, n_steps=1, seed=0)


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("n_pairs", 1),
        ("n_pairs", 3.5),
        ("n_pairs", True),
        ("n_steps", 0),
        ("n_steps", 2.5),
        ("seed", -1),
        ("seed", 0.5),
        ("confidence_z", 0.0),
        ("confidence_z", math.inf),
        ("confidence_z", math.nan),
        ("option_type", "straddle"),
    ],
)
def test_rejects_invalid_estimator_settings(name: str, value: object) -> None:
    kwargs: dict[str, object] = {"n_pairs": 2, "n_steps": 1, "seed": 0}
    kwargs[name] = value
    with pytest.raises(ValueError):
        heston_mc_price(100.0, 100.0, 1.0, 0.04, 1.5, 0.04, 0.4, -0.7, **kwargs)  # type: ignore[arg-type]


def test_rejects_numerical_overflow_instead_of_clipping() -> None:
    with pytest.raises(FloatingPointError):
        heston_mc_price(
            100.0,
            100.0,
            1.0,
            0.04,
            1.5,
            0.04,
            0.4,
            -0.7,
            1e4,
            n_pairs=2,
            n_steps=1,
            seed=0,
        )
