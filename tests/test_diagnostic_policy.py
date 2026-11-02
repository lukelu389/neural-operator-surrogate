"""Numerical-policy fixtures only; no experiment data or learned diagnostics."""

import copy
import math
import runpy
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def diagnostics() -> dict[str, Any]:
    return runpy.run_path(str(ROOT / "scripts/validate_diagnostics.py"))


@pytest.fixture
def config() -> dict[str, Any]:
    return yaml.safe_load((ROOT / "config/diagnostics.yaml").read_text())


def test_reference_settings_agree_with_validated_pricer(config: dict[str, Any]) -> None:
    numerics = yaml.safe_load((ROOT / "config/data_numerics.yaml").read_text())
    policy = config["greek_sensitivities"]
    assert config["parameter_box"] == numerics["parameter_box"]
    assert policy["iv"] == numerics["iv"]
    assert policy["cos"] == {
        "terms": numerics["cos"]["reference_terms"],
        "width": numerics["cos"]["reference_width"],
    }


def test_declared_fixtures_pass_refinement_and_independent_arbitrage_checks(
    diagnostics: dict[str, Any], config: dict[str, Any]
) -> None:
    original_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(config["validation"]["torch_threads"])
        result = diagnostics["run"](config)
    finally:
        torch.set_num_threads(original_threads)
    assert result["passed"]
    assert result["greek"]["attempted_stencils"] == 21
    assert result["greek"]["reliable_stencils"] == 21
    assert result["arbitrage"]["bs_butterfly_checks"] == 36
    assert result["arbitrage"]["bs_calendar_checks"] == 45
    checks = result["arbitrage"]["constructed_checks"]
    assert [row["detected"] for row in checks] == [False, True, False, True]


def test_calendar_interpolation_never_bridges_invalid_nodes_or_extrapolates(
    diagnostics: dict[str, Any],
) -> None:
    result, valid = diagnostics["_interpolate_total_variance"](
        np.array([-1.0, 0.0, 1.0, 2.0]),
        np.array([0.02, np.nan, 0.04, 0.06]),
        np.array([True, False, True, True]),
        np.array([-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 3.0]),
    )
    np.testing.assert_array_equal(
        valid, [False, True, False, False, False, True, True, True, False]
    )
    np.testing.assert_allclose(result[valid], [0.02, 0.04, 0.05, 0.06])
    assert np.isnan(result[~valid]).all()


def test_calendar_compares_same_k_on_overlap_with_unequal_native_grids(
    diagnostics: dict[str, Any],
) -> None:
    earlier_k, later_k = np.array([-1.0, 0.0, 1.0]), np.array([-2.0, 0.0, 2.0])
    result, valid = diagnostics["_calendar_differences"](
        earlier_k,
        0.03 + 0.01 * earlier_k,
        np.ones(3, dtype=np.bool_),
        later_k,
        0.05 + 0.02 * later_k,
        np.ones(3, dtype=np.bool_),
        15,
    )
    assert valid.all()
    np.testing.assert_allclose(result, 0.02 + 0.01 * np.linspace(-1.0, 1.0, 15))


def test_butterfly_uses_native_neighbors_with_unequal_strike_spacing(
    diagnostics: dict[str, Any],
) -> None:
    result, valid = diagnostics["_butterfly_differences"](
        np.array([0.8, 0.9, 1.1, 1.4, 1.5, 1.7]),
        np.array([0.4, 0.31, 0.15, 0.09, 0.08, 0.07]),
        np.array([True, True, False, True, True, True]),
    )
    np.testing.assert_array_equal(valid, [False, False, False, True])
    assert result[-1] == pytest.approx(0.05)


@pytest.mark.parametrize("coordinate,value", [(0, 0.0099), (4, -0.951), (6, 0.0)])
def test_invalid_stencil_domain_is_masked_before_pricing(
    diagnostics: dict[str, Any],
    config: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
    coordinate: int,
    value: float,
) -> None:
    def forbidden(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("invalid stencil reached the pricer")

    monkeypatch.setitem(diagnostics["_quote"].__globals__, "heston_cos_price", forbidden)
    point = [0.04, 1.5, 0.04, 0.3, -0.7, 0.0, 1.0]
    point[coordinate] = value
    result = diagnostics["_quote"](config, point)
    assert not result["valid"] and result["iv"] is None
    assert result["status"] == "STENCIL_OUTSIDE_DOMAIN"


def test_maturity_stencil_recomputes_strike_at_fixed_forward_k(
    diagnostics: dict[str, Any],
    config: dict[str, Any],
) -> None:
    point = [0.04, 1.5, 0.04, 0.3, -0.7, 0.03, 1.0]
    base = diagnostics["_quote"](config, point)
    point[-1] += config["greek_sensitivities"]["coordinate_steps"]["T"]
    shifted = diagnostics["_quote"](config, point)
    assert base["valid"] and shifted["valid"]
    assert math.log(shifted["strike"] / base["strike"]) == pytest.approx(0.02 * 0.0001)


@pytest.mark.parametrize("failure", ["invalid_quote", "unstable_derivative"])
def test_failed_refinement_keeps_attempted_count_and_masks_reference(
    diagnostics: dict[str, Any],
    config: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
) -> None:
    config = copy.deepcopy(config)
    config["greek_sensitivities"]["fixtures"] = [config["greek_sensitivities"]["fixtures"][0]]

    def quote(_config: dict[str, Any], point: list[float]) -> dict[str, Any]:
        valid = not (failure == "invalid_quote" and point[5] < 0)
        # A cubic has a known central-difference truncation error proportional to h^2.
        iv = 0.2 + 1e8 * point[5] ** 3 if valid else None
        return {"valid": valid, "iv": iv}

    monkeypatch.setitem(diagnostics["_greek_fixtures"].__globals__, "_quote", quote)
    result = diagnostics["_greek_fixtures"](config)
    assert result["attempted_stencils"] == 7
    assert result["reliable_stencils"] == 6
    assert not result["passed"]
    row = next(row for row in result["rows"] if row["coordinate"] == "k")
    assert row["reference_derivative"] is None
    assert row["all_quotes_valid"] == (failure == "unstable_derivative")
