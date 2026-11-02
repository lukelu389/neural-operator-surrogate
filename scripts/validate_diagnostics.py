"""Validate predeclared diagnostic numerics on explicit, non-dataset fixtures.

This checks IV finite-difference refinement and the units/signs of arbitrage
tolerances before model results. It is not the Gate 4 evaluation suite and
does not certify derivative accuracy throughout the parameter domain.
"""

import argparse
import hashlib
import json
import math
import platform
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import torch
import yaml

from nop.pricing.bs import black_scholes_price
from nop.pricing.heston_cos import heston_cos_price
from nop.pricing.iv import IVSettings, IVStatus, implied_volatility

FloatArray = npt.NDArray[np.float64]
BoolArray = npt.NDArray[np.bool_]


def _butterfly_differences(
    strikes: FloatArray, calls: FloatArray, valid: BoolArray
) -> tuple[FloatArray, BoolArray]:
    slopes = np.diff(calls) / np.diff(strikes)
    differences = np.diff(slopes)
    usable = valid[:-2] & valid[1:-1] & valid[2:] & np.isfinite(differences)
    return differences, usable


def _interpolate_total_variance(
    k: FloatArray, variance: FloatArray, valid: BoolArray, query: FloatArray
) -> tuple[FloatArray, BoolArray]:
    """Interpolate original adjacent nodes; an exact node only needs itself."""
    result = np.full_like(query, np.nan)
    usable = np.zeros(query.shape, dtype=np.bool_)
    for index, coordinate in enumerate(query):
        right = int(np.searchsorted(k, coordinate, side="left"))
        if right < len(k) and k[right] == coordinate:
            if valid[right] and math.isfinite(variance[right]):
                result[index], usable[index] = variance[right], True
        elif 0 < right < len(k) and valid[right - 1] and valid[right]:
            weight = (coordinate - k[right - 1]) / (k[right] - k[right - 1])
            result[index] = (1 - weight) * variance[right - 1] + weight * variance[right]
            usable[index] = math.isfinite(result[index])
    return result, usable


def _calendar_differences(
    k_early: FloatArray,
    w_early: FloatArray,
    valid_early: BoolArray,
    k_later: FloatArray,
    w_later: FloatArray,
    valid_later: BoolArray,
    count: int,
) -> tuple[FloatArray, BoolArray]:
    start, stop = max(k_early[0], k_later[0]), min(k_early[-1], k_later[-1])
    if start >= stop:
        return np.empty(0, dtype=np.float64), np.empty(0, dtype=np.bool_)
    common_k = np.linspace(start, stop, count)
    earlier, earlier_valid = _interpolate_total_variance(k_early, w_early, valid_early, common_k)
    later, later_valid = _interpolate_total_variance(k_later, w_later, valid_later, common_k)
    return later - earlier, earlier_valid & later_valid


def _quote(config: dict[str, Any], point: list[float]) -> dict[str, Any]:
    parameters, k, maturity = point[:5], point[5], point[6]
    bounds = config["parameter_box"]
    if (
        any(not math.isfinite(value) for value in point)
        or maturity <= 0
        or any(
            not bounds[name][0] <= value <= bounds[name][1]
            for name, value in zip(config["parameter_order"], parameters, strict=True)
        )
    ):
        return {"valid": False, "status": "STENCIL_OUTSIDE_DOMAIN", "iv": None}
    units, policy = config["units"], config["greek_sensitivities"]
    strike = units["spot"] * math.exp((units["rate"] - units["dividend_yield"]) * maturity + k)
    v0, kappa, theta, sigma, rho = parameters
    try:
        price = heston_cos_price(
            units["spot"],
            strike,
            maturity,
            v0,
            kappa,
            theta,
            sigma,
            rho,
            units["rate"],
            units["dividend_yield"],
            n_terms=policy["cos"]["terms"],
            truncation_width=policy["cos"]["width"],
            option_type="put",
        )
    except FloatingPointError as error:
        return {"valid": False, "status": "COS_FAILURE", "iv": None, "error": str(error)}
    iv = implied_volatility(
        price,
        units["spot"],
        strike,
        maturity,
        units["rate"],
        units["dividend_yield"],
        settings=IVSettings(**policy["iv"]),
        option_type="put",
    )
    return {
        "valid": bool(iv.valid),
        "status": IVStatus(int(iv.status)).name,
        "iv": float(iv.volatility) if bool(iv.valid) else None,
        "strike": strike,
    }


def _greek_fixtures(config: dict[str, Any]) -> dict[str, Any]:
    policy = config["greek_sensitivities"]
    names = config["parameter_order"] + ["k", "T"]
    steps = {**policy["parameter_steps"], **policy["coordinate_steps"]}
    rows: list[dict[str, Any]] = []
    for fixture in policy["fixtures"]:
        point = fixture["parameters"] + [fixture["k"], fixture["T"]]
        base = _quote(config, point)
        for coordinate, name in enumerate(names):
            step = steps[name]
            refined_step = step * policy["refinement_factor"]
            quotes: list[dict[str, Any]] = []
            for delta in (-step, step, -refined_step, refined_step):
                shifted = list(point)
                shifted[coordinate] += delta
                quotes.append(_quote(config, shifted))
            valid = base["valid"] and all(quote["valid"] for quote in quotes)
            coarse = (quotes[1]["iv"] - quotes[0]["iv"]) / (2 * step) if valid else None
            fine = (quotes[3]["iv"] - quotes[2]["iv"]) / (2 * refined_step) if valid else None
            discrepancy = abs(coarse - fine) if coarse is not None and fine is not None else None
            tolerance = (
                policy["reliability_absolute_tolerance"]
                + policy["reliability_relative_tolerance"] * abs(fine)
                if fine is not None
                else None
            )
            reliable = (
                discrepancy is not None and tolerance is not None and discrepancy <= tolerance
            )
            rows.append(
                {
                    "fixture": fixture["name"],
                    "coordinate": name,
                    "step": step,
                    "base": base,
                    "stencils_minus_h_plus_h_minus_half_h_plus_half_h": quotes,
                    "all_quotes_valid": valid,
                    "coarse_derivative": coarse,
                    "refined_derivative": fine,
                    "reference_derivative": fine if reliable else None,
                    "absolute_refinement_difference": discrepancy,
                    "reliability_tolerance": tolerance,
                    "reliable": reliable,
                }
            )
    differences = [row["absolute_refinement_difference"] for row in rows if row["all_quotes_valid"]]
    return {
        "passed": bool(rows) and all(row["reliable"] for row in rows),
        "attempted_stencils": len(rows),
        "valid_stencils": sum(row["all_quotes_valid"] for row in rows),
        "reliable_stencils": sum(row["reliable"] for row in rows),
        "max_absolute_refinement_difference": max(differences, default=None),
        "rows": rows,
    }


def _arbitrage_fixtures(config: dict[str, Any]) -> dict[str, Any]:
    units, fixtures, policy = config["units"], config["validation"], config["arbitrage"]
    butterfly_tolerance = policy["butterfly"]["absolute_tolerance"]
    calendar_tolerance = policy["calendar"]["absolute_tolerance"]
    z = np.array(fixtures["bs_z"], dtype=np.float64)
    vol = fixtures["bs_volatility"]
    surface: list[tuple[FloatArray, FloatArray, BoolArray]] = []
    bs_butterfly: list[float] = []
    bs_calendar: list[float] = []
    for maturity in fixtures["bs_maturities"]:
        k = z * fixtures["bs_k_scale"] * math.sqrt(maturity)
        strikes = units["spot"] * np.exp((units["rate"] - units["dividend_yield"]) * maturity + k)
        calls = black_scholes_price(
            units["spot"],
            torch.from_numpy(strikes),
            maturity,
            vol,
            units["rate"],
            units["dividend_yield"],
        ).numpy()
        valid = np.ones(k.shape, dtype=np.bool_)
        differences, usable = _butterfly_differences(strikes, calls, valid)
        bs_butterfly.extend(differences[usable].tolist())
        surface.append((k, np.full_like(k, maturity * vol**2), valid))
    for earlier, later in zip(surface[:-1], surface[1:], strict=True):
        differences, usable = _calendar_differences(
            *earlier, *later, policy["calendar"]["common_k_points"]
        )
        bs_calendar.extend(differences[usable].tolist())
    constructed: list[dict[str, Any]] = []
    strikes = np.array(fixtures["constructed_butterfly_strikes"], dtype=np.float64)
    for magnitude in fixtures["constructed_butterfly_magnitudes"]:
        calls = np.array(fixtures["constructed_butterfly_calls"], dtype=np.float64)
        # Raising the center introduces this negative change in successive slopes.
        calls[1] += magnitude / (1 / (strikes[1] - strikes[0]) + 1 / (strikes[2] - strikes[1]))
        values, usable = _butterfly_differences(strikes, calls, np.ones(3, dtype=np.bool_))
        observed = -float(values[0])
        detected = bool(values[0] < -butterfly_tolerance)
        constructed.append(
            {
                "kind": "butterfly",
                "known_negative_magnitude": magnitude,
                "observed_negative_magnitude": observed,
                "detected": detected,
                "passed": bool(usable[0])
                and math.isclose(observed, magnitude, rel_tol=1e-6, abs_tol=1e-14)
                and detected == (magnitude > butterfly_tolerance),
            }
        )
    earlier_k, _, valid = surface[0]
    later_k, _, later_valid = surface[1]
    variance = fixtures["constructed_calendar_total_variance"]
    for magnitude in fixtures["constructed_calendar_magnitudes"]:
        values, usable = _calendar_differences(
            earlier_k,
            np.full_like(earlier_k, variance),
            valid,
            later_k,
            np.full_like(later_k, variance - magnitude),
            later_valid,
            policy["calendar"]["common_k_points"],
        )
        detected = bool(np.all(values < -calendar_tolerance))
        constructed.append(
            {
                "kind": "calendar",
                "known_negative_magnitude": magnitude,
                "observed_negative_magnitude": float(-values.min()),
                "detected": detected,
                "passed": bool(usable.all())
                and bool(np.allclose(-values, magnitude, rtol=1e-6, atol=1e-16))
                and detected == (magnitude > calendar_tolerance),
            }
        )
    return {
        "passed": min(bs_butterfly) >= -butterfly_tolerance
        and min(bs_calendar) >= -calendar_tolerance
        and all(row["passed"] for row in constructed),
        "bs_butterfly_checks": len(bs_butterfly),
        "bs_calendar_checks": len(bs_calendar),
        "bs_minimum_butterfly_slope_difference": min(bs_butterfly),
        "bs_minimum_calendar_total_variance_difference": min(bs_calendar),
        "constructed_checks": constructed,
    }


@torch.no_grad()
def run(config: dict[str, Any]) -> dict[str, Any]:
    """Validate the configured finite differences and arbitrage numerical units."""
    if config["version"] != 1 or config["parameter_order"] != [
        "v0",
        "kappa",
        "theta",
        "sigma",
        "rho",
    ]:
        raise ValueError("unsupported diagnostic config or parameter order")
    policy = config["greek_sensitivities"]
    for name, bounds in config["parameter_box"].items():
        expected = policy["parameter_step_fraction_of_ID_width"] * (bounds[1] - bounds[0])
        if not math.isclose(policy["parameter_steps"][name], expected, rel_tol=1e-12):
            raise ValueError(f"{name} step disagrees with the declared ID width fraction")
    greek, arbitrage = _greek_fixtures(config), _arbitrage_fixtures(config)
    return {
        "passed": greek["passed"] and arbitrage["passed"],
        "greek": greek,
        "arbitrage": arbitrage,
    }


def validation_metadata(root: Path, config_bytes: bytes) -> dict[str, Any]:
    """Identify source, settings, and environment for deterministic cached studies."""
    files = [
        Path(__file__),
        root / "src/nop/pricing/heston_cos.py",
        root / "src/nop/pricing/iv.py",
        root / "src/nop/pricing/bs.py",
    ]
    return {
        "config_hash": hashlib.sha256(config_bytes).hexdigest(),
        "source_hash": hashlib.sha256(b"".join(path.read_bytes() for path in files)).hexdigest(),
        "source_hashes": {
            str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in files
        },
        "seed": None,
        "sampling": "none_explicit_numerical_fixtures",
        "python": platform.python_version(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "pyyaml": yaml.__version__,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "torch_threads": torch.get_num_threads(),
        "dtype": "float64",
        "device": "cpu",
    }


def main(argv: list[str] | None = None) -> int:
    """Run or reuse the same completed numerical study, retaining failures too."""
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=root / "config/diagnostics.yaml")
    parser.add_argument(
        "--output", type=Path, default=root / "reports/tables/diagnostics_validation.json"
    )
    args = parser.parse_args(argv)
    config_bytes = args.config.read_bytes()
    config = yaml.safe_load(config_bytes)
    torch.set_num_threads(config["validation"]["torch_threads"])
    metadata = validation_metadata(root, config_bytes)
    cache_key = hashlib.sha256(json.dumps(metadata, sort_keys=True).encode()).hexdigest()
    if args.output.exists():
        previous = json.loads(args.output.read_text())
        if previous.get("cache_key") == cache_key:
            print(f"Cached diagnostic validation: passed={previous['passed']} ({args.output})")
            return 0 if previous["passed"] else 1
        # Keep any already-seen study intact when implementation/settings change.
        prior_path = args.output.with_name(f"{args.output.stem}_{previous['cache_key']}.json")
        if not prior_path.exists():
            prior_path.write_bytes(args.output.read_bytes())
    result = run(config)
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True, capture_output=True, check=True
    ).stdout.strip()
    result.update(
        {
            "cache_key": cache_key,
            "metadata": metadata,
            "code_commit": commit,
            "config": config,
            "scope": "explicit_numerical_fixtures_not_experiment_data",
        }
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    temporary.replace(args.output)
    print(f"Diagnostic validation passed={result['passed']}; report: {args.output}")
    print(
        json.dumps(
            {key: value for key, value in result["greek"].items() if key != "rows"}, indent=2
        )
    )
    print(json.dumps(result["arbitrage"], indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
