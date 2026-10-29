"""Reproduce T1.3 IV failures and price roundtrips on deterministic fixtures.

This is a corner/center numerical diagnostic, not a sampled data split or an
estimate of training/validation coverage. Keep upstream COS failures in the
attempted-quote denominator. Successful roundtrips establish inversion accuracy
against supplied prices, not Heston label accuracy. Cache failed studies too.
"""

import argparse
import hashlib
import itertools
import json
import math
import platform
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from nop.pricing.bs import OptionType, black_scholes_price
from nop.pricing.heston_cos import heston_cos_price
from nop.pricing.iv import IVSettings, IVStatus, implied_volatility


def _finite(value: torch.Tensor) -> float | None:
    result = float(value)
    return result if math.isfinite(result) else None


def _cos_prices(
    parameters: list[float],
    strikes: torch.Tensor,
    maturities: torch.Tensor,
    config: dict[str, Any],
    option_type: OptionType,
    terms: int,
    width: float,
) -> tuple[torch.Tensor, list[str | None]]:
    """Try batches, then isolate numerical failures without dropping quotes."""
    prices = torch.full_like(strikes, torch.nan)
    failures: list[str | None] = [None] * len(strikes)
    batch_size = config["cos"]["quote_batch_size"]
    v0, kappa, theta, sigma, rho = parameters

    def price(strike: torch.Tensor, maturity: torch.Tensor) -> torch.Tensor:
        return heston_cos_price(
            config["spot"],
            strike,
            maturity,
            v0,
            kappa,
            theta,
            sigma,
            rho,
            config["rate"],
            config["dividend_yield"],
            n_terms=terms,
            truncation_width=width,
            option_type=option_type,
        )

    for start in range(0, len(strikes), batch_size):
        stop = min(start + batch_size, len(strikes))
        try:
            prices[start:stop] = price(strikes[start:stop], maturities[start:stop])
        except FloatingPointError:
            for index in range(start, stop):
                try:
                    prices[index] = price(strikes[index], maturities[index])
                except FloatingPointError as error:
                    failures[index] = str(error)
    return prices, failures


def _summary(rows: list[dict[str, Any]], tolerance: float) -> dict[str, Any]:
    statuses = Counter(row["status"] for row in rows)
    accepted = [row for row in rows if row["status"] == IVStatus.SUCCESS.name]
    residuals = [row["normalized_roundtrip_residual"] for row in accepted]
    finite_residuals = [value for value in residuals if value is not None]
    differences = [
        row["normalized_selected_reference_difference"]
        for row in rows
        if row["normalized_selected_reference_difference"] is not None
    ]
    attempted = len(rows)
    failure_count = attempted - len(accepted)
    roundtrip_failures = sum(value is None or value > tolerance for value in residuals)
    return {
        "attempted": attempted,
        "accepted": len(accepted),
        "failed": failure_count,
        "failure_fraction": failure_count / attempted,
        "status_counts": dict(sorted(statuses.items())),
        "selected_cos_failures": sum(row["selected_cos_error"] is not None for row in rows),
        "reference_cos_failures": sum(row["reference_cos_error"] is not None for row in rows),
        "accepted_roundtrip_failures": roundtrip_failures,
        "accepted_finite_roundtrip_residuals": len(finite_residuals),
        "p99_normalized_roundtrip_residual": (
            float(np.quantile(finite_residuals, 0.99, method="linear"))
            if finite_residuals
            else None
        ),
        "max_normalized_roundtrip_residual": max(finite_residuals, default=None),
        "selected_reference_comparable_quotes": len(differences),
        "p99_normalized_selected_reference_difference": (
            float(np.quantile(differences, 0.99, method="linear")) if differences else None
        ),
        "max_normalized_selected_reference_difference": max(differences, default=None),
    }


@torch.no_grad()
def run(config: dict[str, Any]) -> dict[str, Any]:
    """Evaluate fixed corner/center fixtures; retain every failure and stratum."""
    if config["version"] != 1 or config["fixtures"] != "corners_plus_arithmetic_center":
        raise ValueError("unsupported numerical-validation config")
    if config["parameter_order"] != ["v0", "kappa", "theta", "sigma", "rho"]:
        raise ValueError("unexpected Heston parameter order")
    if config["option_types"] != ["call", "put"]:
        raise ValueError("diagnostic requires both call and put")
    batch_size = config["cos"]["quote_batch_size"]
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("quote_batch_size must be a positive integer")
    tolerance = config["roundtrip"]["max_normalized_price_residual"]
    if not math.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("roundtrip tolerance must be finite and positive")
    ranges = [config["parameter_box"][key] for key in config["parameter_order"]]
    if any(len(bounds) != 2 or bounds[0] >= bounds[1] for bounds in ranges):
        raise ValueError("each parameter range must contain an increasing pair")
    fixtures = [list(parameters) for parameters in itertools.product(*ranges)]
    fixtures.append([sum(bounds) / 2 for bounds in ranges])
    fractions = config["grid"]["maturity_year_fractions"]
    maturity_values = [numerator / denominator for numerator, denominator in fractions]
    z_config = config["grid"]["z_linspace"]
    z_values = torch.linspace(
        z_config["start"], z_config["stop"], z_config["count"], dtype=torch.float64
    )
    maturities = torch.tensor(maturity_values, dtype=torch.float64).repeat_interleave(len(z_values))
    z = z_values.repeat(len(maturity_values))
    k = z * config["grid"]["k_scale"] * maturities.sqrt()
    forwards = config["spot"] * torch.exp((config["rate"] - config["dividend_yield"]) * maturities)
    strikes = forwards * torch.exp(k)
    normalization = torch.maximum(
        config["spot"] * torch.exp(-config["dividend_yield"] * maturities),
        strikes * torch.exp(-config["rate"] * maturities),
    )
    settings = IVSettings(**config["iv"])
    cos_settings = config["cos"]
    rows: list[dict[str, Any]] = []
    option_types: tuple[OptionType, ...] = ("call", "put")
    for fixture_index, parameters in enumerate(fixtures):
        for option_type in option_types:
            selected, selected_errors = _cos_prices(
                parameters,
                strikes,
                maturities,
                config,
                option_type,
                cos_settings["selected_terms"],
                cos_settings["selected_width"],
            )
            reference, reference_errors = _cos_prices(
                parameters,
                strikes,
                maturities,
                config,
                option_type,
                cos_settings["reference_terms"],
                cos_settings["reference_width"],
            )
            result = implied_volatility(
                selected,
                config["spot"],
                strikes,
                maturities,
                config["rate"],
                config["dividend_yield"],
                settings=settings,
                option_type=option_type,
            )
            roundtrip = torch.full_like(selected, torch.nan)
            valid = result.valid
            if bool(valid.any()):
                try:
                    roundtrip[valid] = black_scholes_price(
                        config["spot"],
                        strikes[valid],
                        maturities[valid],
                        result.volatility[valid],
                        config["rate"],
                        config["dividend_yield"],
                        option_type=option_type,
                    )
                except FloatingPointError:
                    for index in torch.where(valid)[0].tolist():
                        try:
                            roundtrip[index] = black_scholes_price(
                                config["spot"],
                                strikes[index],
                                maturities[index],
                                result.volatility[index],
                                config["rate"],
                                config["dividend_yield"],
                                option_type=option_type,
                            )
                        except FloatingPointError:
                            pass  # Retained as a failed roundtrip, never an accepted residual.
            for index in range(len(strikes)):
                upstream_failed = selected_errors[index] is not None
                status = (
                    "UPSTREAM_COS_FAILURE"
                    if upstream_failed
                    else IVStatus(int(result.status[index])).name
                )
                rows.append(
                    {
                        "fixture_index": fixture_index,
                        "option_type": option_type,
                        "maturity_index": index // len(z_values),
                        "z_index": index % len(z_values),
                        "maturity": float(maturities[index]),
                        "z": float(z[index]),
                        "k": float(k[index]),
                        "strike": float(strikes[index]),
                        "selected_price": _finite(selected[index]),
                        "reference_price": _finite(reference[index]),
                        "selected_cos_error": selected_errors[index],
                        "reference_cos_error": reference_errors[index],
                        "status": status,
                        "volatility": _finite(result.volatility[index]),
                        "iterations": int(result.iterations[index]),
                        "normalized_vega": _finite(result.vega[index] / normalization[index]),
                        "signed_iv_price_residual": _finite(result.price_residual[index]),
                        "normalized_roundtrip_residual": _finite(
                            (roundtrip[index] - selected[index]).abs() / normalization[index]
                        ),
                        "normalized_selected_reference_difference": _finite(
                            (selected[index] - reference[index]).abs() / normalization[index]
                        ),
                    }
                )
        print(f"IV fixture {fixture_index + 1}/{len(fixtures)} complete", flush=True)
    overall = _summary(rows, tolerance)
    groups: dict[str, Any] = {}
    for key, values in (
        ("option_type", ["call", "put"]),
        ("maturity_index", list(range(len(maturity_values)))),
        ("z_index", list(range(len(z_values)))),
    ):
        groups[key] = {
            str(value): _summary([row for row in rows if row[key] == value], tolerance)
            for value in values
        }
    passed = overall["accepted"] > 0 and overall["accepted_roundtrip_failures"] == 0
    return {
        "passed": passed,
        "overall": overall,
        "by_stratum": groups,
        "parameter_fixtures": fixtures,
        "rows": rows,
        "notes": [
            "Deterministic corners plus center; failure fractions are not population estimates.",
            "All attempted quotes stay in the denominator, including selected COS failures.",
            "Reference COS failure affects the price comparison, not selected-price inversion.",
            "Roundtrip success measures inversion accuracy, not Heston label accuracy.",
            "Residuals and vega normalize by max(S exp(-qT), K exp(-rT)), matching IVSettings.",
            "Failure masks are reported without tuning thresholds or replacing COS settings.",
        ],
    }


def main() -> int:
    """Run or reuse the identical config/source/environment diagnostic."""
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=root / "config/iv_numerics.yaml")
    parser.add_argument("--output", type=Path, default=root / "reports/tables/iv_validation.json")
    args = parser.parse_args()
    config_bytes = args.config.read_bytes()
    config = yaml.safe_load(config_bytes)
    files = [
        Path(__file__),
        root / "src/nop/pricing/iv.py",
        root / "src/nop/pricing/bs.py",
        root / "src/nop/pricing/heston_cos.py",
    ]
    source_hash = hashlib.sha256(b"".join(path.read_bytes() for path in files)).hexdigest()
    metadata = {
        "config_hash": hashlib.sha256(config_bytes).hexdigest(),
        "source_hash": source_hash,
        "seed": None,
        "sampling": "none_deterministic_numerical_fixtures",
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
    cache_key = hashlib.sha256(json.dumps(metadata, sort_keys=True).encode()).hexdigest()
    if args.output.exists():
        previous = json.loads(args.output.read_text())
        if previous.get("cache_key") == cache_key:
            print(f"Cached validation: passed={previous['passed']} ({args.output})")
            return 0 if previous["passed"] else 1
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
            "scope": "deterministic_numerical_fixtures_not_experiment_data",
        }
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    temporary.replace(args.output)
    print(f"Validation passed={result['passed']}; report: {args.output}")
    print(json.dumps(result["overall"], indent=2, allow_nan=False))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
