"""Validate adaptive labels against fixed-resolution quotes on numerical fixtures.

Use all 32 ID-box corners and their arithmetic center, never sampled data or
holdouts. Every adaptive-valid quote must have a valid fixed-reference IV and
pass both price and actual IV discrepancy checks. Retain all mask disagreements.
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
from validate_iv import _cos_prices

from nop.data.grids import build_grid
from nop.data.labels import LabelSettings, LabelStatus, label_surfaces
from nop.pricing.iv import IVSettings, IVStatus, implied_volatility


def _finite(value: torch.Tensor) -> float | None:
    result = float(value)
    return result if math.isfinite(result) else None


def _status_name(code: int) -> str:
    return IVStatus(code).name if code <= 10 else LabelStatus(code).name


def _summary(
    rows: list[dict[str, Any]], price_tolerance: float, iv_tolerance: float
) -> dict[str, Any]:
    accepted = [row for row in rows if row["adaptive_valid"]]
    comparable = [row for row in accepted if row["fixed_valid"]]
    unverified = len(accepted) - len(comparable)
    price_errors = [row["normalized_price_error"] for row in comparable]
    iv_errors = [row["absolute_iv_error"] for row in comparable]
    price_failures = sum(value is None or value > price_tolerance for value in price_errors)
    iv_failures = sum(value is None or value > iv_tolerance for value in iv_errors)
    finite_prices = [value for value in price_errors if value is not None]
    finite_ivs = [value for value in iv_errors if value is not None]
    return {
        "attempted": len(rows),
        "adaptive_valid": len(accepted),
        "adaptive_invalid": len(rows) - len(accepted),
        "adaptive_invalid_fraction": (len(rows) - len(accepted)) / len(rows),
        "fixed_valid": sum(row["fixed_valid"] for row in rows),
        "fixed_invalid": sum(not row["fixed_valid"] for row in rows),
        "both_valid": len(comparable),
        "adaptive_valid_fixed_invalid": unverified,
        "adaptive_invalid_fixed_valid": sum(
            not row["adaptive_valid"] and row["fixed_valid"] for row in rows
        ),
        "validity_mask_disagreements": sum(
            row["adaptive_valid"] != row["fixed_valid"] for row in rows
        ),
        "adaptive_status_counts": dict(
            sorted(Counter(row["adaptive_status"] for row in rows).items())
        ),
        "fixed_status_counts": dict(sorted(Counter(row["fixed_status"] for row in rows).items())),
        "status_pair_counts": dict(
            sorted(
                Counter(
                    row["adaptive_status"] + " -> " + row["fixed_status"] for row in rows
                ).items()
            )
        ),
        "fixed_cos_failures": sum(row["fixed_cos_error"] is not None for row in rows),
        "price_comparison_failures": price_failures,
        "iv_comparison_failures": iv_failures,
        "finite_price_comparisons": len(finite_prices),
        "p99_normalized_price_error": (
            float(np.quantile(finite_prices, 0.99, method="linear")) if finite_prices else None
        ),
        "max_normalized_price_error": max(finite_prices, default=None),
        "finite_iv_comparisons": len(finite_ivs),
        "p99_absolute_iv_error": (
            float(np.quantile(finite_ivs, 0.99, method="linear")) if finite_ivs else None
        ),
        "max_absolute_iv_error": max(finite_ivs, default=None),
        "passed": len(accepted) > 0
        and unverified == 0
        and price_failures == 0
        and iv_failures == 0,
    }


@torch.no_grad()
def run(config: dict[str, Any], numerics: dict[str, Any]) -> dict[str, Any]:
    """Compare configured adaptive puts with independent fixed quote pricing."""
    design, units, labels = config["data_design"], config["units"], config["labels"]
    if design["parameter_order"] != ["v0", "kappa", "theta", "sigma", "rho"]:
        raise ValueError("unexpected Heston parameter order")
    if labels["option_type"] != "put" or labels["iv"] != numerics["iv"]:
        raise ValueError("adaptive and fixed diagnostics require the same put/IV policy")
    if design["id_box"] != numerics["parameter_box"]:
        raise ValueError("adaptive and fixed numerical studies must cover the same ID box")
    ranges = [design["id_box"][key] for key in design["parameter_order"]]
    fixtures = [list(values) for values in itertools.product(*ranges)]
    fixtures.append([sum(bounds) / 2 for bounds in ranges])
    parameters = torch.tensor(fixtures, dtype=torch.float64)
    grid_config = design["main_grid"]
    maturities = tuple(a / b for a, b in grid_config["maturity_year_fractions"])
    z_config = grid_config["z_linspace"]
    z = tuple(
        torch.linspace(
            z_config["start"], z_config["stop"], z_config["count"], dtype=torch.float64
        ).tolist()
    )
    grid = build_grid(
        maturities, z, grid_config["k_scale"], units["spot"], units["rate"], units["dividend_yield"]
    )
    settings = LabelSettings(
        term_levels=tuple(labels["term_levels"]),
        width=labels["width"],
        reference_width=labels["reference_width"],
        price_rtol=labels["price_rtol"],
        iv_error_tolerance=labels["iv_error_tolerance"],
        iv=IVSettings(**labels["iv"]),
    )
    cos = numerics["cos"]
    quote_config = {**units, "cos": cos}
    scale = torch.maximum(
        units["spot"] * torch.exp(-units["dividend_yield"] * grid.maturity),
        grid.strike * torch.exp(-units["rate"] * grid.maturity),
    )
    batch_size = config["storage"]["pricing_batch_size"]
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("pricing_batch_size must be a positive integer")
    rows: list[dict[str, Any]] = []
    term_counts: list[int] = []
    for start in range(0, len(fixtures), batch_size):
        stop = min(start + batch_size, len(fixtures))
        adaptive = label_surfaces(
            parameters[start:stop],
            grid.strike,
            grid.maturity[:, 0],
            units["spot"],
            units["rate"],
            units["dividend_yield"],
            settings=settings,
        )
        term_counts.extend(adaptive.terms.tolist())
        for local_index, fixture_index in enumerate(range(start, stop)):
            fixed, fixed_errors = _cos_prices(
                fixtures[fixture_index],
                grid.strike.flatten(),
                grid.maturity.flatten(),
                quote_config,
                "put",
                cos["reference_terms"],
                cos["reference_width"],
            )
            fixed = fixed.reshape_as(grid.strike)
            fixed_iv = implied_volatility(
                fixed,
                units["spot"],
                grid.strike,
                grid.maturity,
                units["rate"],
                units["dividend_yield"],
                settings=settings.iv,
                option_type="put",
            )
            for maturity_index in range(len(maturities)):
                for z_index in range(len(z)):
                    index = (maturity_index, z_index)
                    adaptive_index = (local_index, maturity_index, z_index)
                    fixed_error = fixed_errors[maturity_index * len(z) + z_index]
                    rows.append(
                        {
                            "fixture_index": fixture_index,
                            "maturity_index": maturity_index,
                            "z_index": z_index,
                            "maturity": maturities[maturity_index],
                            "z": z[z_index],
                            "strike": float(grid.strike[index]),
                            "adaptive_price": _finite(adaptive.price[adaptive_index]),
                            "fixed_price": _finite(fixed[index]),
                            "adaptive_iv": _finite(adaptive.iv[adaptive_index]),
                            "fixed_iv": _finite(fixed_iv.volatility[index]),
                            "adaptive_valid": bool(adaptive.valid[adaptive_index]),
                            "fixed_valid": bool(fixed_iv.valid[index]),
                            "adaptive_status": _status_name(int(adaptive.status[adaptive_index])),
                            "fixed_status": (
                                "COS_FAILURE"
                                if fixed_error is not None
                                else IVStatus(int(fixed_iv.status[index])).name
                            ),
                            "fixed_cos_error": fixed_error,
                            "adaptive_finest_terms": int(adaptive.terms[local_index]),
                            "normalized_price_error": _finite(
                                (adaptive.price[adaptive_index] - fixed[index]).abs() / scale[index]
                            ),
                            "absolute_iv_error": _finite(
                                (adaptive.iv[adaptive_index] - fixed_iv.volatility[index]).abs()
                            ),
                        }
                    )
            print(f"Adaptive fixture {fixture_index + 1}/{len(fixtures)} complete", flush=True)
    overall = _summary(rows, settings.price_rtol, settings.iv_error_tolerance)
    return {
        "passed": overall["passed"],
        "overall": overall,
        "by_maturity": {
            str(index): _summary(
                [row for row in rows if row["maturity_index"] == index],
                settings.price_rtol,
                settings.iv_error_tolerance,
            )
            for index in range(len(maturities))
        },
        "by_z": {
            str(index): _summary(
                [row for row in rows if row["z_index"] == index],
                settings.price_rtol,
                settings.iv_error_tolerance,
            )
            for index in range(len(z))
        },
        "finest_term_counts_by_surface": dict(sorted(Counter(term_counts).items())),
        "parameter_fixtures": fixtures,
        "rows": rows,
        "notes": [
            "Deterministic corners/center are numerical fixtures, not sampled population coverage.",
            "Every adaptive-valid label needs a valid fixed IV and passing price/actual-IV errors.",
            "Adaptive-invalid/fixed-valid quotes are reported; no failure is silently omitted.",
            "Fixed prices use the quote pricer, independent of the grid-pricing adapter.",
            "No train, validation, test, or OOD dataset was generated or read.",
        ],
    }


def validation_metadata(root: Path, config_bytes: bytes, numerics_bytes: bytes) -> dict[str, Any]:
    """Return the exact source/config/environment identity used by the CLI cache."""
    files = [
        Path(__file__),
        root / "scripts/validate_iv.py",
        root / "src/nop/data/labels.py",
        root / "src/nop/data/grids.py",
        root / "src/nop/pricing/heston_grid.py",
        root / "src/nop/pricing/heston_cos.py",
        root / "src/nop/pricing/iv.py",
        root / "src/nop/pricing/bs.py",
    ]
    return {
        "config_hash": hashlib.sha256(config_bytes).hexdigest(),
        "numerics_config_hash": hashlib.sha256(numerics_bytes).hexdigest(),
        "source_hash": hashlib.sha256(b"".join(path.read_bytes() for path in files)).hexdigest(),
        "source_hashes": {
            str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in files
        },
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


def main(argv: list[str] | None = None) -> int:
    """Validate or reuse an identical completed study, including failure reports."""
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=root / "config/data.yaml")
    parser.add_argument("--numerics-config", type=Path, default=root / "config/data_numerics.yaml")
    parser.add_argument("--output", type=Path, default=root / "reports/tables/adaptive_labels.json")
    args = parser.parse_args(argv)
    config_bytes, numerics_bytes = args.config.read_bytes(), args.numerics_config.read_bytes()
    config, numerics = yaml.safe_load(config_bytes), yaml.safe_load(numerics_bytes)
    torch.set_num_threads(config["storage"]["torch_threads"])
    metadata = validation_metadata(root, config_bytes, numerics_bytes)
    cache_key = hashlib.sha256(json.dumps(metadata, sort_keys=True).encode()).hexdigest()
    if args.output.exists():
        previous = json.loads(args.output.read_text())
        if previous.get("cache_key") == cache_key:
            print(f"Cached adaptive validation: passed={previous['passed']} ({args.output})")
            return 0 if previous["passed"] else 1
    result = run(config, numerics)
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True, capture_output=True, check=True
    ).stdout.strip()
    result.update(
        {
            "cache_key": cache_key,
            "metadata": metadata,
            "code_commit": commit,
            "config": config,
            "numerics_config": numerics,
            "scope": "deterministic_numerical_fixtures_not_experiment_data",
        }
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    temporary.replace(args.output)
    print(f"Adaptive validation passed={result['passed']}; report: {args.output}")
    print(json.dumps(result["overall"], indent=2, allow_nan=False))
    print("Finest term counts:", result["finest_term_counts_by_surface"])
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
