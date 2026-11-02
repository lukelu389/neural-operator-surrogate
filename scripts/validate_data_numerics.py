"""Check fixed COS convergence before T1.4 data generation.

Reuse the T1.3 quote diagnostic so its full denominator, failure reasons, and
roundtrip definitions remain consistent. Add price/vega-conditioned convergence
and an independent Runge--Kutta solution of the Heston Riccati ODE. These fixed
fixtures do not open data splits or certify arbitrary parameter values.
"""

import argparse
import hashlib
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
from numpy.typing import NDArray
from validate_iv import run as run_iv

from nop.pricing.heston_cos import heston_characteristic_function


def _riccati_solution(
    frequencies: NDArray[np.float64],
    maturity: float,
    parameters: list[float],
    rate: float,
    dividend_yield: float,
    steps: int,
) -> NDArray[np.complex128]:
    """Integrate affine coefficients directly, independently of the closed CF."""
    v0, kappa, theta, sigma, rho = parameters
    c = np.zeros_like(frequencies, dtype=np.complex128)
    d = c.copy()
    dt = maturity / steps
    iu = 1j * frequencies

    def rhs(value: NDArray[np.complex128]) -> tuple[NDArray[np.complex128], NDArray[np.complex128]]:
        return (
            iu * (rate - dividend_yield) + kappa * theta * value,
            0.5 * sigma**2 * value**2
            + (rho * sigma * iu - kappa) * value
            - 0.5 * (frequencies**2 + iu),
        )

    with np.errstate(over="raise", invalid="raise", divide="raise"):
        for _ in range(steps):
            c1, d1 = rhs(d)
            c2, d2 = rhs(d + dt * d1 / 2)
            c3, d3 = rhs(d + dt * d2 / 2)
            c4, d4 = rhs(d + dt * d3)
            c += dt * (c1 + 2 * c2 + 2 * c3 + c4) / 6
            d += dt * (d1 + 2 * d2 + 2 * d3 + d4) / 6
        result = np.exp(c + v0 * d)
    if not np.isfinite(result).all():
        raise FloatingPointError("nonfinite independent Riccati result")
    return result


def _riccati_check(config: dict[str, Any]) -> dict[str, Any]:
    settings = config["riccati"]
    steps = settings["step_counts"]
    if len(steps) != 2 or not 1 <= steps[0] < steps[1]:
        raise ValueError("Riccati check requires two increasing positive step counts")
    frequencies = np.asarray(settings["frequencies"], dtype=np.float64)
    v0, kappa, theta, sigma, rho = settings["parameters"]
    closed = heston_characteristic_function(
        torch.from_numpy(frequencies),
        settings["maturity"],
        v0,
        kappa,
        theta,
        sigma,
        rho,
        config["rate"],
        config["dividend_yield"],
    ).numpy()
    estimates = [
        _riccati_solution(
            frequencies,
            settings["maturity"],
            settings["parameters"],
            config["rate"],
            config["dividend_yield"],
            count,
        )
        for count in steps
    ]
    refinement = np.abs(estimates[0] - estimates[1])
    discrepancy = np.abs(closed - estimates[1])
    tolerance = settings["max_absolute_error"]
    return {
        "passed": bool(np.all(refinement <= tolerance) and np.all(discrepancy <= tolerance)),
        "max_refinement_difference": float(refinement.max()),
        "max_closed_form_difference": float(discrepancy.max()),
        "rows": [
            {
                "frequency": float(frequency),
                "closed_form_real": float(closed[index].real),
                "closed_form_imaginary": float(closed[index].imag),
                "fine_ode_real": float(estimates[1][index].real),
                "fine_ode_imaginary": float(estimates[1][index].imag),
                "refinement_difference": float(refinement[index]),
                "closed_form_difference": float(discrepancy[index]),
            }
            for index, frequency in enumerate(frequencies)
        ],
    }


def run(config: dict[str, Any]) -> dict[str, Any]:
    """Keep all attempted quotes and assess convergence of accepted IV labels."""
    limits = config["convergence"]
    if any(not math.isfinite(value) or value <= 0 for value in limits.values()):
        raise ValueError("convergence tolerances must be finite and positive")
    result = run_iv(config)
    ratios = []
    comparable = 0
    for row in result["rows"]:
        row["price_difference_over_vega"] = None
        row["numerically_converged"] = None
        row["label_status"] = row["status"]
        if row["status"] != "SUCCESS":
            continue
        discrepancy, vega = (
            row["normalized_selected_reference_difference"],
            row["normalized_vega"],
        )
        valid_comparison = discrepancy is not None and vega is not None and vega > 0
        accepted = False
        if valid_comparison:
            comparable += 1
            ratio = discrepancy / vega
            ratios.append(ratio)
            row["price_difference_over_vega"] = ratio
            accepted = (
                discrepancy <= limits["max_normalized_price_difference"]
                and ratio <= limits["max_price_difference_over_vega"]
            )
        row["numerically_converged"] = accepted
        if not accepted:
            row["label_status"] = "COS_NOT_CONVERGED"
    status_counts = Counter(row["label_status"] for row in result["rows"])
    missing = result["overall"]["accepted"] - comparable
    numerical_failures = status_counts["COS_NOT_CONVERGED"]
    result["numerical_convergence"] = {
        "attempted": len(result["rows"]),
        "iv_accepted": result["overall"]["accepted"],
        "iv_accepted_with_comparable_reference": comparable,
        "iv_accepted_missing_comparison": missing,
        "numerical_comparison_failures": numerical_failures,
        "label_status_counts": dict(sorted(status_counts.items())),
        "p99_price_difference_over_vega": (
            float(np.quantile(ratios, 0.99, method="linear")) if ratios else None
        ),
        "max_price_difference_over_vega": max(ratios, default=None),
        "passed": comparable > 0 and numerical_failures == 0,
    }
    try:
        result["riccati"] = _riccati_check(config)
    except FloatingPointError as error:
        result["riccati"] = {"passed": False, "error": str(error)}
    result["passed"] &= result["numerical_convergence"]["passed"] and result["riccati"]["passed"]
    result["notes"].extend(
        [
            "Price/vega measures local IV sensitivity, not a rigorous global IV error bound.",
            "Only IV-accepted quotes with successful numerical comparisons are accepted labels.",
            "COS_NOT_CONVERGED retains failed or unavailable comparisons in the denominator.",
            "Fixed convergence fixtures do not certify all labels or change any failure threshold.",
        ]
    )
    return result


def main(argv: list[str] | None = None) -> int:
    """Cache completed studies, including failed checks, by exact provenance."""
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=root / "config/data_numerics.yaml")
    parser.add_argument("--output", type=Path, default=root / "reports/tables/data_numerics.json")
    args = parser.parse_args(argv)
    config_bytes = args.config.read_bytes()
    config = yaml.safe_load(config_bytes)
    files = [
        Path(__file__),
        root / "scripts/validate_iv.py",
        root / "src/nop/pricing/iv.py",
        root / "src/nop/pricing/bs.py",
        root / "src/nop/pricing/heston_cos.py",
    ]
    metadata = {
        "config_hash": hashlib.sha256(config_bytes).hexdigest(),
        "source_hash": hashlib.sha256(b"".join(path.read_bytes() for path in files)).hexdigest(),
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
            print(f"Cached data numerics: passed={previous['passed']} ({args.output})")
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
    print(f"Data numerics passed={result['passed']}; report: {args.output}")
    print(json.dumps(result["numerical_convergence"], indent=2, allow_nan=False))
    print(json.dumps(result["riccati"], indent=2, allow_nan=False))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
