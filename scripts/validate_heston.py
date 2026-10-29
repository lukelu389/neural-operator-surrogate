"""Reproduce T1.2 COS convergence and independent Monte Carlo cross-checks.

Only explicit numerical fixtures are used. No data split or holdout is opened.
Cache results by config, source, and environment hashes, including failed runs.
"""

import argparse
import hashlib
import json
import math
import platform
import subprocess
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from nop.pricing.heston_cos import heston_cos_price
from nop.pricing.mc import heston_mc_price


def run(config: dict[str, Any]) -> dict[str, Any]:
    """Run the configured deterministic fixtures; raise for malformed settings."""
    if config["version"] != 1:
        raise ValueError("unsupported numerical-validation config version")
    settings = config["cos"]
    convergence = []
    passed = True
    for case in config["cases"]:
        args = list(case["parameters"])
        args[1] = torch.tensor(case["strikes"], dtype=torch.float64)
        reference = heston_cos_price(
            *args, n_terms=settings["reference_terms"], truncation_width=settings["reference_width"]
        )
        rows = []
        for width in settings["truncation_widths"]:
            for terms in settings["term_counts"]:
                try:
                    price = heston_cos_price(*args, n_terms=terms, truncation_width=width)
                    error: float | None = float((price - reference).abs().max())
                except FloatingPointError:
                    error = None
                rows.append({"terms": terms, "width": width, "max_abs_error": error})
        selected = heston_cos_price(
            *args, n_terms=settings["selected_terms"], truncation_width=settings["selected_width"]
        )
        error = float((selected - reference).abs().max())
        accepted = error <= settings["absolute_tolerance"]
        passed &= accepted
        convergence.append(
            {
                "name": case["name"],
                "rows": rows,
                "selected_error": error,
                "passed": accepted,
                "reference_prices": reference.tolist(),
            }
        )
        print(f"COS {case['name']}: selected max error {error:.3g}; pass={accepted}", flush=True)
    mc = config["monte_carlo"]
    comparisons = []
    for case in mc["cases"]:
        cos = heston_cos_price(
            *case["parameters"],
            n_terms=settings["reference_terms"],
            truncation_width=settings["reference_width"],
        ).item()
        estimates = []
        for steps in mc["step_counts"]:
            estimate = heston_mc_price(
                *case["parameters"],
                n_pairs=mc["n_pairs"],
                n_steps=steps,
                seed=case["seed"] + steps,
                confidence_z=mc["confidence_z"],
            )
            estimates.append(estimate)
            print(
                f"MC {case['name']} steps={steps}: {estimate.estimate:.9g}, "
                f"CI [{estimate.confidence_low:.9g}, {estimate.confidence_high:.9g}], "
                f"COS {cos:.9g}",
                flush=True,
            )
        fine, coarse = estimates[-1], estimates[-2]
        in_interval = fine.confidence_low <= cos <= fine.confidence_high
        difference = abs(fine.estimate - coarse.estimate)
        refinement_limit = mc["confidence_z"] * math.hypot(
            fine.standard_error, coarse.standard_error
        )
        refined = difference <= refinement_limit
        passed &= in_interval and refined
        comparisons.append(
            {
                "name": case["name"],
                "cos": cos,
                "estimates": [asdict(e) for e in estimates],
                "cos_in_fine_sampling_interval": in_interval,
                "refinement_difference": difference,
                "refinement_limit": refinement_limit,
                "refinement_passed": refined,
            }
        )
    return {"passed": passed, "convergence": convergence, "monte_carlo": comparisons}


def main() -> int:
    """Read YAML, reuse matching cached results, or run and record validation."""
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=root / "config/heston_numerics.yaml")
    parser.add_argument(
        "--output", type=Path, default=root / "reports/tables/heston_validation.json"
    )
    args = parser.parse_args()
    config_bytes = args.config.read_bytes()
    config = yaml.safe_load(config_bytes)
    files = [Path(__file__), root / "src/nop/pricing/heston_cos.py", root / "src/nop/pricing/mc.py"]
    source_hash = hashlib.sha256(b"".join(path.read_bytes() for path in files)).hexdigest()
    metadata = {
        "config_hash": hashlib.sha256(config_bytes).hexdigest(),
        "source_hash": source_hash,
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
            "scope": "numerical_fixtures_not_experiment_data",
        }
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    temporary.replace(args.output)
    print(f"Validation passed={result['passed']}; report: {args.output}")
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
