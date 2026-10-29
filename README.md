# Neural surrogates for option pricing and calibration

This research project will compare MLP, DeepONet-style, Transformer, and SSM
surrogates for synthetic Heston implied-volatility surfaces, then evaluate their
use in calibration. Accuracy, tail errors, invalid quotes, arbitrage, Greeks,
speed, and variability across seeds are part of the planned comparison.

**Status: Gate 0 complete; T1.1–T1.2 Black–Scholes and Heston COS numerical checks complete.
No Heston dataset, trained model, or experimental benchmark result exists yet.**
[PLAN.md](PLAN.md) defines the
sequential gates; [TODO.md](TODO.md) records verified progress. Start with the
[reading index](docs/reading/README.md) for the eight core papers and source
limitations. The [protocol draft](config/eval_protocol.yaml) and its
[design notes](config/README.md) specify targets, OOD recipes, and unresolved
freeze requirements. Next is T1.3, IV inversion with explicit failure masks.
Track B and the demo are optional.

## Setup and verification

Python 3.11 or newer is required. From a clean checkout:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -e '.[dev]'
make check
```

`make check` runs Ruff formatting and lint checks, strict mypy, and pytest.
`make test`, `make lint`, and `make typecheck` run them separately. To use another
environment, pass `make check PYTHON=/absolute/path/to/python`. Dependencies are
declared in `pyproject.toml`; exact experiment environments must also be recorded
with future research artifacts.

The package uses a `src/` layout. Install it before running tests; this checks
the installed package rather than relying on a test-only import-path override.

## Black–Scholes pricing and Greeks

After installation, use the dividend-adjusted European pricer directly:

```python
from nop.pricing.bs import black_scholes_greeks, black_scholes_price

price = black_scholes_price(100.0, 100.0, 1.0, 0.2, rate=0.05)
greeks = black_scholes_greeks(100.0, 100.0, 1.0, 0.2, rate=0.05)
print(price.item())  # approximately 10.4505835722
print(greeks.delta.item())  # approximately 0.6368306512
```

Set `option_type="put"` for puts and `dividend_yield` for a continuous dividend
yield. Maturity is in years; rates, yields, and volatility are decimal annual
inputs. Vega and rho differentiate one decimal input unit; multiply by 0.01 for
a one-percentage-point sensitivity. Theta is `-dV/dT` per year with spot and
strike fixed. Delta and gamma are the first and second spot derivatives.

Scalars return CPU float64 tensors. Tensor inputs broadcast, preserve autograd,
and must share a float32 or float64 dtype and device; scalar arguments adopt
that dtype/device. Prefer float64 for reference calculations. Prices support
expiry and zero volatility via exact limiting payoffs. Analytic Greeks require
positive maturity and volatility; boundary autograd derivatives are not promised.
Nonfinite inputs and invalid domains raise errors. Very small tail prices may
still underflow at the selected precision. Nonfinite outputs or negative computed
prices raise `FloatingPointError`; float32 cancellation can trigger this at very
small total volatility. Use float64 for those cases; prices are never clipped to
hide cancellation.

Run `.venv/bin/python -m pytest tests/test_bs.py` for the focused checks or
`make check` for the full suite. Tests include published values from
[QuantLib's Haug fixtures](https://github.com/lballabio/QuantLib/blob/v1.38/test-suite/europeanoption.cpp),
independent payoff integration, dividend-adjusted parity, finite differences,
and first/second autodiff checks. Greek conventions were cross-checked against
[QuantLib's implementation](https://github.com/lballabio/QuantLib/blob/v1.38/ql/pricingengines/blackcalculator.cpp).
These verify Black–Scholes behavior on the tested cases. Heston checks are
documented below; IV inversion and accelerator verification remain pending.

## Heston COS pricing

`nop.pricing.heston_cos` provides a batched PyTorch characteristic function,
log-return cumulants, and European call/put prices. Choose `n_terms` and
`truncation_width` explicitly and verify convergence for the required quotes:

```python
from nop.pricing.heston_cos import heston_cos_price

price = heston_cos_price(
    1.0, 1.0, 1.0, 0.04, 1.5, 0.04, 0.3, -0.7,
    rate=0.02, dividend_yield=0.01, n_terms=4096, truncation_width=28.0,
)
```

The argument order is spot, strike, maturity, v0, kappa, theta, sigma, rho,
followed by optional rate/dividend yield. Run
`.venv/bin/python scripts/validate_heston.py` to reproduce the configured
convergence and independent Monte Carlo checks; matching results are cached.
[Validation notes](docs/heston-validation.md) record methods, observed results,
the failed initial MC run, and limits. Passing those fixtures does not certify
all future labels or IV errors. No experiment configuration is frozen yet.

## Reproducible randomness

Use `nop.reproducibility.seed_everything(seed)` before constructing models or
datasets, and `nop.reproducibility.seed_worker` as the PyTorch DataLoader worker
callback. Pass an explicitly seeded `torch.Generator` to each loader. The utility
seeds Python, NumPy's legacy random generator, and PyTorch, and enables
deterministic algorithms. Independently created NumPy `Generator` objects need
their own explicit seeds.

Repeatability is scoped to a fixed environment and device. Record Python,
dependency versions, hardware, seed, and configuration hash with future artifacts.
Python hash randomization and CUDA's cuBLAS workspace settings are launch-time
concerns; the utility does not pretend to change them retroactively.

## Commands and gate boundaries

| Command | Available behavior |
| --- | --- |
| `make install` | Install the package and development tools |
| `make check` | Formatting, lint, typing, and tests |
| `make data` | Exits with an explicit pending-ticket error until T1.4 |
| `make final-eval` | Always refuses at this stage; evaluation and frozen-hash verification arrive at T3.4 |
| `make reproduce` | Exits with an explicit pending-ticket error until T7.2 |

No placeholder command generates data or results. Gate 1 must validate the
Heston pricer before labels are generated. Final evaluation may only open test
and OOD sets after the Gate 3 freeze and successful frozen-hash verification.
Training and model selection must use training and validation data only.

## Layout

- `src/nop/`: pricing, data, models, training, evaluation, and calibration modules.
- `config/`: draft evaluation protocol and future experiment configurations;
  `frozen/`: future frozen-file hashes.
- `scripts/`: reproducible entry points; `tests/`: automated checks.
- `docs/`: decisions, deviations, reading notes, and interview preparation.
- `reports/`: future report, tables, and figures; `notebooks/`: figures only.
- `data/`: generated datasets, excluded from Git.

## Limitations

Heston already has a fast semi-analytic pricer. The potential value of a surrogate
is differentiability, batched calibration and risk loops, and a template for
models without closed-form pricers. Speed and accuracy improvements must be
measured. Calibration will use synthetic surfaces; real-market performance and
tradable profit are not claims of this project.

The draft assigns later Greeks/arbitrage diagnostics and calibration to validation
parameters to preserve the single final holdout pass. These will be labeled as
validation analyses, not independent test evidence; the sequencing interpretation
and its limits are recorded in [DEVIATIONS.md](docs/DEVIATIONS.md).
