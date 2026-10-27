# Neural surrogates for option pricing and calibration

This research project will compare MLP, DeepONet-style, Transformer, and SSM
surrogates for synthetic Heston implied-volatility surfaces, then evaluate their
use in calibration. Accuracy, tail errors, invalid quotes, arbitrage, Greeks,
speed, and variability across seeds are part of the planned comparison.

**Status: Gate 0, T0.1 scaffold. No pricer, dataset, trained model, or research
result exists yet.** [PLAN.md](PLAN.md) defines the sequential gates;
[TODO.md](TODO.md) records verified progress. Track B and the demo are optional.

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
- `config/`: future experiment YAML files; `frozen/`: future frozen-file hashes.
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
