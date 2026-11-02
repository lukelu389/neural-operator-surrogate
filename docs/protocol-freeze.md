# Gate 1 protocol v1

T1.5 completes the preregistration after numerical validation and inspection of
training/validation label coverage, before any model training or test/OOD access.
The effective freeze is the committed `frozen/manifest.json`, not the YAML status
field alone. It records file hashes and Gate 1 evidence; it does not authorize
final evaluation. Model configurations and all final checkpoints still require
the separate Gate 3 freeze.

## Fixed comparisons

`config/training.yaml` gives M0–M3 twelve paired random-search trials, the same
AdamW search ranges, batch size 256, 100-epoch cap, five-epoch warmup, cosine
schedule, norm clipping, ten-epoch early-stopping patience, and final seeds
11/29/47. Architectural widths may be set using parameter counts and shape tests,
without label scores; each architecture is fixed before its first trial. Failed
trials count. Model selection uses validation only. Vega-weighted, data-efficiency,
and withheld-coordinate runs reuse selected hyperparameters without extra tuning.
The price-error ablation weights standardized residuals by `(vega * IV_std)^2`,
normalized by one fixed valid-training-quote mean.

Coordinate normalization interpolates training means and log standard deviations
bilinearly in `(z,T)`. Compute z from the actual query T before clamping the moment
lookup independently to each training axis. The model still receives the actual
query coordinates. Withheld-coordinate models use only their observed-coordinate
training moments. The M0 adapter interpolates total variance with natural cubic
splines in forward k and linearly in T; short-T extrapolation uses the first two
rows at fixed k. Strike and long-T extrapolation are forbidden. Invalid native
predictions invalidate the affected spline row; adapter failures never shrink the
common reference mask. Short-T extrapolation may fail and carries no accuracy
guarantee.

`config/diagnostics.yaml` fixes finite-difference steps and refinement reliability,
as well as dimensioned butterfly/calendar tolerances. IV sensitivities use the
first 32 validation surfaces at midpoints of the native T/z axes, avoiding
normalization knots. Stencils must stay in the same normalization cell. For
coordinate-dependent models, other query coordinates remain fixed when taking an
output's derivative. Arbitrage checks cover the full validation grid and report
excluded checks. These later analyses use validation, not independent test data. M0 parameter
sensitivities differentiate through its spline adapter at the same coordinates
and are labeled `M0_plus_interpolation`; M0 coordinate sensitivities are excluded.

`config/calibration.yaml` fixes 32 validation surface IDs, disjoint fitting/held-out
strikes, all six noise/missingness conditions, three paired model/experiment seeds,
and eight common starts. C0 and C1 use bounded trust-region least squares with
finite-difference versus autodiff Jacobians. Both have the same residual-call cap,
including finite-difference calls, and Jacobian cap. Score selected parameters
with COS; never use COS to choose a C1 start. C2's architecture, augmentation,
training cap, and separate validation-selection rows are fixed with no tuning.
The sample is small: report counts and seed variability without claiming broad
statistical power or real-market performance.

MLflow logging and SciPy least squares are preregistered for their implementation
tickets. They are not installed or used in T1.5; no dependency is added by the
freeze itself.

## Provenance migration

T1.4 embedded the whole draft protocol hash in every archive. Completing the
protocol changes that hash even though the data recipes and numerical labels are
unchanged. `scripts/rebind_data_protocol.py` therefore requires identical old/new
`data_design` and `units`, unchanged generator/config/environment provenance, a
complete verified source, and no existing frozen JSON manifest. It publishes a
new directory atomically, replacing only embedded `protocol_hash` metadata.

It verifies every other array's dtype, shape, raw bytes (including NaN payloads),
and saved NPY payload, then rebuilds archive and combined hashes. The separate
`protocol_migration.json` records old/new identities and per-array evidence. The
original T1.4 directory and historical `docs/data-hashes.json` remain preserved.
A later cached `make data` must reproduce the migrated hash and normalization
before freeze. This is not a new cold numerical generation or new label study.

## Run and verify

After installation, run:

```sh
make check
.venv/bin/python scripts/validate_diagnostics.py
.venv/bin/python scripts/freeze_protocol.py --verify
```

On a clean checkout, `make data` creates training/validation directly with the
frozen protocol; it does not need a metadata migration. Generated data and
numerical reports remain outside Git. Strict cache identity is environment-bound:
a different dependency/platform identity can produce different archive hashes
and must be reported, rather than silently treated as an identical reproduction.
`make final-eval` continues to refuse until T3.4.

A frozen-file change requires Luke's approval and a deviation record. Optional
Track B/demo are unchanged. Gate 4's vectorized GPU COS benchmark remains required
and must be marked incomplete if suitable hardware is unavailable.

## Recorded freeze evidence

- Protocol SHA-256: `da1f11c2ea44b2c28efe41b868c73f5877f28280b707a917516531aba29658af`.
- Migrated dataset SHA-256: `9120ef79ac2c385618cad182ae55d18212b34a591a0579c864fb0d58874797d7`.
- Metadata-migration proof SHA-256: `6bac970f8474c33e4ec23b730325a1c93757806dbb7d59a2b043c189ffad6cd2`.
- `make check`: 595 passed, one CUDA skip; Ruff and strict mypy pass.
- Diagnostic fixtures: 21/21 valid and reliable stencils; maximum refinement
  difference `1.5158245030288953e-6` decimal IV per differentiated unit.
  All 36 butterfly and 45 calendar BS checks, plus four constructed tolerance
  cases, pass. These fixture results do not certify all later model derivatives.

The full cached replay preserved the new hash, then the migration verifier
rechecked all payloads against the original directory. No model or holdout was
used. The manifest records actual train/validation mask counts and report hashes.
