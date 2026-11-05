# Experiment configuration

[eval_protocol.yaml](eval_protocol.yaml) is the T1.5 protocol v1. The committed
[frozen manifest](../frozen/manifest.json) records the effective Gate 1 freeze.
[training.yaml](training.yaml), [diagnostics.yaml](diagnostics.yaml), and
[calibration.yaml](calibration.yaml) complete the shared budgets and numerical
policies. See the [freeze record](../docs/protocol-freeze.md) for verification,
metadata migration, and remaining gate restrictions. Model configurations and
selected checkpoints remain unfrozen until T3.4; no test/OOD data informs these
choices.

`m0.yaml` through `m2.yaml` hold fixed architectures, each committed before its
first tuning trial. M2's continuous query encoding uses normalized raw
coordinates plus sine/cosine Fourier features at 1, 2, 4, and 8 cycles. It
applies full bidirectional attention jointly to the five parameter tokens and
the complete submitted query set, then reads only query-token outputs. The
model is permutation-equivariant over that set; compare coordinates using the
same query set. Run the M2 validation budget with
`.venv/bin/python scripts/train_m2.py`.

[heston_numerics.yaml](heston_numerics.yaml) configures T1.2's reproducible COS
convergence and Monte Carlo checks. It contains explicit numerical fixtures,
not experiment splits or frozen data-production settings. Run it with
`.venv/bin/python scripts/validate_heston.py`; see the
[validation record](../docs/heston-validation.md) for results and limits.

[iv_numerics.yaml](iv_numerics.yaml) configures T1.3's deterministic box-corner
and center diagnostic, including both calls and puts. Run
`.venv/bin/python scripts/validate_iv.py`; the [IV validation record](../docs/iv-validation.md)
reports masks, price roundtrips, and selected/reference COS discrepancies.
This is a numerical study, not a training/validation sample or frozen IV policy.

[data.yaml](data.yaml) implements the training/validation recipes and
records holdout recipes without generating them. It sets adaptive COS effort,
per-label convergence checks, IV masks, and deterministic chunk/worker settings.
[data_numerics.yaml](data_numerics.yaml) controls the wider fixed-resolution
prerequisite study; `scripts/validate_adaptive_labels.py` independently checks
the adaptive grid implementation against fixed quote pricing. `make data`
requires both numerical checks to pass with matching cache identities.

The Gate 1 manifest covers six configurations plus the coordinate-normalization
and M0-interpolation implementations. Model configurations and selected
checkpoints remain unfrozen until T3.4. No test/OOD data informs these choices.

## Targets, units, and comparison rules

IV is stored in decimal units; errors are multiplied by 100 to obtain vol points.
The plan's ID targets are MAE at most 0.05 vol points (0.0005 decimal IV) and p99
absolute error at most 0.3 vol points (0.003 decimal IV). Assess the architecture
selected on validation using the mean of each metric across three final seeds;
report individual seeds and sample standard deviations as well. These are goals,
not measured results. Greek errors and arbitrage rates have no success threshold.

All architectures get the same data, masks, optimizer family, training budget,
12 tuning trials, and final seeds. Failed training trials count toward the budget.
Reference failures use a common mask; a model's nonfinite or nonpositive IV
prediction cannot remove a difficult quote from its score. Report prediction
failures and mark affected metrics undefined, failing target assessment.

## Data and OOD definitions

Use the plan's starting parameter box, 8 by 15 scaled grid, and split sizes.
Do not impose the Feller condition: the experiment intentionally includes that
part of the stated box, subject to pricer validation and explicit failure masks.
Here `k = log(K/F(T))`, with `F(T) = S0 * exp((r-q)T)`; the grid defines k from
z and maturity. Calendar comparisons hold k fixed, not the z index.

| OOD set | Definition |
| --- | --- |
| Wider parameters | Expand kappa to [0.1, 7.5] and sigma to [0.1, 1.5]; retain only samples outside the closed training box |
| Short maturities | ID parameters at T = 1/52 and 2/52 years, below the shortest training maturity |
| Parameter corners | kappa in [0.1, 0.5], sigma in [1.0, 1.5], rho in [-0.99, -0.95]; exclude the shared ID boundary |

The wider and corner sets need not be mutually disjoint; each is reported
separately. Rejection retains the first accepted Sobol points without reseeding.
Holdout seeds and offsets remain recipes, not existing holdout datasets.
Training/validation coverage is recorded in the [data card](../docs/data-card.md).
The Gate 1 stop threshold is 3% overall
or at any maturity on training/validation data.

For query generalization, train on alternating strike coordinates at every native
maturity and evaluate the intervening coordinates. This is a held-out-strike
experiment; unseen maturities are assessed separately by short-maturity OOD.
M0 uses a natural cubic spline of total variance in k. Its short-maturity result
uses linear total-variance extrapolation in T at fixed k and must be labeled
`M0_plus_extrapolation`, using the full-grid headline checkpoint. The held-out
strike experiment uses a separate subset-quote checkpoint. Nonpositive
extrapolated variance is a failure, not clipped away. Train-only off-grid normalization and the adapter are tested; see the freeze record.

## Holdout access and later diagnostics

The supplied plan has two sequencing ambiguities, recorded in
[DEVIATIONS.md](../docs/DEVIATIONS.md). The protocol preserves the stricter access
rule: T1.4 materializes training/validation only and records test/OOD recipes;
`scripts/final_eval.py` alone materializes and reads holdouts after Gate 3 hash
verification. The sampler must not generate a larger prefix containing test
samples during development. Freeze recipe hashes first and record actual holdout
artifact hashes during final evaluation. Subsequent reproduction reads cached
reports rather than reopening holdouts.

Gate 4 diagnostics and Gate 5 calibration use validation parameters and must be
labeled as such. Those parameters are excluded from training but exposed to model
selection, so these are not independent test results. Calibration compares all
methods against the same observations, masks, and parameter bounds; C0/C1 share
starts and start budgets. Score recovered parameters with validated COS for every
method, including C2. Report noisy observed fit, latent surface error, and disjoint
held-out-quote error separately. Noise-dependent fit tolerances are design
choices, not confidence bounds or success guarantees.

C2 trains and fits normalization on training surfaces only. A calibration succeeds
only with valid in-bounds parameters, no optimizer/pricing/inversion failure on
scored quotes, and COS observed-fit RMSE within the shared tolerance. Compare each
surrogate's across-seed mean success rate with C0 separately for every noise and
missingness condition. Derivative diagnostics measure IV sensitivities: the
maturity derivative holds forward k and parameters fixed, so strike changes with
maturity. These are not conventional option-price Greeks.

Requiring holdout artifacts before Gate 3, or independent test diagnostics after
the single final pass, would require resolving the sequencing conflict before
freeze. Do not silently bypass access rules or claim a stronger evaluation.

## Verify the freeze

Run `.venv/bin/python scripts/freeze_protocol.py --verify` and `make check`.
The verifier checks every committed frozen-file hash without requiring generated
artifacts. Creation checks additionally require complete train/validation coverage
and current passing numerical reports. Final evaluation remains unavailable until
T3.4; Gate 1 does not authorize holdout access. Changes to frozen files require
Luke's approval and a deviation record.
