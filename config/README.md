# Experiment configuration

[eval_protocol.yaml](eval_protocol.yaml) is the T0.3 draft. It records proposed
metrics, thresholds, OOD definitions, comparison budgets, and access rules before
any labels or model results exist. It is a design document, not an executable
configuration yet. `make check` tests the existing scaffold, not enforcement of
this protocol. No configuration is frozen at this stage.

T1.4 will create `data.yaml`; T1.5 will freeze it and this protocol only after the
`required_before_G1_freeze` list is resolved. Model configurations and selected
checkpoints are frozen at T3.4. No test/OOD data may inform these choices.

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

## Proposed data and OOD definitions

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
Seeds and offsets specify recipes, not existing datasets. No reference quote
failure rate has been measured. The proposed Gate 1 stop threshold is 3% overall
or at any maturity on training/validation data.

For query generalization, train on alternating strike coordinates at every native
maturity and evaluate the intervening coordinates. This is a held-out-strike
experiment; unseen maturities are assessed separately by short-maturity OOD.
M0 uses a natural cubic spline of total variance in k. Its short-maturity result
uses linear total-variance extrapolation in T at fixed k and must be labeled
`M0_plus_extrapolation`, using the full-grid headline checkpoint. The held-out
strike experiment uses a separate subset-quote checkpoint. Nonpositive
extrapolated variance is a failure, not clipped away. Off-grid normalization and
adapter validation remain freeze blockers.

## Holdout access and later diagnostics

The supplied plan has two sequencing ambiguities, recorded in
[DEVIATIONS.md](../docs/DEVIATIONS.md). This draft preserves the stricter access
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
held-out-quote error separately. Proposed noise-dependent fit tolerances are design
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

## Before freezing

Resolve the YAML's explicit checklist: validated COS and IV settings, failure
coverage, matching data recipes, numerical training/search budgets, off-grid
normalization, derivative steps, arbitrage tolerances, and calibration cases,
quote partitions, optimizer caps, and C2 training budget. Verify these using only
permitted development data. No loader, sampler, metric, or evaluator implementation
is part of T0.3; those belong to later tickets.
