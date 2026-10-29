# T1.3 — IV inversion and explicit failure masks

The solver passes the stated price-roundtrip criterion on accepted numerical
fixtures. **The wider corner study exposes unresolved COS accuracy and coverage
issues; these settings are not ready for production labels or Gate 1 freeze.**
No training, validation, test, or OOD dataset was generated or accessed.

## Reproduce

After installing the project:

```sh
.venv/bin/python -m pytest tests/test_iv.py
.venv/bin/python scripts/validate_iv.py
make check
```

The script reads [iv_numerics.yaml](../config/iv_numerics.yaml) and writes
`reports/tables/iv_validation.json`, excluded from Git. Identical config, source,
and environment hashes reuse the completed report, including failed studies.
Every attempted quote remains in the report with its status and available
diagnostics; nonfinite numbers serialize as null. Config and source hashes,
Git HEAD, versions, hardware, precision, and thread count are recorded.
There is no RNG: the seed field is null and the sampling rule is explicit.

## Solver contract and numerical policy

Use `IVSettings` and `implied_volatility` from `nop.pricing.iv`. All six settings
are explicit. For discounted spot A = S exp(-qT) and strike B = K exp(-rT),
the price scale is max(A, B). Static price bounds are [max(A-B, 0), A] for calls
and [max(B-A, 0), B] for puts. Prices strictly outside the bounds are rejected;
no price is clipped or replaced.

The study uses a price tolerance of `1e-10 * scale`, IV bracket-width tolerance
`1e-10` decimal, annual volatility bracket [0, 5], and at most 100 bisections.
Acceptance requires both price residual and bracket width to meet tolerance,
followed by a normalized-vega check: `vega / scale >= 1e-4`. Vega differentiates
decimal volatility, not vol points. A failed quote returns NaN IV. Scalar calls
use CPU float64; tensors preserve their common float32/float64 dtype and device.
Float32 needs coarser tolerances. Outputs are detached; no IV autodiff is promised.

| Status | Meaning |
| --- | --- |
| SUCCESS | Both convergence tests and the vega floor pass |
| NONFINITE_INPUT | At least one quote input is NaN or infinite |
| INVALID_DOMAIN | Nonpositive spot/strike or negative maturity |
| EXPIRED | Zero maturity does not identify volatility |
| PRICE_OUT_OF_BOUNDS | Price violates a static bound, even within tolerance |
| NEAR_LOWER_BOUND | Price lies within tolerance above the lower bound |
| NEAR_UPPER_BOUND | Price lies within tolerance below the upper bound |
| UNBRACKETED | The finite upper volatility cannot reach the target price |
| LOW_VEGA | A converged root fails the conditioning floor |
| NO_CONVERGENCE | Iteration budget exhausted or arithmetic stagnated |
| NUMERICAL_ERROR | Discounting, pricing, or Greeks cannot be evaluated reliably |

Input checks precede bounds, which precede solving and vega checks. When both
near-bound tests apply, the lower-bound status takes precedence. Malformed
settings, unsupported dtypes, incompatible devices/shapes, and an unknown option
type raise structural errors. A numeric failure is isolated to its quote, even
when the underlying BS API raises for a whole batch. Available signed residuals,
iteration counts, and converged-root vegas are retained for diagnosis.

Tests cover known BS prices, recovery and repricing across broadcast inputs,
negative rates/dividends, strict bounds, low vega, missing brackets, convergence
conditions, float32 stagnation, extreme-scale numerical failures, empty batches,
and detached outputs. No dependency or existing pricer implementation changed.

## Deterministic box study, 2026-10-28

Fixtures are the 32 corners and arithmetic center of the plan's five-dimensional
ID box. Each uses all eight maturities and 15 scaled-moneyness coordinates, for
3,960 calls and 3,960 puts. Spot is 1, rate 0.02, yield 0, and
K = S exp((r-q)T) exp(z * 0.2 * sqrt(T)). These fixtures do not use a research
split or its Sobol stream. Their failure fractions are not population estimates.

Selected COS settings remain N=4096/L=28; the comparison uses N=8192/L=36.
The word "reference" labels the comparison resolution, not exact truth.

| Outcome | Count | Fraction of all 7,920 attempts |
| --- | ---: | ---: |
| Accepted | 7,436 | 93.8889% |
| Low vega | 208 | 2.6263% |
| Near lower bound | 92 | 1.1616% |
| Price out of bounds | 92 | 1.1616% |
| Upstream selected-COS failure | 92 | 1.1616% |
| Total failed | 484 | 6.1111% |

Calls and puts each accept 3,718/3,960. The 92 upstream failures are negative
computed call prices rejected by COS; the corresponding puts violate intrinsic
bounds at inversion. These are preserved as failures, not rounded to intrinsic.
The finer COS run has 72 failures, counted separately from selected-price IV
coverage. Price comparisons require both evaluations: 7,824 comparable quotes.

| Absolute error / max(A, B) | p99 | Maximum | Denominator |
| --- | ---: | ---: | ---: |
| Accepted BS price roundtrip | 2.8712e-11 | 3.9619e-11 | 7,436 |
| Selected versus finer COS price | 1.1056e-6 | 2.3416e-6 | 7,824 |

All accepted roundtrips pass 1e-10. The script's `passed` field means only this
roundtrip condition with at least one accepted quote; it does **not** approve
source-price accuracy or a freeze. Percentiles use NumPy's linear quantile rule.

| Maturity (years) | Failed / 990 | Failure fraction |
| --- | ---: | ---: |
| 1/12 | 74 | 7.4747% |
| 2/12 | 66 | 6.6667% |
| 3/12 | 62 | 6.2626% |
| 6/12 | 56 | 5.6566% |
| 9/12 | 56 | 5.6566% |
| 1 | 56 | 5.6566% |
| 1.5 | 56 | 5.6566% |
| 2 | 58 | 5.8586% |

For ascending z indices 0–14, failures out of 528 are
`[32, 10, 0, 0, 0, 0, 0, 0, 0, 12, 42, 62, 80, 114, 132]`.
Thus the z=2.5 wing loses 25% of these corner fixtures; restricting attention
to overall averages would hide that concentration. The JSON includes status
counts and error tails for each maturity and z index.

Observed with CPU float64, macOS 27.0.1 arm64, Python 3.12.7, NumPy 2.5.4,
PyTorch 2.14.1, PyYAML 6.0.3, and six PyTorch threads. Config SHA-256:
`cfc93551d31a7665d3e4601d27f605567b5aff07888374f004a37f6b33f92a7c`.
The report records the actual source-content hash because these checks ran
before committing T1.3. Accelerator execution remains unverified.

## Limits and next gate

A tiny roundtrip residual establishes inversion of the supplied price, not
accurate Heston IV. On these fixtures, 308/7,436 accepted quotes have a selected
versus finer COS discrepancy above 1e-8 times scale. The largest discrepancy is
at v0=theta=0.01, kappa=0.5, sigma=1, rho=-0.95, T=2, approximately forward ATM.
The narrower T1.2 fixture study did not establish uniform accuracy over this box.
Neither a small price discrepancy nor agreement between two resolutions proves
an error bound; both resolutions can share bias.

First-order IV sensitivity is approximately price uncertainty divided by vega.
With the chosen floor, assumed source uncertainty of 1e-8 times scale could
already imply 1e-4 decimal IV (0.01 vol points); the assumption is not established
across this box. The larger observed discrepancies make that caveat material.
The vega floor is a conditioning choice, not a label-accuracy certificate.

Before production label generation in T1.4, investigate COS convergence on the
exposed difficult fixtures and establish a defensible price/IV accuracy policy.
Retain the present report and settings. Training/validation failure coverage
and the draft 3% overall/per-maturity stop threshold remain to be checked before
T1.5 freeze. The deterministic 6.11% stress rate neither estimates that population
rate nor clears the stop rule. Do not freeze these settings or narrow the grid
silently in response to the observed failures.
