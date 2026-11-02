# T1.4 — Synthetic Heston IV data card

This dataset is synthetic, with no market observations or trading claims.
The data configuration and evaluation protocol remain unfrozen. T1.5 must still
complete every protocol-freeze requirement; T1.4 does not authorize training.

## Recipe and access boundaries

- Parameter order: v0, kappa, theta, sigma, rho.
- Box: v0/theta [0.01, 0.25], kappa [0.5, 5], sigma [0.1, 1], rho [-0.95, -0.1].
- Sampling: one scrambled PyTorch Sobol stream, seed 1729, no Feller filtering.
- Train: 200,000 surfaces, indices [0, 200000). Validation: 20,000, [200000, 220000).
- Maturities: 1/12, 2/12, 3/12, 6/12, 9/12, 1, 1.5, 2 years.
- Fifteen z coordinates from -2.5 to 2.5; k = 0.2 z sqrt(T), K = F(T) exp(k).
- Spot 1, continuously compounded rate 0.02, dividend yield 0. Native order is maturity then z.
- Targets: decimal annual IV, derived from European puts. One vol point is 0.01 decimal IV.

The CLI checks exact agreement with `eval_protocol.yaml` before any generation.
Train and validation windows must end before the test window; the low-level
generator also rejects overlapping windows. Sampling seeks directly to each
requested chunk and never draws a prefix extending into test. The test window
[220000, 240000) and the three OOD recipes are recorded only. Their materializer
remains reserved for `scripts/final_eval.py` after Gate 3 hash verification.
There is no development loader or CLI switch for those splits.

The counts follow the predeclared draft, rather than powers of two. Reproduction
is scoped to the recorded runtime, default tensor dtype, device, and thread
configuration. PyTorch's first Sobol point depends on its default floating dtype;
the generator requires its recorded float32 default while explicitly drawing
and storing float64 parameters. This does not imply IID random sampling.

## Numerical validation and label policy

T1.3 exposed both interval truncation and finite-series errors. Increasing N at
L=28 could not remove the worst-case interval bias. The new fixed study uses
N=16,384/L=64 versus N=32,768/L=80 on 33 deterministic corner/center fixtures,
all 120 quotes, both calls and puts. Among 7,444 accepted IVs of 7,920 attempts,
maximum normalized price discrepancy is 1.8903e-11 and maximum discrepancy/vega
is 4.5004e-8 decimal IV. All 476 failures remain reported. Independent Riccati
integration at the previously worst corner agrees with the characteristic
function to 1.4918e-12; refinement is checked separately. These fixture results
are not population estimates or rigorous universal error bounds.

The grid adapter reuses each parameter/maturity characteristic function across
strikes. It shares the tested numerical primitives with the original pricer;
regression tests compare the adapter against quote pricing. Existing BS/Heston
pricing implementations are unchanged.

For each production parameter surface:

1. Start at N=512; compare (N,L=64), (2N,L=64), and (2N,L=80).
2. Invert the finest price using the T1.3 IV settings: bracket [0,5], 100 iterations,
   price residual 1e-10 times max(discounted spot, discounted strike), IV interval
   width 1e-10, normalized vega at least 1e-4.
3. Require the maximum adjacent price discrepancy to be at most 1e-8 times that
   price scale and, for accepted IVs, at most 1e-5 times vega. Invalid-IV quotes
   require the stricter IV price tolerance for numerical resolution.
4. If any quote is unresolved, double N for that surface, through N=16,384.
   The finest possible evaluation therefore uses 32,768 terms at L=80.
5. Retain each final quote and its status, including COS failure/nonconvergence.
   Never replace failed IVs, clip prices, resample parameters, or drop surfaces.

On 3,960 deterministic put fixtures, the adaptive implementation accepted 3,722,
with zero validity-mask disagreements against independent fixed N=32,768/L=80
quote pricing. Maximum normalized price discrepancy was 2.133e-13; actual IV
differences were zero at the solver's resolution. The 238 masked quotes remain
in the denominator. This supports the adaptive policy on these fixtures only;
each production label still receives its own convergence checks.

Status codes 0–10 retain `IVStatus` meanings. Additional codes are 20
(`COS_FAILURE`) and 21 (`COS_NOT_CONVERGED`). Every failed IV is NaN. The stored
resolution discrepancy is a diagnostic, not a certified error bound. A shared
mask must be used across future architectures; model prediction failures must
not alter the reference mask.

## Normalization and storage

Parameters use per-parameter mean/population standard deviation fitted only on
training rows. IV targets use valid training quotes at each grid point, with
counts stored alongside mean/population standard deviation. Zero variance or
an entirely invalid training coordinate is an error. Validation never contributes
statistics. Transform/inverse-transform preserve NaNs and do not clip predictions.
Data-efficiency subsets and withheld-strike experiments must fit their own
permitted statistics later; the full-data statistics are not a substitute.

Each 256-surface chunk stores parameters, price, IV, status, vega, resolution
discrepancy, and finest term count in deterministic NPZ bytes. Fixed archive
metadata removes timestamp variation. Arrays remain float64/int64, with no
lossy conversion. Generation uses four processes, one CPU thread each, and
16-surface pricing batches. Only plain configuration is passed to workers;
small tensors are rebuilt locally, avoiding shared-memory IPC requirements.

Seed/config/source/environment identities are embedded in artifacts. Sidecar
hashes protect cache reuse, and the manifest records ordered chunk hashes plus
grid/normalization hashes and a combined dataset hash. Loading verifies canonical
paths, contiguous chunk counts, and hash agreement. Changed provenance or corrupt
bytes cause an explicit error. Generated data and reports are excluded from Git;
the configuration, scripts, and reproducibility hashes are version controlled.

## Run and verify

```sh
make data
make check
```

The first command runs or reuses both numerical prerequisite studies before
generation. It resumes valid chunks and verifies cache bytes on repeat calls.
To regenerate independently without overwriting the original, run
`.venv/bin/python scripts/make_data.py --output data/heston_v1_replay`.
The script exits with a stop message if training or validation invalid-quote
coverage exceeds 3% overall or at any maturity. Preserve that report and revisit
numerics/grid before freezing; do not silently relax the criterion.

Tests include two independent cold synthetic generations with identical hashes,
actual serial/parallel byte equality, train-only moments, cache corruption and
manifest/path checks, holdout refusal, and numerical prerequisite failures.
Small test fixtures use their own seeds and bounds; they are not research splits.

## Measured dataset coverage and reproduction

Full generation completed on 2026-11-01. There are 200,000 training and 20,000
validation surfaces; no parameter row was resampled or omitted. Both overall
and every-maturity failure fractions pass the predeclared 3% stop threshold.

| Split | Attempted quotes | Accepted | Masked | Masked fraction |
| --- | ---: | ---: | ---: | ---: |
| Train | 24,000,000 | 23,938,903 | 61,097 | 0.254571% |
| Validation | 2,400,000 | 2,393,937 | 6,063 | 0.252625% |

| Failure reason | Train | Validation |
| --- | ---: | ---: |
| LOW_VEGA | 51,838 | 5,170 |
| NEAR_LOWER_BOUND | 8,783 | 844 |
| PRICE_OUT_OF_BOUNDS | 476 | 49 |
| COS_FAILURE | 0 | 0 |
| COS_NOT_CONVERGED | 0 | 0 |

| Maturity (years) | Train masked | Validation masked |
| --- | ---: | ---: |
| 1/12 | 0.341700% | 0.335333% |
| 2/12 | 0.243633% | 0.246000% |
| 3/12 | 0.210067% | 0.209333% |
| 6/12 | 0.194533% | 0.195333% |
| 9/12 | 0.213667% | 0.212667% |
| 1 | 0.235267% | 0.234000% |
| 1.5 | 0.280567% | 0.277667% |
| 2 | 0.317133% | 0.310667% |

The worst coordinate is the shortest maturity at z=2.5: **3.0165%** masked in
training and **2.9150%** in validation. This is reported even though the declared
stop rule applies overall and per maturity, not per individual coordinate.
Across all maturities the z=2.5 wing loses 2.2086875%/2.1993750% respectively;
the full coordinate mask matrix remains in the manifest and committed hash record.

| Accepted-label discrepancy diagnostic | Train p99 | Train max | Validation p99 | Validation max |
| --- | ---: | ---: | ---: | ---: |
| Price / max(A,B) | 2.2124978e-09 | 9.9978668e-09 | 2.1772161e-09 | 9.9951416e-09 |
| Price discrepancy / vega (decimal IV) | 1.8114969e-08 | 9.9119354e-06 | 1.792509e-08 | 9.0467855e-06 |

These are resolution-comparison diagnostics, not errors against exact prices.
No generated quote required the finest 32768-term cap; 25 training surfaces
and one validation surface used 16384 terms. The complete effort histograms
are recorded with the hashes. Normalization uses 23,938,903 valid training quotes;
validation contributes none of its 2,393,937 accepted quotes to those statistics.

Combined dataset SHA-256:
`50df50f75c727f238e6db44cb57fe44d6173b8f0a80921de8dc0e2b5f452efc7`.
The [hash record](data-hashes.json) stores config/source/environment identities,
split summaries, and grid/normalization hashes. Environment: CPU float64,
macOS 27.0.1 arm64, Python 3.12.7, NumPy 2.5.4, PyTorch 2.14.1, PyYAML 6.0.3.
Accelerator generation remains unverified. This dataset is unfrozen; later
protocol changes require explicit provenance reconciliation before freeze.

A complete second `make data` invocation verified every cached chunk and rebuilt
normalization/summaries with the same combined hash. Four production chunks were
also regenerated cold: training starts 0, 1792 (the first chunk reaching the
largest observed term count), and 199936; validation start 19968. All 608 surfaces
produced byte-identical archives. Unit tests separately verify independent cold
synthetic datasets and serial/parallel identity. A second **full cold** 220,000-
surface generation was not run; cached replay and cold spot checks are reported
separately. `make check` passed 459 tests, Ruff, and strict mypy (25 source files);
one CUDA check was skipped because CUDA hardware is unavailable.
