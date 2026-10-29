# Interview preparation

## T0.1 — Scaffold and reproducibility

**Why is setting one seed insufficient?** Python, NumPy, PyTorch, and each
DataLoader worker maintain distinct random-number state. Seed each source,
pass a seeded generator to loaders, and explicitly seed independently created
NumPy generators. Fixing seeds also does not remove nondeterministic kernels
or differences between hardware and library versions.

**What does `make check` establish?** It checks formatting, lint, types, and
automated behavior of the implemented scaffold, including repeated random
streams and safe rejection of pending research commands. It does not validate
option prices or research claims; those have explicit later gate criteria.

**How do you prevent accidental test-set leakage?** Keep all development
decisions on training and validation data. Restrict holdout access to the final
evaluation script, require frozen-file hash verification before any holdout is
opened, and record deviations. At T0.1 the command refuses every evaluation;
the actual hash-verifying evaluator is still pending T3.4.

## T0.2 — Literature and experimental design

**Why use COS and the stable Heston representation together?** COS supplies a
fast characteristic-function-based route to labels, but quadrature convergence
cannot fix an incorrectly evaluated characteristic function. The Heston branch
issue and the COS truncation/series errors are different failure modes. Gate 1
must check both, with the Black–Scholes limit, parity, convergence studies, and
an independent Monte Carlo comparison before generating training data. See the
[COS](reading/01-cos-method.md) and [Heston](reading/02-little-heston-trap.md) notes.

**Is M1 actually an operator-learning experiment?** Its branch takes five Heston
parameters and its trunk takes contract coordinates. It is a finite-parameter
surrogate with a DeepONet-style factorization, so the original function-space
theorem is not evidence that this model will train or extrapolate successfully.
The optional payoff-function-to-price-function experiment would have the more
direct operator interpretation. See the [DeepONet note](reading/06-deeponet.md).

**Why not promise published neural-network or Mamba speedups?** Timing depends on
the reference solver, complete task, hardware, precision, batching, and kernel
implementation. Speed against Monte Carlo or an IV root finder does not establish
speed against vectorized COS plus IV inversion. Likewise, a transparent PyTorch
selective recurrence is not the paper's fused scan. Measure our implementation
under the frozen protocol and report losses as well as wins. See the
[neural-pricing](reading/04-neural-pricing-and-iv.md) and
[Mamba](reading/08-mamba.md) notes.

## T0.3 — Evaluation protocol draft

**Why commit a draft before freezing the protocol?** Committing establishes the
proposed metrics, targets, budgets, and OOD definitions before model outcomes can
influence them. Gate 1 still needs validated pricer/IV tolerances, mask coverage,
and complete numerical settings. A draft is not an enforceable freeze; the
explicit checklist prevents unresolved settings from being mistaken for one.

**What makes these datasets out of distribution?** The wider-parameter and
corner recipes explicitly reject points inside the closed training box, so
expanding a sampling range cannot quietly mix ID samples into the OOD score.
Short-maturity OOD keeps ID parameters but evaluates unseen maturities. Report
each set separately because these shifts test different generalization claims.

**Why use a common failure mask and per-seed metrics?** A common reference mask
keeps every model accountable for the same valid quotes. Invalid predictions
must be reported as failures, not removed to improve accuracy. Compute metrics
within each seed, then mean and sample standard deviation; pooling predictions
or selecting a best seed conceals run variability. Validation-only later
diagnostics must also disclose their model-selection exposure.

## T1.1 — Black–Scholes prices and Greeks

**Why is put–call parity insufficient to validate a pricer?** Both prices can
share the same error and still satisfy their difference exactly, especially
when one leg is reconstructed by parity. Check independent published prices
and direct risk-neutral payoff integration as well. Parity remains useful for
detecting dividend, discounting, and call/put sign mistakes.

**What are the Greek conventions?** Delta/gamma differentiate spot once/twice.
Vega and rho differentiate decimal volatility/rate, so multiplying by 0.01 gives
the sensitivity to a percentage-point change. Theta is negative maturity
derivative per year with spot and strike fixed. Finite differences and autodiff
verify both signs and units, including nonzero dividends and negative rates.

**How do you handle numerical tails and degenerate inputs?** Use `erfc` for
normal lower tails and price the out-of-the-money leg before recovering its
counterpart by parity. At expiry or zero volatility use exact limiting prices
with safe unused regular branches. Reject analytic Greeks there because some
derivatives are undefined or singular; finite precision can still underflow in
extreme tails. Cancellation can also produce negative float32 prices at tiny
total volatility; raise a numerical error rather than clipping, and use float64
references. Do not claim arbitrary-precision accuracy.

## T1.2 — Heston COS and independent validation

**Why use the Little Heston Trap form?** Algebraically equivalent complex-log
representations can cross different branch cuts in floating-point evaluation.
The decaying-exponential representation avoids the unstable branch behavior;
additional rearrangements remove small-sigma cancellation. Independent Riccati
integration tests the resulting CF rather than merely checking its normalization.

**Why vary both the COS term count and interval width?** More cosine terms reduce
series error on a fixed interval but cannot restore excluded tail probability.
A wider interval reduces truncation error but may need more terms for the same
resolution. Measure both, then cross-check independent prices; agreement between
two resolutions of the same formula alone cannot rule out a shared error.

**What does Monte Carlo agreement within a confidence interval establish?** It
shows consistency with sampling uncertainty for that simulation and numerical
case. The interval does not include timestep bias. Use step refinement and
compute standard errors from independent antithetic pair means, not individual
correlated paths. Preserve failed checks and their settings; increasing numerical
effort is different from retrying seeds or widening intervals until a test passes.

## T1.3 — IV inversion and failure masks

**Why use bisection rather than ordinary Newton iteration?** For a valid quote
and positive maturity, the Black–Scholes price increases with volatility.
A verified finite bracket therefore supports deterministic interval reduction.
Newton divides by vega and can leave the admissible interval or take enormous
steps when vega is small. Bisection costs more evaluations but has a simple
failure contract; require both price residual and interval width to converge.

**Does an accurate price roundtrip imply an accurate Heston IV label?** No.
It verifies inversion of the supplied price. Source-pricer error remains, and
its first-order effect on IV is price error divided by vega. Near-zero time
value or small vega makes this inversion poorly conditioned. Report numerical
convergence, mask coverage, and source-price uncertainty separately rather than
using a small solver residual as evidence of accurate ground truth.

**Why retain invalid quotes instead of dropping them?** Dropping them silently
changes the sampled problem and can hide errors in difficult regions. Keep the
full shape, explicit reason codes, and NaN IVs; use the common reference mask
for every model. Report all attempted quotes in the denominator, including
upstream pricing failures, with maturity/moneyness breakdowns. Deterministic
corner-fixture failure fractions are stress diagnostics, not population rates.
