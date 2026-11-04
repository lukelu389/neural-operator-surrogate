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

## T1.4 — Reproducible data and numerical coverage

**How do you make a dataset reproducible without leaking holdouts?** Declare
separate nonoverlapping Sobol windows, seek directly to train/validation slices,
and leave test/OOD as recipes until final evaluation. Record seeds, configuration,
source/environment identities, and hashes. Fixed archive metadata makes cold
serial/parallel output byte-identical; cache hashes detect corruption rather
than disguising it as a successful rerun.

**Why validate both N and interval width for every accepted label?** Series
resolution and tail truncation are different errors; refining only one can
appear converged while retaining bias. Compare both, relate price discrepancies
to vega, and verify the adaptive implementation against an independent fixed
quote-pricing path. These diagnostics support a numerical policy but are not
rigorous universal error bounds. Preserve unresolved quotes with explicit masks.

**What prevents normalization leakage or mask bias?** Fit parameter and per-grid
IV moments using training data only, retaining valid counts and rejecting
undefined statistics. Apply those fixed moments to validation. Do not resample
failed surfaces or silently drop quote positions; report coverage by maturity
and moneyness and use identical reference masks across architectures. Reduced
data and withheld-coordinate experiments must fit their own allowed statistics.

## T1.5 — Preregistration, interpolation, and immutable evidence

**Why freeze training details as well as headline metrics?** Model comparisons
can be biased through unequal search effort, early stopping, seed selection,
normalization, or masks even when they share a reported metric. Fix those choices,
count failed trials, preserve all seeds, and check parameter budgets. A hash
proves file identity; it does not prove that the protocol is scientifically sound
or that a later implementation actually obeyed it.

**How can protocol metadata change without changing numerical evidence?** Before
freeze, require the same data recipe, numerical code, configuration, and environment.
Copy each archive and replace only its protocol-hash metadata, then compare every
other array's shape, dtype, raw bytes, and NPY payload. Preserve the original
artifacts and both identities. Archive hashes necessarily change; label values do
not. A subsequent cache replay checks consistency, not independent cold repricing.

**Why treat interpolation and normalization as part of the experiment?** They
affect both extrapolated values and derivatives. Interpolating log standard
deviation keeps it positive, while training-only moments prevent leakage.
A natural spline couples its entire row, so invalid knots cannot be silently
bridged. Total-variance extrapolation can become nonpositive and must fail
explicitly. Coordinate derivatives require a stated convention; this protocol
uses interior off-knot diagnostics and differentiates inverse normalization too.

## T2.1 — Training mechanics and exact resume

**Why is multiplying NaN labels by a zero mask unsafe?** IEEE arithmetic leaves
`NaN * 0` as NaN, and masked expressions can still contaminate backward passes.
Index valid entries before subtracting and squaring, preserve every original
quote in coverage counts, and divide by the valid quote count. For vega-weighted
losses, keep that same denominator and fit the weight scale on training data once.

**What makes an interrupted training run reproducible?** Save model, optimizer,
scheduler, all used random states, progress, and source/data/config identity at an
atomic boundary. Replay an unfinished epoch from its previous committed state;
never silently create another tuning trial. Isolate logging RNG effects and
replay committed log records idempotently. Compare uninterrupted and resumed
weights and optimizer states directly, not only approximately equal losses.

**What does overfitting 100 surfaces prove?** It checks the execution path through
normalization, masks, gradients, optimization, selection, and checkpoint restore
on real training labels. It establishes neither generalization nor a winning
architecture. Keep its cap and acceptance criterion fixed, report its actual
reconstruction error, and leave the production tuning budget and unseen-data
assessment to later tickets. A nonlinear AdamW trace need not be monotone;
verify strict descent on a separate controlled convex fixture.

## T2.2 — M0 baseline and fixed tuning budget

**Why count failed trials instead of replacing them?** The twelve draws are the
declared compute and selection budget. Replacing an unstable draw conditions the
search on successful training and grants extra optimization effort. M0 retained
two prediction-failure trials as failures, then selected from the remaining
terminal records using the preregistered validation ordering.

**What does passing the 0.2-vol-point stop rule mean?** It permits moving to the
next architecture under Gate 2; it is not the final 0.05-vol-point target. M0's
three-seed validation mean MAE was 0.021973 vol points, but its mean per-seed
maximum error was 1.237196 vol points. Both summary and tails matter, and
validation has already informed model selection.

**How were invalid quotes and seed variability handled?** Every seed attempted
the same 2.4 million validation quotes; 6,063 reference failures were preserved
in the denominator, while no valid quote had a prediction failure. Metrics were
computed per seed and then summarized with arithmetic means and sample standard
deviations. No predictions were pooled and no best seed was reported.
