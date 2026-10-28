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
