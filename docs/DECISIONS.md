# Decisions

## 2026-10-27 — T0.1

- Copy the supplied PLAN.md, AGENTS.md, and TODO.md into the empty repository so the agreed research protocol is versioned with the implementation.
- Respect the user's explicit clarification to complete only T0.1 this session; later gates remain pending.
- Use a setuptools `src/nop` package, Python 3.11+, and Make targets so development and clean-checkout verification use the same commands.
- Add NumPy and PyTorch for the required random-number and DataLoader seed utilities; defer SciPy, YAML parsing, plotting, and experiment logging dependencies until their tickets need them.
- Add Ruff, pytest, and mypy as development dependencies to enforce formatting, lint, reproducibility checks, and strict types with `make check`.
- Keep the unimplemented data, final-evaluation, and reproduction commands fail-closed with nonzero exits; do not create placeholder datasets, hashes, or research results.
- Use bounded dependency ranges for installation compatibility; record the exact environment used to validate this ticket in its handoff, and require full environment metadata for future experiments.
- Let mypy use the active interpreter version so it can parse the dependency stubs selected for that interpreter; retain Python 3.11 as the package minimum and Ruff language target.
- Preserve the supplied AGENTS.md and PLAN.md byte-for-byte, including existing whitespace, to avoid changing the agreed instructions during scaffolding.

### Verification

- Source commit `d11e4dd` passed `make check` both in the working repository and in a clean local Git clone with a newly created virtual environment and independently installed dependencies: Ruff formatting/lint, strict mypy (11 source files), and 22 passing tests.
- Verified runtime: macOS arm64, Python 3.12.7, NumPy 2.5.4, PyTorch 2.14.1, pytest 9.2.0, Ruff 0.17.3, mypy 1.20.2. CPU checks only; Python 3.11 and accelerator execution are unverified.
- Confirmed `make data`, `make final-eval`, and `make reproduce` return nonzero, and virtual environments, datasets, checkpoints, local secrets, and generated reports are ignored by Git. No research artifacts were created.

## 2026-10-28 — T0.2

- Follow the user's publication instruction: create public `lukelu389/neural-operator-surrogate`, push T0.1 before starting T0.2, and commit/push each completed stage after verification; `main` initially matched T0.1 commit `9a980c9`.
- Write eight compact core-paper notes with primary-source links, versions, actual baselines, and separate critical assessments so published results are not presented as our experimental evidence.
- Record preprint versus journal dates and source-access limits explicitly; use accessible complete manuscripts or identified copies rather than inventing unavailable publication details.
- Select a bidirectional Mamba-style selective SSM as the M3 implementation direction: an explicit PyTorch recurrence is easier to inspect for this short coordinate sequence than full S4 structured-kernel machinery; neither accuracy nor fused-kernel speed is assumed.
- Leave model dimensions, hyperparameters, dataset choices, and protocol thresholds to their designated gates; reading the papers does not authorize additional architecture trials or Track B implementation.
- Defer reuse of Luke's Poseidon notes because no source was supplied or found in the repository/targeted connected-note search; all eight core-paper notes can be completed without fabricating that personal research context.
- Reinstall the existing editable package after Python's verbose startup showed its hidden `.pth` file was being skipped; changing the flag alone did not persist. `.venv/bin/python -m pip install --no-deps --no-build-isolation -e .` restored imports without tracked packaging or application code changes.

### Verification

- `make check` passed Ruff formatting/lint, strict mypy (11 source files), and all 22 tests after the local editable-install repair; no dependency was added or upgraded for T0.2.
- All eight notes contain methods, reported claims/comparisons, and independent critical/project assessments; relative Markdown links resolve, and publication versions/source-access limits are disclosed.
- No paper benchmark was reproduced, and no research data, model, frozen configuration, or test/OOD result was created or accessed.

## 2026-10-28 — T0.3

- Draft the evaluation protocol without implementing future tickets; retain `status: draft` and explicit freeze blockers rather than inventing validated numerical settings.
- Adopt the plan's starting ID box, scaled grid, split sizes, and accuracy targets; do not impose Feller filtering, so the declared box is assessed with explicit numerical failure coverage.
- Exclude the closed ID box from wider-parameter and corner OOD recipes; short-maturity OOD uses ID parameters at unseen maturities, keeping the source of distribution shift identifiable.
- Use fixed Sobol recipes and split offsets; materialize only training/validation before Gate 3 and reserve holdout generation/read access for `final_eval.py`, following the stricter AGENTS.md rule.
- Keep later diagnostics/calibration on validation and disclose selection exposure; this preserves ticket ordering and the single final holdout pass, with the tradeoff recorded in DEVIATIONS.md.
- Select the target-assessed model on validation, compute metrics separately for all three seeds, then report mean and sample standard deviation; common reference masks and explicit prediction failures prevent selective scoring.
- Declare 12 tuning trials per architecture including failed training trials, shared training seeds, and parameter counts within 10% of M0; exact search ranges and training caps remain pre-freeze requirements.
- Use nested training subsets with their own normalization statistics for data efficiency; train and freeze every required run before the single final holdout evaluation.
- Define query generalization as withheld strikes at known maturities; label M0's short-maturity total-variance extrapolation explicitly and treat nonpositive variance as failure.
- Score every calibration method's recovered parameters with COS; separate observed noisy fit, latent surface error, and held-out-quote error so a surrogate cannot grade itself.
- Quantify the plan's pre-freeze "few percent" failure-mask stop rule as 3% overall or at any maturity on training/validation; this is a draft choice with no observed failure rates.
- Add no dependency for a descriptive YAML draft; syntax and structural checks use an already installed external PyYAML interpreter, while runtime parsing belongs to implementation tickets.

### Verification

- Parsed the YAML with PyYAML 6.0.1 from `/opt/anaconda3/bin/python`, rejecting duplicate keys; checked split offsets, range/grid ordering, query partitions, seeds, unit conversion, draft state, and local Markdown links. These checks do not establish future runtime enforcement.
- `make check` passed Ruff formatting/lint, strict mypy (11 source files), and 22 tests. Reinstalled the existing editable package using the previously documented local repair; no package dependency was added or upgraded.
- Independent read-only protocol and numerical reviews found no blocker to the unfrozen draft; incorporated explicit C2 training provenance, condition-wise calibration success, derivative conventions, and M0 checkpoint scope.
- No pricing labels, dataset samples, model results, holdout accesses, or frozen hashes were produced. Gate 1 remains unstarted.

## 2026-10-28 — T1.1

- Implement only Black–Scholes prices and analytic Greeks in `src/nop/pricing/bs.py`; preserve the one-ticket boundary before Heston COS, IV inversion, or data generation.
- Use the existing PyTorch dependency so prices broadcast and retain first/second autodiff support alongside the planned batched Heston implementation; add no dependencies or experiment configuration changes.
- Use float64 for scalar-only reference calculations, and require tensor inputs to share float32/float64 dtype and device so implicit conversions do not hide precision or device differences.
- Evaluate normal lower tails with `erfc`, price the out-of-the-money leg directly, and recover the other leg by put–call parity to reduce tail cancellation.
- Return exact expiry and zero-volatility prices with safe unused regular branches; reject analytic Greeks at these singular boundaries rather than inventing derivatives. Boundary price autodiff is outside the contract.
- Define theta as calendar decay per year at fixed spot/strike, and vega/rho per decimal input unit, preventing time-sign and factor-of-100 ambiguities.
- Validate against the pinned QuantLib v1.38 Haug fixtures, independent risk-neutral payoff integration, parity, finite differences, and autodiff rather than relying on parity alone or copying the implementation into expected values.
- Keep experiment holdouts sealed and the protocol draft unchanged; these are numerical unit-test cases, not training labels or model-evaluation results.
- A numerical review exposed negative float32 prices from cancellation at tiny total volatility; reject nonfinite outputs and negative prices with `FloatingPointError`, add regression coverage, and recommend float64 rather than silently clipping or changing computation precision.

### Verification

- `make check` passed Ruff formatting/lint, strict mypy (12 source files), and 117 tests, including 95 Black–Scholes cases. Repaired the existing editable installation immediately before this standard command; no test-path override, new dependency, or package upgrade was used for the final run.
- Published four-decimal fixtures use absolute tolerance 1e-4. Separate risk-neutral payoff quadrature checks, including small OTM prices, use relative tolerance 2e-9 on the selected cases; these are case-specific checks, not a domain-wide accuracy guarantee.
- Analytic Greeks match central finite differences and PyTorch first/second derivatives, including the forward-ATM parity branch. Explicit tests cover zero-time/zero-volatility limits, mixed batches, dividends, negative rates, float32/float64 broadcasting, cancellation errors, and overflow.
- Verified on CPU with the existing Python 3.12.7 environment. Accelerator behavior is unverified; test/OOD experiment datasets remain untouched and no frozen manifest changed.
