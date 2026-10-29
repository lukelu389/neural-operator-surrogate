# TODO: Neural Surrogates for Option Pricing and Calibration

One ticket per session. Tick when the exit criteria in PLAN.md are met. Add handoff notes under each finished ticket.

## Gate 0: Setup and pre-registration

- [x] T0.1 Repo scaffold, Makefile, `make check`, seed utilities
  - Done when: `make check` passes from a clean clone
  - Handoff notes:
    - 2026-10-27: Added the installable `src/nop` scaffold, Make commands, deterministic Python/NumPy/PyTorch seeds and DataLoader worker seeding.
    - Verified source commit `d11e4dd` from a clean local clone with a fresh virtual environment: Ruff formatting/lint, strict mypy (11 source files), and all 22 tests passed; the clone remained clean.
    - Run: `python3 -m venv .venv`, `.venv/bin/python -m pip install -e '.[dev]'`, then `make check`; setup and limitations are in README.md.
    - Verified on macOS arm64, Python 3.12.7, NumPy 2.5.4, PyTorch 2.14.1, pytest 9.2.0, Ruff 0.17.3, and mypy 1.20.2; GPU execution and Python 3.11 remain untested.
    - Next: T0.2 reading notes. No open scope questions; data generation, final evaluation, and reproduction intentionally refuse until their tickets. No experiments or holdout access occurred.
- [x] T0.2 Reading notes (COS, Little Heston Trap, Deep Learning Volatility, Liu et al., FNO, DeepONet, S4/Mamba)
  - Done when: one page per paper in docs/reading/
  - Handoff notes:
    - 2026-10-28: Added eight approximately one-page notes with verified references, methods, actual comparisons, critical assessments, and source-access limits; start at docs/reading/README.md.
    - Selected a bidirectional Mamba-style selective SSM direction for T3.3; no model/configuration, pricer, dataset, or protocol implementation was started.
    - Verification: `make check` passed Ruff, strict mypy, and 22 tests; all relative Markdown links resolve. Reinstalled the existing editable package to repair a local hidden `.pth` import issue.
    - Review by following each note's source/section links. No paper experiments were reproduced; the Heston manuscript/final-typesetting distinction and unavailable personal Poseidon notes are explicitly disclosed.
    - T0.1 was pushed first to public lukelu389/neural-operator-surrogate. This stage is committed/pushed separately; next is T0.3 (draft protocol, thresholds, OOD definitions), which remains unchecked.
- [x] T0.3 Draft eval_protocol.yaml, thresholds, OOD definitions
  - Done when: committed, not yet frozen
  - Handoff notes:
    - 2026-10-28: Added config/eval_protocol.yaml with ID targets, three explicit OOD recipes, seeds/budgets, common-mask and seed-aggregation rules, calibration scoring, and pre-freeze requirements; see config/README.md for interpretation.
    - The protocol is a committed draft, not frozen or executable. No pricer, data, model, or numerical result was created; final evaluation still refuses to run.
    - Verification: YAML duplicate-key and structural checks, units, split offsets, query partitions, and Markdown links passed; `make check` passed Ruff, strict mypy, and 22 tests after reinstalling the existing editable package.
    - Recorded two sequencing interpretations in docs/DEVIATIONS.md: holdout recipes are frozen before final-only generation, and later diagnostics/calibration use explicitly labeled validation data rather than independent test evidence.
    - Next: T1.1 Black–Scholes closed form and Greeks. Resolve the protocol's numerical/normalization/budget checklist before T1.5; no Gate 1 work was started in this ticket.



## Gate 1: Ground truth validated and data frozen

- [x] T1.1 Black-Scholes closed form and Greeks
  - Done when: unit tests (known values, parity) pass
  - Handoff notes:
    - 2026-10-28: Added broadcastable PyTorch European call/put pricing and analytic delta, gamma, vega, annual theta, and rho in src/nop/pricing/bs.py; README.md documents units and usage.
    - Prices handle dividends, negative rates, expiry, and zero volatility. Greeks reject singular boundaries; invalid computed prices/nonfinite outputs raise numerical errors rather than being silently clipped.
    - Verification: `make check` passed Ruff, strict mypy (12 source files), and 117 tests, including 95 Black–Scholes cases covering published values, independent payoff integration, parity, finite differences, autodiff, limits, and precision failures.
    - Run the README example or `.venv/bin/python -m pytest tests/test_bs.py`; float64 is recommended for reference pricing. Verified on CPU; accelerator execution and extreme-tail relative accuracy remain unverified. The existing editable installation was repaired without dependency changes.
    - Next: T1.2 Heston COS pricer with its required numerical validations. No Heston data generation, IV inversion, model training, holdout access, or protocol freeze occurred in T1.1.
- [x] T1.2 Heston COS pricer (torch, batched)
  - Done when: BS-limit test, put-call parity, convergence in N and range, Monte Carlo agreement within CI
  - Handoff notes:
    - 2026-10-28: Implemented the batched PyTorch Heston CF/cumulants/COS pricer and independent antithetic full-truncation Monte Carlo; explicit N/L settings and numerical error handling preserve honest failure reporting.
    - `make check` passed Ruff, strict mypy (15 source files), and 277 tests; one CUDA comparison was skipped for unavailable hardware. Checks include independent Riccati integration/reference prices, BS limits, parity, convergence, and first/second autodiff.
    - Run `.venv/bin/python scripts/validate_heston.py`; config/heston_numerics.yaml and docs/heston-validation.md record the cached numerical study. Selected N=4096/L=28 differed from N=8192/L=36 by at most 8.72e-11 across 12 spot=1 quotes.
    - Both 200,000-pair, 2048-step MC intervals contained COS at nominal 95%; refinement checks passed. The initial smaller run failed and is disclosed in docs/DEVIATIONS.md. Sampling intervals exclude timestep bias; these fixtures do not certify every future IV label.
    - Next: T1.3 robust IV inversion and explicit failure masks. No training/validation/holdout datasets or frozen hashes were created; GPU execution remains unverified. PyYAML and its typing stubs were added for the required YAML-configured study.
- [x] T1.3 IV inversion with failure mask
  - Done when: roundtrip error below stated tolerance; failure fraction reported
  - Handoff notes:
    - 2026-10-28: Added bounded batched BS IV inversion, explicit status/NaN masks, residual/vega diagnostics, and 69 focused tests; no existing pricing implementation or dependency changed.
    - Run `.venv/bin/python scripts/validate_iv.py` for the cached YAML-configured corner/center study. Of 7,920 attempts, 7,436 pass; 484 fail (6.1111%), including upstream COS failures. Accepted normalized roundtrip p99/max are 2.872e-11/3.962e-11, below 1e-10.
    - `make check` passed Ruff, strict mypy (17 source files), and 346 tests; one CUDA check was skipped. Cache reuse and report denominators passed; the existing editable installation was repaired without dependency changes.
    - COS selected/finer-resolution differences reach 2.342e-6 times price scale; 308 accepted quotes exceed 1e-8. See docs/iv-validation.md and docs/DEVIATIONS.md. A successful roundtrip does not certify Heston label accuracy, and these settings must not be frozen as-is.
    - Next T1.4 must first investigate source-price convergence before generating production labels, then measure actual training/validation mask coverage before T1.5's stop check. No research split or holdout was created/read; the 6.11% deterministic stress rate is not a population estimate.
- [ ] T1.4 Sampling, grids, splits, OOD sets, normalization, data card
  - Done when: `make data` reproduces identical hashes
  - Handoff notes:
- [ ] T1.5 Freeze protocol v1 (hashes in frozen/)
  - Done when: frozen files committed; stop rule on failure mask checked
  - Handoff notes:



## Gate 2: Baseline trained

- [ ] T2.1 Training loop, logging, sanity tests (overfit 100 samples)
  - Done when: sanity tests pass
  - Handoff notes:
- [ ] T2.2 M0 MLP with tuning budget, 3 seeds
  - Done when: validation metrics, learning curve, error heatmaps; stop rule checked
  - Handoff notes:



## Gate 3: Architecture comparison

- [ ] T3.1 M1 DeepONet-style (parameter count within 10% of M0, tests)
  - Done when: trained with the same tuning budget
  - Handoff notes:
- [ ] T3.2 M2 Transformer
  - Done when: trained with the same tuning budget
  - Handoff notes:
- [ ] T3.3 M3 SSM (bidirectional or two-pass, documented)
  - Done when: trained with the same tuning budget
  - Handoff notes:
- [ ] T3.4 Freeze model configs; run final_eval once (test, OOD, data efficiency, 3 seeds)
  - Done when: results table with mean and std; no retuning after this point
  - Handoff notes:



## Gate 4: Greeks, arbitrage, speed

- [ ] T4.1 Greeks via autodiff vs COS finite differences
  - Done when: error distributions reported
  - Handoff notes:
- [ ] T4.2 Arbitrage diagnostics (butterfly, calendar)
  - Done when: violation rates and magnitudes reported
  - Handoff notes:
- [ ] T4.3 Speed benchmarks incl. vectorized GPU COS
  - Done when: table with hardware details and honest discussion
  - Handoff notes:



## Gate 5: Calibration

- [ ] T5.1 C0 classical calibration (COS plus least squares, multi-start)
  - Done when: baseline success rate, error, time
  - Handoff notes:
- [ ] T5.2 C1 surrogate-based gradient calibration
  - Done when: same metrics as C0
  - Handoff notes:
- [ ] T5.3 C2 inverse network (plus optional refinement)
  - Done when: same metrics
  - Handoff notes:
- [ ] T5.4 Noise, missing quotes, held-out quotes, identifiability analysis
  - Done when: comparison table and identifiability figure
  - Handoff notes:



## Gate 6 (optional): Track B payoff-to-price operator

- [ ] T6.1 BS operator data with exact labels, cross-checked with Crank-Nicolson
  - Handoff notes:
- [ ] T6.2 FNO, DeepONet, Transformer, SSM
  - Handoff notes:
- [ ] T6.3 Resolution transfer, OOD payoffs, Greeks
  - Handoff notes:



## Gate 7: Write-up and release

- [ ] T7.1 Report (6 to 8 pages)
  - Handoff notes:
- [ ] T7.2 README with headline table, key figure, `make reproduce`
  - Handoff notes:
- [ ] T7.3 Optional Streamlit demo
  - Handoff notes:
- [ ] T7.4 Final checks: clean-clone reproduction, no data or secrets committed, DEVIATIONS.md complete, citations verified
  - Handoff notes:

