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
- [ ] T1.2 Heston COS pricer (torch, batched)
  - Done when: BS-limit test, put-call parity, convergence in N and range, Monte Carlo agreement within CI
  - Handoff notes:
- [ ] T1.3 IV inversion with failure mask
  - Done when: roundtrip error below stated tolerance; failure fraction reported
  - Handoff notes:
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

