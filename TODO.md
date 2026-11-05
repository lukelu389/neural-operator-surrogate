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
- [x] T1.4 Sampling, grids, splits, OOD sets, normalization, data card
  - Done when: `make data` reproduces identical hashes
  - Handoff notes:
    - 2026-11-01: Added validated adaptive COS labels, deterministic Sobol/grid utilities, resumable hashed chunks, strict masks, training-only normalization, and a data card; `make data` generates only train/validation and keeps test/OOD recipes inert.
    - Generated 200,000/20,000 surfaces. Mask rates are 0.254571%/0.252625%; every maturity passes 3%. The worst train coordinate is 3.0165% masked and is disclosed. All 26.4 million attempted quotes remain accounted for.
    - A full cached replay and four cold production-chunk regenerations preserve hashes; combined SHA-256 is 50df50f75c727f238e6db44cb57fe44d6173b8f0a80921de8dc0e2b5f452efc7. See docs/data-card.md and docs/data-hashes.json; a second full cold replay was not run.
    - `make check` passed Ruff, strict mypy (25 sources), and 459 tests, with one CUDA skip. Both fixed/adaptive numerical prerequisite studies pass. No dependency was added; generated datasets/reports remain ignored by Git.
    - Next: T1.5, completing every freeze checklist item. Current archives embed the whole draft protocol hash: resolve subsequent provenance changes explicitly with regeneration or verified metadata migration before freezing. No test/OOD data, model training, or frozen manifest was created.
- [x] T1.5 Freeze protocol v1 (hashes in frozen/)
  - Done when: frozen files committed; stop rule on failure mask checked
  - Handoff notes:
    - 2026-11-02: Completed shared training/calibration budgets, tested train-only off-grid normalization/M0 adapter, fixed diagnostic policies, and froze eight files after current numerical evidence and actual train/validation coverage passed.
    - Explicit protocol metadata migration preserved all nonmetadata array bytes; original T1.4 artifacts remain at data/heston_v1_t14. Full cached replay and migration recheck agree on current dataset hash 9120ef79ac2c385618cad182ae55d18212b34a591a0579c864fb0d58874797d7.
    - Run `.venv/bin/python scripts/freeze_protocol.py --verify`, `.venv/bin/python scripts/validate_diagnostics.py`, and `make check`. Ruff, strict mypy (31 sources), and 595 tests pass; one CUDA test is skipped. All 21 finite-difference fixtures and 81 analytic arbitrage checks pass.
    - See docs/protocol-freeze.md and docs/DEVIATIONS.md for migration evidence, derivative conventions, numerical limits, and three interview answers in docs/interview.md. No dependency, model result, or test/OOD access was introduced.
    - Next: T2.1 training loop, MLflow local logging, and the frozen 100-sample sanity check. Gate 3 model/checkpoint freeze remains mandatory; GPU speed evidence will require suitable hardware at Gate 4.



## Gate 2: Baseline trained

- [x] T2.1 Training loop, logging, sanity tests (overfit 100 samples)
  - Done when: sanity tests pass
  - Handoff notes:
    - 2026-11-02: Added shared masked/vega losses, vol-point metrics, AdamW training with validation selection, atomic epoch checkpoints and exact resume, plus explicit local MLflow logging. All eight frozen files remain unchanged.
    - Run `.venv/bin/python scripts/validate_training.py`: the declared 100-training-row fixture passed at 5,000 updates with standardized masked MSE 1.19850756e-5 <=1e-4; 11,967/12,000 quotes are valid. This is reconstruction, not unseen-data performance or an M0 result.
    - Cached replay preserved report/checkpoint hashes, the same MLflow run, and exactly 5,000 loss records. Interrupted synthetic training matches uninterrupted weights, optimizer/scheduler, RNG and metric history exactly; no production tuning trial was used.
    - `make check` passed Ruff, strict mypy (36 sources), and 699 tests, with one CUDA skip. Added mlflow-skinny >=3.17.2,<4 for the frozen logger requirement; existing numerical package versions stayed unchanged. See docs/training-loop.md and docs/interview.md.
    - Next: T2.2, define M0 before its first trial, enforce twelve paired search draws, train three selected-config seeds, and check validation MAE <=0.2 vol points before other architectures. CUDA/MPS execution is unverified; no test/OOD access occurred.
- [x] T2.2 M0 MLP with tuning budget, 3 seeds
  - Done when: validation metrics, learning curve, error heatmaps; stop rule checked
  - Handoff notes:
    - 2026-11-03 implementation: fixed a 1,013,160-parameter M0 before production trials, added exact twelve-draw orchestration, three-seed reporting, lock-protected resume, and validation-only figures. Code is committed before training; no budget or frozen file changed.
    - Run `.venv/bin/python scripts/train_m0.py`; repeat the same command after interruption, preserving runs/m0_v1. No extra trial-count option exists; selected-config and cache drift are refused. See docs/m0-baseline.md.
    - Pre-production verification: `make check` passed Ruff, strict mypy (40 sources), and 783 tests, with one CUDA skip; all eight frozen hashes verify.
    - Production completed: all 12 paired draws reached terminal states; trial_00 and trial_04 failed for 3,208 and 6,574 nonfinite or nonpositive valid-quote predictions and remained counted. Trial_02 was selected with validation MAE 0.003914 vol points; three final seeds completed with mean MAE 0.021973 ± 0.007027 vol points and mean p99 0.105630 ± 0.031661. The frozen 0.2-vol-point stop rule passed.
    - Seed-level validation metrics, reference-mask counts, full per-maturity/z/cell metrics, checkpoint/report hashes, and figure hashes are recorded in reports/tables/m0_validation.json (SHA-256 a607995a29f8c82b8991d98799d0f011e07daed37efcb128fc7e51340ac3544); plots are reports/figures/m0/learning_curves.png and validation_error_heatmaps.png. Cached rerun preserved these report and figure hashes and MLflow records.
    - Final `make check` passed Ruff, strict mypy (40 sources), and 783 tests, with one CUDA skip; all eight frozen protocol hashes verify. See docs/m0-baseline.md for full metrics, coverage, tail errors, failures, and reproduction command.
    - Next: T3.1 DeepONet-style M1. No test/OOD data was accessed. M0's max validation error is materially larger than its mean (per-seed maxima 1.010–1.600 vol points); validation results are development evidence, not final or market performance.



## Gate 3: Architecture comparison

- [x] T3.1 M1 DeepONet-style (parameter count within 10% of M0, tests)
  - Done when: trained with the same tuning budget
  - Handoff notes:
    - 2026-11-04 implementation: fixed M1 before validation as a branch(theta)/trunk(k,T) DeepONet with three 480-wide SiLU hidden layers per tower, rank 128, and inverse-square-root-rank dot-product scaling. It has 1,050,977 trainable parameters (3.73% above selected M0); parameter count is checked against M0's selected config at runtime.
    - `scripts/train_m1.py` uses the same verified train/validation data, frozen AdamW schedule, exact paired twelve-trial draws, trial-failure accounting, three selected-config seeds, local MLflow, resumable checkpoints, and validation-only reports. Run `.venv/bin/python scripts/train_m1.py`; repeat the same command to resume. `--tune-only` stops after all twelve trials.
    - Model tests cover native/query shapes, coordinate permutation behavior and gradients, positive-IV output, serialized normalization buffers, exact parameter count, and the ±10% M0 budget. Synthetic runner coverage verifies all twelve trials precede the three seeds and the selected hyperparameters are reused.
    - Production completed with all 12 paired trials terminal (11 early-stopped, one completed; none failed), then seeds 11/29/47. Trial_08 was selected. Validation mean MAE was 0.016920 ± 0.006073 vol points; mean p99 0.110340 ± 0.028856 and mean per-seed max 1.497752 ± 0.151406. Each seed had 6,063/2,400,000 reference failures (0.252625%) and zero prediction failures. See docs/m1-baseline.md.
    - Full validation details and figure/report hashes are recorded in ignored reports/tables/m1_validation.json. The initial plots inherited the helper's M0 default title; a post-run figures-only correction regenerated M1-labeled plots from cached validation histories. Checkpoints, predictions, optimizer logs, selections and metrics did not change; `.venv/bin/python scripts/train_m1.py --figures-only` replays that correction idempotently. The original run identity remains the code used for training.
    - Cached replay before the title correction preserved the report, all 15 run IDs/checkpoints/per-run reports, seed prediction files, figures and MLflow loss counts. Final `make check` passed Ruff, strict mypy (42 sources), and 791 tests, with one CUDA skip; all eight frozen hashes verify.
    - Next: T3.2 Transformer. No test/OOD data was accessed; M1 validation results are development evidence and do not establish final architecture ranking, the final accuracy target, or market performance.
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

