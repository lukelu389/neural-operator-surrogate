# Neural Surrogates for Option Pricing and Calibration: Strict Plan

Project type: research-style finance ML. Duration: about 6 weeks (hard-moderate).
Builds on: your PDE / SSM / Transformer research.

---

## 0. How to follow this plan strictly

1. **Gates are sequential.** Do not start gate N+1 until gate N's exit criteria are met and ticked in TODO.md.
2. **Freeze before you look.** The evaluation protocol (section 6.5) and the success thresholds (section 6.5.3) are written and committed at Gate 0/1, **before any model is trained**. After that they change only through an entry in `docs/DEVIATIONS.md` (what changed, why, what you had already seen).
3. **The test set is opened once.** All decisions use the validation set. The test and out-of-distribution sets are evaluated by `scripts/final_eval.py`, only after the model configs are frozen (Gate 3).
4. **Equal footing.** Every model gets the same data, optimizer family, parameter budget (within 10%), tuning budget and seeds.
5. **Report whatever happens.** A plain MLP may win. That is a legitimate result if you can explain where each model wins or loses.
6. **One ticket per working session.** Each session ends with handoff notes in TODO.md and decisions logged in `docs/DECISIONS.md`.

---

## 1. Project in one paragraph

Build and rigorously benchmark neural surrogates that map model parameters and contract coordinates to option prices (expressed as implied volatilities). Use the Heston model as the main testbed because it has a fast semi-analytic pricer (the COS method) that gives accurate labels cheaply. Compare an MLP baseline, a DeepONet-style coordinate model, a Transformer and a State Space Model (SSM) on accuracy, out-of-distribution robustness, data efficiency, Greeks quality, arbitrage violations and speed. Then use the best surrogates inside a calibration pipeline and compare against classical calibration. An optional second track learns the payoff-to-price operator for Black-Scholes (and local volatility) with FNO, DeepONet, Transformer and SSM, tying directly to your PDE research.

**Claims you may make only if the evidence supports them** (decide the real wording at the end):
- "Surrogate X reproduces Heston implied volatility surfaces to within Y vol points (mean) and Z (99th percentile), and is N times faster per surface than vectorized COS at batch size B."
- "Calibration through the surrogate reaches comparable fit error to classical calibration in T seconds vs T' seconds, with success rate S%."
- "Architecture A beats B out of distribution, and loses in-distribution, by these margins across 3 seeds."

---

## 2. Research questions and hypotheses

| RQ | Question | Hypothesis (to test, not assume) |
|---|---|---|
| RQ1 | How accurate can each architecture be in-distribution at a fixed parameter and data budget? | A well-tuned MLP is very competitive in-distribution. |
| RQ2 | Which architecture generalizes best out of distribution (wider parameter ranges, short maturities, deep wings)? | Coordinate-query models (DeepONet/Transformer/SSM) degrade more gracefully than the MLP, but this is uncertain. |
| RQ3 | How data-efficient is each model? | Models with more structure need fewer samples. |
| RQ4 | Are surrogate Greeks and surfaces arbitrage-consistent? | Pointwise accuracy does not guarantee smooth, arbitrage-free surfaces or accurate derivatives. |
| RQ5 | Does a surrogate speed up and stabilize calibration versus COS plus a classical optimizer? | Speed-up is real for the gradient-based loop at large batch sizes; accuracy near the optimum matters more than average accuracy. |
| RQ6 (Track B) | For the Black-Scholes payoff-to-price operator, which architecture generalizes across resolution and payoff families? | FNO excels because the BS solution operator is a convolution in log-price. |

---

## 3. Scope

**In scope:** Black-Scholes sanity checks; Heston European options (calls/puts via implied volatility); calibration on synthetic surfaces; Greeks and arbitrage diagnostics; speed benchmarks; Track B (optional).

**Out of scope until everything else is done:** real market data calibration, rough volatility models, American options, multi-asset products. Listed in stretch goals.

**Honest limitation to state in the README:** Heston already has a fast semi-analytic pricer, so the surrogate's value is mainly in differentiability, batched calibration/risk loops and as a template for models without closed-form pricers (rough volatility, local volatility). Do not oversell speed.

---

## 4. Pre-work reading (2 to 3 days, one-page notes each in `docs/reading/`)

Verify each citation and details yourself before relying on them.

- Fang and Oosterlee, the COS method for European option pricing (2008).
- Albrecher, Mayer, Schoutens, Tistaert, "The Little Heston Trap" (2007), for a numerically stable characteristic function.
- Horvath, Muguruza, Tomas, "Deep Learning Volatility" (two-stage approach: learn pricing map, then calibrate).
- Liu, Oosterlee, Bohte, "Pricing options and computing implied volatilities using neural networks" (2019).
- Li et al., Fourier Neural Operator for parametric PDEs; Lu et al., DeepONet.
- Gu et al. on structured state spaces (S4) and Gu and Dao on Mamba (pick the SSM variant you will implement).
- Your own Poseidon-related research notes, to reuse code and lessons where applicable.

Reading exit criterion: for each paper, a note answering "what is the method, what did they claim, what baseline did they compare to, what could be missing?"

---

## 5. Repository layout

```
neural-operator-pricing/
  README.md  PLAN.md  AGENTS.md  TODO.md
  Makefile  pyproject.toml  .gitignore  .env.example
  config/
    data.yaml            frozen at G1
    models/*.yaml        frozen at G3
    eval_protocol.yaml   frozen at G1
  frozen/                sha256 hashes of frozen files
  data/README.md         generated data is not committed; seeds + hashes are
  src/nop/
    pricing/             bs.py, heston_cos.py, mc.py, iv.py, fd_bs.py
    data/                sampling.py, grids.py, dataset.py, normalize.py
    models/              mlp.py, deeponet.py, transformer.py, ssm.py, fno.py
    train/               loop.py, losses.py, tune.py
    eval/                metrics.py, arbitrage.py, greeks.py, ood.py, speed.py
    calibrate/           classical.py, surrogate.py, inverse_net.py
  scripts/               make_data.py, train.py, final_eval.py, calibrate.py
  tests/
  notebooks/             figures only, importing from src
  reports/               figures/, tables/, report.md
  docs/                  DECISIONS.md, DEVIATIONS.md, reading/
```

---

## 6. Technical specification

### 6.1 Ground-truth pricers

1. **Black-Scholes closed form** (calls/puts, Greeks analytic).
2. **Heston COS pricer** in PyTorch (GPU-capable, batched):
   - Use a numerically stable characteristic function formulation (the "little Heston trap" form).
   - Truncation range from cumulants; number of terms N chosen by a convergence study and recorded.
   - Cross-checks: reduces to Black-Scholes when vol-of-vol is tiny and v0 = theta; put-call parity; agrees with Monte Carlo within confidence intervals (use a variance-reduced scheme and document the discretization).
3. **Implied volatility inversion:** robust root-finder with a documented failure mask (price below intrinsic plus tolerance, or vega too small). Roundtrip test: price to IV to price.
4. Optional: Crank-Nicolson finite-difference BS solver (needed for Track B and as an independent check).

Dynamics: dS = (r - q) S dt + sqrt(v) S dW1, dv = kappa (theta - v) dt + sigma sqrt(v) dW2, corr(dW1, dW2) = rho. Fix r and q in the main experiment (for example r = 0.02, q = 0.0; record in config). Parameters: theta_vec = (v0, kappa, theta, sigma, rho).

### 6.2 Dataset

- **Sampling:** Sobol sequence over a parameter box. Suggested starting box (finalize and freeze at G1): v0 in [0.01, 0.25], kappa in [0.5, 5], theta in [0.01, 0.25], sigma in [0.1, 1.0], rho in [-0.95, -0.1]. Decide whether to impose the Feller condition and record the decision.
- **Output grid:** maturities T in {1/12, 2/12, 3/12, 6/12, 9/12, 1, 1.5, 2}; per maturity, scaled log-moneyness k = z * 0.2 * sqrt(T) with z in linspace(-2.5, 2.5, 15). The scaled grid keeps wings comparable across maturities. Decision to freeze at G1.
- **Target:** implied volatility on the grid (decimal); mask for invalid points.
- **Sizes (suggested):** train 200k, validation 20k, test 20k, OOD sets 20k each.
- **OOD sets (defined now, never trained on):** (a) wider sigma and kappa ranges than training; (b) shorter maturities (for example T in {1/52, 2/52}); (c) parameter-corner samples (high sigma with low kappa, strongly negative rho).
- **Normalization:** parameters standardized or min-max scaled using train statistics only; IV targets standardized per grid point using train statistics only.
- **Versioning:** every dataset file carries a config hash and seed; `make data` reproduces it exactly.

### 6.3 Models

All models output IV on queried coordinates, parameter budget about 1M (within 10%), same optimizer and schedule family.

| ID | Model | Notes |
|---|---|---|
| M0 | MLP: theta to flattened grid | Strong baseline from the literature. |
| M1 | DeepONet-style: branch(theta), trunk(k, T) with dot-product head | Can be queried at arbitrary (k, T). |
| M2 | Transformer: parameter tokens plus query tokens (k, T), bidirectional attention, output at query tokens | Document positional encoding choice for continuous coordinates. |
| M3 | SSM (S4/Mamba-style): sequence of query tokens ordered maturity-major, theta embedding concatenated or FiLM-conditioned | SSMs are directional; use a bidirectional design or two passes and document it. |
| M4 (Track B) | FNO | Only for the payoff-operator track. |

Fairness rules: identical train/val data; identical tuning budget (for example 12 random-search trials per model over small, pre-declared ranges); best configuration selected on validation only; 3 seeds minimum for final numbers; parameter counts reported.

### 6.4 Training protocol

- AdamW, cosine schedule with warmup, gradient clipping, optional mixed precision.
- Loss: MSE on standardized IV; test also a vega-weighted loss as an ablation (it approximates price error). Record which loss is the headline.
- Early stopping on validation; log everything to MLflow or W&B; save checkpoints with config hash.
- Sanity checks before full runs: overfit 100 samples to near zero error; loss decreases monotonically on a tiny run; shapes and mask tests.

### 6.5 Evaluation protocol (frozen at G1)

**6.5.1 Accuracy (vol points; 1 vol point = 0.01 IV):** mean absolute error, RMSE, 99th percentile, max; reported overall, by maturity, by moneyness, in-distribution and per OOD set.

**6.5.2 Further diagnostics:**
- **Data efficiency:** error vs training set size (5k, 20k, 50k, 200k).
- **Query generalization (M1 to M3):** train on a subset of grid points, evaluate on held-out (k, T); compare with spline interpolation of the MLP output.
- **Greeks:** autodiff derivatives with respect to theta, and with respect to coordinates where applicable; compare with finite differences of the COS pricer. Report error distributions.
- **Arbitrage:** butterfly check (convexity of call prices in strike per maturity) and calendar check (nondecreasing total variance in maturity at fixed k, using interpolation of total variance in k; document the interpolation). Report violation rates and magnitudes.
- **Speed:** surfaces per second at batch sizes 1, 256, 8192 on CPU and GPU for surrogates, versus the COS pricer (single-thread numpy and vectorized GPU). Report hardware, precision and warm-up handling.
- **Seed variability:** mean and standard deviation over 3 seeds; no best-seed reporting.

**6.5.3 Pre-registered targets (adjust at G1 only, then freeze):**
- In-distribution mean absolute IV error at most 0.05 vol points and 99th percentile at most 0.3 vol points for the best model.
- Arbitrage violation rate and Greek error reported with no target; the goal is to characterize them.
- Surrogate calibration success rate at least that of classical calibration within a comparable fit tolerance.
These are goals, not guarantees. Missing them is reportable.

### 6.6 Calibration experiments

Inputs: synthetic surfaces from held-out parameters with additive Gaussian IV noise at three levels (for example 0.0, 0.1, 0.3 vol points), optionally with random missing quotes.

| ID | Method |
|---|---|
| C0 | COS pricer plus least-squares optimizer (multi-start, same budget of starts). |
| C1 | Surrogate plus gradient-based optimizer (autodiff, multi-start, batched across starts). |
| C2 | Inverse network (surface to theta), then optional C1 refinement. |

Metrics: success rate (surface fit error below a frozen tolerance), normalized parameter error, surface fit error, wall-clock time, number of function evaluations. **Identifiability analysis:** show examples where very different parameter vectors give nearly identical surfaces (typical for kappa and theta) and report how parameter error behaves compared with surface error. Held-out quote test: calibrate on a subset of quotes and predict the rest.

### 6.7 Track B (optional, only after Gate 5): payoff-to-price operator

- Black-Scholes in log-price x with fixed sigma, r, T. Map payoff g(x) on a grid to V(0, x).
- Payoff families: random sums of calls, puts and smoothed digitals. Held-out family: higher kink density and different strike ranges.
- Labels: exact by quadrature or FFT convolution with the lognormal kernel (BS price is a convolution in log-price), cross-checked with Crank-Nicolson.
- Models: FNO, DeepONet, Transformer, SSM. Tests: zero-shot resolution transfer (train at 128 points, test at 256 and 512), error near kinks, Delta and Gamma via autodiff vs analytic.
- Extension (stretch): local volatility with Crank-Nicolson labels, sigma(x, t) as an additional input function.

---

## 7. Gates, tickets and exit criteria

Each ticket is one working session. Exit criteria are mandatory. If a **stop rule** triggers, stop, log it in DEVIATIONS.md and decide deliberately before continuing.

### Gate 0: Setup and pre-registration (days 1 to 3)
- **T0.1** Repo scaffold, Makefile (`data`, `test`, `lint`, `check`, `final-eval`), CI-style `make check`, seed utilities. *Exit:* `make check` passes from a clean clone.
- **T0.2** Reading notes for the core papers. *Exit:* one page per paper in `docs/reading/`.
- **T0.3** Draft `config/eval_protocol.yaml`, success thresholds and OOD definitions. *Exit:* committed (not yet frozen).

### Gate 1: Ground truth validated and data frozen (week 1)
- **T1.1** BS closed form and Greeks with unit tests (known values, put-call parity).
- **T1.2** Heston COS pricer (batched, torch). *Exit:* BS-limit test, put-call parity, convergence in N and truncation range, Monte Carlo agreement within confidence intervals.
- **T1.3** IV inversion with failure mask. *Exit:* price to IV to price roundtrip error below a stated tolerance across the box; failure fraction reported.
- **T1.4** Sampling, grids, splits, OOD sets, normalization. *Exit:* `make data` reproduces datasets with identical hashes; data card written (ranges, grid, failure rates).
- **T1.5** Freeze: write hashes into `frozen/`, commit "freeze protocol v1". *Stop rule:* if failure mask removes more than a few percent of the grid, revisit the grid before freezing.

### Gate 2: Baseline trained (week 2)
- **T2.1** Training loop, logging, sanity tests (overfit 100 samples). 
- **T2.2** M0 MLP trained with the tuning budget. *Exit:* validation metrics, learning curve, 3 seeds, per-maturity and per-moneyness error heatmaps.
- *Stop rule:* if M0 cannot reach the order of 0.2 vol points mean error on validation after the tuning budget, debug data and normalization before building other models.

### Gate 3: Architecture comparison (week 2 to 3)
- **T3.1** M1 DeepONet-style. **T3.2** M2 Transformer. **T3.3** M3 SSM. Each with unit tests (shape, permutation/coordinate behavior, parameter count within 10% of M0) and the same tuning budget.
- **T3.4** Freeze model configs; commit hashes. Run `scripts/final_eval.py` once: test and OOD results for all models, 3 seeds. *Exit:* results table with mean and standard deviation, OOD tables, data-efficiency curves. 
- *Stop rule:* after final eval, no retuning. Additional experiments go in a separate "post-hoc" section of the report.

### Gate 4: Greeks, arbitrage, speed (week 4)
- **T4.1** Greeks via autodiff vs COS finite differences. **T4.2** Arbitrage diagnostics. **T4.3** Speed benchmarks (including the vectorized GPU COS baseline). *Exit:* tables and figures with hardware details; honest discussion of when the surrogate is actually faster.
- Optional ablation: add butterfly/calendar penalties to the loss and report the tradeoff with accuracy.

### Gate 5: Calibration (week 4 to 5)
- **T5.1** C0 classical calibration. **T5.2** C1 surrogate calibration. **T5.3** C2 inverse network. **T5.4** Noise, missing quotes, held-out quote tests, identifiability analysis. *Exit:* table of success rate, error and time per method and noise level; identifiability figure.
- *Stop rule:* if surrogate calibration systematically lands in different regions than C0, investigate surrogate error near the optimum (local Jacobian error) before drawing conclusions.

### Gate 6 (optional): Track B (week 5 to 6)
- **T6.1** BS operator data (exact labels, cross-checked with Crank-Nicolson). **T6.2** FNO, DeepONet, Transformer, SSM. **T6.3** Resolution transfer, OOD payoffs, Greeks. *Exit:* table and figures; compare with hypothesis RQ6.
- Only start if Gates 1 to 5 are complete. Skip rather than rush.

### Gate 7: Write-up and release (week 6)
- **T7.1** Report (6 to 8 pages): problem, method, protocol, results, limitations, deviations. 
- **T7.2** README with headline table, one key figure, one-command reproduction (`make reproduce`).
- **T7.3** Optional small Streamlit demo: choose parameters, see true vs surrogate surface and calibrate a noisy surface.
- **T7.4** Final checks: clean-clone reproduction, no data or secrets committed, DEVIATIONS.md complete, citations verified.

---

## 8. Timeline summary

| Week | Focus | Gate |
|---|---|---|
| 0 (days 1-3) | Setup, reading, draft protocol | G0 |
| 1 | Pricers, data, freeze | G1 |
| 2 | Baseline, training infrastructure, start M1 | G2 |
| 3 | M1 to M3, final evaluation | G3 |
| 4 | Greeks, arbitrage, speed, start calibration | G4 |
| 5 | Calibration, optional Track B | G5/G6 |
| 6 | Report, README, demo | G7 |

Buffer: if you are more than a week behind at Gate 3, drop Track B and the demo; never drop the frozen protocol or the honest reporting.

---

## 9. Deliverables

- Public repo with README leading with: the problem, headline results table, one key figure, one-command reproduction, limitations.
- Report PDF or `reports/report.md`.
- Results tables (CSV) and figures: error heatmaps, OOD comparison, data-efficiency curves, Greeks error distributions, arbitrage violation plots, speed table, calibration table, identifiability figure.
- `docs/DECISIONS.md` and `docs/DEVIATIONS.md`.
- Optional demo and Track B results.

---

## 10. Failure modes and honest-reporting rules

- **Evaluating on training distribution only.** Always include the OOD sets.
- **Reporting the mean only.** Always include tail errors and max; calibration cares about worst cases.
- **Unfair speed comparison.** Compare with the vectorized GPU COS, not only a slow loop.
- **Tuning leakage.** If the test set influenced any decision, log it in DEVIATIONS.md and flag the affected results.
- **Cherry-picked seeds.** Mean and standard deviation across seeds.
- **Overclaiming novelty.** State what is replicated from published work and what is your own comparison; cite the papers.
- **Ignoring IV inversion failures.** Report the mask and its effect.
- **Treating synthetic calibration as market calibration.** Say clearly that real-data behavior is untested.

---

## 11. Interview preparation (write short answers in `docs/interview.md` as you go)

- Why use the COS method for labels, and how did you validate it?
- Why predict implied volatility rather than price? What breaks at short maturities and deep wings?
- What is the difference between an operator-learning model and a parametric surrogate?
- Where did the MLP beat the fancier models, and why?
- Why can an accurate surrogate still calibrate badly?
- What does it mean that kappa and theta are weakly identified?
- How did you check for arbitrage, and what did you find?
- When is the surrogate genuinely faster than the classical pricer, and when not?

---

## 12. Resume bullet templates (fill in real numbers only)

- Built and benchmarked MLP, DeepONet-style, Transformer and SSM surrogates for Heston implied-volatility surfaces on N parameter samples; best model achieved mean absolute error X vol points (99th percentile Y) in distribution and Z on out-of-distribution regimes, across 3 seeds.
- Validated a batched PyTorch COS pricer against Black-Scholes limits and Monte Carlo; quantified Greeks error and static-arbitrage violations of learned surfaces.
- Implemented gradient-based calibration through the surrogate, achieving success rate S% versus C% for classical calibration at a T-times lower wall-clock cost; analyzed parameter identifiability.
- (If Track B) Compared FNO, DeepONet, Transformer and SSM for the Black-Scholes payoff-to-price operator, including zero-shot resolution transfer.

---

## 13. Stretch goals (only after Gate 7)

- Rough Bergomi or other model without closed-form pricing, with Monte Carlo labels.
- Calibration to a public real option chain, clearly labeled as an exploratory demonstration.
- Local volatility operator with Crank-Nicolson labels.
- Conformal or ensemble uncertainty estimates on surrogate predictions.
- Include r and q as inputs; American options extension.
