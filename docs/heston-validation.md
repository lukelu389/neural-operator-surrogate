# T1.2 Heston numerical validation

Verified on 2026-10-28 using explicit numerical fixtures, not training,
validation, test, or OOD experiment datasets. The Heston implementation is in
`src/nop/pricing/heston_cos.py`; the independent Monte Carlo implementation is
in `src/nop/pricing/mc.py`.

## Reproduce

```sh
.venv/bin/python -m pip install -e '.[dev]'
make check
.venv/bin/python scripts/validate_heston.py
```

The study reads [heston_numerics.yaml](../config/heston_numerics.yaml), with
parameter order `(spot, strike, T, v0, kappa, theta, sigma, rho, r, q)`. It writes
`reports/tables/heston_validation.json`, excluded from Git, and reuses a matching
cache. The report records config/source hashes, Git HEAD, package versions,
hardware, precision, seeds, all convergence rows, and Monte Carlo intervals.
Source-content hashes identify the actual executed files, including uncommitted
edits when a run precedes its stage commit. Completed failures are cached too.

For the recorded passing run, config SHA-256 is
`66716375e1eed06312c47a40a75e61f91c649ec3d9681869faf2eb417ff89e86`.
Environment: macOS arm64, Python 3.12.7, NumPy 2.5.4, PyTorch 2.14.1,
PyYAML 6.0.3, CPU float64, six PyTorch threads. CUDA comparison is an optional
unit test and was skipped because CUDA hardware was unavailable.

## Formula and numerical choices

The CF describes `log(S_T/S_0)`, including `(r-q)T`. For real frequency u, use
`w = u² + iu`, `beta = kappa - rho*sigma*iu`, and the principal square root
`d = sqrt(beta² + sigma²*w)` with positive real part. The decaying-exponential
Little Heston Trap representation avoids the unstable exponential branch.
The rearrangement `h = -w/(beta+d)` replaces `(beta-d)/sigma²`; `expm1` and
small-argument Taylor ratios avoid cancellation. The same formula handles
sigma=0 with the full mean-reverting deterministic variance path; small positive
sigma is not silently replaced with Black–Scholes.

Let `I = integral(v dt)` and `M = integral(sqrt(v) dW_S)`. The truncation cumulants
are `c1 = (r-q)T - E[I]/2` and `c2 = E[I] + Var(I)/4 - Cov(I,M)`, evaluated using
closed integral kernels with small-`kappa*T` series. A negative computed variance
raises an error rather than being made positive with `abs`.

The COS interval is `c1 +/- L*sqrt(c2)`, shifted by `log(S0/K)` for payoff
integration. Evaluate the bounded put payoff over its actual support, halve the
zero-frequency term once, and recover calls by parity. This avoids exponentially
large call-payoff coefficients. Handle the zero-frequency integral analytically;
this also avoids an undefined second derivative in PyTorch's `sinc` at zero.
Expiry and identically zero variance return deterministic prices. Nonfinite or
negative prices raise; no numerical price clipping is used.

Sources: [Fang–Oosterlee, §§3 and 5.3](https://ir.cwi.nl/pub/13283/13283D.pdf)
uses first/second cumulants for Heston; [The Little Heston Trap, §3](https://www.ma.imperial.ac.uk/~ajacquie/IC_Num_Methods/IC_Num_Methods_Docs/Literature/HestonTrap.pdf)
gives the stable representation. Independent fixtures come from
[QuantLib v1.38's Heston tests](https://github.com/lballabio/QuantLib/blob/v1.38/test-suite/hestonmodel.cpp#L1698-L1736).
These are references for numerical methods, not project benchmark results.

## Deterministic checks

Tests compare the CF with an independent RK4 solution of its Riccati ODE,
including positive/negative correlation and long maturities. They check CF
normalization/conjugacy, cumulants, sigma=0 and small-sigma Black–Scholes limits,
dividend-adjusted parity, broadcasting, limits, and first/second autodiff.
Four independent Heston price fixtures agree within 1e-10 price units; observed
maximum discrepancy was approximately 5.3e-13 for those fixtures.

The configured study varies N in {128, 512, 2048, 4096} and L in {12, 20, 28, 36}.
Selected fixture settings are N=4096, L=28, compared with N=8192, L=36.
Acceptance is absolute price difference <=1e-8 with spot=1.

| Fixture (three strikes each) | Maximum selected/reference difference |
| --- | ---: |
| Regular | 2.50e-16 |
| Short maturity, low Feller ratio | 8.71e-11 |
| Long maturity, high variance | 2.78e-16 |
| Negative rate, low Feller ratio | 1.67e-16 |

The term count and width must both be checked. An earlier regular-case test at
spot=100 found about 3.58e-8 error with L=12 even after increasing N; increasing
width resolved that truncation error. Agreement of two COS settings is not an
independent accuracy proof, hence the ODE, published-price, and MC checks.

## Independent Monte Carlo

Use full-truncation Euler for an auxiliary variance and log-Euler for stock;
negative auxiliary variance is retained in the recurrence, while its positive
part enters coefficients. This is the documented simulation scheme, not a
modification of analytical prices. See
[Lord, Koekkoek and van Dijk (2010)](https://repub.eur.nl/pub/18571).
Antithetic paths reduce variance; independent observations for the standard
error are pair means. Each run uses 200,000 pairs (400,000 paths). Step counts
are 1024 and 2048 with independent streams from the declared `seed + n_steps`
rule. No seed was selected based on results.

| Fixture | COS | Fine-step MC estimate | Fine-step nominal 95% sampling interval |
| --- | ---: | ---: | --- |
| Regular | 0.0799642924 | 0.0798139754 | [0.0795938134, 0.0800341374] |
| Low Feller ratio | 0.0908907778 | 0.0907516444 | [0.0904936107, 0.0910096781] |

Both fine-step intervals contain COS. Absolute coarse/fine differences were
7.99e-5 and 3.01e-4, within their respective 95% combined-sampling-error limits
3.11e-4 and 3.65e-4. The regular coarse interval narrowly excludes COS, and is
retained in the report. These intervals cover sampling error only; refinement
is evidence about discretization error, not a proof of zero bias or simultaneous
coverage across every check.

The initial 50,000-pair run at 512/1024 steps **failed** both fine-step COS
interval checks and the low-Feller refinement check. It is preserved locally as
`reports/tables/heston_validation_initial.json`; its full config is embedded,
and [DEVIATIONS.md](DEVIATIONS.md) records the settings and observed results.
The response was to increase paths and time resolution while retaining the
confidence multiplier, seed rule, numerical fixtures, and acceptance rules.

## Limits carried forward

These are numerical validation cases, not a uniform error certificate over the
future parameter box or every tail quote. Price tolerance is not IV tolerance:
T1.3 must account for small vega and explicit inversion failures. The conservative
fixture N/L values are not frozen data-production settings; T1.4/T1.5 must finish
the required label/mask checks before freeze. Float32 and extreme tails can lose
accuracy. CPU verification does not establish accelerator support or speed; no
Heston training data or research holdout has been generated or inspected.
