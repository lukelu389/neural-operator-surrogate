# T3.1 M1 DeepONet-style validation workflow and results

M1 is a DeepONet-style synthetic-Heston IV surrogate. This page records the
complete validation-only architecture comparison stage; it contains no test,
OOD, calibration, or market evidence. Validation was used for trial selection
and early stopping, so these values are development results, not an unbiased
final comparison.

## Fixed architecture and equal budget

`config/m1.yaml` fixes a branch on the five training-standardized Heston
parameters and a trunk on forward log-moneyness/maturity coordinates. Each tower
has three 480-wide SiLU hidden layers and a 128-dimensional output. Their dot
product is scaled by `1/sqrt(128)`. The model has **1,050,977 trainable
parameters**, 3.73% above selected M0's 1,013,160. The runner checks that M0's
selected config still hashes to its committed model architecture and enforces
the frozen 10% parameter budget before training.

The branch uses the same parameter standardization as M0. The trunk uses fixed
coordinate scales from the native grid. Its score is mapped through softplus to
positive decimal IV, then standardized using the same training-only coordinate
mean and log-standard-deviation moments. Arbitrary query coordinates are
supported by `DeepONet.query(parameters, k, T)`; the shared training loop uses
the native-grid `forward(parameters)` path.

M1 used the same verified 200,000-surface training split, 20,000-surface
validation split, frozen AdamW/warmup-cosine schedule, 12 paired PCG64 draws,
training seed 101, and final seeds 11/29/47 as M0. All trials were retained; no
trial failed and no draw was replaced. Selection used validation MAE with the
frozen p99, parameter-count, and lexical tie-breaks. M1 has no additional
architecture stop threshold; no result authorized access to test/OOD.

## Tuning results

| Trial | Status | Best epoch | Validation MAE | p99 | Max |
|---|---|---:|---:|---:|---:|
| 00 | early stopped | 9 | 0.082753 | 0.454550 | 2.556303 |
| 01 | early stopped | 53 | 0.016138 | 0.085017 | 1.037675 |
| 02 | early stopped | 40 | 0.026992 | 0.176045 | 1.818117 |
| 03 | early stopped | 25 | 0.030697 | 0.172919 | 1.801050 |
| 04 | early stopped | 11 | 0.078069 | 0.416614 | 2.736145 |
| 05 | early stopped | 29 | 0.031949 | 0.161453 | 1.565626 |
| 06 | early stopped | 48 | 0.017439 | 0.120472 | 1.587808 |
| 07 | early stopped | 15 | 0.050131 | 0.284402 | 2.169913 |
| 08 | completed | 99 | 0.009724 | 0.077249 | 1.173928 |
| 09 | early stopped | 41 | 0.021338 | 0.139388 | 1.805422 |
| 10 | early stopped | 47 | 0.019031 | 0.126803 | 1.630827 |
| 11 | early stopped | 41 | 0.037976 | 0.264848 | 2.203258 |

Errors are in vol points; 1 vol point equals 0.01 decimal IV. Trial_08 was
selected, with learning rate `0.0003771660758558053` and weight decay
`0.00002622503937088`. It completed 100 epochs/78,200 optimizer updates. Its
validation metrics were MAE 0.009724, RMSE 0.019945, p99 0.077249, and max
1.173928 vol points.

## Three-seed validation results

| Seed | Best epoch (completed) | MAE | RMSE | p99 | Max |
|---:|---:|---:|---:|---:|---:|
| 11 | 100 (100) | 0.009924 | 0.020053 | 0.077073 | 1.327422 |
| 29 | 48 (58) | 0.020842 | 0.035062 | 0.128592 | 1.617044 |
| 47 | 52 (62) | 0.019994 | 0.034145 | 0.125355 | 1.548791 |
| Mean ± sample SD | — | 0.016920 ± 0.006073 | 0.029753 ± 0.008413 | 0.110340 ± 0.028856 | 1.497752 ± 0.151406 |

Each seed attempted 2,400,000 validation quotes. There were 2,393,937 valid
reference quotes and 6,063 reference failures (0.252625%) per seed. All three
seeds had zero prediction failures on valid quotes. Metrics and tails are
computed per seed before arithmetic-mean/sample-SD aggregation (`ddof=1`); quote
predictions are not pooled. The mean max error of 1.497752 vol points remains
material despite the lower mean MAE.

The full report includes metrics per maturity, scaled moneyness, and native cell,
all trial and seed records, mask counts, run IDs, checkpoint/report/prediction
hashes, and environment provenance. The validation report is
`reports/tables/m1_validation.json` (SHA-256
`a38ce291cde3615c65b2970ac95b4c8bd8a4e8e6b771d85426ed787b338e655a`). Its
ignored static figures are:

- `reports/figures/m1/learning_curves.png` — SHA-256
  `b9b3c2c51c62402898f1c6e22127432559de1598e997b3da4b55809a64b6b0db`
- `reports/figures/m1/validation_error_heatmaps.png` — SHA-256
  `11100100b754173582c1a1548db9bde071c5506734c68602ae32fa976e944705`

The selected configuration is tracked in `config/m1_selected.yaml` (SHA-256
`4177bb72bf9b40b64a4d96c84950f31dbb8b98f8f55f28f632b67724ca33f40c`). The
architecture config hash is `74aedbc041a32a4f53441cd492c23c9270fc317eb3c2bd95a1f8fdd9e520d42a`.
Generated reports, figures, checkpoints, predictions, and MLflow files remain
outside Git.

## Reproduction and reporting correction

In a clean checkout, install the package and run:

```sh
.venv/bin/python -m pip install -e '.[dev]'
make check
.venv/bin/python scripts/train_m1.py
```

This runs the complete 12-trial budget followed by the three selected-config
seeds; repeating it after interruption resumes from the same verified caches.
`--tune-only` stops after all twelve terminal trials. In the current workspace,
training was committed and run at `c7072e3`. Visual QA later found that the M1
plot call had inherited the shared helper's M0 title. Commit `d8dddd0` fixed the
label and added `--figures-only`, which verifies cached per-run report hashes and
the train/validation-only dataset identity, then regenerates the figures without
training. That correction changed no checkpoints, predictions, trial selection,
metrics, or MLflow log records; its scope and source hash are recorded in the
validation report. Running the full trainer against the pre-correction cache
will correctly refuse the changed source identity. Use
`.venv/bin/python scripts/train_m1.py --figures-only` for the current cached
report, or run the full command in a clean checkout to reproduce from scratch.

`make check` passed Ruff, strict mypy (42 source files), and 791 tests; one CUDA
test was skipped because the hardware was unavailable. All eight frozen hashes
verify. No test/OOD data was opened. T3.2 Transformer is next; Gate 3 final
evaluation remains blocked until all three architectures and T3.4 are complete.
