# T2.2 M0 baseline workflow and validation results

The implementation and search budget were fixed before production trials. This
page records validation-only development evidence; it contains no independent
test, OOD, or market evidence.

## Architecture and budget

`config/m0.yaml` defines four width-560 SiLU hidden layers, five standardized
parameter inputs, and 120 native-grid outputs: **1,013,160 trainable parameters**.
Decimal IV is `softplus(raw_score)`, then standardized with the saved training
mean/std for the shared loss. Zero output weights and inverse-softplus(mean)
bias initialize the output at the training mean. This parameterization enforces
positive IV without clipping predictions after inference; underflow/nonfinite
values still trigger explicit failure. M1–M3 will use the same output contract.

`config/training.yaml` supplies twelve paired PCG64 learning-rate/weight-decay
draws, seed 101, AdamW, warmup/cosine scheduling, clipping, epoch cap, and early
stopping. Every terminal numerical failure consumes its existing trial slot.
Selection requires the entire declared budget and follows MAE, p99 error,
parameter count, then lexical trial ID. No architecture or extra hyperparameter
search is performed here.

After all twelve trials, the selected tuning run must have validation MAE no
greater than the frozen 0.2-vol-point stop threshold. Otherwise the command
records the stop and does not launch seed retraining. On passing, the selected
configuration is saved once to `config/m0_selected.yaml`, then trained with seeds
11/29/47. All three seeds must succeed and their arithmetic-mean validation MAE
must pass the same threshold. No best-seed or complete-case average can pass the
gate. A stop requires investigation and a deliberate decision, not extra trials.

## Execute or resume

```sh
.venv/bin/python -m pip install -e '.[dev]'
make check
.venv/bin/python scripts/train_m0.py
```

Use `--tune-only` to stop after the same twelve-trial budget; running the ordinary
command later verifies and reuses those trials before continuing the three seeds.
There is no trial-count override. The default CPU execution uses four torch
threads, set in `config/m0_run.yaml` independently of model choice.

An OS-held process lock rejects overlapping writers, and a killed process releases
the lock automatically. Keep the existing run directory and repeat the same
command after interruption. Each run verifies actual data/initial-weight/config/
source identities, checkpoint hashes, and saved RNG/optimizer/scheduler state.
A completed trial is reused; stale or corrupt state raises rather than retrains.
The original commit is preserved across later source-identical stage commits.

Files to inspect:

- `runs/m0_v1/experiment.json`: paired draws, terminal trial records, current phase,
  selected configuration, three-seed records, and gate outcome.
- `runs/m0_v1/trial_*/checkpoint.json` and `seed_*/checkpoint.json`: last/best
  committed epoch, checkpoint hashes, and resumable state. Per-run `report.json`
  includes full learning histories and numerical failures.
- `mlruns`: local MLflow configs, source/data hashes, epoch metrics, checkpoints,
  and reports; no tracking server or external service is used.
- `reports/tables/m0_validation.json`: final validation-only summary or explicit
  tuning-stop evidence.
- `reports/figures/m0`: all-trial/all-seed learning curves and native T×z error
  heatmaps. Axes label maturity in years, z, and the associated k convention.

Generated data, predictions, logs, and checkpoints remain outside Git. Only
scripts, configs, hashes, verification evidence, and documentation are committed.

## Interpretation and remaining gates

Metrics are computed separately for every seed overall and by maturity, z, and
native cell; means and sample standard deviations use `ddof=1`. In particular,
p99 and maximum errors are not computed from pooled seeds. Reference masks and
prediction-failure counts remain explicit. Missing/undefined seeds prevent the
affected aggregate metric from being reported as a complete comparison.

These validation surfaces inform selection, so the reported errors are development
results. No test/OOD data is opened. The Gate 3 freeze, data-efficiency/withheld-query
runs, vega ablation, and final evaluation remain separate future requirements.
A passed M0 gate permits the next architecture; it does not establish the final
0.05-vol-point test target, market performance, or an inference speed-up.

## Completed validation results

All twelve paired draws completed or failed terminally, and all failures consumed
their declared slots. Trial_00 failed after 10 epochs/8,040 optimizer steps with
3,208 valid-quote predictions reported as nonfinite or nonpositive. Trial_04
failed after 8 epochs/6,967 steps with 6,574 such predictions. The failure report
does not distinguish the two causes. No replacement trials were run.

Trial_02 was selected by the declared validation-only ordering. Its learning rate
was 2.72119067e-4 and weight decay 4.90797956e-3; it completed 100 epochs and
78,200 optimizer steps. Its validation MAE was 0.003914 vol points (p99 0.027515,
max 0.531349), passing the frozen 0.2-vol-point tuning stop check. The required
seeds 11, 29, and 47 all completed and the arithmetic-mean MAE passed the same
threshold; no seed was replaced or selected as a best seed.

| Seed | Best epoch (completed) | MAE (vol pts) | RMSE (vol pts) | p99 (vol pts) | Max (vol pts) |
|---:|---:|---:|---:|---:|---:|
| 11 | 25 (35) | 0.018670 | 0.027138 | 0.085602 | 1.010367 |
| 29 | 24 (34) | 0.017206 | 0.026384 | 0.089158 | 1.101190 |
| 47 | 13 (23) | 0.030043 | 0.044777 | 0.142131 | 1.600030 |
| Mean ± sample SD | — | 0.021973 ± 0.007027 | 0.032766 ± 0.010409 | 0.105630 ± 0.031661 | 1.237196 ± 0.317488 |

Each seed attempted 2,400,000 validation quotes; 2,393,937 were valid and 6,063
reference quotes failed (0.252625%). All three models had zero prediction
failures on valid quotes. The worst observed per-seed maximum error was 1.600030
vol points, so the small mean does not describe tail behavior by itself. The
0.2-vol-point rule is a Gate 2 permission to proceed, not the separate final
accuracy target or a guarantee of generalization.

The complete per-seed and per-maturity/z/cell metrics, trial records, selected
configuration, checkpoint/report hashes, and run IDs are retained in the ignored
validation report `reports/tables/m0_validation.json` (SHA-256
`a607995a29f8c82b8991d98799d0f011e07daed37efcb128fc7e51340ac3544`). The static
figures are `reports/figures/m0/learning_curves.png` (SHA-256
`18710c1875e7d964a461dc45a6546e2c82f5659b222c3bc2b1409c3911cbb944`) and
`reports/figures/m0/validation_error_heatmaps.png` (SHA-256
`a028a053a8060147a3c8a191aa82a9342c2c11bf997fc0eeb36d5fbf66624989`). Repeating
`.venv/bin/python scripts/train_m0.py` reused completed runs; the report, figure,
per-run report, MLflow run-ID, and logged training-point identities remained
unchanged. No test/OOD data was accessed.
