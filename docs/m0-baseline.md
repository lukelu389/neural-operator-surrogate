# T2.2 M0 baseline workflow

The implementation is fixed before production trials. Results are pending; this
page describes the declared experiment and how to resume it. It does not contain
independent test evidence.

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
