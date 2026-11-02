# T2.1 training mechanics and sanity checks

The architecture-neutral loop accepts standardized CPU float32 parameter and IV
arrays, explicit Boolean quote masks, and fixed training-derived target moments.
It neither generates data nor opens holdouts. Later architecture tickets must
construct their models and declared trial settings from the frozen protocol.

## Training and selection

Each optimizer update uses the valid-quote mean of squared standardized residuals,
optionally multiplied by fixed training-derived vega weights. Invalid positions
are indexed out before arithmetic so NaN labels cannot poison gradients. Empty
batches skip both the optimizer and schedule. Epoch loss uses quote counts, so a
short final batch or different mask coverage cannot change its weighting.

AdamW, linear warmup followed by cosine decay, global gradient-norm clipping,
seeded epoch shuffles, and validation-MAE early stopping follow the shared policy.
Validation uses the training moments and reports MAE, RMSE, linear-interpolated
p99 absolute error, maximum error, and explicit failure counts in vol points.
An invalid prediction on a valid reference quote fails the trial; it cannot shrink
that model's evaluation mask. Tied validation scores retain the earlier checkpoint.

The low-level loop is generic for sanity tests. It is not a tuning-budget manager;
T2.2's orchestration must enforce the twelve declared trials and frozen settings.
No production architecture or search is selected by T2.1.

## Resume and artifact identity

Checkpoints include model, optimizer, scheduler, Python/NumPy/torch RNG state,
completed epochs, history, and provenance. An atomic generation manifest commits
one epoch at a time and retains only the latest and best payloads. A process
interrupted inside an epoch replays from the prior committed boundary. The same
initialized model, actual data bytes, settings, source hashes, and environment are
required; corrupt or stale caches raise instead of silently retraining.

Completed runs restore the earliest best validation weights. Resume and cached
runs replay committed log records idempotently, repairing a logging interruption.
Logging is isolated from training RNG so adding a logger cannot change dropout or
other stochastic updates. Failed numerical trials remain failed on cache replay.
A caller must not launch concurrent writers for the same run directory.

MLflow writes to explicitly local tracking, registry, workspace, and artifact
URIs. Identity JSON preserves full configuration, seed, dataset/source hashes,
commit, and environment; epoch metrics and final reports/checkpoints are local
artifacts. Current MLflow requires a scoped FileStore compatibility opt-in;
telemetry is disabled before import and during calls. See the official
[FileStore documentation](https://mlflow.org/docs/latest/self-hosting/migrate-from-file-store/)
and [usage tracking documentation](https://mlflow.org/docs/latest/community/usage-tracking/).
The lightweight client package supplies this required logger; no server/account
or external tracking service is used.

## Reproduce the sanity checks

```sh
.venv/bin/python -m pip install -e '.[dev]'
make check
.venv/bin/python scripts/validate_training.py
```

`config/training_sanity.yaml` specifies one fixture: the first 100 training rows,
normalization fitted only to those rows, a three-layer width-256 SiLU network,
zero-initialized output head, seed 101, and at most 5,000 optimizer updates. The
frozen success criterion is standardized masked MSE at most 1e-4. The assessment
uses those same 100 rows; internal `validation_metrics` names in the reusable
loop mean reconstruction metrics here, not unseen-data performance. A separate
fixed linear fixture checks strict monotone descent; nonlinear AdamW losses are
not asserted to decrease on every update.

The script verifies Gate 1 hashes and dataset identity before loading training
labels. Reports live in `reports/tables/training_sanity.json`, checkpoints in
`runs/training_sanity_v1`, and logs in `mlruns`. These are ignored by Git. An
identical rerun verifies and reuses checkpoints and the same MLflow run; it does
not spend another training trial. A changed source/config keeps the old evidence
and fails for investigation instead of overwriting it.

Limits: this small reconstruction exercise validates optimization mechanics,
not model comparison, generalization, calibration, accelerator performance, or
production training speed. CPU and fixed-environment replay are the tested scope.

## Observed fixture results (2026-11-02)

The single configured CPU run passed: 5,000 updates, 163,960 trainable parameters,
standardized masked MSE `1.198507561639417e-5` (criterion `1e-4`). It retained all
12,000 quote positions, with 11,967 valid and 33 reference failures. The best
reconstruction epoch was 5,000, with zero invalid predictions.

| Same-100-row reconstruction metric | Vol points |
| --- | ---: |
| MAE | 0.02094764249 |
| RMSE | 0.02923971134 |
| p99 absolute error | 0.09757363796 |
| Maximum absolute error | 0.25949031115 |

These are explicitly not held-out results. Training-loop time was approximately
144.89 seconds on CPU with four torch threads, including validation, checkpoint
and epoch logging overhead; this is not a speed benchmark. The separate linear
fixture fell monotonically from 0.234375 to `2.0502360070552507e-8`.

- Sanity report SHA-256: `8aa84439282554c1c487a754fd619121b5daeee2475acd3be7f8b0272ab43ca0`.
- Best checkpoint SHA-256: `89d36c7dc4fda7d03c65cc6035ad07fe2e21886111a739b93748828f0b163e22`.
- MLflow run: `9e0ba187b89b47378511ce508141299d`.
- Verification: 699 tests passed, one CUDA skip; Ruff and strict mypy passed.

A cached command replay preserved the report bytes, checkpoint hash, run ID, and
exactly 5,000 loss-history records. A separate interrupted synthetic fixture
verified bitwise equality of model/optimizer/scheduler/RNG states and metric
history; a second cold 5,000-update reconstruction run was not performed.

The run records Python 3.12.7, macOS arm64, PyTorch 2.14.1, NumPy 2.5.4, PyYAML
6.0.3, and mlflow-skinny 3.17.2. Local hidden editable `.pth` behavior required a
regular package install for child-process verification; no source-path workaround
was added. Reinstall the package after source edits in that local environment.
