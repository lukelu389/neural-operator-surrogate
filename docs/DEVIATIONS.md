# Deviations

No protocol deviations at T0.1. No data, metrics, or model results have been seen.
The evaluation protocol and data/model configurations have not been drafted or frozen.

Record future deviations here with the date, affected files, change, reason,
approval where required, and results already observed. Never silently relax a
gate, stop rule, tuning budget, or evaluation restriction.

## T0.2 — reading/source limitations

No protocol deviation. Eight core-paper notes are complete; no data were
generated, no model was trained, and no test/OOD set was opened. The protocol
and configurations remain undrafted/unfrozen.

Source limitations are documented in `docs/reading/README.md` and the individual
notes: inaccessible publisher pages were replaced by accessible paper texts;
The Little Heston Trap's final magazine typesetting/DOI were not verified.
Luke's own Poseidon notes were unavailable and have not been represented as read.
These are disclosure limits, not evidence that the papers' experiments were
reproduced or that any numerical gate has passed.

## 2026-10-28 — T0.3 draft sequencing interpretations

Affected files: `config/eval_protocol.yaml`, `config/README.md`, and README status
and limitations. PLAN.md and AGENTS.md remain unchanged. These choices are
unfrozen design proposals, made before any dataset, model, or numerical result
exists; no holdout was generated or read and no frozen file was changed.

1. **Data creation versus access restriction.** T1.4 describes generating the
   splits/OOD datasets, but AGENTS.md allows test/OOD sets to be touched only by
   `scripts/final_eval.py` after Gate 3. The draft follows that stricter rule:
   T1.4 reproduces training/validation and records deterministic holdout recipes;
   Gate 1 freezes recipe hashes. Final evaluation generates holdout artifacts and
   records their hashes after verifying the frozen protocol, configs, and
   checkpoints. Actual holdout-file hashes cannot be checked before the files
   exist. A requirement for all split artifacts at T1.4 would need an explicit
   resolution of this conflict before freeze.
2. **One final pass versus later diagnostics.** T3.4 opens holdouts once, while
   Greeks/arbitrage and calibration implementations arrive at Gates 4 and 5.
   The draft assigns those later analyses to validation parameters, with frozen
   settings and explicit validation-only labels. This satisfies parameter
   exclusion from training, but validation was used for model selection and is
   not independent test evidence. A later claim requiring independent test
   diagnostics or calibration would need a revised evaluation schedule before
   freeze, not an unreported second holdout pass.

The unresolved numerical choices are enumerated in
`required_before_G1_freeze`. T0.3 completion is not permission to freeze this
draft as-is. Any later approved change to frozen files must still record its
reason, approval, and all results already seen.

## 2026-10-28 — T1.1

No protocol deviation. Implemented and tested the Black–Scholes reference pricer
and analytic Greeks on explicit unit-test inputs. No Heston labels, datasets,
training runs, or test/OOD experiment artifacts were generated or accessed.
The draft evaluation protocol and all gate boundaries remain unchanged.

## 2026-10-28 — T1.2 numerical refinement before freeze

No frozen-protocol change. T1.2 used explicit numerical fixtures, not research
split data. Two implementation/test issues were corrected: a narrower COS
interval failed its requested tolerance despite increasing N, and the zero-term
`torch.sinc` path produced NaN second derivatives. Width refinement and an
analytic zero-frequency integral resolved them without weakening tolerances.

The first configured MC run used 50,000 antithetic pairs at 512/1024 steps,
`z=1.959963984540054` (nominal 95%), and the fixed case seed plus step-count rule.
Its config hash was
`0ffcbc64265e9c7bd16f170a41f38f51d4966fdd19294e3fef3c0cabce17df11`.
Both fine-step comparisons failed: regular COS 0.0799642924 was below
[0.0799906341, 0.0808698918]; low-Feller COS 0.0908907778 was above
[0.0897853744, 0.0908170134]. The low-Feller refinement check also failed.
These results were seen before increasing the numerical effort and are preserved
locally in `reports/tables/heston_validation_initial.json` (including its config).

Changed only the study effort to 200,000 pairs and 1024/2048 steps; retained the
seed rule/base seeds, 95% confidence multiplier, cases, pricing tolerances, and
acceptance rules. Both fine-step comparisons and refinement checks then passed.
The final regular coarse-step interval still narrowly excludes COS and remains
reported. The initial run is reproducible by restoring those two effort settings
in a copy of the committed config; no seed search or confidence widening was used.

Sampling uncertainty and timestep bias remain distinct. Sequential numerical
refinement and these few fixtures do not establish uniform accuracy, simultaneous
95% coverage, or production IV-label validity. No Heston dataset was generated,
and T1.3/T1.4 requirements remain pending.
