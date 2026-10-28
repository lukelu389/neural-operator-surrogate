# Decisions

## 2026-10-27 — T0.1

- Copy the supplied PLAN.md, AGENTS.md, and TODO.md into the empty repository so the agreed research protocol is versioned with the implementation.
- Respect the user's explicit clarification to complete only T0.1 this session; later gates remain pending.
- Use a setuptools `src/nop` package, Python 3.11+, and Make targets so development and clean-checkout verification use the same commands.
- Add NumPy and PyTorch for the required random-number and DataLoader seed utilities; defer SciPy, YAML parsing, plotting, and experiment logging dependencies until their tickets need them.
- Add Ruff, pytest, and mypy as development dependencies to enforce formatting, lint, reproducibility checks, and strict types with `make check`.
- Keep the unimplemented data, final-evaluation, and reproduction commands fail-closed with nonzero exits; do not create placeholder datasets, hashes, or research results.
- Use bounded dependency ranges for installation compatibility; record the exact environment used to validate this ticket in its handoff, and require full environment metadata for future experiments.
- Let mypy use the active interpreter version so it can parse the dependency stubs selected for that interpreter; retain Python 3.11 as the package minimum and Ruff language target.
- Preserve the supplied AGENTS.md and PLAN.md byte-for-byte, including existing whitespace, to avoid changing the agreed instructions during scaffolding.

### Verification

- Source commit `d11e4dd` passed `make check` both in the working repository and in a clean local Git clone with a newly created virtual environment and independently installed dependencies: Ruff formatting/lint, strict mypy (11 source files), and 22 passing tests.
- Verified runtime: macOS arm64, Python 3.12.7, NumPy 2.5.4, PyTorch 2.14.1, pytest 9.2.0, Ruff 0.17.3, mypy 1.20.2. CPU checks only; Python 3.11 and accelerator execution are unverified.
- Confirmed `make data`, `make final-eval`, and `make reproduce` return nonzero, and virtual environments, datasets, checkpoints, local secrets, and generated reports are ignored by Git. No research artifacts were created.

## 2026-10-28 — T0.2

- Follow the user's publication instruction: create public `lukelu389/neural-operator-surrogate`, push T0.1 before starting T0.2, and commit/push each completed stage after verification; `main` initially matched T0.1 commit `9a980c9`.
- Write eight compact core-paper notes with primary-source links, versions, actual baselines, and separate critical assessments so published results are not presented as our experimental evidence.
- Record preprint versus journal dates and source-access limits explicitly; use accessible complete manuscripts or identified copies rather than inventing unavailable publication details.
- Select a bidirectional Mamba-style selective SSM as the M3 implementation direction: an explicit PyTorch recurrence is easier to inspect for this short coordinate sequence than full S4 structured-kernel machinery; neither accuracy nor fused-kernel speed is assumed.
- Leave model dimensions, hyperparameters, dataset choices, and protocol thresholds to their designated gates; reading the papers does not authorize additional architecture trials or Track B implementation.
- Defer reuse of Luke's Poseidon notes because no source was supplied or found in the repository/targeted connected-note search; all eight core-paper notes can be completed without fabricating that personal research context.
- Reinstall the existing editable package after Python's verbose startup showed its hidden `.pth` file was being skipped; changing the flag alone did not persist. `.venv/bin/python -m pip install --no-deps --no-build-isolation -e .` restored imports without tracked packaging or application code changes.

### Verification

- `make check` passed Ruff formatting/lint, strict mypy (11 source files), and all 22 tests after the local editable-install repair; no dependency was added or upgraded for T0.2.
- All eight notes contain methods, reported claims/comparisons, and independent critical/project assessments; relative Markdown links resolve, and publication versions/source-access limits are disclosed.
- No paper benchmark was reproduced, and no research data, model, frozen configuration, or test/OOD result was created or accessed.
