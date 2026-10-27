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
