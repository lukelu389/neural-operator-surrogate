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
