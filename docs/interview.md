# Interview preparation

## T0.1 — Scaffold and reproducibility

**Why is setting one seed insufficient?** Python, NumPy, PyTorch, and each
DataLoader worker maintain distinct random-number state. Seed each source,
pass a seeded generator to loaders, and explicitly seed independently created
NumPy generators. Fixing seeds also does not remove nondeterministic kernels
or differences between hardware and library versions.

**What does `make check` establish?** It checks formatting, lint, types, and
automated behavior of the implemented scaffold, including repeated random
streams and safe rejection of pending research commands. It does not validate
option prices or research claims; those have explicit later gate criteria.

**How do you prevent accidental test-set leakage?** Keep all development
decisions on training and validation data. Restrict holdout access to the final
evaluation script, require frozen-file hash verification before any holdout is
opened, and record deviations. At T0.1 the command refuses every evaluation;
the actual hash-verifying evaluator is still pending T3.4.
