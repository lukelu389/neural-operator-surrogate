# Generated datasets

No data exist at T0.1. T1.4 will implement deterministic generation after the
Gate 1 pricers pass validation. Dataset metadata must record seed, configuration
hash, content hash, parameter ranges, grid, and explicit invalid-quote masks.
Normalization statistics must use training data only.

Generated data are ignored by Git. Record reproducibility metadata and the data
card in version-controlled documentation. Test and OOD data may only be opened
by `scripts/final_eval.py`, after the Gate 3 freeze and frozen-hash verification.
