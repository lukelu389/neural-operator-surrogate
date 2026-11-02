# Generated datasets

Run `make data` after installing the project. The command validates the numerical
prerequisites, then generates only the configured training and validation Sobol
windows. Completed chunks are verified and reused; interrupted generation resumes
from valid chunks. A source/config/environment mismatch or corrupt cache is an
error, not permission to overwrite an artifact silently.

The default directory is `data/heston_v1/`, containing `train/` and `validation/`
chunk archives and metadata, `grid.npz`, `normalization.npz`, and `manifest.json`.
Each archive records the seed, config hash, source hashes, and environment;
the manifest records exact file hashes and a combined dataset hash. Failed IVs
remain NaN with an explicit status. Normalization is fitted on valid training
quotes only, with per-grid-point counts and population standard deviations.

For a separate cold regeneration, use
`.venv/bin/python scripts/make_data.py --output data/heston_v1_replay`.
This recomputes the entire configured dataset and can be expensive. Repeating
`make data` verifies cached bytes and reconstructs the summaries and normalization.
Do not delete the original artifacts merely to run a check.

See [the data card](../docs/data-card.md) for the numerical policy, coverage,
reproducibility evidence, limitations, and the distinction between fixtures and
research splits. All labels are synthetic Heston IVs in decimal units.

Generated data are ignored by Git. Record reproducibility metadata and the data
card in version-controlled documentation. Test and OOD data may only be opened
by `scripts/final_eval.py`, after the Gate 3 freeze and frozen-hash verification.
