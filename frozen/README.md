# Frozen protocol manifests

[manifest.json](manifest.json) records the T1.5 Gate 1 freeze: six configuration
files and two numerical adapter implementations, plus verified data coverage and
numerical-report identities. Run `.venv/bin/python scripts/freeze_protocol.py --verify`
to check every listed file hash. This works without generated artifacts.

The freeze-time commit identifies the parent checkout; exact new file identities
are the manifest hashes. The commit containing this manifest is the published
freeze. Generated reports/datasets remain outside Git; see
[the freeze record](../docs/protocol-freeze.md) for their provenance.

Gate 1 does not authorize test/OOD access. T3.4 must separately freeze selected
model configurations and all final checkpoints, then verify both gates before
`scripts/final_eval.py` alone opens holdouts. Until that evaluator exists,
`make final-eval` always refuses.

Never edit a frozen file without Luke's approval and a corresponding entry in
`docs/DEVIATIONS.md` documenting the change, reason, and results already seen.
