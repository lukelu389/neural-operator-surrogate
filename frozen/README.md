# Frozen protocol manifests

No protocol or configuration is frozen yet. T1.5 will record SHA-256 hashes of
the data configuration and evaluation protocol; T3.4 will add model configuration
hashes. Final evaluation must verify every required frozen file before opening
test or OOD data. Until that evaluator exists, `make final-eval` always fails.

Never edit a frozen file without Luke's approval and a corresponding entry in
`docs/DEVIATIONS.md` documenting the change, reason, and results already seen.
