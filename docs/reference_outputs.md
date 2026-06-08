# Reference Outputs

The reference smoke is a deterministic check that the public verifier behaves
as expected in a fresh install:

```bash
uv run python scripts/reference_smoke.py \
  --output-dir reference_runs/synthetic-smoke
```

The script creates a tiny synthetic task in the requested output directory,
runs the canonical repair-style verifier on a deterministic candidate, writes
`reference_summary.json`, and compares selected metrics to
`reference_outputs/synthetic_smoke_expected.json`.

This check is intentionally small and independent of the hosted market data
bundle. It is a verifier sanity check, not a model-quality benchmark.
