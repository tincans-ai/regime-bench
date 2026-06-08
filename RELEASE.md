# RegimeBench v0.1.2

RegimeBench `v0.1.2` is the public report release for the distribution-shift
signal benchmark.

## Release Identity

- Source repository: https://github.com/tincans-ai/regime-bench
- Source tag: `v0.1.2`
- Data repository: https://huggingface.co/datasets/amitvpatel06/regime-bench
- Data revision: `eb359fb35abc99930b76cb1bf3495949b98d5376`
- Data bundle: `regime-bench-base-data`
- Data payload: 36 parquet files, 1,221,241,781 bytes
- License: Apache-2.0 for source and base data bundle

## Changes Since v0.1.1

- Adds the companion report, "When Hill Climbing Isn't Enough: RegimeBench,"
  as Markdown, LaTeX, and a release PDF.
- Adds the final report figures used by the companion paper.
- Recreates the public repository history from the audited release tree so
  stale exploratory notebooks and internal notebook outputs are not part of the
  public Git history.

## Carried Forward From v0.1.1

- The public runner and verifier CLI now use the repair-style saved-code replay
  as the default scoring engine.
- The default replay mode is `strict`, with hidden/OOS current return labels
  withheld from `build_signal(data_path)`.
- The bar-level causality gate remains enabled by default and is part of the
  canonical public verifier path.
- The data bundle revision is unchanged.

## Scope

This release includes:

- 18 public `signal-*` tasks.
- Base train/test parquet files for all 18 tasks.
- The canonical public repair-style PnL verifier and causality gate.
- Local task runner and split materialization tooling.
- A deterministic synthetic verifier smoke expectation.

This release excludes:

- Reference solutions.
- Private orchestration, worker infrastructure, credentials, and raw provider
  logs.
- A blind hosted leaderboard.

## Verification

Expected local release checks:

```bash
uv run --extra data --extra test pytest -q
uv run python scripts/data_bundle.py verify --bundle-dir .release-data
uv run python scripts/reference_smoke.py --output-dir reference_runs/synthetic-smoke
```

Because hidden/OOS labels are included in the public data bundle, results from
this release should be described as reproducibility or local-method comparison
results rather than blind leaderboard scores.
