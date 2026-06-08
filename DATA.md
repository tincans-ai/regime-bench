# Data

RegimeBench keeps large generated parquet files out of Git. The source checkout
contains task definitions, verifier code, and docs. To run the benchmark, fetch
and restore the external data bundle:

```bash
uv run python scripts/data_bundle.py hf-download \
  --repo-id amitvpatel06/regime-bench \
  --revision eb359fb35abc99930b76cb1bf3495949b98d5376 \
  --output-dir .release-data

uv run python scripts/data_bundle.py verify --bundle-dir .release-data
uv run python scripts/data_bundle.py restore --bundle-dir .release-data --repo-root . --force
```

The pinned `v0.1.2` data revision is
`eb359fb35abc99930b76cb1bf3495949b98d5376`.

The source repository is Apache-2.0. Terms for the external parquet bundle are
Apache-2.0, matching the repository. The data bundle contains processed
benchmark parquet files for research and benchmark reproducibility; it is not
investment advice and should not be interpreted as a live trading signal
dataset.

The bundle intentionally includes hidden/OOS labels. This release is therefore
a local reproducibility benchmark, not a blind leaderboard. Any future blind
evaluation should use a separate withheld data service or private evaluation
set.

The primary data bundle contains only the 18 base task train/test parquet files.
Split tasks are generated locally from those files:

```bash
uv run python scripts/materialize_splits.py \
  --profile split_profiles/report.toml \
  --output task_splits/report-profile \
  --force
```
