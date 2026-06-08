# RegimeBench

RegimeBench is a benchmark for agent-built trading signals under market regime
shift. A submission implements one function, `build_signal(data_path)`, and is
scored by replaying the resulting signal through a deterministic PnL verifier
with a bar-level causality gate.

Current release: `v0.1.2`. The pinned data bundle revision is
`eb359fb35abc99930b76cb1bf3495949b98d5376` from
[amitvpatel06/regime-bench](https://huggingface.co/datasets/amitvpatel06/regime-bench).
This is a reproducibility benchmark release: the data bundle includes the
hidden/OOS labels needed to reproduce local verifier scores, and is not a blind
hosted leaderboard.

The repository contains the public benchmark surface:

- `tasks/`: 18 canonical generated `signal-*` Harbor-style tasks.
- `task_configs/` and `task_templates/`: task metadata and templates used to
  document the generated task surface.
- `runner/run_task.py`: local one-task smoke runner.
- `verifier/pnl_verifier.py`: canonical repair-style PnL replay CLI.
- `scripts/data_bundle.py`: data bundle download, verify, restore, stage, and
  upload tooling.
- `scripts/materialize_splits.py`: deterministic split-task generation from
  restored base data.
- `split_profiles/report.toml`: the frozen split profile used by the companion
  report.
- `docs/`: task interface, scoring, and split-profile notes.
- `reference_outputs/`: deterministic verifier smoke expectations.
- `reports/regimebench-2026/`: companion paper source/PDF and final figures.

Large parquet task data and report artifacts are distributed separately through
the RegimeBench data bundle, not committed to Git.

## Companion Paper

The companion report is included in this repository:

- [When Hill Climbing Isn't Enough: RegimeBench](reports/regimebench-2026/report.pdf)
- [Markdown source](reports/regimebench-2026/report.md)
- [LaTeX source](reports/regimebench-2026/report.tex)

The paper studies whether LLM coding agents can turn limited in-sample feedback
into out-sample research judgment, and summarizes the baseline, split
robustness, train/validation/test, and stopping-policy experiments.

## Quick Start

Install dependencies:

```bash
git clone https://github.com/tincans-ai/regime-bench.git
cd regime-bench
uv sync --extra data --extra test
```

Download and restore the data bundle:

```bash
uv run python scripts/data_bundle.py hf-download \
  --repo-id amitvpatel06/regime-bench \
  --revision eb359fb35abc99930b76cb1bf3495949b98d5376 \
  --output-dir .release-data

uv run python scripts/data_bundle.py verify \
  --bundle-dir .release-data

uv run python scripts/data_bundle.py restore \
  --bundle-dir .release-data \
  --repo-root . \
  --force
```

Run the included example candidate on one task:

```bash
uv run python runner/run_task.py \
  --task tasks/signal-volume \
  --candidate examples/simple_momentum.py \
  --output-dir verifier_runs/signal-volume
```

Run only the canonical verifier. By default this uses the repair-style replay
with the bar-level causality gate enabled:

```bash
uv run python verifier/pnl_verifier.py score-code \
  --task-dir tasks/signal-volume \
  --code-path examples/simple_momentum.py \
  --output-dir verifier_runs/signal-volume-verifier
```

## Candidate API

Submissions are Python files that define:

```python
def build_signal(data_path):
    ...
    return dataframe_with_timestamp_ticker_signal
```

The returned frame must contain `timestamp_agg`, `ticker`, and finite numeric
`signal` values. Higher signal means higher expected next-period residual
return. During hidden/OOS scoring, current hidden return labels are withheld
from the feature frame and the verifier scores only prediction-period rows.

See [docs/task_interface.md](docs/task_interface.md) and
[docs/scoring.md](docs/scoring.md) for details.

## Reference Smoke

Run the deterministic verifier smoke to confirm that your local install matches
the release expectation:

```bash
uv run python scripts/reference_smoke.py \
  --output-dir reference_runs/synthetic-smoke
```

The smoke builds a tiny synthetic task, runs the public verifier, and compares
selected metrics to [reference_outputs/synthetic_smoke_expected.json](reference_outputs/synthetic_smoke_expected.json).

## Split Tasks

The main data bundle ships only base task parquet files. Report split tasks are
derived artifacts and can be regenerated after restoring base data:

```bash
uv run python scripts/materialize_splits.py \
  --profile split_profiles/report.toml \
  --output task_splits/report-profile \
  --force
```

This creates `task_splits/report-profile/tasks/`, plus
`split_manifest.json`, `split_manifest.parquet`, and `metadata.json`.

## Data Bundle Profiles

The bundle tool currently supports:

- `benchmark-source`: source files needed to recreate this repository.
- `base-task-data`: `train.parquet` and `test.parquet` for the 18 base tasks.
- `report-split-tasks`: optional derived split-profile task materialization if
  you have already generated it locally.

To stage an audited bundle from a checkout that already has data restored:

```bash
uv run python scripts/data_bundle.py stage \
  --repo-root . \
  --output-dir .release-data \
  --include base-task-data \
  --force
```

Before upload, audit the staged payload for secrets, raw provider streams,
worker env files, absolute local paths, and unaudited logs.

## Release Policy

The first RegimeBench release is a reproducibility package, not a blind
leaderboard. The audited data bundle includes hidden/OOS labels so public users
can reproduce the report and verify saved code snapshots locally. Future blind
evaluations should use a separate withheld data service.

See [docs/release_policy.md](docs/release_policy.md) for the public benchmark
mode and [DATA.md](DATA.md) for data terms.

## License

The repository source and RegimeBench base data bundle are released under the
Apache License 2.0. See [DATA.md](DATA.md) for data-specific notes.
