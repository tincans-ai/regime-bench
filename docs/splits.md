# Split Profile

The bundled RegimeBench split profile matches the current report package:

- Base tasks: 18 canonical `signal-*` tasks.
- Split families: `rolling`, `random_block`, `stripe`.
- Split count: `6` per base task.
- Seed: `0`.
- Purge window: `24` hours.
- Validation split: `none`.

After restoring base task data, generate split tasks with:

```bash
uv run python scripts/materialize_splits.py \
  --profile split_profiles/report.toml \
  --output task_splits/report-profile \
  --force
```

This creates:

```text
task_splits/report-profile/
  metadata.json
  split_manifest.json
  split_manifest.parquet
  tasks/
```

The report profile contains 108 generated task variants. It is bundled because
the report figures and tables use this exact split materialization. The derived
split task parquets are not part of the primary data bundle.
