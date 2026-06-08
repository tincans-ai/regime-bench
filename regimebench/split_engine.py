from __future__ import annotations

import json
import random
import shutil
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import polars as pl

TIMESTAMP_COLUMN = "timestamp"
TIMESTAMP_CANDIDATES = ("timestamp", "timestamp_agg", "date", "datetime")
DEFAULT_SPLIT_PRESET = "none"
SUPPORTED_SPLIT_FAMILIES = {"fixed", "rolling", "random_block", "stripe"}
DEFAULT_SPLIT_FAMILIES = ["fixed", "rolling", "random_block", "stripe"]
DEFAULT_STRIPE_UNITS = ["month", "week"]
DEFAULT_VALIDATION_SPLIT = "none"
SUPPORTED_VALIDATION_SPLITS = {"none", "is-month-stripe"}
VALIDATION_INSTRUCTION_MARKER = "<!-- regimebench-validation-mode -->"
VALIDATION_INSTRUCTION_BLOCK = f"""
{VALIDATION_INSTRUCTION_MARKER}

## Validation Checker

This benchmark variant provides `/check-val`, a separate validation checker
for choosing among promising candidates before final submission. Use `/check`
for normal in-sample iteration; when a validation budget remains, run
`/check-val` on your current best candidate before final submission.

During `/check-val`, your code receives a train-plus-validation feature view.
Validation prediction rows have `is_prediction_period = True`; current return
labels on those rows are withheld or null. Do not read validation files
directly, and do not treat validation feedback as a new training set. Final
scoring still uses hidden held-out test data.
"""


@dataclass(frozen=True, slots=True)
class SplitConfig:
    preset: str = DEFAULT_SPLIT_PRESET
    families: list[str] = field(default_factory=lambda: list(DEFAULT_SPLIT_FAMILIES))
    count: int = 1
    seed: int = 0
    stripe_units: list[str] = field(default_factory=lambda: list(DEFAULT_STRIPE_UNITS))
    purge_hours: int = 24

    def enabled(self) -> bool:
        return self.preset != "none"

    def normalized_families(self) -> list[str]:
        families = [family.strip() for family in self.families if family.strip()]
        unknown = sorted(set(families) - SUPPORTED_SPLIT_FAMILIES)
        if unknown:
            raise ValueError(f"Unsupported split families: {', '.join(unknown)}")
        return families or list(DEFAULT_SPLIT_FAMILIES)


@dataclass(frozen=True, slots=True)
class ValidationConfig:
    split: str = DEFAULT_VALIDATION_SPLIT
    stripe_unit: str = "month"
    stripe_modulus: int = 2
    stripe_holdout: int = 1

    def enabled(self) -> bool:
        return self.split != "none"


@dataclass(frozen=True, slots=True)
class SplitSpec:
    base_task: str
    task_name: str
    split_id: str
    split_family: str
    train_start: str | None
    train_end: str | None
    test_start: str | None
    test_end: str | None
    purge_hours: int = 0
    seed: int = 0
    stripe_unit: str | None = None
    stripe_modulus: int | None = None
    stripe_holdout: int | None = None
    leakage_risk: str = "low"
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def split_config_from_values(
    *,
    preset: str = DEFAULT_SPLIT_PRESET,
    families: list[str] | str | None = None,
    count: int = 1,
    seed: int = 0,
    stripe_units: list[str] | str | None = None,
    purge_hours: int = 24,
) -> SplitConfig:
    if isinstance(families, str):
        family_values = [item.strip() for item in families.split(",") if item.strip()]
    elif families is None:
        family_values = list(DEFAULT_SPLIT_FAMILIES)
    else:
        family_values = list(families)

    if isinstance(stripe_units, str):
        stripe_values = [item.strip() for item in stripe_units.split(",") if item.strip()]
    elif stripe_units is None:
        stripe_values = list(DEFAULT_STRIPE_UNITS)
    else:
        stripe_values = list(stripe_units)

    config = SplitConfig(
        preset=str(preset or DEFAULT_SPLIT_PRESET),
        families=family_values,
        count=max(1, int(count)),
        seed=int(seed),
        stripe_units=stripe_values or list(DEFAULT_STRIPE_UNITS),
        purge_hours=max(0, int(purge_hours)),
    )
    config.normalized_families()
    if config.preset not in {"none", "volume-variety"}:
        raise ValueError("split_preset must be one of: none, volume-variety")
    return config


def validation_config_from_values(
    *,
    split: str = DEFAULT_VALIDATION_SPLIT,
    stripe_unit: str = "month",
    stripe_modulus: int = 2,
    stripe_holdout: int = 1,
) -> ValidationConfig:
    split_value = str(split or DEFAULT_VALIDATION_SPLIT)
    if split_value not in SUPPORTED_VALIDATION_SPLITS:
        expected = ", ".join(sorted(SUPPORTED_VALIDATION_SPLITS))
        raise ValueError(f"validation_split must be one of: {expected}")
    if stripe_unit not in {"month", "week", "day", "hour"}:
        raise ValueError("validation_stripe_unit must be one of: month, week, day, hour")
    modulus = max(2, int(stripe_modulus))
    holdout = int(stripe_holdout)
    if holdout < 0 or holdout >= modulus:
        raise ValueError("validation_stripe_holdout must be in [0, validation_stripe_modulus)")
    return ValidationConfig(
        split=split_value,
        stripe_unit=str(stripe_unit),
        stripe_modulus=modulus,
        stripe_holdout=holdout,
    )


def read_task_frames(task_dir: Path) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    train_path = task_dir / "environment" / "data" / "train.parquet"
    test_path = task_dir / "tests" / "test.parquet"
    if not train_path.exists():
        raise FileNotFoundError(f"Missing train parquet: {train_path}")
    if not test_path.exists():
        raise FileNotFoundError(f"Missing test parquet: {test_path}")

    train = pl.read_parquet(train_path)
    test = pl.read_parquet(test_path)
    timestamp_column = _timestamp_column(train, train_path)
    _timestamp_column(test, test_path)
    full = (
        pl.concat([train, test], how="diagonal_relaxed")
        .unique()
        .sort(timestamp_column)
    )
    return train, test, full


def build_split_specs(
    *,
    base_task: str,
    source_train: pl.DataFrame,
    source_test: pl.DataFrame,
    full_data: pl.DataFrame,
    config: SplitConfig,
) -> list[SplitSpec]:
    if not config.enabled():
        return []
    _timestamp_column(full_data, Path(base_task))

    families = config.normalized_families()
    family_candidates: dict[str, list[SplitSpec]] = {}
    for family in families:
        if family == "fixed":
            family_candidates[family] = _fixed_specs(
                base_task, source_train, source_test, config, config.count
            )
        elif family == "rolling":
            family_candidates[family] = _rolling_specs(base_task, full_data, config, config.count)
        elif family == "random_block":
            family_candidates[family] = _random_block_specs(
                base_task, full_data, config, config.count
            )
        elif family == "stripe":
            family_candidates[family] = _stripe_specs(base_task, full_data, config, config.count)

    specs: list[SplitSpec] = []
    max_candidates = max((len(items) for items in family_candidates.values()), default=0)
    for index in range(max_candidates):
        for family in families:
            candidates = family_candidates.get(family) or []
            if index < len(candidates):
                specs.append(candidates[index])
                if len(specs) >= config.count:
                    return specs
    return specs[: config.count]


def materialize_split_tasks(
    *,
    source_tasks_root: Path,
    output_tasks_root: Path,
    base_task_names: list[str],
    config: SplitConfig,
    validation_config: ValidationConfig | None = None,
    manifest_path: Path | None = None,
    manifest_parquet_path: Path | None = None,
) -> tuple[list[Path], list[SplitSpec]]:
    output_tasks_root.mkdir(parents=True, exist_ok=True)
    generated_dirs: list[Path] = []
    generated_specs: list[SplitSpec] = []

    for base_task in base_task_names:
        base_task_dir = source_tasks_root / base_task
        source_train, source_test, full_data = read_task_frames(base_task_dir)
        specs = build_split_specs(
            base_task=base_task,
            source_train=source_train,
            source_test=source_test,
            full_data=full_data,
            config=config,
        )
        for spec in specs:
            train_df, test_df = slice_data_for_spec(full_data, spec)
            train_df, val_df, validation_metadata = apply_validation_split(
                train_df,
                task_name=spec.task_name,
                config=validation_config,
            )
            if validation_metadata is not None:
                spec = replace(
                    spec,
                    details={
                        **spec.details,
                        "validation": validation_metadata,
                    },
                )
            target_dir = output_tasks_root / spec.task_name
            _copy_task_template(base_task_dir, target_dir)
            (target_dir / "environment" / "data").mkdir(parents=True, exist_ok=True)
            (target_dir / "tests").mkdir(parents=True, exist_ok=True)
            train_df.write_parquet(target_dir / "environment" / "data" / "train.parquet")
            test_df.write_parquet(target_dir / "tests" / "test.parquet")
            if val_df is not None:
                val_df.write_parquet(target_dir / "tests" / "val.parquet")
                _inject_validation_instruction(target_dir)
            (target_dir / "split_metadata.json").write_text(
                json.dumps(spec.to_dict(), indent=2, sort_keys=True) + "\n"
            )
            generated_dirs.append(target_dir)
            generated_specs.append(spec)

    if manifest_path is not None:
        write_split_manifest(generated_specs, manifest_path, manifest_parquet_path)
    return generated_dirs, generated_specs


def materialize_validation_tasks(
    *,
    source_tasks_root: Path,
    output_tasks_root: Path,
    base_task_names: list[str],
    validation_config: ValidationConfig,
    manifest_path: Path | None = None,
    manifest_parquet_path: Path | None = None,
) -> list[Path]:
    if not validation_config.enabled():
        return []

    output_tasks_root.mkdir(parents=True, exist_ok=True)
    generated_dirs: list[Path] = []
    manifest_rows: list[dict[str, Any]] = []

    for base_task in base_task_names:
        base_task_dir = source_tasks_root / base_task
        source_train, source_test, _full_data = read_task_frames(base_task_dir)
        train_df, val_df, validation_metadata = apply_validation_split(
            source_train,
            task_name=base_task,
            config=validation_config,
        )
        if val_df is None or validation_metadata is None:
            continue

        target_dir = output_tasks_root / base_task
        _copy_task_template(base_task_dir, target_dir)
        (target_dir / "environment" / "data").mkdir(parents=True, exist_ok=True)
        (target_dir / "tests").mkdir(parents=True, exist_ok=True)
        train_df.write_parquet(target_dir / "environment" / "data" / "train.parquet")
        val_df.write_parquet(target_dir / "tests" / "val.parquet")
        source_test.write_parquet(target_dir / "tests" / "test.parquet")
        _inject_validation_instruction(target_dir)
        metadata = {
            "base_task": base_task,
            "task_name": base_task,
            "split_id": "base",
            "split_family": "base",
            "validation": validation_metadata,
        }
        (target_dir / "split_metadata.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n"
        )
        manifest_rows.append(metadata)
        generated_dirs.append(target_dir)

    if manifest_path is not None:
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(manifest_rows, indent=2, sort_keys=True) + "\n")
    if manifest_parquet_path is not None:
        manifest_parquet_path.parent.mkdir(parents=True, exist_ok=True)
        if manifest_rows:
            pl.DataFrame([_flatten_manifest_row(row) for row in manifest_rows]).write_parquet(
                manifest_parquet_path
            )
        else:
            pl.DataFrame(
                schema={
                    "base_task": pl.String,
                    "task_name": pl.String,
                    "split_id": pl.String,
                    "split_family": pl.String,
                }
            ).write_parquet(manifest_parquet_path)

    return generated_dirs


def write_split_manifest(
    specs: list[SplitSpec],
    manifest_path: Path,
    manifest_parquet_path: Path | None = None,
) -> None:
    rows = [spec.to_dict() for spec in specs]
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n")
    if manifest_parquet_path is not None:
        manifest_parquet_path.parent.mkdir(parents=True, exist_ok=True)
        if rows:
            flat_rows = [_flatten_manifest_row(row) for row in rows]
            pl.DataFrame(flat_rows).write_parquet(manifest_parquet_path)
        else:
            pl.DataFrame(
                schema={
                    "base_task": pl.String,
                    "task_name": pl.String,
                    "split_id": pl.String,
                    "split_family": pl.String,
                }
            ).write_parquet(manifest_parquet_path)


def slice_data_for_spec(full_data: pl.DataFrame, spec: SplitSpec) -> tuple[pl.DataFrame, pl.DataFrame]:
    train_mask, test_mask = _masks_for_spec(full_data, spec)
    timestamp_column = _timestamp_column(full_data, Path(spec.task_name))
    train_df = full_data.filter(train_mask).sort(timestamp_column)
    test_df = full_data.filter(test_mask).sort(timestamp_column)
    if train_df.height == 0:
        raise ValueError(f"Split {spec.task_name} produced an empty train set")
    if test_df.height == 0:
        raise ValueError(f"Split {spec.task_name} produced an empty test set")
    return train_df, test_df


def apply_validation_split(
    train_df: pl.DataFrame,
    *,
    task_name: str,
    config: ValidationConfig | None,
) -> tuple[pl.DataFrame, pl.DataFrame | None, dict[str, Any] | None]:
    if config is None or not config.enabled():
        return train_df, None, None
    if config.split != "is-month-stripe":
        raise ValueError(f"Unsupported validation split: {config.split}")

    timestamp_column = _timestamp_column(train_df, Path(task_name))
    ts = pl.col(timestamp_column)
    unit_column = "_qr_eval_validation_unit"
    ordinal_column = "_qr_eval_validation_ordinal"
    indexed = train_df.with_columns(_stripe_value_expr(ts, config.stripe_unit).alias(unit_column))
    units = indexed.select(unit_column).unique().sort(unit_column).with_row_index(ordinal_column)
    indexed = indexed.join(units, on=unit_column, how="left")
    val_mask = (pl.col(ordinal_column) % config.stripe_modulus) == config.stripe_holdout
    visible_train = indexed.filter(~val_mask).drop([unit_column, ordinal_column]).sort(
        timestamp_column
    )
    val_df = indexed.filter(val_mask).drop([unit_column, ordinal_column]).sort(timestamp_column)
    if visible_train.height == 0:
        raise ValueError(f"Validation split {task_name} produced an empty train set")
    if val_df.height == 0:
        raise ValueError(f"Validation split {task_name} produced an empty validation set")

    train_min, train_max = _timestamp_range(visible_train)
    val_min, val_max = _timestamp_range(val_df)
    metadata = {
        "split": config.split,
        "stripe_unit": config.stripe_unit,
        "stripe_modulus": config.stripe_modulus,
        "stripe_holdout": config.stripe_holdout,
        "stripe_basis": "current_is_ordinal",
        "train_start": train_min,
        "train_end": train_max,
        "val_start": val_min,
        "val_end": val_max,
        "train_rows": visible_train.height,
        "val_rows": val_df.height,
    }
    return visible_train, val_df, metadata


def _timestamp_column(df: pl.DataFrame, path: Path) -> str:
    for candidate in TIMESTAMP_CANDIDATES:
        if candidate in df.columns:
            return candidate
    candidates = ", ".join(repr(candidate) for candidate in TIMESTAMP_CANDIDATES)
    raise ValueError(f"{path} is missing a timestamp column; tried {candidates}")


def _copy_task_template(source: Path, target: Path) -> None:
    if target.exists():
        shutil.rmtree(target)

    def ignore(_dir: str, names: list[str]) -> set[str]:
        ignored = {"__pycache__"}
        if "train.parquet" in names:
            ignored.add("train.parquet")
        if "val.parquet" in names:
            ignored.add("val.parquet")
        if "test.parquet" in names:
            ignored.add("test.parquet")
        return ignored & set(names)

    shutil.copytree(source, target, ignore=ignore)


def _inject_validation_instruction(task_dir: Path) -> None:
    instruction_path = task_dir / "instruction.md"
    if not instruction_path.exists():
        return
    instruction = instruction_path.read_text()
    if VALIDATION_INSTRUCTION_MARKER in instruction:
        return
    instruction_path.write_text(instruction.rstrip() + "\n\n" + VALIDATION_INSTRUCTION_BLOCK.lstrip())


def _flatten_manifest_row(row: dict[str, Any]) -> dict[str, Any]:
    flattened = dict(row)
    details = flattened.pop("details", {}) or {}
    for key, value in details.items():
        flattened[f"details_{key}"] = json.dumps(value, sort_keys=True) if isinstance(
            value, dict | list
        ) else value
    return flattened


def _fixed_specs(
    base_task: str,
    train: pl.DataFrame,
    test: pl.DataFrame,
    config: SplitConfig,
    quota: int,
) -> list[SplitSpec]:
    train_start, train_end = _timestamp_range(train)
    test_start, test_end = _timestamp_range(test)
    spec = SplitSpec(
        base_task=base_task,
        task_name=f"{base_task}__fixed-001",
        split_id="fixed-001",
        split_family="fixed",
        train_start=train_start,
        train_end=train_end,
        test_start=test_start,
        test_end=test_end,
        purge_hours=0,
        seed=config.seed,
        details={"source": "original_train_test"},
    )
    return [spec][:quota]


def _rolling_specs(
    base_task: str,
    full_data: pl.DataFrame,
    config: SplitConfig,
    quota: int,
) -> list[SplitSpec]:
    min_ts, max_ts = _timestamp_range(full_data)
    start_month = _month_floor(_parse_ts(min_ts))
    end_month = _month_floor(_parse_ts(max_ts))
    train_months = 36
    test_months = 12
    step_months = 12
    candidates: list[SplitSpec] = []
    cursor = start_month
    while True:
        train_start = cursor
        train_end_exclusive = _add_months(train_start, train_months)
        test_start = train_end_exclusive
        test_end_exclusive = _add_months(test_start, test_months)
        if test_start > end_month or test_end_exclusive <= start_month:
            break
        if test_start <= _parse_ts(max_ts):
            index = len(candidates) + 1
            candidates.append(
                SplitSpec(
                    base_task=base_task,
                    task_name=f"{base_task}__rolling-{index:03d}",
                    split_id=f"rolling-{index:03d}",
                    split_family="rolling",
                    train_start=_iso(train_start),
                    train_end=_iso(train_end_exclusive - timedelta(microseconds=1)),
                    test_start=_iso(test_start),
                    test_end=_iso(test_end_exclusive - timedelta(microseconds=1)),
                    purge_hours=config.purge_hours,
                    seed=config.seed,
                    leakage_risk="low",
                    details={
                        "train_months": train_months,
                        "test_months": test_months,
                        "step_months": step_months,
                    },
                )
            )
        cursor = _add_months(cursor, step_months)
        if cursor > end_month:
            break
    return candidates[:quota]


def _random_block_specs(
    base_task: str,
    full_data: pl.DataFrame,
    config: SplitConfig,
    quota: int,
) -> list[SplitSpec]:
    months = _available_months(full_data)
    if len(months) < 2:
        return []
    rng = random.Random(config.seed)
    specs: list[SplitSpec] = []
    block_months = 2
    target_blocks_per_split = 12
    blocks_per_split = min(target_blocks_per_split, max(1, len(months) // block_months))
    for index in range(1, quota + 1):
        starts = _sample_disjoint_block_starts(
            month_count=len(months),
            block_months=block_months,
            block_count=blocks_per_split,
            rng=rng,
        )
        block_ranges: list[tuple[datetime, datetime]] = []
        for start_index in starts:
            block_start = months[start_index]
            block_end = _add_months(block_start, block_months)
            block_ranges.append((block_start, block_end))
        test_start = min(start for start, _ in block_ranges)
        test_end = max(end for _, end in block_ranges)
        specs.append(
            SplitSpec(
                base_task=base_task,
                task_name=f"{base_task}__random-block-{index:03d}",
                split_id=f"random-block-{index:03d}",
                split_family="random_block",
                train_start=None,
                train_end=None,
                test_start=_iso(test_start),
                test_end=_iso(test_end - timedelta(microseconds=1)),
                purge_hours=config.purge_hours,
                seed=config.seed,
                leakage_risk="medium",
                details={
                    "block_months": block_months,
                    "blocks_per_split": len(block_ranges),
                    "target_blocks_per_split": target_blocks_per_split,
                    "blocks": [
                        {
                            "start": _iso(start),
                            "end": _iso(end - timedelta(microseconds=1)),
                        }
                        for start, end in block_ranges
                    ],
                },
            )
        )
    return specs


def _sample_disjoint_block_starts(
    *,
    month_count: int,
    block_months: int,
    block_count: int,
    rng: random.Random,
) -> list[int]:
    offsets = [
        offset
        for offset in range(block_months)
        if len(range(offset, month_count - block_months + 1, block_months)) >= block_count
    ]
    if not offsets:
        offsets = [0]
    offset = rng.choice(offsets)
    candidates = list(range(offset, month_count - block_months + 1, block_months))
    return sorted(rng.sample(candidates, k=min(block_count, len(candidates))))


def _stripe_specs(
    base_task: str,
    full_data: pl.DataFrame,
    config: SplitConfig,
    quota: int,
) -> list[SplitSpec]:
    specs: list[SplitSpec] = []
    min_ts, max_ts = _timestamp_range(full_data)
    for unit in config.stripe_units:
        for holdout in (0, 1):
            index = len(specs) + 1
            leakage = "medium" if unit in {"month", "week"} else "high"
            specs.append(
                SplitSpec(
                    base_task=base_task,
                    task_name=f"{base_task}__stripe-{unit}-mod2-h{holdout}",
                    split_id=f"stripe-{unit}-mod2-h{holdout}",
                    split_family="stripe",
                    train_start=min_ts,
                    train_end=max_ts,
                    test_start=min_ts,
                    test_end=max_ts,
                    purge_hours=config.purge_hours,
                    seed=config.seed,
                    stripe_unit=unit,
                    stripe_modulus=2,
                    stripe_holdout=holdout,
                    leakage_risk=leakage,
                    details={"sequence": index},
                )
            )
    return specs[:quota]


def _masks_for_spec(full_data: pl.DataFrame, spec: SplitSpec) -> tuple[pl.Expr, pl.Expr]:
    ts = pl.col(_timestamp_column(full_data, Path(spec.task_name)))
    if spec.split_family == "fixed":
        train_mask = _closed_range_expr(ts, spec.train_start, spec.train_end)
        test_mask = _closed_range_expr(ts, spec.test_start, spec.test_end)
        return train_mask, test_mask

    if spec.split_family == "rolling":
        train_mask = _closed_range_expr(ts, spec.train_start, spec.train_end)
        test_mask = _closed_range_expr(ts, spec.test_start, spec.test_end)
        return _apply_purge(ts, train_mask, [(spec.test_start, spec.test_end)], spec.purge_hours), test_mask

    if spec.split_family == "random_block":
        blocks = [
            (str(block["start"]), str(block["end"]))
            for block in spec.details.get("blocks", [])
            if isinstance(block, dict)
        ]
        test_mask = pl.lit(False)
        for start, end in blocks:
            test_mask = test_mask | _closed_range_expr(ts, start, end)
        train_mask = ~test_mask
        return _apply_purge(ts, train_mask, blocks, spec.purge_hours), test_mask

    if spec.split_family == "stripe":
        unit = spec.stripe_unit or "month"
        modulus = int(spec.stripe_modulus or 2)
        holdout = int(spec.stripe_holdout or 0)
        stripe_value = _stripe_value_expr(ts, unit)
        test_mask = (stripe_value % modulus) == holdout
        train_mask = ~test_mask
        return train_mask, test_mask

    raise ValueError(f"Unsupported split family: {spec.split_family}")


def _closed_range_expr(column: pl.Expr, start: str | None, end: str | None) -> pl.Expr:
    mask = pl.lit(True)
    if start is not None:
        mask = mask & (column >= _parse_ts(start))
    if end is not None:
        mask = mask & (column <= _parse_ts(end))
    return mask


def _apply_purge(
    column: pl.Expr,
    train_mask: pl.Expr,
    test_ranges: list[tuple[str | None, str | None]],
    purge_hours: int,
) -> pl.Expr:
    if purge_hours <= 0:
        return train_mask
    purge_delta = timedelta(hours=purge_hours)
    purged = pl.lit(False)
    for start, end in test_ranges:
        if start is None or end is None:
            continue
        purged = purged | (
            (column >= (_parse_ts(start) - purge_delta))
            & (column <= (_parse_ts(end) + purge_delta))
        )
    return train_mask & ~purged


def _stripe_value_expr(column: pl.Expr, unit: str) -> pl.Expr:
    if unit == "month":
        return column.dt.year() * 12 + column.dt.month()
    if unit == "week":
        return column.dt.year() * 53 + column.dt.week()
    if unit == "day":
        return column.dt.year() * 366 + column.dt.ordinal_day()
    if unit == "hour":
        return (column.dt.year() * 366 + column.dt.ordinal_day()) * 24 + column.dt.hour()
    raise ValueError(f"Unsupported stripe unit: {unit}")


def _timestamp_range(df: pl.DataFrame) -> tuple[str, str]:
    timestamp_column = _timestamp_column(df, Path("dataframe"))
    row = df.select(
        pl.col(timestamp_column).min().alias("start"),
        pl.col(timestamp_column).max().alias("end"),
    ).row(0, named=True)
    return _iso(row["start"]), _iso(row["end"])


def _available_months(full_data: pl.DataFrame) -> list[datetime]:
    min_ts, max_ts = _timestamp_range(full_data)
    cursor = _month_floor(_parse_ts(min_ts))
    end = _month_floor(_parse_ts(max_ts))
    months: list[datetime] = []
    while cursor <= end:
        months.append(cursor)
        cursor = _add_months(cursor, 1)
    return months


def _parse_ts(value: str | datetime) -> datetime:
    if isinstance(value, datetime):
        return value.replace(tzinfo=None)
    text = str(value).replace("Z", "+00:00")
    parsed = datetime.fromisoformat(text)
    return parsed.replace(tzinfo=None)


def _iso(value: Any) -> str:
    if isinstance(value, datetime):
        return value.replace(tzinfo=None).isoformat()
    if hasattr(value, "to_pydatetime"):
        return value.to_pydatetime().replace(tzinfo=None).isoformat()
    return str(value)


def _month_floor(value: datetime) -> datetime:
    return datetime(value.year, value.month, 1)


def _add_months(value: datetime, months: int) -> datetime:
    month_index = value.month - 1 + months
    year = value.year + month_index // 12
    month = month_index % 12 + 1
    day = min(value.day, _days_in_month(year, month))
    return value.replace(year=year, month=month, day=day)


def _days_in_month(year: int, month: int) -> int:
    if month == 12:
        next_month = datetime(year + 1, 1, 1)
    else:
        next_month = datetime(year, month + 1, 1)
    return (next_month - datetime(year, month, 1)).days
