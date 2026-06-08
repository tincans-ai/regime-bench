#!/usr/bin/env python3
"""Materialize RegimeBench split tasks from restored base task data."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tomllib
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from regimebench.split_engine import (  # noqa: E402
    materialize_split_tasks,
    split_config_from_values,
    validation_config_from_values,
)


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def load_profile(path: Path) -> dict:
    with path.open("rb") as handle:
        return tomllib.load(handle)


def discover_base_tasks(tasks_root: Path) -> list[str]:
    return sorted(path.name for path in tasks_root.glob("signal-*") if path.is_dir())


def parse_task_names(value: str | None, tasks_root: Path) -> list[str]:
    if not value or value == "all":
        return discover_base_tasks(tasks_root)
    return sorted(item.strip() for item in value.split(",") if item.strip())


def materialize(args: argparse.Namespace) -> int:
    profile_path = args.profile.resolve()
    profile = load_profile(profile_path)
    profile_name = str(profile.get("name") or profile_path.stem)
    tasks_root = args.tasks_root.resolve()
    output_root = args.output.resolve() if args.output else REPO_ROOT / "task_splits" / profile_name
    output_tasks_root = output_root / "tasks"

    if output_root.exists():
        if not args.force:
            raise SystemExit(f"Refusing to overwrite existing split output without --force: {output_root}")
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True)

    split_values = profile.get("split", {})
    validation_values = profile.get("validation", {})
    split_config = split_config_from_values(
        preset=split_values.get("preset", "none"),
        families=split_values.get("families"),
        count=split_values.get("count", 1),
        seed=split_values.get("seed", 0),
        stripe_units=split_values.get("stripe_units"),
        purge_hours=split_values.get("purge_hours", 24),
    )
    validation_config = validation_config_from_values(
        split=validation_values.get("split", "none"),
        stripe_unit=validation_values.get("stripe_unit", "month"),
        stripe_modulus=validation_values.get("stripe_modulus", 2),
        stripe_holdout=validation_values.get("stripe_holdout", 1),
    )
    task_names = parse_task_names(args.task_names, tasks_root)
    if not task_names:
        raise SystemExit(f"No base tasks found under {tasks_root}")

    generated_dirs, specs = materialize_split_tasks(
        source_tasks_root=tasks_root,
        output_tasks_root=output_tasks_root,
        base_task_names=task_names,
        config=split_config,
        validation_config=validation_config,
        manifest_path=output_root / "split_manifest.json",
        manifest_parquet_path=output_root / "split_manifest.parquet",
    )
    metadata = {
        "created_at": utc_now(),
        "profile_path": profile_path.relative_to(REPO_ROOT).as_posix()
        if profile_path.is_relative_to(REPO_ROOT)
        else profile_path.as_posix(),
        "profile": profile,
        "base_task_names": task_names,
        "split_config": asdict(split_config),
        "validation_config": asdict(validation_config),
        "generated_task_count": len(generated_dirs),
    }
    (output_root / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    print(f"materialized {len(generated_dirs)} split tasks into {output_root}")
    print(f"wrote manifest: {output_root / 'split_manifest.json'}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile",
        type=Path,
        default=REPO_ROOT / "split_profiles" / "report.toml",
        help="Path to a split profile TOML file.",
    )
    parser.add_argument(
        "--tasks-root",
        type=Path,
        default=REPO_ROOT / "tasks",
        help="Directory containing restored base signal tasks.",
    )
    parser.add_argument(
        "--task-names",
        default="all",
        help="Comma-separated base task names, or 'all' for every signal-* task.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output directory. Defaults to task_splits/<profile-name>.",
    )
    parser.add_argument("--force", action="store_true", help="Replace an existing output directory.")
    return parser


def main(argv: list[str] | None = None) -> int:
    return materialize(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
