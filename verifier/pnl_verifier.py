#!/usr/bin/env python3
"""Public CLI for canonical RegimeBench repair-style PnL verification."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import polars as pl

try:
    from .repair_verifier import score_code, write_json
except ImportError:  # Allow running as a file.
    from repair_verifier import score_code, write_json


def _read_parquet(path: Path) -> pl.DataFrame:
    return pl.read_parquet(path) if path.exists() else pl.DataFrame()


def command_score_code(args: argparse.Namespace) -> int:
    result = score_code(
        task_dir=args.task_dir.resolve(),
        code_path=args.code_path.resolve(),
        output_dir=args.output_dir.resolve(),
        causality_gate=not args.no_causality_gate,
        gate_days=args.gate_days,
        min_gate_coverage=args.min_gate_coverage,
        replay_mode=args.replay_mode,
    )
    manifest = {
        "kind": "pnl_verifier_score_code",
        "status": result.get("status"),
        "task_dir": str(args.task_dir.resolve()),
        "code_path": str(args.code_path.resolve()),
        "output_dir": str(args.output_dir.resolve()),
        "causality_gate": not args.no_causality_gate,
        "replay_mode": args.replay_mode,
        "results": str(args.output_dir.resolve() / "results.json"),
        "daily_pnl": str(args.output_dir.resolve() / "daily_pnl.parquet"),
        "score": result.get("score"),
        "passed": result.get("passed"),
    }
    write_json(args.output_dir.resolve() / "pnl_verifier_manifest.json", manifest)
    print(json.dumps(result, indent=2, sort_keys=True, default=str))
    return 0 if result.get("status") == "completed" else 1


def _resolve_code_path(experiment: Path, row: dict[str, Any]) -> Path | None:
    for key in ("code_path", "source_path", "relative_path"):
        value = row.get(key)
        if not value:
            continue
        path = Path(str(value))
        if path.is_absolute() and path.exists():
            return path
        candidate = experiment / path
        if candidate.exists():
            return candidate
    return None


def command_replay_experiment(args: argparse.Namespace) -> int:
    experiment = args.experiment.resolve()
    output_dir = args.output_dir.resolve()
    manifest = _read_parquet(experiment / "attempt_code_manifest.parquet")
    if manifest.is_empty():
        raise SystemExit(f"Missing or empty attempt_code_manifest.parquet: {experiment}")

    rows = manifest.to_dicts()
    if args.limit is not None:
        rows = rows[: max(0, args.limit)]
    output_dir.mkdir(parents=True, exist_ok=True)
    attempt_rows: list[dict[str, Any]] = []
    daily_frames: list[pl.DataFrame] = []

    for index, row in enumerate(rows, start=1):
        task_name = str(row.get("task_name") or "")
        code_path = _resolve_code_path(experiment, row)
        attempt = {
            "index": index,
            "task_name": task_name,
            "status": "started",
            "code_path": str(code_path) if code_path else None,
        }
        try:
            if not task_name:
                raise RuntimeError("attempt manifest row is missing task_name")
            if code_path is None:
                raise RuntimeError("attempt manifest row does not point to an existing code file")
            task_dir = args.tasks_root.resolve() / task_name
            replay_dir = output_dir / f"{index:05d}-{task_name}"
            result = score_code(
                task_dir=task_dir,
                code_path=code_path,
                output_dir=replay_dir,
                causality_gate=not args.no_causality_gate,
                gate_days=args.gate_days,
                min_gate_coverage=args.min_gate_coverage,
                replay_mode=args.replay_mode,
            )
            attempt.update(
                {
                    "status": result.get("status"),
                    "score": result.get("score"),
                    "passed": result.get("passed"),
                    "daily_pnl_path": result.get("daily_pnl_path"),
                }
            )
            daily = _read_parquet(replay_dir / "daily_pnl.parquet")
            if not daily.is_empty():
                context = {key: row.get(key) for key in row if key in {"trial_id", "family_name", "model_name", "max_checks", "repeat_idx", "task_name", "check_num"}}
                daily_frames.append(daily.with_columns([pl.lit(value).alias(key) for key, value in context.items()]))
        except Exception as exc:
            attempt.update({"status": "failed", "error_type": type(exc).__name__, "error_message": str(exc)})
        attempt_rows.append(attempt)

    attempts = pl.DataFrame(attempt_rows) if attempt_rows else pl.DataFrame()
    attempts.write_parquet(output_dir / "replay_attempts.parquet")
    daily_output = pl.concat(daily_frames, how="diagonal_relaxed") if daily_frames else pl.DataFrame()
    daily_output.write_parquet(output_dir / "pnl_streams_out_sample.parquet")
    summary = {
        "kind": "pnl_verifier_replay_experiment",
        "status": "completed",
        "experiment": str(experiment),
        "tasks_root": str(args.tasks_root.resolve()),
        "output_dir": str(output_dir),
        "attempt_count": len(attempt_rows),
        "completed_count": sum(1 for row in attempt_rows if row.get("status") == "completed"),
        "failed_count": sum(1 for row in attempt_rows if row.get("status") == "failed"),
        "outputs": {
            "replay_attempts": str(output_dir / "replay_attempts.parquet"),
            "pnl_streams_out_sample": str(output_dir / "pnl_streams_out_sample.parquet"),
        },
    }
    write_json(output_dir / "pnl_verifier_manifest.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["failed_count"] == 0 else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    score = subparsers.add_parser("score-code")
    score.add_argument("--task-dir", type=Path, required=True)
    score.add_argument("--code-path", type=Path, required=True)
    score.add_argument("--output-dir", type=Path, required=True)
    score.add_argument("--no-causality-gate", action="store_true")
    score.add_argument("--gate-days", type=int, default=1)
    score.add_argument("--min-gate-coverage", type=float, default=0.01)
    score.add_argument("--replay-mode", choices=["strict", "full_labeled"], default="strict")
    score.set_defaults(func=command_score_code)

    replay = subparsers.add_parser("replay-experiment")
    replay.add_argument("--experiment", type=Path, required=True)
    replay.add_argument("--tasks-root", type=Path, required=True)
    replay.add_argument("--output-dir", type=Path, required=True)
    replay.add_argument("--scope", choices=["all", "missing"], default="all")
    replay.add_argument("--limit", type=int)
    replay.add_argument("--no-causality-gate", action="store_true")
    replay.add_argument("--gate-days", type=int, default=1)
    replay.add_argument("--min-gate-coverage", type=float, default=0.01)
    replay.add_argument("--replay-mode", choices=["strict", "full_labeled"], default="strict")
    replay.set_defaults(func=command_replay_experiment)

    return parser


def main() -> int:
    args = build_parser().parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
