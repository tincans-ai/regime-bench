#!/usr/bin/env python3
"""Run one RegimeBench task against one candidate file."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path


BENCHMARK_ROOT = Path(__file__).resolve().parents[1]
if str(BENCHMARK_ROOT) not in sys.path:
    sys.path.insert(0, str(BENCHMARK_ROOT))

from verifier.repair_verifier import score_code, write_json  # noqa: E402


def run_visible_smoke(task_dir: Path, candidate: Path, output_dir: Path) -> dict:
    """Run the candidate once on visible train data to catch API/schema errors."""
    smoke_dir = output_dir / "visible_smoke"
    starter = task_dir / "environment" / "starter_code" / "code.py"
    original = starter.read_text() if starter.exists() else None
    smoke_dir.mkdir(parents=True, exist_ok=True)
    try:
        starter.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(candidate, starter)
        cmd = [sys.executable, str(task_dir / "run_local.py"), "--starter"]
        proc = subprocess.run(
            cmd,
            cwd=task_dir,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=300,
        )
        (smoke_dir / "stdout.txt").write_text(proc.stdout)
        return {
            "status": "completed" if proc.returncode == 0 else "failed",
            "returncode": proc.returncode,
            "stdout_path": str(smoke_dir / "stdout.txt"),
        }
    finally:
        if original is not None:
            starter.write_text(original)


def command_run(args: argparse.Namespace) -> int:
    task_dir = args.task.resolve()
    candidate = args.candidate.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    visible = None
    if not args.skip_visible_smoke:
        visible = run_visible_smoke(task_dir, candidate, output_dir)
        if visible["status"] != "completed" and not args.keep_going:
            result = {"status": "failed", "visible_smoke": visible}
            write_json(output_dir / "run_task_manifest.json", result)
            print(json.dumps(result, indent=2, sort_keys=True))
            return 1

    verifier = score_code(
        task_dir=task_dir,
        code_path=candidate,
        output_dir=output_dir / "verifier",
        causality_gate=not args.no_causality_gate,
        gate_days=args.gate_days,
        min_gate_coverage=args.min_gate_coverage,
        replay_mode=args.replay_mode,
    )
    result = {
        "status": verifier.get("status"),
        "task": str(task_dir),
        "candidate": str(candidate),
        "visible_smoke": visible,
        "verifier": verifier,
    }
    write_json(output_dir / "run_task_manifest.json", result)
    print(json.dumps(result, indent=2, sort_keys=True, default=str))
    return 0 if verifier.get("status") == "completed" else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", type=Path, required=True, help="Path to benchmark/tasks/<task-name>")
    parser.add_argument("--candidate", type=Path, required=True, help="Python file defining build_signal(data_path)")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--skip-visible-smoke", action="store_true")
    parser.add_argument("--keep-going", action="store_true")
    parser.add_argument("--no-causality-gate", action="store_true")
    parser.add_argument("--gate-days", type=int, default=1)
    parser.add_argument("--min-gate-coverage", type=float, default=0.01)
    parser.add_argument("--replay-mode", choices=["strict", "full_labeled"], default="strict")
    parser.set_defaults(func=command_run)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
