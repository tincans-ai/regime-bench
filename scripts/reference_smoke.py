#!/usr/bin/env python3
"""Run the deterministic RegimeBench verifier reference smoke."""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import polars as pl


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from verifier.repair_verifier import score_code, write_json  # noqa: E402


EXPECTED_PATH = REPO_ROOT / "reference_outputs" / "synthetic_smoke_expected.json"
FLOAT_TOLERANCE = 1e-9
SUMMARY_KEYS = [
    "status",
    "score",
    "passed",
    "cheating_detected",
    "return_correlation",
    "mean_ic",
    "ic_tstat_dk",
    "signal_return_sharpe",
    "backtest_sharpe",
    "total_return",
    "max_drawdown",
    "turnover_ann",
    "n_timestamps",
    "coverage_pct",
]


def make_synthetic_task(output_dir: Path) -> tuple[Path, Path]:
    task_dir = output_dir / "synthetic-signal"
    if task_dir.exists():
        shutil.rmtree(task_dir)
    (task_dir / "environment" / "data").mkdir(parents=True)
    (task_dir / "tests").mkdir(parents=True)

    tickers = ["BTC", "ETH", "SOL", "XRP", "ADA"]
    start = datetime(2024, 1, 1)
    rows: list[dict[str, Any]] = []
    for hour in range(24 * 12):
        timestamp = start + timedelta(hours=hour)
        wave = math.sin(hour / 9.0)
        seasonal = math.cos(hour / 17.0)
        for index, ticker in enumerate(tickers):
            center = index - (len(tickers) - 1) / 2
            feature_alpha = center + 0.3 * wave + 0.05 * ((hour + index) % 3 - 1)
            deterministic_noise = 0.001 * math.sin((hour + 1) * (index + 2) * 0.37)
            deterministic_noise += 0.00033 * seasonal * math.cos(index + 1)
            residual_return = 0.00035 * feature_alpha + deterministic_noise
            rows.append(
                {
                    "timestamp_agg": timestamp,
                    "ticker": ticker,
                    "close": 100.0 + hour * 0.05 + index + feature_alpha * 0.1,
                    "volume": 1000.0 + 10.0 * index + hour,
                    "feature_alpha": feature_alpha,
                    "return": residual_return,
                    "residual_return": residual_return,
                }
            )

    frame = pl.DataFrame(rows)
    split_rows = 24 * 7 * len(tickers)
    frame.slice(0, split_rows).write_parquet(task_dir / "environment" / "data" / "train.parquet")
    frame.slice(split_rows).write_parquet(task_dir / "tests" / "test.parquet")

    candidate = output_dir / "synthetic_candidate.py"
    candidate.write_text(
        "import polars as pl\n\n"
        "def build_signal(data_path):\n"
        "    df = pl.read_parquet(data_path)\n"
        "    return df.select([\n"
        "        'timestamp_agg',\n"
        "        'ticker',\n"
        "        pl.col('feature_alpha').alias('signal'),\n"
        "    ])\n"
    )
    (task_dir / "tests" / "evaluate.py").write_text(
        "import importlib.util\n"
        "import math\n"
        "import sys\n"
        "from pathlib import Path\n\n"
        "import numpy as np\n"
        "import polars as pl\n\n"
        "from verifier import pnl_engine as engine\n\n"
        "EPS = engine.EPS\n"
        "TSTAT_THRESHOLD = engine.TSTAT_THRESHOLD\n"
        "SHARPE_THRESHOLD = engine.SHARPE_THRESHOLD\n"
        "SignalMetrics = engine.SignalMetrics\n"
        "_source_policy_violations = engine.source_policy_violations\n"
        "_sort_columns_for = engine.sort_columns_for\n"
        "_return_columns_for = engine.return_columns_for\n"
        "_with_safe_return_lags = engine.with_safe_return_lags\n"
        "_final_eval_feature_frame = engine.final_eval_feature_frame\n"
        "_filter_signal_period = engine.filter_signal_period\n"
        "_preprocess_signal = engine.preprocess_signal\n"
        "compute_ic_per_timestamp = engine.compute_ic_per_timestamp\n"
        "compute_dk_tstat = engine.compute_dk_tstat\n"
        "compute_daily_sharpe = engine.compute_daily_sharpe\n"
        "compute_autocorrelation = engine.compute_autocorrelation\n"
        "compute_quantile_metrics = engine.compute_quantile_metrics\n"
        "compute_ic_decay = engine.compute_ic_decay\n"
        "compute_return_correlation = engine.compute_return_correlation\n"
        "run_backtest_with_alphalib = engine.run_backtest_with_public_engine\n\n"
        "def compute_combined_score(metrics, judge_result):\n"
        "    return engine.combined_score(metrics)\n\n"
        "def _all_finite(values):\n"
        "    return all(value is not None and math.isfinite(float(value)) for value in values)\n\n"
        "def _build_signal_from_code_snapshot(code_path: Path, data_path: Path, module_name: str) -> pl.DataFrame:\n"
        "    if module_name in sys.modules:\n"
        "        del sys.modules[module_name]\n"
        "    spec = importlib.util.spec_from_file_location(module_name, code_path)\n"
        "    module = importlib.util.module_from_spec(spec)\n"
        "    sys.modules[module_name] = module\n"
        "    spec.loader.exec_module(module)\n"
        "    return module.build_signal(str(data_path))\n"
    )
    return task_dir, candidate


def summarize(result: dict[str, Any]) -> dict[str, Any]:
    return {key: result.get(key) for key in SUMMARY_KEYS}


def compare_summary(actual: dict[str, Any], expected: dict[str, Any]) -> list[str]:
    failures: list[str] = []
    for key in SUMMARY_KEYS:
        actual_value = actual.get(key)
        expected_value = expected.get(key)
        if isinstance(expected_value, float):
            if not isinstance(actual_value, (float, int)):
                failures.append(f"{key}: expected float {expected_value}, got {actual_value!r}")
            elif not math.isclose(float(actual_value), expected_value, rel_tol=FLOAT_TOLERANCE, abs_tol=FLOAT_TOLERANCE):
                failures.append(f"{key}: expected {expected_value}, got {actual_value}")
        elif actual_value != expected_value:
            failures.append(f"{key}: expected {expected_value!r}, got {actual_value!r}")
    return failures


def command_run(args: argparse.Namespace) -> int:
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    task_dir, candidate = make_synthetic_task(output_dir)
    result = score_code(
        task_dir=task_dir,
        code_path=candidate,
        output_dir=output_dir / "verifier",
        causality_gate=True,
    )
    summary = summarize(result)
    write_json(output_dir / "reference_summary.json", summary)

    if args.update_expected:
        EXPECTED_PATH.parent.mkdir(parents=True, exist_ok=True)
        write_json(EXPECTED_PATH, summary)
        print(f"updated {EXPECTED_PATH}")
        return 0

    expected = json.loads(EXPECTED_PATH.read_text())
    failures = compare_summary(summary, expected)
    if failures:
        print("reference smoke mismatch:", file=sys.stderr)
        for failure in failures:
            print(f"  - {failure}", file=sys.stderr)
        return 1
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--update-expected", action="store_true")
    parser.set_defaults(func=command_run)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
