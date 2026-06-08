#!/usr/bin/env python
"""Local runner for testing the signal-ml task without Docker.

Usage:
    # Run the starter code (will fail until implemented)
    python run_local.py --starter

    # Run evaluation only (after signal.parquet exists)
    python run_local.py --eval-only
"""

import argparse
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import polars as pl
from scipy import stats

# Paths
TASK_DIR = Path(__file__).parent
DATA_DIR = TASK_DIR / "environment" / "data"
OUTPUT_DIR = TASK_DIR / "output"


def run_starter():
    """Run the starter code."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "signal_module", TASK_DIR / "environment" / "starter_code" / "code.py"
    )
    signal_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(signal_module)
    build_signal = signal_module.build_signal

    # Concatenate train + test for proper rolling calculations
    print("Loading data...")
    train_df = pl.read_parquet(DATA_DIR / "train.parquet")
    test_path = TASK_DIR / "tests" / "test.parquet"
    if test_path.exists():
        test_df = pl.read_parquet(test_path)
        all_df = pl.concat([train_df, test_df]).sort(["ticker", "timestamp_agg"])
        print(f"Loaded {len(all_df):,} rows ({len(train_df):,} train + {len(test_df):,} test)")
    else:
        all_df = train_df.sort(["ticker", "timestamp_agg"])
        print(f"Loaded {len(all_df):,} rows (train only, test not found)")

    # Write to temp file and call build_signal with path
    with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
        temp_path = f.name
    all_df.write_parquet(temp_path)

    print("\nBuilding signal...")
    signal_df = build_signal(temp_path)

    # Cleanup temp file
    Path(temp_path).unlink()

    # Validate
    assert "timestamp_agg" in signal_df.columns, "Missing timestamp_agg column"
    assert "ticker" in signal_df.columns, "Missing ticker column"
    assert "signal" in signal_df.columns, "Missing signal column"

    signal_df = signal_df.filter(
        pl.col("signal").is_not_null()
        & pl.col("signal").is_not_nan()
        & pl.col("signal").is_finite()
    )

    print(f"Generated {len(signal_df):,} valid signals")

    OUTPUT_DIR.mkdir(exist_ok=True)
    signal_df.write_parquet(OUTPUT_DIR / "signal.parquet")
    print(f"\nSaved to {OUTPUT_DIR / 'signal.parquet'}")


def run_evaluation() -> dict:
    """Evaluate the generated signal against test data."""
    signal_path = OUTPUT_DIR / "signal.parquet"
    test_path = TASK_DIR / "tests" / "test.parquet"

    if not signal_path.exists():
        return {"error": "No signal file found", "passed": False}

    if not test_path.exists():
        return {"error": "No test data found", "passed": False}

    # Load data
    signal_df = pl.read_parquet(signal_path)
    test_df = pl.read_parquet(test_path)

    print(f"Signal: {len(signal_df):,} rows")
    print(f"Test:   {len(test_df):,} rows")

    # Join signal with test returns
    joined = signal_df.join(
        test_df.select(["timestamp_agg", "ticker", "return"]),
        on=["timestamp_agg", "ticker"],
        how="inner",
    ).filter(
        pl.col("signal").is_not_null()
        & pl.col("signal").is_not_nan()
        & pl.col("signal").is_finite()
        & pl.col("return").is_not_null()
    )

    print(f"Joined: {len(joined):,} rows")

    if len(joined) == 0:
        return {"error": "No overlapping data", "passed": False}

    # Compute IC per timestamp
    ic_values = []
    for _, group in joined.group_by("timestamp_agg"):
        if len(group) < 3:
            continue
        signals = group["signal"].to_numpy()
        returns = group["return"].to_numpy()
        ic, _ = stats.spearmanr(signals, returns)
        if not np.isnan(ic):
            ic_values.append(ic)

    if len(ic_values) == 0:
        return {"error": "No valid IC computations", "passed": False}

    mean_ic = float(np.mean(ic_values))
    ic_std = float(np.std(ic_values))
    ic_positive_pct = float(np.mean([ic > 0 for ic in ic_values]))

    # Cheating check
    signals = joined["signal"].to_numpy()
    returns = joined["return"].to_numpy()
    return_corr, _ = stats.pearsonr(signals, returns)
    return_corr = float(return_corr) if not np.isnan(return_corr) else 0.0

    cheating = abs(return_corr) > 0.90
    passed = (mean_ic > 0.003) and not cheating
    score = min(1.5, max(-1.0, mean_ic * 100)) if not cheating else -1.0

    return {
        "ic": mean_ic,
        "ic_std": ic_std,
        "ic_positive_pct": ic_positive_pct,
        "n_timestamps": len(ic_values),
        "return_correlation": return_corr,
        "cheating_detected": cheating,
        "passed": passed,
        "score": score,
    }


def main():
    parser = argparse.ArgumentParser(description="Run signal-ml task locally")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--starter", action="store_true", help="Run starter code")
    group.add_argument("--eval-only", action="store_true", help="Run evaluation only")
    args = parser.parse_args()

    # Check data exists
    if not (DATA_DIR / "train.parquet").exists():
        print(f"Training data not found at {DATA_DIR}/train.parquet")
        print("\nRestore the data bundle from the repository root before running this task.")
        print("See DATA.md in the RegimeBench checkout.")
        sys.exit(1)

    print("=" * 60)
    print("SIGNAL-ML")
    print("=" * 60)

    # Run signal generation
    if args.starter:
        print("Mode: Starter Code\n")
        try:
            run_starter()
        except NotImplementedError as e:
            print(f"\nNotImplementedError: {e}")
            print("   Implement build_signal() in starter_code/code.py")
            sys.exit(1)
        except Exception as e:
            print(f"\nError: {e}")
            import traceback

            traceback.print_exc()
            sys.exit(1)

    # Run evaluation
    test_path = TASK_DIR / "tests" / "test.parquet"
    if not test_path.exists():
        print("\nTest data not found - skipping evaluation")
        sys.exit(0)

    print("\n" + "=" * 60)
    print("EVALUATION")
    print("=" * 60)
    results = run_evaluation()

    # Save results
    OUTPUT_DIR.mkdir(exist_ok=True)
    with open(OUTPUT_DIR / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    if "error" in results:
        print(f"\nError: {results['error']}")
        sys.exit(1)

    print(f"\nIC:                 {results['ic']:.6f}")
    print(f"IC Std:             {results['ic_std']:.6f}")
    print(f"IC Positive %:      {results['ic_positive_pct']:.1%}")
    print(f"Timestamps:         {results['n_timestamps']:,}")
    print(f"Return Correlation: {results['return_correlation']:.4f}")
    print(f"Cheating:           {results['cheating_detected']}")
    print(f"Score:              {results['score']:.4f}")

    print("\n" + "=" * 60)
    if results["passed"]:
        print("PASSED")
    else:
        print("FAILED")
        if results["cheating_detected"]:
            print("   Reason: Signal correlated with returns (cheating)")
        elif results["ic"] <= 0.003:
            print(f"   Reason: IC ({results['ic']:.6f}) below threshold (0.003)")
    print("=" * 60)

    sys.exit(0 if results["passed"] else 1)


if __name__ == "__main__":
    main()
