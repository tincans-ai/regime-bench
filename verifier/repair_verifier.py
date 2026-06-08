#!/usr/bin/env python3
"""Saved-code replay verifier used as RegimeBench's default public scorer.

This is the public-facing slice of the internal QR-Eval PnL repair replay:
load a task's evaluator, build the strict hidden/OOS feature frame, run the
submitted ``build_signal(data_path)`` snapshot, apply the bar-level causality
gate, and replay the resulting signal through the deterministic PnL evaluator.
"""

from __future__ import annotations

import gc
import hashlib
import importlib.util
import json
import math
import sys
import traceback
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from typing import Any

import polars as pl


class CausalityGateFailure(RuntimeError):
    def __init__(self, result: dict[str, Any]) -> None:
        self.result = result
        super().__init__(str(result.get("failure_reason") or "causality gate failed"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_as_jsonable(payload), indent=2, sort_keys=True) + "\n")


def _as_jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _as_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_as_jsonable(item) for item in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            return str(value)
    return value


def _safe_float(value: Any, default: float | None = None) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def _load_evaluator(task_dir: Path) -> Any:
    path = task_dir / "tests" / "evaluate.py"
    if not path.exists():
        raise FileNotFoundError(f"missing evaluator for {task_dir.name}: {path}")
    digest = hashlib.sha1(str(path.resolve()).encode("utf-8")).hexdigest()[:10]
    module_name = f"regimebench_repair_eval_{task_dir.name.replace('-', '_')}_{digest}"
    if module_name in sys.modules:
        return sys.modules[module_name]
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not import evaluator: {path}")
    module = importlib.util.module_from_spec(spec)
    old_path = list(sys.path)
    try:
        sys.path.insert(0, str(path.parent))
        sys.path.insert(0, str(task_dir.parent.parent))
        sys.modules.pop("pnl_engine", None)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
    finally:
        sys.path[:] = old_path
    return module


def _daily_pnl(backtest_results: pl.DataFrame | None) -> pl.DataFrame:
    if backtest_results is None or backtest_results.is_empty() or "net_ret" not in backtest_results.columns:
        return pl.DataFrame(schema={"date": pl.Date, "daily_ret": pl.Float64})
    return (
        backtest_results.with_columns(pl.col("timestamp_agg").dt.date().alias("date"))
        .group_by("date")
        .agg(pl.col("net_ret").sum().alias("daily_ret"))
        .sort("date")
    )


def _timestamp_dtype(frame: pl.DataFrame) -> Any | None:
    return frame.schema.get("timestamp_agg") if "timestamp_agg" in frame.columns else None


def _timestamp_lit(value: Any, dtype: Any | None) -> pl.Expr:
    return pl.lit(value, dtype=dtype) if dtype is not None else pl.lit(value)


def _align_timestamp_dtype(frame: pl.DataFrame, dtype: Any | None) -> pl.DataFrame:
    if dtype is None or frame.is_empty() or "timestamp_agg" not in frame.columns:
        return frame
    if frame.schema.get("timestamp_agg") == dtype:
        return frame
    return frame.with_columns(pl.col("timestamp_agg").cast(dtype).alias("timestamp_agg"))


def _dk_timestamp_frame(frame: pl.DataFrame) -> pl.DataFrame:
    dtype = _timestamp_dtype(frame)
    if isinstance(dtype, pl.Datetime) and dtype.time_unit != "us":
        return _align_timestamp_dtype(frame, pl.Datetime(time_unit="us", time_zone=dtype.time_zone))
    return frame


def _compute_dk_tstat(evaluator: Any, signal_df: pl.DataFrame, test_df: pl.DataFrame) -> float:
    # Some exported evaluators build a datetime replacement map that Polars
    # treats as microsecond precision. Normalize only for this compatibility
    # call so joins/backtests keep the task's original timestamp dtype.
    return evaluator.compute_dk_tstat(_dk_timestamp_frame(signal_df), _dk_timestamp_frame(test_df))


def _feature_frame_full_labels(
    evaluator: Any,
    train_df: pl.DataFrame | None,
    hidden_df: pl.DataFrame,
) -> pl.DataFrame:
    frames: list[pl.DataFrame] = []
    if train_df is not None:
        frames.append(train_df.with_columns(pl.lit(False).alias("is_prediction_period")))
    frames.append(hidden_df.with_columns(pl.lit(True).alias("is_prediction_period")))
    feature_df = pl.concat(frames, how="diagonal_relaxed")
    sort_cols = evaluator._sort_columns_for(feature_df)
    if sort_cols:
        feature_df = feature_df.sort(sort_cols)
    return_columns = evaluator._return_columns_for(feature_df)
    return evaluator._with_safe_return_lags(feature_df, return_columns)


def _prediction_return_masked_frame_at_timestamp(
    evaluator: Any,
    train_df: pl.DataFrame | None,
    hidden_df: pl.DataFrame,
    *,
    gate_timestamp: Any,
    include_future: bool,
    future_timestamps: list[Any] | None = None,
    adversarial_current: bool = False,
    adversarial_future: bool = False,
) -> pl.DataFrame:
    timestamp_dtype = _timestamp_dtype(hidden_df)
    gate_lit = _timestamp_lit(gate_timestamp, timestamp_dtype)
    if include_future and future_timestamps:
        future_mask = pl.any_horizontal(
            [pl.col("timestamp_agg") == _timestamp_lit(value, timestamp_dtype) for value in future_timestamps]
        )
        selected_hidden = pl.concat(
            [
                hidden_df.filter(pl.col("timestamp_agg") <= gate_lit),
                hidden_df.filter(future_mask),
            ],
            how="diagonal_relaxed",
        ).unique(maintain_order=True)
    elif include_future:
        selected_hidden = hidden_df
    else:
        selected_hidden = hidden_df.filter(pl.col("timestamp_agg") <= gate_lit)
    selected_hidden = selected_hidden.with_columns(
        (pl.col("timestamp_agg") == gate_lit).alias("is_prediction_period")
    )

    frames: list[pl.DataFrame] = []
    if train_df is not None:
        frames.append(train_df.with_columns(pl.lit(False).alias("is_prediction_period")))
    frames.append(selected_hidden)
    feature_df = pl.concat(frames, how="diagonal_relaxed")
    sort_cols = evaluator._sort_columns_for(feature_df)
    if sort_cols:
        feature_df = feature_df.sort(sort_cols)
    return_columns = evaluator._return_columns_for(feature_df)
    feature_df = evaluator._with_safe_return_lags(feature_df, return_columns)
    if not return_columns:
        return feature_df

    feature_df = feature_df.with_row_index("_gate_row_nr")
    exprs: list[pl.Expr] = []
    for column in return_columns:
        null_value = pl.lit(None, dtype=feature_df.schema[column])
        randomish = (
            (((pl.col("_gate_row_nr") * 1103515245 + 12345) % 2000000) / 1000000.0)
            - 1.0
        ) * 0.25
        value = pl.col(column)
        if adversarial_future:
            value = pl.when(pl.col("timestamp_agg") > gate_lit).then(randomish).otherwise(value)
        if adversarial_current:
            value = pl.when(pl.col("timestamp_agg") == gate_lit).then(randomish).otherwise(value)
        else:
            value = pl.when(pl.col("timestamp_agg") == gate_lit).then(null_value).otherwise(value)
        exprs.append(value.alias(column))
    return feature_df.with_columns(exprs).drop("_gate_row_nr")


def _future_sentinel_timestamps(
    timestamps: list[Any],
    gate_timestamp: Any,
    *,
    count: int = 2,
) -> list[Any]:
    future = [value for value in timestamps if value > gate_timestamp]
    if not future or count <= 0:
        return []
    selected: list[Any] = [future[0]]
    if count > 1 and future[-1] != selected[0]:
        selected.append(future[-1])
    return selected[:count]


def _build_signal_from_feature_frame(
    *,
    evaluator: Any,
    code_path: Path,
    feature_df: pl.DataFrame,
    feature_path: Path,
    module_suffix: str,
) -> pl.DataFrame:
    feature_path.parent.mkdir(parents=True, exist_ok=True)
    feature_df.write_parquet(feature_path)
    try:
        signal_df = evaluator._build_signal_from_code_snapshot(
            code_path,
            feature_path,
            f"regimebench_repair_code_{module_suffix}",
        )
    finally:
        try:
            feature_path.unlink(missing_ok=True)
        except OSError:
            pass
    required_cols = {"timestamp_agg", "ticker", "signal"}
    missing_cols = sorted(required_cols - set(signal_df.columns))
    if missing_cols:
        raise RuntimeError(f"invalid signal schema; missing columns: {missing_cols}")
    return signal_df


def _prediction_signal(
    *,
    evaluator: Any,
    code_path: Path,
    feature_df: pl.DataFrame,
    feature_path: Path,
    module_suffix: str,
) -> pl.DataFrame:
    signal_df_full = _build_signal_from_feature_frame(
        evaluator=evaluator,
        code_path=code_path,
        feature_df=feature_df,
        feature_path=feature_path,
        module_suffix=module_suffix,
    )
    signal_df_full = _align_timestamp_dtype(signal_df_full, _timestamp_dtype(feature_df))
    signal_df = evaluator._filter_signal_period(signal_df_full, feature_df, prediction=True)
    if signal_df.is_empty():
        return signal_df
    signal_df = evaluator._preprocess_signal(signal_df).select(["timestamp_agg", "ticker", "signal"])
    return _align_timestamp_dtype(signal_df, _timestamp_dtype(feature_df))


def _finite_signal_count(signal_df: pl.DataFrame) -> int:
    if signal_df.is_empty() or "signal" not in signal_df.columns:
        return 0
    return int(
        signal_df.filter(
            pl.col("signal").is_not_null()
            & pl.col("signal").is_not_nan()
            & pl.col("signal").is_finite()
        ).height
    )


def _signal_delta(left: pl.DataFrame, right: pl.DataFrame) -> dict[str, Any]:
    if left.is_empty() or right.is_empty():
        return {
            "shared_rows": 0,
            "max_abs_delta": None,
            "changed_rows": None,
            "missing_rows": max(left.height, right.height),
        }
    timestamp_dtype = _timestamp_dtype(left) or _timestamp_dtype(right)
    left = _align_timestamp_dtype(left, timestamp_dtype)
    right = _align_timestamp_dtype(right, timestamp_dtype)
    joined = (
        left.rename({"signal": "signal_left"})
        .join(
            right.rename({"signal": "signal_right"}),
            on=["timestamp_agg", "ticker"],
            how="inner",
        )
        .with_columns((pl.col("signal_left") - pl.col("signal_right")).abs().alias("abs_delta"))
    )
    shared = joined.height
    max_abs_delta = float(joined["abs_delta"].max()) if shared else None
    changed = int(joined.filter(pl.col("abs_delta") > 1e-8).height) if shared else None
    return {
        "shared_rows": shared,
        "max_abs_delta": max_abs_delta,
        "changed_rows": changed,
        "missing_rows": max(left.height, right.height) - shared,
    }


def run_causality_gate(
    *,
    evaluator: Any,
    code_path: Path,
    train_df: pl.DataFrame | None,
    hidden_df: pl.DataFrame,
    work_dir: Path,
    gate_days: int,
    min_coverage: float,
) -> dict[str, Any]:
    timestamps = hidden_df.select("timestamp_agg").unique().sort("timestamp_agg")["timestamp_agg"].to_list()
    if not timestamps:
        return {
            "status": "failed",
            "failure_reason": "hidden OOS frame has no timestamps",
            "gate_days": [],
            "gate_unit": "bar",
        }
    if gate_days <= 1:
        selected_timestamps = [timestamps[min(1, len(timestamps) - 1)]]
    else:
        indices = sorted({min(1, len(timestamps) - 1), len(timestamps) // 2, max(0, len(timestamps) - 2)})[:gate_days]
        selected_timestamps = [timestamps[index] for index in indices]

    day_results: list[dict[str, Any]] = []
    for idx, gate_timestamp in enumerate(selected_timestamps, start=1):
        gate_dir = work_dir / f"causality_gate_{idx:02d}"
        timestamp_dtype = _timestamp_dtype(hidden_df)
        hidden_bar = hidden_df.filter(pl.col("timestamp_agg") == _timestamp_lit(gate_timestamp, timestamp_dtype))
        expected_rows = hidden_bar.select(["timestamp_agg", "ticker"]).unique().height
        min_rows = max(1, int(expected_rows * min_coverage))
        try:
            causal_feature = _prediction_return_masked_frame_at_timestamp(
                evaluator,
                train_df,
                hidden_df,
                gate_timestamp=gate_timestamp,
                include_future=False,
                adversarial_current=False,
                adversarial_future=False,
            )
            causal_signal = _prediction_signal(
                evaluator=evaluator,
                code_path=code_path,
                feature_df=causal_feature,
                feature_path=gate_dir / "causal_feature.parquet",
                module_suffix=f"gate_{idx:02d}_causal",
            )
            finite_rows = _finite_signal_count(causal_signal)
            if finite_rows < min_rows:
                day_results.append(
                    {
                        "gate_timestamp": str(gate_timestamp),
                        "status": "failed",
                        "failure_reason": "insufficient causal prediction rows",
                        "expected_rows": expected_rows,
                        "minimum_rows": min_rows,
                        "finite_rows": finite_rows,
                    }
                )
                continue

            current_feature = _prediction_return_masked_frame_at_timestamp(
                evaluator,
                train_df,
                hidden_df,
                gate_timestamp=gate_timestamp,
                include_future=False,
                adversarial_current=True,
                adversarial_future=False,
            )
            current_signal = _prediction_signal(
                evaluator=evaluator,
                code_path=code_path,
                feature_df=current_feature,
                feature_path=gate_dir / "current_adversarial_feature.parquet",
                module_suffix=f"gate_{idx:02d}_current",
            )
            current_delta = _signal_delta(causal_signal, current_signal)
            del current_feature, current_signal
            gc.collect()

            future_sentinels = _future_sentinel_timestamps(timestamps, gate_timestamp, count=2)
            future_feature = _prediction_return_masked_frame_at_timestamp(
                evaluator,
                train_df,
                hidden_df,
                gate_timestamp=gate_timestamp,
                include_future=True,
                future_timestamps=future_sentinels,
                adversarial_current=False,
                adversarial_future=True,
            )
            future_signal = _prediction_signal(
                evaluator=evaluator,
                code_path=code_path,
                feature_df=future_feature,
                feature_path=gate_dir / "future_adversarial_feature.parquet",
                module_suffix=f"gate_{idx:02d}_future",
            )
            future_delta = _signal_delta(causal_signal, future_signal)
            del causal_feature, future_feature, future_signal
            gc.collect()

            failed_reason = None
            if current_delta["missing_rows"] or (current_delta["max_abs_delta"] or 0.0) > 1e-8:
                failed_reason = "signal changes when current prediction bar labels are perturbed"
            elif future_delta["missing_rows"] or (future_delta["max_abs_delta"] or 0.0) > 1e-8:
                failed_reason = "signal changes when future labels/rows are perturbed"
            day_results.append(
                {
                    "gate_timestamp": str(gate_timestamp),
                    "status": "failed" if failed_reason else "passed",
                    "failure_reason": failed_reason,
                    "expected_rows": expected_rows,
                    "minimum_rows": min_rows,
                    "finite_rows": finite_rows,
                    "future_sentinel_timestamps": [str(value) for value in future_sentinels],
                    "current_label_delta": current_delta,
                    "future_label_delta": future_delta,
                }
            )
        except Exception as exc:
            day_results.append(
                {
                    "gate_timestamp": str(gate_timestamp),
                    "status": "failed",
                    "failure_reason": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc(limit=8),
                }
            )

    failed = [item for item in day_results if item.get("status") != "passed"]
    result = {
        "status": "failed" if failed else "passed",
        "failure_reason": failed[0].get("failure_reason") if failed else None,
        "gate_days": day_results,
        "gate_day_count": len(day_results),
        "gate_unit": "bar",
        "llm_audit_status": "not_requested",
    }
    write_json(work_dir / "causality_gate.json", result)
    return result


def _build_verifier_results(
    *,
    evaluator: Any,
    signal_df_full: pl.DataFrame,
    feature_df: pl.DataFrame,
    train_df: pl.DataFrame,
    hidden_df: pl.DataFrame,
    backtest_results: pl.DataFrame | None,
    backtest_metrics: dict[str, Any],
    symbols: list[str],
) -> tuple[dict[str, Any], pl.DataFrame]:
    signal_df_full = _align_timestamp_dtype(signal_df_full, _timestamp_dtype(feature_df))
    signal_df = evaluator._filter_signal_period(signal_df_full, feature_df, prediction=True)
    if signal_df.is_empty():
        raise RuntimeError("no prediction-period signal rows returned by build_signal")
    signal_df = evaluator._preprocess_signal(signal_df)
    signal_df = _align_timestamp_dtype(signal_df, _timestamp_dtype(hidden_df))

    ic_df = evaluator.compute_ic_per_timestamp(signal_df, hidden_df)
    if ic_df.is_empty():
        raise RuntimeError("no valid IC computations")
    ic_values = ic_df["ic"].to_numpy()
    mean_ic = float(evaluator.np.mean(ic_values))
    ic_std = float(evaluator.np.std(ic_values))
    ic_positive_pct = float((ic_values > 0).mean())
    ic_tstat_dk = _compute_dk_tstat(evaluator, signal_df, hidden_df)
    signal_return_sharpe = evaluator.compute_daily_sharpe(backtest_results)
    max_drawdown = _safe_float(backtest_metrics.get("max_drawdown"), 0.0)
    turnover_ann = _safe_float(backtest_metrics.get("turnover_ann"), 0.0)

    autocorrelation = evaluator.compute_autocorrelation(signal_df)
    quantile_metrics = evaluator.compute_quantile_metrics(signal_df, hidden_df)
    ic_decay = evaluator.compute_ic_decay(signal_df, hidden_df)
    return_corr = evaluator.compute_return_correlation(signal_df, hidden_df)
    cheating_detected = abs(float(return_corr or 0.0)) > 0.90

    n_timestamps = len(ic_df)
    n_symbols_mean = len(symbols) if symbols else 0
    coverage_pct = len(signal_df) / (n_timestamps * n_symbols_mean) if n_timestamps > 0 and n_symbols_mean > 0 else 0.0

    is_metrics: dict[str, Any] = {}
    try:
        signal_df_is = evaluator._filter_signal_period(signal_df_full, feature_df, prediction=False)
        if not signal_df_is.is_empty():
            signal_df_is = evaluator._preprocess_signal(signal_df_is)
            signal_df_is = _align_timestamp_dtype(signal_df_is, _timestamp_dtype(train_df))
            ic_df_is = evaluator.compute_ic_per_timestamp(signal_df_is, train_df)
            if not ic_df_is.is_empty():
                ic_values_is = ic_df_is["ic"].to_numpy()
                is_metrics["mean_ic"] = float(evaluator.np.mean(ic_values_is))
                is_metrics["ic_std"] = float(evaluator.np.std(ic_values_is))
                is_metrics["ic_positive_pct"] = float((ic_values_is > 0).mean())
                is_metrics["ic_tstat_dk"] = _compute_dk_tstat(evaluator, signal_df_is, train_df)
                is_backtest_results, is_backtest_metrics, _, _ = evaluator.run_backtest_with_alphalib(
                    signal_df_is,
                    train_df,
                )
                is_metrics["sharpe_net"] = evaluator.compute_daily_sharpe(is_backtest_results)
                is_metrics["max_drawdown"] = (is_backtest_metrics or {}).get("max_drawdown", 0.0)
                if abs(is_metrics["mean_ic"]) > evaluator.EPS:
                    is_metrics["ic_gap_pct"] = (is_metrics["mean_ic"] - mean_ic) / abs(is_metrics["mean_ic"]) * 100
                if is_metrics["sharpe_net"] > evaluator.EPS:
                    is_metrics["sharpe_gap_pct"] = (
                        (is_metrics["sharpe_net"] - signal_return_sharpe)
                        / is_metrics["sharpe_net"]
                        * 100
                    )
    except Exception as exc:
        is_metrics["error"] = str(exc)

    metrics = evaluator.SignalMetrics(
        mean_ic=mean_ic,
        ic_std=ic_std,
        ic_positive_pct=ic_positive_pct,
        ic_tstat_dk=ic_tstat_dk,
        signal_return_sharpe=signal_return_sharpe,
        autocorrelation=autocorrelation,
        long_short_spread=quantile_metrics["long_short_spread"],
        monotonicity=quantile_metrics["monotonicity"],
        n_timestamps=n_timestamps,
        n_symbols_mean=float(n_symbols_mean),
        coverage_pct=coverage_pct,
        ic_decay=ic_decay,
    )
    judge_result = {
        "decision": "skip",
        "score": 0.5,
        "feedback": "Skipped during saved-code deterministic repair replay; judge weight is zero.",
        "issues": [],
        "suggestions": [],
    }
    finite_values = [
        mean_ic,
        ic_std,
        ic_positive_pct,
        ic_tstat_dk,
        signal_return_sharpe,
        autocorrelation,
        quantile_metrics["long_short_spread"],
        quantile_metrics["monotonicity"],
        n_timestamps,
        n_symbols_mean,
        coverage_pct,
        return_corr,
        max_drawdown,
        turnover_ann,
    ]
    if not evaluator._all_finite(finite_values):
        final_score = -1.0
        passed = False
    else:
        final_score, passed = evaluator.compute_combined_score(metrics, judge_result)
        if cheating_detected:
            final_score = 0.0
            passed = False
            judge_result = {
                "decision": "reject",
                "score": 0.0,
                "feedback": f"Cheating detected: signal has {return_corr:.2%} correlation with returns",
                "issues": ["Signal directly uses return values"],
                "suggestions": ["Build signal from features only, not from return column"],
            }

    results = {
        "ic": mean_ic,
        "ic_std": ic_std,
        "ic_positive_pct": ic_positive_pct,
        "tstat_threshold": evaluator.TSTAT_THRESHOLD,
        "sharpe_threshold": evaluator.SHARPE_THRESHOLD,
        "n_timestamps": n_timestamps,
        "return_correlation": return_corr,
        "cheating_detected": cheating_detected,
        **metrics.to_dict(),
        "backtest_sharpe": signal_return_sharpe,
        "backtest_max_drawdown": max_drawdown,
        "backtest_turnover_ann": turnover_ann,
        **backtest_metrics,
        "in_sample": is_metrics if is_metrics else None,
        "judge_decision": judge_result.get("decision", "skip"),
        "judge_score": judge_result.get("score", 0.5),
        "judge_feedback": judge_result.get("feedback", ""),
        "judge_issues": judge_result.get("issues", []),
        "judge_suggestions": judge_result.get("suggestions", []),
        "score": final_score,
        "passed": passed,
    }
    return results, signal_df


def _replay_task(
    *,
    evaluator: Any,
    task_dir: Path,
    code_path: Path,
    work_dir: Path,
    replay_mode: str,
    causality_gate: bool,
    gate_days: int,
    min_gate_coverage: float,
) -> tuple[pl.DataFrame, pl.DataFrame, dict[str, Any]]:
    train_path = task_dir / "environment" / "data" / "train.parquet"
    hidden_path = task_dir / "tests" / "test.parquet"
    if not train_path.exists():
        raise FileNotFoundError(f"missing train parquet: {train_path}")
    if not hidden_path.exists():
        raise FileNotFoundError(f"missing hidden OOS parquet: {hidden_path}")

    policy_violations = evaluator._source_policy_violations(code_path)
    if policy_violations:
        raise RuntimeError("policy violation: " + "; ".join(policy_violations))

    work_dir.mkdir(parents=True, exist_ok=True)
    log_path = work_dir / "verifier_stdout.txt"
    with log_path.open("w") as log, redirect_stdout(log), redirect_stderr(log):
        train_df = pl.read_parquet(train_path)
        hidden_df = pl.read_parquet(hidden_path)
        gate_result: dict[str, Any] | None = None
        if causality_gate:
            gate_result = run_causality_gate(
                evaluator=evaluator,
                code_path=code_path,
                train_df=train_df,
                hidden_df=hidden_df,
                work_dir=work_dir,
                gate_days=gate_days,
                min_coverage=min_gate_coverage,
            )
            if gate_result.get("status") != "passed":
                raise CausalityGateFailure(gate_result)

        if replay_mode == "full_labeled":
            feature_df = _feature_frame_full_labels(evaluator, train_df, hidden_df)
        elif replay_mode == "strict":
            feature_df = evaluator._final_eval_feature_frame(train_df, hidden_df)
        else:
            raise ValueError(f"unknown replay_mode={replay_mode!r}")

        module_digest = hashlib.sha1(str(code_path).encode("utf-8")).hexdigest()[:12]
        signal_df_full = _build_signal_from_feature_frame(
            evaluator=evaluator,
            code_path=code_path,
            feature_df=feature_df,
            feature_path=work_dir / "feature.parquet",
            module_suffix=f"replay_{module_digest}",
        )
        signal_df_full = _align_timestamp_dtype(signal_df_full, _timestamp_dtype(feature_df))
        signal_df = evaluator._filter_signal_period(signal_df_full, feature_df, prediction=True)
        if signal_df.is_empty():
            raise RuntimeError("no prediction-period signal rows returned by build_signal")
        signal_df = evaluator._preprocess_signal(signal_df)
        signal_df = _align_timestamp_dtype(signal_df, _timestamp_dtype(hidden_df))
        backtest_results, backtest_metrics, _timestamps, symbols = evaluator.run_backtest_with_alphalib(
            signal_df,
            hidden_df,
        )
        daily = _daily_pnl(backtest_results)
        if daily.is_empty():
            raise RuntimeError("replay produced no daily PnL rows")
        results, signal_df = _build_verifier_results(
            evaluator=evaluator,
            signal_df_full=signal_df_full,
            feature_df=feature_df,
            train_df=train_df,
            hidden_df=hidden_df,
            backtest_results=backtest_results,
            backtest_metrics=backtest_metrics or {},
            symbols=symbols,
        )
        results["replay_mode"] = replay_mode
        results["causality_gate"] = gate_result
        results["verifier_log_path"] = str(log_path)
    return daily, signal_df, results


def score_code(
    *,
    task_dir: Path,
    code_path: Path,
    output_dir: Path,
    causality_gate: bool = True,
    gate_days: int = 1,
    min_gate_coverage: float = 0.01,
    replay_mode: str = "strict",
) -> dict[str, Any]:
    task_dir = task_dir.resolve()
    code_path = code_path.resolve()
    output_dir = output_dir.resolve()
    evaluator = _load_evaluator(task_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        daily, signal, result = _replay_task(
            evaluator=evaluator,
            task_dir=task_dir,
            code_path=code_path,
            work_dir=output_dir / "_work",
            replay_mode=replay_mode,
            causality_gate=causality_gate,
            gate_days=gate_days,
            min_gate_coverage=min_gate_coverage,
        )
    except CausalityGateFailure as exc:
        result = {
            "status": "failed",
            "task_name": task_dir.name,
            "code_path": str(code_path),
            "replay_mode": replay_mode,
            "error": "causality gate failed",
            "causality_gate": exc.result,
            "score": 0.0,
            "passed": False,
        }
        write_json(output_dir / "results.json", result)
        return result

    daily_path = output_dir / "daily_pnl.parquet"
    signal_path = output_dir / "signal.parquet"
    daily.write_parquet(daily_path)
    signal.write_parquet(signal_path)
    result.update(
        {
            "status": "completed",
            "task_name": task_dir.name,
            "code_path": str(code_path),
            "daily_pnl_path": str(daily_path),
            "signal_path": str(signal_path),
        }
    )
    write_json(output_dir / "results.json", result)
    return result
