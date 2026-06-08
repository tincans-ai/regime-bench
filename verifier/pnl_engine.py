#!/usr/bin/env python3
"""Deterministic PnL scoring utilities for RegimeBench.

This module intentionally vendors only the small backtest surface needed by the
signal tasks and canonical verifier. It avoids the private internal ``alphalib``
dependency used while the benchmark was developed.
"""

from __future__ import annotations

import importlib.util
import json
import math
import re
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from scipy import stats

try:
    from statsmodels.api import OLS, add_constant
except Exception:  # pragma: no cover - optional but expected in release envs
    OLS = None
    add_constant = None


EPS = 1e-8
RETURN_COLUMN_NAME = "residual_return"
RESIDUAL_RETURN_COLUMN_NAME = "residual_return"
SAFE_RETURN_LAGS = (1, 2, 3, 6, 12, 24)
TSTAT_THRESHOLD = 1.0
SHARPE_THRESHOLD = 0.7
AUTOCORR_OPTIMAL_LOW = 0.50
AUTOCORR_OPTIMAL_HIGH = 0.999
AUTOCORR_PENALTY_MAX = 0.40
AUTOCORR_PENALTY_FLOOR = 0.10
WEIGHT_TSTAT = 0.40
WEIGHT_SHARPE = 0.60
MIN_SCORE_VALID_OUTPUT = 0.05


@dataclass
class BacktestConfig:
    max_gross: float = 2.0
    cost_model: str = "none"
    periodicity: str = "1h"


@dataclass
class BacktestMetrics:
    n_periods: int
    total_return: float
    cagr: float
    ann_vol_net: float
    sharpe_net: float
    sharpe_gross: float
    max_drawdown: float
    turnover_ann: float
    turnover_cost_bps_ann: float
    gross_leverage_mean: float
    funding_pnl_ann_bps: float = 0.0

    def to_dict(self) -> dict[str, float | int]:
        return {
            "n_periods": self.n_periods,
            "total_return": self.total_return,
            "CAGR": self.cagr,
            "ann_vol_net": self.ann_vol_net,
            "sharpe_net": self.sharpe_net,
            "sharpe_gross": self.sharpe_gross,
            "max_drawdown": self.max_drawdown,
            "turnover_ann": self.turnover_ann,
            "turnover_cost_bps_ann": self.turnover_cost_bps_ann,
            "gross_leverage_mean": self.gross_leverage_mean,
            "funding_pnl_ann_bps": self.funding_pnl_ann_bps,
        }


@dataclass
class WideMatrix:
    data: pl.DataFrame
    timestamp_col: str = "timestamp_agg"
    value_name: str = "value"

    @property
    def symbol_cols(self) -> list[str]:
        return [column for column in self.data.columns if column != self.timestamp_col]

    @property
    def symbols(self) -> list[str]:
        return self.symbol_cols

    @classmethod
    def from_long(
        cls,
        df: pl.DataFrame | pl.LazyFrame,
        timestamp_col: str = "timestamp_agg",
        symbol_col: str = "ticker",
        value_col: str = "value",
        aggregate_function: str = "first",
        fill_null: float | None = None,
    ) -> "WideMatrix":
        if isinstance(df, pl.LazyFrame):
            df = df.collect()
        wide = (
            df.select([timestamp_col, symbol_col, value_col])
            .pivot(
                index=timestamp_col,
                on=symbol_col,
                values=value_col,
                aggregate_function=aggregate_function,
            )
            .sort(timestamp_col)
        )
        if fill_null is not None:
            wide = wide.fill_null(fill_null)
        return cls(wide, timestamp_col=timestamp_col, value_name=value_col)

    def select_symbols(self, symbols: list[str]) -> "WideMatrix":
        columns = [self.timestamp_col] + [symbol for symbol in symbols if symbol in self.symbol_cols]
        return WideMatrix(self.data.select(columns), self.timestamp_col, self.value_name)

    def align_with(self, other: "WideMatrix") -> tuple["WideMatrix", "WideMatrix"]:
        symbols = sorted(set(self.symbol_cols) & set(other.symbol_cols))
        left = self.select_symbols(symbols)
        right = other.select_symbols(symbols)
        timestamp_dtype = left.data.schema[self.timestamp_col]
        right_data = right.data.with_columns(pl.col(other.timestamp_col).cast(timestamp_dtype))
        if other.timestamp_col != self.timestamp_col:
            right_data = right_data.rename({other.timestamp_col: self.timestamp_col})
        common = (
            left.data.select(self.timestamp_col)
            .join(right_data.select(self.timestamp_col), on=self.timestamp_col, how="inner")
            .unique()
        )
        return (
            WideMatrix(left.data.join(common, on=self.timestamp_col, how="inner"), self.timestamp_col, self.value_name),
            WideMatrix(right_data.join(common, on=self.timestamp_col, how="inner"), self.timestamp_col, other.value_name),
        )


class Backtester:
    def __init__(self, config: BacktestConfig | None = None) -> None:
        self.config = config or BacktestConfig()

    def run(self, strategy_wide: WideMatrix, return_wide: WideMatrix) -> tuple[pl.DataFrame, BacktestMetrics]:
        strategy_aligned, return_aligned = strategy_wide.align_with(return_wide)
        timestamp_col = strategy_aligned.timestamp_col
        symbols = strategy_aligned.symbols
        if not symbols or strategy_aligned.data.height < 2:
            raise ValueError("not enough aligned strategy/return data for backtest")

        timestamps = strategy_aligned.data[timestamp_col].to_list()
        weights = strategy_aligned.data.select(symbols).fill_null(0.0).to_numpy()
        returns = return_aligned.data.select(symbols).fill_null(0.0).to_numpy()

        gross = np.abs(weights).sum(axis=1)
        scale = np.where(gross > self.config.max_gross, self.config.max_gross / np.maximum(gross, EPS), 1.0)
        weights = weights * scale[:, None]
        w_start = np.vstack([np.zeros((1, weights.shape[1])), weights[:-1]])
        port_ret = (w_start * returns).sum(axis=1)
        w_end = w_start * (returns + 1.0)
        w_end = w_end * (1.0 / np.maximum(port_ret + 1.0, EPS))[:, None]
        trades = weights - w_end
        turnover_2sided = np.abs(trades).sum(axis=1)
        if turnover_2sided.size:
            turnover_2sided[0] = 0.0
        gross_leverage = np.abs(w_start).sum(axis=1)
        equity_curve = np.cumsum(port_ret) + 1.0

        results = pl.DataFrame(
            {
                timestamp_col: timestamps,
                "gross_returns": port_ret,
                "net_ret": port_ret,
                "turnover": turnover_2sided / 2.0,
                "cost": np.zeros_like(port_ret),
                "funding_pnl": np.zeros_like(port_ret),
                "gross_leverage": gross_leverage,
                "equity_curve": equity_curve,
            }
        )
        return results, calculate_metrics(results)


@dataclass
class SignalMetrics:
    mean_ic: float
    ic_std: float
    ic_positive_pct: float
    ic_tstat_dk: float
    signal_return_sharpe: float
    autocorrelation: float
    long_short_spread: float
    monotonicity: float
    n_timestamps: int
    n_symbols_mean: float
    coverage_pct: float
    ic_decay: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


def calculate_metrics(results: pl.DataFrame, ann_factor: float = 8760.0) -> BacktestMetrics:
    df = results.select(
        [
            pl.col("net_ret").cast(pl.Float64),
            pl.col("gross_returns").cast(pl.Float64),
            pl.col("turnover").cast(pl.Float64),
            pl.col("cost").cast(pl.Float64),
            pl.col("gross_leverage").cast(pl.Float64),
            pl.col("equity_curve").cast(pl.Float64),
        ]
    ).drop_nulls()
    if df.height < 2:
        raise ValueError("not enough backtest rows")
    net = df["net_ret"].to_numpy()
    gross = df["gross_returns"].to_numpy()
    turnover = df["turnover"].to_numpy()
    costs = df["cost"].to_numpy()
    leverage = df["gross_leverage"].to_numpy()
    equity = df["equity_curve"].to_numpy()

    def sharpe(values: np.ndarray) -> float:
        std = float(np.std(values, ddof=1))
        return float(np.mean(values) / std * math.sqrt(ann_factor)) if std > 0 else 0.0

    peak = np.maximum.accumulate(equity)
    drawdowns = equity / np.maximum(peak, EPS) - 1.0
    years = len(net) / ann_factor
    final_equity = float(equity[-1])
    cagr = final_equity ** (1.0 / years) - 1.0 if final_equity > 0 and years > 0 else 0.0
    return BacktestMetrics(
        n_periods=int(len(net)),
        total_return=final_equity - 1.0,
        cagr=float(cagr),
        ann_vol_net=float(np.std(net, ddof=1) * math.sqrt(ann_factor)),
        sharpe_net=sharpe(net),
        sharpe_gross=sharpe(gross),
        max_drawdown=float(np.min(drawdowns)),
        turnover_ann=float(np.mean(turnover) * ann_factor),
        turnover_cost_bps_ann=float(np.mean(costs) * ann_factor * 10000.0),
        gross_leverage_mean=float(np.mean(leverage)),
    )


def source_policy_violations(code_path: Path) -> list[str]:
    try:
        source = code_path.read_text(errors="replace")
    except Exception as exc:
        return [f"could not read submitted code for policy scan: {exc}"]
    checks = [
        (re.compile(r"['\"][^'\"]*/tests/[^'\"]*['\"]"), "direct /tests path access is not allowed"),
        (re.compile(r"['\"][^'\"]*test\.parquet[^'\"]*['\"]"), "direct test.parquet access is not allowed"),
        (re.compile(r"['\"][^'\"]*val\.parquet[^'\"]*['\"]"), "direct val.parquet access is not allowed"),
    ]
    return [message for pattern, message in checks if pattern.search(source)]


def return_columns_for(df: pl.DataFrame) -> list[str]:
    return list(
        dict.fromkeys(
            column
            for column in [RETURN_COLUMN_NAME, RESIDUAL_RETURN_COLUMN_NAME, "return"]
            if column in df.columns
        )
    )


def sort_columns_for(df: pl.DataFrame) -> list[str]:
    return [column for column in ["ticker", "timestamp_agg"] if column in df.columns]


def with_safe_return_lags(df: pl.DataFrame, return_columns: list[str]) -> pl.DataFrame:
    lag_exprs: list[pl.Expr] = []
    for column in return_columns:
        for lag in SAFE_RETURN_LAGS:
            lag_exprs.append(pl.col(column).shift(lag).over("ticker").alias(f"{column}_lag{lag}"))
    return df.with_columns(lag_exprs) if lag_exprs else df


def sanitize_prediction_returns(df: pl.DataFrame, return_columns: list[str]) -> pl.DataFrame:
    exprs: list[pl.Expr] = []
    for column in return_columns:
        exprs.append(
            pl.when(pl.col("is_prediction_period"))
            .then(pl.lit(None, dtype=df.schema[column]))
            .otherwise(pl.col(column))
            .alias(column)
        )
    return df.with_columns(exprs) if exprs else df


def final_eval_feature_frame(train_df: pl.DataFrame | None, hidden_df: pl.DataFrame) -> pl.DataFrame:
    frames: list[pl.DataFrame] = []
    if train_df is not None:
        frames.append(train_df.with_columns(pl.lit(False).alias("is_prediction_period")))
    frames.append(hidden_df.with_columns(pl.lit(True).alias("is_prediction_period")))
    feature_df = pl.concat(frames, how="diagonal_relaxed")
    sort_cols = sort_columns_for(feature_df)
    if sort_cols:
        feature_df = feature_df.sort(sort_cols)
    returns = return_columns_for(feature_df)
    feature_df = with_safe_return_lags(feature_df, returns)
    return sanitize_prediction_returns(feature_df, returns)


def filter_signal_period(signal_df: pl.DataFrame, feature_df: pl.DataFrame, *, prediction: bool) -> pl.DataFrame:
    keys = ["timestamp_agg", "ticker"]
    if any(key not in signal_df.columns for key in keys) or "signal" not in signal_df.columns:
        raise RuntimeError("build_signal must return timestamp_agg, ticker, and signal columns")
    periods = feature_df.select([*keys, "is_prediction_period"]).unique()
    signal_df = align_timestamp_dtype(signal_df, timestamp_dtype(feature_df))
    periods = align_timestamp_dtype(periods, timestamp_dtype(signal_df))
    return (
        signal_df.join(periods, on=keys, how="inner")
        .filter(pl.col("is_prediction_period") == prediction)
        .select([*keys, "signal"])
    )


def preprocess_signal(signal_df: pl.DataFrame) -> pl.DataFrame:
    return signal_df.with_columns(
        (
            (pl.col("signal") - pl.col("signal").mean().over("timestamp_agg"))
            / pl.col("signal").std().over("timestamp_agg").clip(lower_bound=EPS)
        )
        .clip(-4.0, 4.0)
        .fill_nan(0.0)
        .fill_null(0.0)
        .alias("signal")
    )


def build_signal_from_code(code_path: Path, feature_df: pl.DataFrame, work_dir: Path, module_name: str) -> pl.DataFrame:
    feature_path = work_dir / "feature.parquet"
    feature_path.parent.mkdir(parents=True, exist_ok=True)
    feature_df.write_parquet(feature_path)
    spec = importlib.util.spec_from_file_location(module_name, code_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load code module: {code_path}")
    module = importlib.util.module_from_spec(spec)
    old_path = list(sys.path)
    sys.path.insert(0, str(code_path.parent))
    try:
        spec.loader.exec_module(module)
        if not hasattr(module, "build_signal"):
            raise RuntimeError("submitted code does not define build_signal(data_path)")
        result = module.build_signal(str(feature_path))
    finally:
        sys.path[:] = old_path
    if not isinstance(result, pl.DataFrame):
        try:
            result = pl.DataFrame(result)
        except Exception as exc:
            raise RuntimeError("build_signal must return a Polars-compatible DataFrame") from exc
    missing = sorted({"timestamp_agg", "ticker", "signal"} - set(result.columns))
    if missing:
        raise RuntimeError("invalid signal schema; missing columns: " + ", ".join(missing))
    return result


def run_backtest_with_public_engine(signal_df: pl.DataFrame, test_df: pl.DataFrame) -> tuple[pl.DataFrame | None, dict[str, Any], list[Any], list[str]]:
    select_cols = ["timestamp_agg", "ticker", RETURN_COLUMN_NAME]
    joined = signal_df.join(test_df.select(select_cols), on=["timestamp_agg", "ticker"], how="inner").filter(
        pl.col("signal").is_not_null()
        & pl.col("signal").is_not_nan()
        & pl.col("signal").is_finite()
        & pl.col(RETURN_COLUMN_NAME).is_not_null()
        & pl.col(RETURN_COLUMN_NAME).is_finite()
    )
    if joined.is_empty():
        return None, {}, [], []
    joined = joined.with_columns(pl.col("signal").fill_nan(0.0).fill_null(0.0).alias("weight"))
    weights = WideMatrix.from_long(joined, value_col="weight", fill_null=0.0)
    returns = WideMatrix.from_long(joined, value_col=RETURN_COLUMN_NAME, fill_null=0.0)
    results, metrics = Backtester(BacktestConfig()).run(weights, returns)
    return results, metrics.to_dict(), joined["timestamp_agg"].unique().sort().to_list(), sorted(joined["ticker"].unique().to_list())


def compute_daily_sharpe(backtest_results: pl.DataFrame | None) -> float:
    if backtest_results is None or backtest_results.is_empty() or "net_ret" not in backtest_results.columns:
        return 0.0
    daily = (
        backtest_results.with_columns(pl.col("timestamp_agg").dt.date().alias("date"))
        .group_by("date")
        .agg(pl.col("net_ret").sum().alias("daily_ret"))
        .sort("date")
    )
    if daily.height < 2:
        return 0.0
    values = daily["daily_ret"].to_numpy()
    std = float(np.std(values, ddof=1))
    return float(np.mean(values) / std * math.sqrt(365)) if std > 0 else 0.0


def daily_pnl(backtest_results: pl.DataFrame | None) -> pl.DataFrame:
    if backtest_results is None or backtest_results.is_empty() or "net_ret" not in backtest_results.columns:
        return pl.DataFrame(schema={"date": pl.Date, "daily_ret": pl.Float64})
    return (
        backtest_results.with_columns(pl.col("timestamp_agg").dt.date().alias("date"))
        .group_by("date")
        .agg(pl.col("net_ret").sum().alias("daily_ret"))
        .sort("date")
    )


def compute_ic_per_timestamp(signal_df: pl.DataFrame, test_df: pl.DataFrame) -> pl.DataFrame:
    test_next = test_df.sort(["ticker", "timestamp_agg"]).with_columns(
        pl.col(RETURN_COLUMN_NAME).shift(-1).over("ticker").alias("return_next")
    )
    joined = signal_df.join(
        test_next.select(["timestamp_agg", "ticker", "return_next"]),
        on=["timestamp_agg", "ticker"],
        how="inner",
    ).filter(
        pl.col("signal").is_not_null()
        & pl.col("signal").is_not_nan()
        & pl.col("signal").is_finite()
        & pl.col("return_next").is_not_null()
        & pl.col("return_next").is_finite()
    )
    if joined.is_empty():
        return pl.DataFrame({"timestamp_agg": [], "ic": []})
    return (
        joined.group_by("timestamp_agg")
        .agg((pl.col("signal") * pl.col("return_next")).mean().alias("ic"))
        .filter(pl.col("ic").is_not_null() & pl.col("ic").is_finite())
        .sort("timestamp_agg")
    )


def compute_return_correlation(signal_df: pl.DataFrame, test_df: pl.DataFrame) -> float:
    joined = signal_df.join(
        test_df.select(["timestamp_agg", "ticker", RETURN_COLUMN_NAME]),
        on=["timestamp_agg", "ticker"],
        how="inner",
    ).filter(
        pl.col("signal").is_not_null()
        & pl.col("signal").is_finite()
        & pl.col(RETURN_COLUMN_NAME).is_not_null()
        & pl.col(RETURN_COLUMN_NAME).is_finite()
    )
    if joined.height < 10:
        return 0.0
    corr = np.corrcoef(joined["signal"].to_numpy(), joined[RETURN_COLUMN_NAME].to_numpy())[0, 1]
    return float(corr) if np.isfinite(corr) else 0.0


def compute_dk_tstat(signal_df: pl.DataFrame, test_df: pl.DataFrame) -> float:
    joined = signal_df.join(
        test_df.select(["timestamp_agg", "ticker", RETURN_COLUMN_NAME]),
        on=["timestamp_agg", "ticker"],
        how="inner",
    ).filter(
        pl.col("signal").is_not_null()
        & pl.col("signal").is_finite()
        & pl.col(RETURN_COLUMN_NAME).is_not_null()
        & pl.col(RETURN_COLUMN_NAME).is_finite()
    )
    if joined.height < 20:
        return 0.0
    if OLS is not None and add_constant is not None:
        try:
            time_codes = joined["timestamp_agg"].rank("dense").cast(pl.Int64).to_numpy()
            x = add_constant(joined["signal"].to_numpy())
            y = joined[RETURN_COLUMN_NAME].to_numpy()
            n_times = max(1, int(np.unique(time_codes).size))
            model = OLS(y, x).fit(cov_type="hac-groupsum", cov_kwds={"time": time_codes, "maxlags": int(np.ceil(n_times ** (1 / 3)))})
            return float(model.tvalues[1])
        except Exception:
            pass
    ic = compute_ic_per_timestamp(signal_df, test_df)
    if ic.height < 2:
        return 0.0
    values = ic["ic"].to_numpy()
    std = float(np.std(values, ddof=1))
    return float(np.mean(values) / (std / math.sqrt(len(values)))) if std > 0 else 0.0


def compute_autocorrelation(signal_df: pl.DataFrame) -> float:
    lagged = signal_df.sort(["ticker", "timestamp_agg"]).with_columns(
        pl.col("signal").shift(1).over("ticker").alias("signal_lag")
    ).drop_nulls()
    if lagged.is_empty():
        return 0.0
    corr = lagged.select(pl.corr("signal", "signal_lag")).item()
    return float(corr) if corr is not None and math.isfinite(float(corr)) else 0.0


def compute_quantile_metrics(signal_df: pl.DataFrame, test_df: pl.DataFrame) -> dict[str, float]:
    test_next = test_df.sort(["ticker", "timestamp_agg"]).with_columns(
        pl.col(RETURN_COLUMN_NAME).shift(-1).over("ticker").alias("return_next")
    )
    joined = signal_df.join(
        test_next.select(["timestamp_agg", "ticker", "return_next"]),
        on=["timestamp_agg", "ticker"],
        how="inner",
    ).filter(pl.col("signal").is_finite() & pl.col("return_next").is_finite())
    if joined.height < 20:
        return {"long_short_spread": 0.0, "monotonicity": 0.0}
    ranked = joined.with_columns(
        pl.col("signal").rank().over("timestamp_agg").alias("_rank"),
        pl.len().over("timestamp_agg").alias("_n"),
    ).with_columns(((pl.col("_rank") - 1) / pl.col("_n") * 5).floor().clip(0, 4).alias("_q"))
    qret = ranked.group_by("_q").agg(pl.col("return_next").mean().alias("ret")).sort("_q")
    if qret.height < 2:
        return {"long_short_spread": 0.0, "monotonicity": 0.0}
    returns = qret["ret"].to_numpy()
    spread = float(returns[-1] - returns[0])
    monotonicity = float(stats.spearmanr(np.arange(len(returns)), returns).correlation or 0.0)
    return {"long_short_spread": spread, "monotonicity": monotonicity}


def compute_ic_decay(signal_df: pl.DataFrame, test_df: pl.DataFrame, horizons: tuple[int, ...] = (1, 2, 3, 6, 12, 24)) -> dict[str, float]:
    results: dict[str, float] = {}
    for horizon in horizons:
        shifted = test_df.sort(["ticker", "timestamp_agg"]).with_columns(
            pl.col(RETURN_COLUMN_NAME).shift(-horizon).over("ticker").alias("return_next")
        )
        joined = signal_df.join(
            shifted.select(["timestamp_agg", "ticker", "return_next"]),
            on=["timestamp_agg", "ticker"],
            how="inner",
        ).filter(pl.col("signal").is_finite() & pl.col("return_next").is_finite())
        if joined.is_empty():
            results[f"T+{horizon}"] = 0.0
            continue
        by_time = joined.group_by("timestamp_agg").agg((pl.col("signal") * pl.col("return_next")).mean().alias("ic"))
        results[f"T+{horizon}"] = float(by_time["ic"].mean() or 0.0)
    return results


def combined_score(metrics: SignalMetrics) -> tuple[float, bool]:
    tstat = metrics.ic_tstat_dk
    sharpe = metrics.signal_return_sharpe
    autocorr = metrics.autocorrelation
    if tstat >= 4.0:
        tstat_score = 1.0
    elif tstat >= 3.0:
        tstat_score = 0.8
    elif tstat >= 2.0:
        tstat_score = 0.6
    elif tstat >= 1.0:
        tstat_score = 0.2
    else:
        tstat_score = 0.0
    if sharpe >= 2.0:
        sharpe_score = 1.0
    elif sharpe >= 1.5:
        sharpe_score = 0.8
    elif sharpe >= 1.0:
        sharpe_score = 0.4
    elif sharpe >= 0.7:
        sharpe_score = 0.2
    else:
        sharpe_score = 0.0
    if AUTOCORR_OPTIMAL_LOW <= autocorr <= AUTOCORR_OPTIMAL_HIGH:
        penalty = 0.0
    elif autocorr > AUTOCORR_OPTIMAL_HIGH:
        penalty = AUTOCORR_PENALTY_MAX
    elif autocorr < AUTOCORR_PENALTY_FLOOR:
        penalty = AUTOCORR_PENALTY_MAX
    else:
        ratio = (AUTOCORR_OPTIMAL_LOW - autocorr) / (AUTOCORR_OPTIMAL_LOW - AUTOCORR_PENALTY_FLOOR + EPS)
        penalty = min(max(ratio, 0.0), 1.0) * AUTOCORR_PENALTY_MAX
    score = max(WEIGHT_TSTAT * tstat_score + WEIGHT_SHARPE * sharpe_score - penalty, MIN_SCORE_VALID_OUTPUT)
    passed = tstat >= TSTAT_THRESHOLD and sharpe >= SHARPE_THRESHOLD and score >= 0.5
    return float(score), bool(passed)


def score_code(
    *,
    task_dir: Path,
    code_path: Path,
    output_dir: Path,
    causality_gate: bool = True,
    gate_days: int = 1,
    min_gate_coverage: float = 0.01,
) -> dict[str, Any]:
    train_path = task_dir / "environment" / "data" / "train.parquet"
    hidden_path = task_dir / "tests" / "test.parquet"
    if not train_path.exists():
        raise FileNotFoundError(f"missing train parquet: {train_path}")
    if not hidden_path.exists():
        raise FileNotFoundError(f"missing hidden/OOS parquet: {hidden_path}")
    violations = source_policy_violations(code_path)
    if violations:
        raise RuntimeError("policy violation: " + "; ".join(violations))

    output_dir.mkdir(parents=True, exist_ok=True)
    train_df = pl.read_parquet(train_path)
    hidden_df = pl.read_parquet(hidden_path)
    gate_result = None
    if causality_gate:
        gate_result = run_causality_gate(
            train_df=train_df,
            hidden_df=hidden_df,
            code_path=code_path,
            output_dir=output_dir / "causality_gate",
            gate_days=gate_days,
            min_coverage=min_gate_coverage,
        )
        if gate_result.get("status") != "passed":
            result = {
                "status": "failed",
                "error": "causality gate failed",
                "causality_gate": gate_result,
                "score": 0.0,
                "passed": False,
            }
            write_json(output_dir / "results.json", result)
            return result

    feature_df = final_eval_feature_frame(train_df, hidden_df)
    signal_full = build_signal_from_code(
        code_path,
        feature_df,
        output_dir / "_work",
        f"qr_eval_public_submission_{abs(hash((str(code_path), str(output_dir))))}",
    )
    signal_full = align_timestamp_dtype(signal_full, timestamp_dtype(feature_df))
    signal = filter_signal_period(signal_full, feature_df, prediction=True)
    if signal.is_empty():
        raise RuntimeError("no prediction-period signal rows returned by build_signal")
    signal = preprocess_signal(signal)
    signal = align_timestamp_dtype(signal, timestamp_dtype(hidden_df))
    backtest_results, backtest_metrics, _timestamps, symbols = run_backtest_with_public_engine(signal, hidden_df)
    daily = daily_pnl(backtest_results)
    if daily.is_empty():
        raise RuntimeError("verifier produced no daily PnL rows")
    daily.write_parquet(output_dir / "daily_pnl.parquet")
    signal.write_parquet(output_dir / "signal.parquet")

    ic_df = compute_ic_per_timestamp(signal, hidden_df)
    if ic_df.is_empty():
        raise RuntimeError("no valid IC computations")
    ic_values = ic_df["ic"].to_numpy()
    mean_ic = float(np.mean(ic_values))
    ic_std = float(np.std(ic_values))
    n_timestamps = int(ic_df.height)
    n_symbols = len(symbols)
    quantiles = compute_quantile_metrics(signal, hidden_df)
    metrics = SignalMetrics(
        mean_ic=mean_ic,
        ic_std=ic_std,
        ic_positive_pct=float((ic_values > 0).mean()),
        ic_tstat_dk=compute_dk_tstat(signal, hidden_df),
        signal_return_sharpe=compute_daily_sharpe(backtest_results),
        autocorrelation=compute_autocorrelation(signal),
        long_short_spread=quantiles["long_short_spread"],
        monotonicity=quantiles["monotonicity"],
        n_timestamps=n_timestamps,
        n_symbols_mean=float(n_symbols),
        coverage_pct=float(signal.height / (max(n_timestamps, 1) * max(n_symbols, 1))),
        ic_decay=compute_ic_decay(signal, hidden_df),
    )
    score, passed = combined_score(metrics)
    return_corr = compute_return_correlation(signal, hidden_df)
    cheating_detected = abs(return_corr) > 0.90
    if cheating_detected:
        score = 0.0
        passed = False
    result = {
        "status": "completed",
        "task_name": task_dir.name,
        "code_path": str(code_path),
        "daily_pnl_path": str(output_dir / "daily_pnl.parquet"),
        "signal_path": str(output_dir / "signal.parquet"),
        "causality_gate": gate_result,
        "return_correlation": return_corr,
        "cheating_detected": cheating_detected,
        "backtest_sharpe": metrics.signal_return_sharpe,
        "backtest_max_drawdown": backtest_metrics.get("max_drawdown", 0.0),
        "backtest_turnover_ann": backtest_metrics.get("turnover_ann", 0.0),
        **metrics.to_dict(),
        **backtest_metrics,
        "score": score,
        "passed": passed,
    }
    write_json(output_dir / "results.json", result)
    return result


def run_causality_gate(
    *,
    train_df: pl.DataFrame,
    hidden_df: pl.DataFrame,
    code_path: Path,
    output_dir: Path,
    gate_days: int,
    min_coverage: float,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamps = hidden_df.select("timestamp_agg").unique().sort("timestamp_agg")["timestamp_agg"].to_list()
    if not timestamps:
        return {"status": "failed", "failure_reason": "hidden frame has no timestamps", "gate_days": []}
    if gate_days <= 1:
        selected = [timestamps[min(1, len(timestamps) - 1)]]
    else:
        indices = sorted({min(1, len(timestamps) - 1), len(timestamps) // 2, max(0, len(timestamps) - 2)})[:gate_days]
        selected = [timestamps[index] for index in indices]
    day_results = []
    for idx, gate_timestamp in enumerate(selected, start=1):
        try:
            baseline = prediction_signal_at_gate(train_df, hidden_df, code_path, output_dir / f"gate_{idx:02d}", gate_timestamp, include_future=False, adversarial_future=False)
            expected = hidden_df.filter(pl.col("timestamp_agg") == gate_timestamp).select(["timestamp_agg", "ticker"]).unique().height
            finite = baseline.filter(pl.col("signal").is_finite()).height if not baseline.is_empty() else 0
            min_rows = max(1, int(expected * min_coverage))
            if finite < min_rows:
                day_results.append({"gate_timestamp": str(gate_timestamp), "status": "failed", "failure_reason": "insufficient causal prediction rows", "expected_rows": expected, "finite_rows": finite})
                continue
            future = prediction_signal_at_gate(train_df, hidden_df, code_path, output_dir / f"gate_{idx:02d}", gate_timestamp, include_future=True, adversarial_future=True)
            delta = signal_delta(baseline, future)
            failed = bool(delta["missing_rows"] or (delta["max_abs_delta"] or 0.0) > 1e-8)
            day_results.append({"gate_timestamp": str(gate_timestamp), "status": "failed" if failed else "passed", "failure_reason": "signal changes when future labels/rows are perturbed" if failed else None, "finite_rows": finite, "future_label_delta": delta})
        except Exception as exc:
            day_results.append({"gate_timestamp": str(gate_timestamp), "status": "failed", "failure_reason": f"{type(exc).__name__}: {exc}"})
    failed = [item for item in day_results if item.get("status") != "passed"]
    result = {
        "status": "failed" if failed else "passed",
        "failure_reason": failed[0].get("failure_reason") if failed else None,
        "gate_unit": "bar",
        "gate_days": day_results,
    }
    write_json(output_dir / "causality_gate.json", result)
    return result


def prediction_signal_at_gate(
    train_df: pl.DataFrame,
    hidden_df: pl.DataFrame,
    code_path: Path,
    work_dir: Path,
    gate_timestamp: Any,
    *,
    include_future: bool,
    adversarial_future: bool,
) -> pl.DataFrame:
    dtype = timestamp_dtype(hidden_df)
    gate_lit = timestamp_literal(gate_timestamp, dtype)
    selected_hidden = hidden_df.filter(pl.col("timestamp_agg") <= gate_lit)
    if include_future:
        future = hidden_df.filter(pl.col("timestamp_agg") > gate_lit)
        if not future.is_empty():
            future_times = future.select("timestamp_agg").unique().sort("timestamp_agg")["timestamp_agg"].to_list()
            sentinel = [future_times[0]]
            if future_times[-1] != sentinel[0]:
                sentinel.append(future_times[-1])
            sentinel_mask = pl.any_horizontal(
                [pl.col("timestamp_agg") == timestamp_literal(value, dtype) for value in sentinel]
            )
            selected_hidden = pl.concat([selected_hidden, hidden_df.filter(sentinel_mask)], how="diagonal_relaxed").unique()
    selected_hidden = selected_hidden.with_columns((pl.col("timestamp_agg") == gate_lit).alias("is_prediction_period"))
    feature = pl.concat([train_df.with_columns(pl.lit(False).alias("is_prediction_period")), selected_hidden], how="diagonal_relaxed")
    sort_cols = sort_columns_for(feature)
    if sort_cols:
        feature = feature.sort(sort_cols)
    returns = return_columns_for(feature)
    feature = with_safe_return_lags(feature, returns)
    if returns:
        feature = feature.with_row_index("_row_nr")
        exprs = []
        randomish = ((((pl.col("_row_nr") * 1103515245 + 12345) % 2000000) / 1000000.0) - 1.0) * 0.25
        for column in returns:
            value = pl.when(pl.col("timestamp_agg") == gate_lit).then(pl.lit(None, dtype=feature.schema[column])).otherwise(pl.col(column))
            if adversarial_future:
                value = pl.when(pl.col("timestamp_agg") > gate_lit).then(randomish).otherwise(value)
            exprs.append(value.alias(column))
        feature = feature.with_columns(exprs).drop("_row_nr")
    signal_full = build_signal_from_code(code_path, feature, work_dir, f"qr_eval_gate_{abs(hash((str(code_path), str(work_dir))))}")
    signal_full = align_timestamp_dtype(signal_full, timestamp_dtype(feature))
    signal = filter_signal_period(signal_full, feature, prediction=True)
    return preprocess_signal(signal).select(["timestamp_agg", "ticker", "signal"]) if not signal.is_empty() else signal


def signal_delta(left: pl.DataFrame, right: pl.DataFrame) -> dict[str, Any]:
    if left.is_empty() or right.is_empty():
        return {"shared_rows": 0, "max_abs_delta": None, "changed_rows": None, "missing_rows": max(left.height, right.height)}
    right = align_timestamp_dtype(right, timestamp_dtype(left))
    joined = left.rename({"signal": "signal_left"}).join(
        right.rename({"signal": "signal_right"}),
        on=["timestamp_agg", "ticker"],
        how="inner",
    ).with_columns((pl.col("signal_left") - pl.col("signal_right")).abs().alias("abs_delta"))
    if joined.is_empty():
        return {"shared_rows": 0, "max_abs_delta": None, "changed_rows": None, "missing_rows": max(left.height, right.height)}
    return {
        "shared_rows": joined.height,
        "max_abs_delta": float(joined["abs_delta"].max()),
        "changed_rows": int(joined.filter(pl.col("abs_delta") > 1e-8).height),
        "missing_rows": max(left.height, right.height) - joined.height,
    }


def timestamp_dtype(df: pl.DataFrame) -> pl.DataType | None:
    return df.schema.get("timestamp_agg") if "timestamp_agg" in df.columns else None


def timestamp_literal(value: Any, dtype: pl.DataType | None) -> pl.Expr:
    expr = pl.lit(value)
    return expr.cast(dtype) if dtype is not None else expr


def align_timestamp_dtype(df: pl.DataFrame, dtype: pl.DataType | None) -> pl.DataFrame:
    if dtype is None or "timestamp_agg" not in df.columns or df.schema.get("timestamp_agg") == dtype:
        return df
    return df.with_columns(pl.col("timestamp_agg").cast(dtype))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")


def smoke_score_task(task_dir: Path, candidate: Path | None = None) -> dict[str, Any]:
    candidate = candidate or task_dir / "solution" / "solve.py"
    with tempfile.TemporaryDirectory(prefix="qr-eval-smoke-") as tmp:
        return score_code(task_dir=task_dir, code_path=candidate, output_dir=Path(tmp), causality_gate=True)
