# ML Signal Task

## Objective

Build a signal using **machine learning** that predicts **forward residual returns**.
Your model should learn patterns from historical data that generalize to predict future returns.

## Available Libraries

You have access to these ML libraries (NO neural networks):
- `sklearn` - General ML algorithms (RandomForest, GradientBoosting, etc.)
- `xgboost` - XGBoost gradient boosting
- `polars` - Data manipulation
- `numpy` - Numerical computing

## Key Challenge: Point-in-Time Compliance

**Critical**: Your model must be trained in a point-in-time compliant manner:
- At time T, you can ONLY use data from times < T to train the model
- You CANNOT train on all data at once and then predict - that's lookahead bias
- Use **rolling window** or **expanding window** training

Example approaches:
1. **Walk-forward**: Train on [T-N, T), predict at T, then slide forward
2. **Expanding window**: Train on [0, T), predict at T
3. **Fixed retrain**: Retrain model every K periods on past data

## Modeling Focus

Prioritize:
- **XGBoost** as the primary nonlinear model for interactions/thresholds
- **Regularized linear regression** (Ridge / Lasso / ElasticNet) as a strong baseline and for stability
- (Optional) LightGBM if available in the environment; otherwise omit


## Data Available

You have access to data in `/app/data/train.parquet` with the following columns:

| Column | Description |
|--------|-------------|
| `timestamp_agg` | Timestamp (hourly intervals) |
| `ticker` | Symbol identifier (e.g., "BTCUSDT") |
| `close` | Closing price |
| `volume` | Trading volume in quote currency |
| `return` | CURRENT period return |
| `return_lag1` | Previous-period return, precomputed safely for prediction rows |
| `funding_rate` | Perpetual futures funding rate |
| `residual_return` | Market-adjusted return for the CURRENT period |
| `residual_return_lag1` | Previous-period market-adjusted return, precomputed safely for prediction rows |
| `open`, `high`, `low` | OHLC prices |
| `turnover` | Total notional traded (price × quantity) |
| `realized_vol` | Standard deviation of prices within bar |
| `vwap` | Volume-weighted average price |
| `ofi_qty` | Order Flow Imbalance (taker buys - taker sells) |
| `vol_maker_up`, `vol_maker_down` | Maker volume on upticks/downticks |
| `vol_taker_up`, `vol_taker_down` | Taker volume on upticks/downticks |
| `notional_bid_1` to `notional_bid_5` | Bid-side order book liquidity (levels 1-5) |
| `notional_ask_1` to `notional_ask_5` | Ask-side order book liquidity (levels 1-5) |
| `total_liquidity` | Sum of all bid/ask notional |
| `imbalance_L1` | Order book imbalance: (bid_1 - ask_1) / (bid_1 + ask_1) |
| `amihud` | Amihud illiquidity: (high-low)/open / turnover |
| `bid_replenishment`, `ask_replenishment` | Liquidity replenishment metrics |
| `market_return` | Equal-weighted BTC + ETH returns |
| `vol_1d`, `vol_7d` | Annualized rolling volatility (1d, 7d windows) |
| `vol_1d_chg`, `vol_7d_chg` | Percentage change in volatility |
| `log_volume_1d`, `log_volume_7d` | Log rolling volume |
| `log_volume_1d_chg`, `log_volume_7d_chg` | Change in log volume |
| `mom_1h`, `mom_12h`, `mom_1d`, `mom_7d` | Rolling sum of returns (momentum) |
| `volume_taker_ratio` | taker_up / (taker_up + taker_down) |
| `volume_maker_ratio` | maker_up / (maker_up + maker_down) |
| `predicted_return` | Fitted factor exposure (from residualization) |

**Important**: You may use lagging versions of `return` or `residual_return` but NOT the current period OR future period returns - that is cheating. On rows marked `is_prediction_period = True`, current-period return labels are withheld/null. Use provided safe lag columns such as `return_lag1` and `residual_return_lag1` for one-bar return features on prediction rows; do not depend on shifting null prediction-period labels.

## Requirements

Implement the `build_signal` function in `starter_code/code.py`:

```python
def build_signal(data_path: str) -> pl.DataFrame:
    """
    ML Signal Task

    Args:
        data_path: Path to parquet file with columns: timestamp_agg, ticker,
                   close, volume, return, return_lag1, funding_rate,
                   residual_return, residual_return_lag1, is_prediction_period

    Returns:
        DataFrame with columns: timestamp_agg, ticker, signal
        - signal should be a float representing your prediction
        - higher signal = expect higher returns
    """
```

Your function must return a DataFrame with exactly these columns:
- `timestamp_agg`: Timestamp matching input data
- `ticker`: Symbol identifier
- `signal`: Your predicted signal value (float). **Higher signal = higher expected return.** Automatically cross-sectionally z-scored and clipped at ±4σ before evaluation.

**Document your signal**: Add a comment or docstring with a LaTeX formula describing your signal computation. For example:
```python
# Signal formula: $s_{i,t} = -\text{zscore}(f_{i,t})$ where $f$ is funding rate
```
Also include a brief justification explaining how the signal maps to the task's stated hypothesis and why each major component belongs.

## Point-In-Time Rules and Examples

Your submitted `build_signal(data_path)` must be point-in-time for every row it predicts. Treat current same-row `return`, `residual_return`, and task evaluation return columns as labels, not deployable features. You may use return labels for visible training analysis only after shifting/lagging them, or when fitting on rows strictly earlier than the rows being predicted.

Final held-out evaluation may pass a combined train-plus-prediction feature view. In that view, rows to predict are marked with `is_prediction_period = True`; return-like labels on those rows are withheld or null, while safe lag features such as `return_lag1` and `residual_return_lag1` are provided when available. Your code must handle those null labels and still emit signals for every prediction row.

When using one-bar lagged returns, prefer the provided safe lag columns on prediction rows. Do not rely only on expressions such as `pl.col("return").shift(1).over("ticker")` or `pl.col("residual_return").shift(1).over("ticker")`, because those shifts can read withheld/null prediction-period labels during held-out evaluation. A robust pattern is to coalesce the safe lag column first, then fall back to a shifted raw label only for ordinary visible rows.

You may construct forward-return targets only for visible training/check analysis, and only if every fitted model is trained on rows strictly earlier than the rows it predicts. When `is_prediction_period` is present, fit only on rows where it is false or absent, then predict rows where it is true. Do not train a supervised model on labels derived from the same `data_path` that you are predicting for unless the training window is strictly earlier than every predicted row.

Acceptable examples:
- Lagged return features that use provided safe lag columns such as `return_lag1` or `residual_return_lag1`, optionally falling back to `pl.col("residual_return").shift(1).over("ticker")` or `pl.col("return").shift(1).over("ticker")` only for visible non-prediction rows; rolling return statistics shifted by at least one bar; or lagged volatility/volume features.
- Walk-forward or expanding-window supervised fitting where `target = residual_return.shift(-1)` is used only inside a past training window, and predictions are produced only for later timestamps.
- Train/predict splits using `is_prediction_period`: fit or tune on non-prediction rows, then score the prediction rows without reading their current return labels.
- Calendar or seasonality features built from historical observations only, such as same-hour averages shifted by one prior occurrence before being joined back to the predicted row.

Not acceptable examples:
- Using current or future returns directly as the signal, such as `signal = residual_return`, `signal = return`, same-row cross-sectional ranks of `return` or `residual_return`, `residual_return.shift(-1)`, `return.shift(-1)`, or `.lead()` on return-like columns.
- Computing `target = residual_return.shift(-1)` on the full `data_path`, fitting a model on all rows with that target, and then predicting those same rows. That is transductive label leakage, even if the target is used only for model fitting.
- Reading or probing hidden files or paths such as `/tests/test.parquet`, `/tests`, hidden-OOS debug artifacts, verifier outputs, or any file other than the `data_path` argument and normal starter-code outputs.

## Evaluation

Your signal will be evaluated on a **held-out test set** (not visible to you) using:

1. **Information Coefficient (IC)**: Cross-sectional correlation between signal and forward returns
   - Target: Mean IC > 0.001


2. **Statistical Significance**: Driscoll-Kraay adjusted t-statistic
   - Target: > 2.0
   
3. **Anti-cheating Check**: Correlation between signal and actual returns
   - Your signal must NOT have > 90% correlation with returns (that would mean you're using future information)

## Iterating on Your Solution

**IMPORTANT: Use the provided checker - do NOT write your own evaluation code.**

Run `/check` to evaluate your signal on training data.

This gives you:
- Key metrics (IC, t-stat, Sharpe) with pass/fail thresholds
- Calibration table showing signal quintiles vs returns
- IC decay across horizons
- Backtest performance summary

Run the checker after each change to your signal. Use the full `/check` budget available in the task configuration before marking the task complete, unless the checker itself is unavailable because of an infrastructure failure. Track the best-performing version across attempts, and restore that best version before final submission. Do not write custom evaluation scripts - the checker handles everything.

Stay focused on the task's stated hypothesis and allowed feature family. Do not drift into generic factor mining unless the change directly tests or refines that hypothesis.

Note: Final evaluation uses held-out test data - do not overfit to the in-sample data.


## Ideas to Explore


- Your target will be a forward shifted `residual_return` (e.g., `y = residual_return.shift(-H)` per ticker), with multiple horizon sweeps H (1h, 4h, 12h, 24h, 3d, 7d).

- Also test stabilized targets: (a) clipped/winsorized forward residual_return, (b) cross-sectional rank/quantile of forward residual_return, (c) sign/3-class bins (up/down/flat).

- Also test normalizing target by vol

- Use rolling or expanding window training (walk-forward validation). Retrain on past-only data, predict the next block, roll forward. Add an embargo gap ~H to avoid leakage from overlapping labels.

- Fixed retrain variant: retrain every K periods (e.g., daily/weekly) on a trailing window of N days; between retrains, score out-of-sample only.

- All feature engineering must be computed using information available strictly before prediction time (lagged, rolling/EMA with proper shift). DO NOT use current/future returns as features.

- Use all available columns. Core columns: `timestamp_agg`, `ticker`, `close`, `volume`, `return`, `funding_rate`, `residual_return`.

- Expanded columns (available in dataset): OHLC (`open`,`high`,`low`), `turnover`, `realized_vol`, `vwap`, order flow (`ofi_qty`, maker/taker up/down volumes), order book liquidity (`notional_bid_1..5`, `notional_ask_1..5`, `total_liquidity`, `imbalance_L1`), `amihud`, `bid_replenishment`, `ask_replenishment`, `market_return`, `vol_1d`, `vol_7d`, `vol_1d_chg`, `vol_7d_chg`, `log_volume_1d`, `log_volume_7d`, `log_volume_1d_chg`, `log_volume_7d_chg`, momentum (`mom_1h`, `mom_12h`, `mom_1d`, `mom_7d`), `volume_taker_ratio`, `volume_maker_ratio`, and `predicted_return`.

- **Lagged returns**: create `return_lag1`, `return_lag4`, `return_lag24` (1h, 4h, 24h ago) per ticker; likewise `residual_return_lag*` and `funding_rate_lag*`.

- **Rolling statistics**: create `vol_24` (24-hour rolling volatility) using returns (per ticker), plus rolling/EMA versions of realized_vol and residual_return volatility.

- **Funding rate features**: `funding_zscore` as cross-sectional z-score of funding_rate at each timestamp; also test rolling/EMA z-scores per ticker.

- **Volume features**: `vol_ratio` = current volume / rolling mean volume (per ticker); also log(volume) and volume shock indicators vs rolling median.

- **Momentum**: rolling mean/sum of lagged returns (you already have `mom_*`; also build EMA momentum variants).

- **Mean reversion**: distance from rolling mean price: `(close - rolling_mean(close,n))/rolling_std(close,n)`; also distance from EMA(close).

- **Cross-sectional ranks**: rank-transform key features across tickers at each timestamp (rank of residual shock, funding, ofi, liquidity, vol_ratio, etc.).

- Create multiple EMA half-life variants for key quantities (close, return, residual_return, volume, turnover, realized_vol, ofi_qty, imbalance_L1, total_liquidity, amihud, funding_rate). Include EMA spreads (fast - slow) and EMA slope/diff features.

- Run a sweep over normalizations: (a) time-series z-score per ticker, (b) cross-sectional rank per timestamp, (c) robust clipping/winsorization before model fit, (d) scale residual_return features by volatility (`resid / vol_24` or `resid / vol_1d`).

- Primary nonlinear model: **XGBoost regressor** for y=forward residual_return with strong regularization and shallow trees; tune max_depth, min_child_weight, subsample, colsample_bytree, reg_alpha/reg_lambda, learning_rate with early stopping inside walk-forward.

- Baseline model: **regularized linear regression** (Ridge/Lasso/ElasticNet). Standardize features (critical), tune alpha (and l1_ratio) in walk-forward CV; compare coefficient stability across time.

- If LightGBM is available in the runtime, add it as a second GBDT implementation with conservative `num_leaves`, large `min_data_in_leaf`, feature/bagging fractions, and early stopping; otherwise skip.

- Convert predictions into a cross-sectional long/short signal: long top X% predicted forward residual_return, short bottom X%, or continuous sizing via rank(pred) mapped to [-1,1].

- Add a confidence/no-trade band to control turnover: trade only if |pred| exceeds threshold or if rank distance from median exceeds cutoff; use hysteresis (enter threshold > exit threshold).

- Liquidity-aware sizing: cap position by `turnover` or `total_liquidity`, downweight high `amihud` names, and optionally scale by inverse realized_vol to equalize risk.

- Evaluate both ML metrics (IC, rank-IC, MSE on clipped targets) and trading metrics (net Sharpe, drawdown, turnover, cost sensitivity).

- Slice performance by regimes using available features: high/low `realized_vol` or `vol_1d`, high/low `vol_ratio`, thin/deep `total_liquidity`, extreme/normal `funding_zscore`, and strong/weak order flow (`ofi_qty`, imbalance_L1).

- Stress-test failure modes: regime overfit, excessive turnover, microstructure feature instability, and dependence on a single feature family (e.g., funding-only or volume-only).

- DO NOT use current/future returns as features. Ensure all rolling/EMA features are shifted so that feature_t uses only data <= t-1 (or <= t-H depending on label definition).

- Note that turnover may be an issue. If you are finding fitting to 1 period returns is too fast, calculate longer term forward returns and use that to fit any rolling fit signals. Make sure to correctly mask returns though. For example if your forward horizon is K, your rolling window for time T should end data at time T-K.



## Constraints

- Your signal must be **point-in-time** (no look-ahead bias)
- You can only use information available at each timestamp
- Your signal is automatically cross-sectionally z-scored and clipped at ±4σ before being used as portfolio weights in the backtest


## Submission

Run your implementation:

```bash
python starter_code/code.py
```

This will:
1. Load data from `/app/data/train.parquet`
2. Call your `build_signal()` function
3. Validate the output schema
4. Save results to `/app/output/signal.parquet`

The evaluation script will then run automatically on the held-out test set.
