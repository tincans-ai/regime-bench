# Top Trader Positioning Signal

## Objective

Build a signal using Binance top trader positioning data.

Top traders are identified by Binance as profitable traders. You have raw positioning
metrics for top traders and retail "takers":
- Top trader long/short ratios (sum and count aggregations)
- Taker (retail) long/short volume ratios
- Open interest values

Your goal: Discover which features and transformations are predictive of future returns.
Consider deriving new features, combining raw metrics, or applying ML techniques.


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
| `sum_toptrader_long_short_ratio` | Sum of top trader long/short ratios |
| `count_toptrader_long_short_ratio` | Count-based top trader ratio |
| `sum_taker_long_short_vol_ratio` | Taker (retail) long/short volume ratio |
| `count_long_short_ratio` | Count-based long/short ratio |
| `sum_open_interest` | Total open interest (contracts) |
| `sum_open_interest_value` | Total open interest (notional USD) |


**Important**: You may use `sum_toptrader_long_short_ratio`, `count_toptrader_long_short_ratio`, `sum_taker_long_short_vol_ratio`, `count_long_short_ratio`, `sum_open_interest`, `sum_open_interest_value`, `volume`, `close`, and `timestamp_agg` to construct your signal.
**Important**: You may use lagging versions of `return` or `residual_return` but NOT the current period OR future period returns - that is cheating. On rows marked `is_prediction_period = True`, current-period return labels are withheld/null. Use provided safe lag columns such as `return_lag1` and `residual_return_lag1` for one-bar return features on prediction rows; do not depend on shifting null prediction-period labels.

## Requirements

Implement the `build_signal` function in `starter_code/code.py`:

```python
def build_signal(data_path: str) -> pl.DataFrame:
    """
    Top Trader Positioning Signal

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


-     Raw data handling and feature hygiene:
    - Hourly positioning data may contain NaNs, discontinuities, or exchange-specific artifacts; explicitly handle missing values and verify alignment with price data.
    - Forward-fill only where economically justified (e.g., open interest), and avoid filling directional ratios across long gaps.
    - Clip extreme ratio values to reduce sensitivity to reporting glitches or thin markets.
    

-     Relative positioning and divergence features:
    - Compare top trader long/short ratios to taker (retail) long/short ratios; divergences may indicate smart-money vs crowd positioning.
    - Construct spread features such as (top trader ratio – taker ratio) or normalized differences.
    - Test whether extreme divergence predicts reversals (crowded trades) or continuation (informed positioning).
    

-     Positioning level vs flow dynamics:
    - Distinguish between positioning levels (absolute long/short ratios) and changes in positioning (flows).
    - Compute rate of change, acceleration, and persistence of positioning shifts over multiple horizons.
    - Hypothesis: rapid changes in top trader positioning may be more informative than static levels.
    

-     Open interest and funding interactions:
    - Combine positioning metrics with open interest changes to separate price moves driven by new risk-taking vs short covering / long liquidation.
    - Test joint conditions (e.g., top traders increasing net longs while open interest rises).
    - Normalize OI changes by recent volatility or average OI to ensure comparability across assets.
    - also look at funding as a sign of market consensus crowding
    

-     Cross-sectional normalization and ranking:
    - Apply cross-sectional z-scores or ranks to positioning features to make signals comparable across tickers.
    - Hypothesis: relative positioning within the cross-section is more informative than absolute levels.
    - Test whether tails of the cross-sectional distribution carry most of the predictive power.
    

-     Time-series normalization and regime sensitivity:
    - Normalize positioning metrics by their own historical distributions (rolling z-scores, percentile ranks).
    - Examine whether predictive power depends on volatility regime, trend regime, or funding conditions.
    - Positioning signals may behave differently in trending vs mean-reverting markets.
    

-     Asymmetry and nonlinear effects:
    - Test whether long-side and short-side positioning have asymmetric predictive power.
    - Explore nonlinear transformations (e.g., squared divergence, thresholds) to capture crowdedness effects.
    - Hypothesis: extremely one-sided positioning may predict reversals, while moderate positioning predicts continuation.
    

-     Market-wide vs asset-specific signals:
    - Aggregate positioning across assets to build market-wide sentiment indicators.
    - Condition individual asset signals on the state of aggregate top trader or retail positioning.
    - Hypothesis: individual signals are stronger when aligned with or contrarian to market-wide positioning extremes.
    

-     ML-assisted feature discovery:
    - Use scikit-learn models (e.g., rolling regressions, tree-based models) to discover nonlinear interactions between positioning, OI, and returns.
    - Treat ML outputs as feature generators or signal components rather than final black-box predictors.
    - Regularize aggressively to avoid overfitting noisy high-frequency positioning data.
    

-     Signal composition and ensembling:
    - Combine multiple positioning-derived features (divergence, flow, OI interaction) into an ensemble signal.
    - Evaluate whether ensembles improve stability and reduce drawdowns relative to single-feature signals.
    - Allow different components to dominate under different market conditions.
    

-     Diagnostics and falsification tests:
    - Verify that apparent predictability is not driven by lookahead bias, data alignment errors, or overlapping returns.
    - Check whether performance concentrates in a small subset of assets or time periods.
    - Stress-test sensitivity to feature definitions, normalization windows, and clipping thresholds.
    

-     Turnover and horizon alignment considerations:
    - Monitor turnover explicitly; positioning signals can change rapidly and incur high trading costs.
    - If fitting to 1-period-ahead returns is too noisy, calculate longer-horizon forward returns and use those as targets.
    - When using a forward horizon of K, ensure all rolling features at time T only use data up to T−K to avoid leakage.
    



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
