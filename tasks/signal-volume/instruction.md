# Volume Signal Task

## Objective

Build a predictive signal for forward returns using volume-derived information (and its interaction with price and funding). The goal is to identify whether volume contains incremental information about future price movements beyond what is already reflected in recent returns.

The core hypothesis is that volume proxies for the intensity and composition of trading: informed trading, retail attention shocks, liquidation/forced-flow events, and changes in risk appetite. Volume can therefore predict returns through multiple mechanisms:
- Continuation: high volume confirms directional conviction and trend persistence.
- Reversal: abnormal volume reflects exhaustion, forced unwinds, or temporary imbalances that subsequently mean-revert.
- Regime dependence: the same volume pattern may imply different outcomes depending on volatility/trend state and funding conditions.

This task should discover robust transformations of volume (levels, changes, anomalies, and nonlinear thresholds) and evaluate whether their predictive relationship with forward returns is stable out-of-sample and economically meaningful after accounting for turnover. Key failure modes include signals that simply proxy for volatility, rely on rare spikes, or collapse once normalized properly across assets and regimes.


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

**Important**: You may use `close`, `volume`, `return`, `funding_rate`, `residual_return`, and `timestamp_agg` to construct your signal.
**Important**: You may use lagging versions of `return` or `residual_return` but NOT the current period OR future period returns - that is cheating. On rows marked `is_prediction_period = True`, current-period return labels are withheld/null. Use provided safe lag columns such as `return_lag1` and `residual_return_lag1` for one-bar return features on prediction rows; do not depend on shifting null prediction-period labels.

## Requirements

Implement the `build_signal` function in `starter_code/code.py`:

```python
def build_signal(data_path: str) -> pl.DataFrame:
    """
    Volume Signal Task

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


-     Relative volume / abnormal activity:
    - Use relative volume = current volume divided by a rolling baseline (mean/median/EMA) to detect attention or activity shocks.
    - Compare time-series z-scores, percentile ranks, and log-ratio transforms to stabilize heavy-tailed volume.
    - Hypothesis: unusually high relative volume predicts either continuation (confirmation) or reversal (exhaustion), depending on context.
    

-     Volume momentum vs volume mean reversion:
    - Test whether sustained elevated volume over multiple bars predicts trend continuation (volume momentum).
    - Test whether single-bar or short burst spikes predict reversal (exhaustion or liquidation flush).
    - Separate “persistent high volume” from “transient spike” using multi-horizon features (level + change + persistence).
    

-     Volume breakouts and threshold effects:
    - Identify breakout events where volume exceeds a high percentile of its own history.
    - Compare outcomes for moderate breakouts vs extreme blow-off tops (nonlinear response).
    - Add post-spike decay features: how quickly volume normalizes may correlate with continuation strength.
    

-     Price–volume confirmation and divergence:
    - Combine return sign and relative volume: up move on high volume may be more persistent than up move on low volume.
    - Look for divergence cases: price up on falling volume (potential weakening) or price down on rising volume (capitulation).
    - Construct composite features like sign(return) * relative_volume or return / volume to capture “efficiency of price movement.”
    

-     Funding rate interaction (positioning / leverage proxy):
    - Condition volume signals on funding extremes: high volume when funding is very positive may indicate crowded long positioning and potential reversal risk.
    - Conversely, high volume with very negative funding may indicate short crowding / squeeze potential.
    - Use funding as a regime conditioner that flips the interpretation of volume spikes (informed flow vs forced flow).
    

-     Flow vs noise decomposition:
    - Use changes in volume (delta, pct change) and “volume acceleration” to identify sudden participation shifts.
    - Hypothesis: changes in participation predict near-term return autocorrelation more than raw volume levels.
    - Smooth volume features (EMA) and clip extreme values to reduce sensitivity to one-off prints.
    

-     Cross-sectional comparability:
    - Normalize volume features cross-sectionally (z-scores/ranks per timestamp) to compare signals across assets with different baseline liquidity.
    - Hypothesis: “high relative volume compared to peers” captures market-wide attention allocation and may be more predictive than token-level anomalies alone.
    

-     Regime conditioning and state dependence:
    - Evaluate volume signals separately in trend vs range regimes (proxied by price momentum strength) and in high vs low volatility regimes (proxied by return dispersion).
    - Hypothesis: volume confirms trends in trending regimes but marks exhaustion in choppy regimes.
    - Implement conditioning via continuous scaling (weights) rather than hard filters where possible.
    

-     Signal composability and ensembling:
    - Combine multiple volume-derived components: anomaly (relative volume), persistence, breakout indicator, and price-volume divergence.
    - Consider simple ensembles (weighted sums of ranks) to improve stability and reduce reliance on any single pattern.
    - Measure overlap with momentum/reversal-only baselines to ensure incremental value from volume.
    

-     Diagnostics, robustness, and falsification:
    - Stress test lookbacks, normalization choices (log vs raw), clipping thresholds, and event definitions for breakouts.
    - Verify the signal is not just a volatility proxy (since vol and volume co-move); test vol-adjusted versions implicitly via return scaling.
    - Check concentration: whether PnL is driven by rare spikes vs frequent small edges.
    

-     Note that turnover may be an issue. If you are finding fitting to 1 period returns is too fast, calculate longer term forward returns and use that to fit any rolling fit signals. Make sure to correctly mask returns though. For example if your forward horizon is K, your rolling window for time T should end data at time T-K.
    



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
