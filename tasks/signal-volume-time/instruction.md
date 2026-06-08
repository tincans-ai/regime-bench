# Volume-Time Signal

## Objective

Build a predictive signal from the volume–time relationship: not just how much volume trades, but how quickly it accumulates and how that rate changes. The key hypothesis is that the “volume clock” (time measured in traded volume units) captures shifts in participation and urgency that precede price moves.

Economically and microstructurally, rapid volume accumulation can indicate informed trading, attention shocks, or forced flows (liquidations), while unusually slow volume can reflect lack of conviction, fragile price moves, or pending regime changes. Changes in volume velocity and acceleration may therefore predict whether price action will continue, reverse, or transition into a new state.

This task aims to transform raw volume and price into features that measure (i) volume velocity, (ii) volume acceleration/deceleration, and (iii) deviations from expected intraday accumulation patterns, and to test whether these features forecast forward returns in a robust, out-of-sample manner. Key failure modes include signals that collapse into simple volatility proxies, are dominated by rare spikes, or rely on unstable time-of-day artifacts rather than persistent participation dynamics.


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
    Volume-Time Signal

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


-     Volume clock / cumulative volume dynamics:
    - Work in cumulative volume space to measure how “fast the market is trading” independent of calendar time.
    - Compute volume velocity as the first difference of cumulative volume (and its rolling normalization).
    - Hypothesis: bursts in the volume clock correspond to participation shocks that can precede directional moves.
    

-     Volume acceleration and deceleration:
    - Use second differences of cumulative volume (acceleration) to detect ramping participation vs fading interest.
    - Hypothesis: accelerating volume during price trends confirms continuation; accelerating volume after sharp moves may indicate exhaustion/turning points.
    - Compare acceleration computed on raw volume vs log(volume + 1) to reduce heavy-tail dominance.
    

-     Abnormal volume relative to expected patterns:
    - Define expected volume based on recent history and/or time-of-day baselines; signal is deviation from expectation.
    - Use rolling percentiles, rolling z-scores, or percentile-of-hour features (e.g., current hour volume relative to same-hour history).
    - Hypothesis: abnormal positive deviations indicate attention/information shocks; abnormal negative deviations indicate low conviction / fragile moves.
    

-     Multi-horizon structure (fast vs slow participation):
    - Construct fast and slow measures of volume velocity/acceleration and take their spread (fast minus slow).
    - Hypothesis: when fast participation ramps above slow baseline, short-horizon predictability increases.
    - Evaluate whether the signal is best at predicting very short horizon returns or slightly longer horizons (to avoid microstructure noise).
    

-     Price confirmation vs divergence under volume-time signals:
    - Combine return sign with volume velocity/acceleration: price up + accelerating volume may imply continuation; price up + decelerating volume may imply weakening.
    - Build divergence features such as (return strength) vs (participation strength) to detect “price moving without participation.”
    - Hypothesis: divergence predicts reversal or increased volatility rather than continuation.
    

-     VWAP-style deviations using available features:
    - Approximate VWAP dynamics from close and volume by constructing volume-weighted price proxies over rolling windows.
    - Use deviations of price from rolling volume-weighted price as a “who is in control” measure (price above/below VWAP under rising participation).
    - Hypothesis: persistent deviation with rising volume clock indicates informed directional flow.
    

-     Volume bursts as event signals:
    - Detect burst events (volume velocity above a high percentile) and measure post-event drift vs reversal.
    - Separate single-bar bursts from sustained bursts (burst persistence) since they may imply different mechanisms (liquidation spike vs trend ignition).
    - Apply clipping/winsorization to prevent single extreme prints from dominating features.
    

-     Normalization and comparability:
    - Use time-series normalization (rolling z-score, rolling percentile) for each asset to stabilize volume features.
    - Consider cross-sectional ranking of “participation shocks” per timestamp to identify which assets are receiving disproportionate attention.
    - Log-transform volume features to reduce scale effects and heavy tails.
    

-     Participation-state regimes:
    - Define regimes based on participation intensity (high vs low volume-clock speed) and test whether return predictability changes by regime.
    - Hypothesis: momentum effects may strengthen when participation is rising; mean reversion may dominate after extreme participation spikes.
    - Use regime as a conditioner for signal strength rather than a hard filter where possible.
    

-     Noise reduction and data quality:
    - Handle missing volume prints explicitly; avoid forward-filling where it creates artificial smoothness in velocity/acceleration.
    - Clip implausible spikes and verify that volume units are consistent over time.
    - Smooth velocity/acceleration estimates (EMA) to reduce churn and stabilize inference.
    

-     Diagnostics and falsification:
    - Check whether performance is merely capturing volatility (since vol and volume often co-move); compare to volatility-scaled variants implicitly via return normalization.
    - Test stability across lookbacks, clipping thresholds, and time-of-day adjustments; robust effects should degrade smoothly.
    - Verify signal is not an artifact of overlapping windows or leakage from forward return construction.
    

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
