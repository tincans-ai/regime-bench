# Price-Volume Correlation Signal Task

## Objective

Build a predictive signal from the relationship between price changes and trading volume. The goal is to test whether the strength, sign, and stability of price–volume linkage contains information about future returns, beyond what is captured by price-only or volume-only signals.

The core hypothesis is microstructural and behavioral: in healthy trending markets, price moves tend to be “confirmed” by volume (participation and conviction), while weak or manipulated markets can exhibit price changes that are poorly supported by genuine volume. Additionally, shifts in the price–volume relationship may indicate regime transitions (trend formation, exhaustion, or reversal).

This task evaluates three questions:
1) Does high price–volume correlation predict continuation (trend persistence)?
2) Does low or negative correlation predict reversal or underperformance (weak conviction, noisy prints, potential wash-trading-like behavior)?
3) Do changes in correlation act as early warnings of regime shifts and trend reversals?

The signal should extract robust features from rolling/EWM estimates of price–volume coupling, be evaluated cross-sectionally and over time, and remain stable out-of-sample under reasonable changes to windows, normalization, and noise controls. Failure modes include correlations driven purely by volatility scaling, sensitivity to outliers/spikes, or effects that disappear after conditioning on liquidity and turnover.


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
    Price-Volume Correlation Signal Task

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


-     Rolling correlation as a baseline feature:
    - Compute rolling or EWM correlations between volume and (abs(return), abs(log-return), or range proxies derived from close).
    - Test multiple windows (e.g., 24–168 hours) to identify whether microstructure vs slower behavioral effects dominate.
    - Use robust correlation variants (rank correlation) to reduce outlier sensitivity.
    

-     Directional price–volume correlation:
    - Use signed returns instead of abs returns to capture “buying pressure on up moves” vs “selling pressure on down moves.”
    - Hypothesis: positive corr(signed return, volume) indicates trend-following conviction; negative may indicate distribution or absorption.
    - Test asymmetric correlations: corr(volume, positive returns only) vs corr(volume, negative returns only).
    

-     Correlation change as a regime-shift indicator:
    - Construct signals from the change in correlation (delta corr) or the divergence between fast and slow correlation estimates.
    - Hypothesis: rising correlation indicates trend formation / increasing participation; falling correlation indicates trend exhaustion.
    - Evaluate whether sharp breaks in correlation precede reversals or volatility expansions.
    

-     High correlation = continuation vs exhaustion (nonlinear mapping):
    - Test both interpretations: (a) high correlation implies continuation (trend confirmation), (b) extremely high correlation implies late-stage blowoff/exhaustion.
    - Use thresholded or piecewise logic: mid-high correlation supports momentum; extreme correlation triggers caution/contrarian.
    - Validate whether the relationship is monotonic or hump-shaped.
    

-     Low correlation as a “quality / validity” filter:
    - Treat low or unstable price–volume coupling as a red flag for noisy markets (weak participation, thin liquidity, potential wash-like patterns).
    - Use low correlation to downweight other signals (e.g., momentum) rather than as a standalone short.
    - Test whether low correlation predicts higher future volatility and worse returns after costs.
    

-     Correlation stability and consistency:
    - Measure stability of correlation estimates over time (vol-of-corr, rolling std of corr, drawdown in corr).
    - Hypothesis: assets with stable price–volume coupling have more reliable trend dynamics and cleaner execution.
    - Penalize unstable assets or scale positions inversely with corr-instability.
    

-     Volume conditioning and reliability:
    - Only trust correlation estimates when volume is sufficiently high relative to baseline (relative volume filter).
    - Weight correlation observations by volume or use volume-weighted covariance/correlation to emphasize higher-information periods.
    - Downweight low-volume intervals where correlation is dominated by discrete prints.
    

-     Volume-weighted price change and “efficiency” measures:
    - Construct features like volume * return, return / volume, or return per unit volume to capture price impact / efficiency.
    - Hypothesis: large returns on low volume may be fragile (revert), while large returns on high volume may persist (confirm).
    - Combine with correlation: price impact metrics may explain when correlation is informative.
    

-     Cross-sectional ranking and portfolio construction:
    - Rank assets cross-sectionally by correlation metrics each hour and test tails: high-corr bucket vs low-corr bucket.
    - Compare performance conditional on recent returns: does high corr improve momentum Sharpe? does low corr improve reversal Sharpe?
    - Ensure cross-sectional normalization to remove market-wide volume/volatility regime effects.
    

-     Interaction with volatility (avoid accidental vol timing):
    - Since abs(return) is tied to volatility, verify the signal isn’t just “high vol assets behave differently.”
    - Test volatility-scaled returns or use rank correlations to reduce mechanical coupling between vol and correlation estimates.
    - Compare corr(abs(return), volume) vs corr(|return|/rolling_vol, volume) to isolate “volume response” from “vol level.”
    

-     Diagnostics and falsification tests:
    - Stress test lookbacks, EWM decay rates, clipping of returns/volume spikes, and correlation estimator choice.
    - Check whether signal is driven by a few extreme events (spikes) versus broadly distributed behavior.
    - Evaluate robustness across liquidity cohorts (high-volume vs low-volume assets) to ensure generality.
    

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
