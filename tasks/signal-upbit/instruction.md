# Cross-Venue Signal: Upbit (Korea) vs Binance (Global)

## Objective

Build a predictive signal using **cross-venue** information from Upbit (Korean retail exchange) and Binance (global exchange).

**Context:**
- **Upbit** is Korea's largest crypto exchange, dominated by retail traders
- **Binance** is the world's largest exchange with global institutional flow
- Price differences ("kimchi premium") and volume patterns between venues may contain predictive information
- Korean retail trading activity often differs from global markets

**Data Available:**
You have hourly data with prices and volumes from both exchanges:
- Upbit prices converted to USD (using KRW/USD FX rate)
- Binance prices in USD
- Volume from both venues
- Pre-computed kimchi premium (Upbit price vs Binance price difference)

**IMPORTANT: The Upbit data is contemporaneous with Binance data (same timestamps). In production, Upbit data arrives with ~1 hour delay. You MUST lag all Upbit-derived features by at least 1 hour (e.g., `.shift(1).over("ticker")`) to avoid look-ahead bias.**

**Your Goal:**
Discover cross-venue patterns that predict future Binance returns. Consider:
- Price dislocations between venues
- Volume patterns and divergences
- How Korean retail activity relates to subsequent global price moves
- Whether certain patterns are stronger during specific market conditions

The signal should predict **forward Binance returns** (next hour).


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
| `upbit_close_usd` | Upbit close price converted to USD |
| `binance_close_usd` | Binance close price in USD |
| `upbit_volume` | Upbit trading volume (base asset) |
| `binance_volume` | Binance trading volume (base asset) |
| `kimchi_premium_pct` | (Upbit - Binance) / Binance * 100, clipped to [-10, 10] |
| `volume_ratio` | Upbit volume / Binance volume |


**Important**: You may use `upbit_close_usd`, `binance_close_usd`, `upbit_volume`, `binance_volume`, `kimchi_premium_pct`, `volume_ratio`, `close`, `volume`, and `timestamp_agg` to construct your signal.
**Important**: You may use lagging versions of `return` or `residual_return` but NOT the current period OR future period returns - that is cheating. On rows marked `is_prediction_period = True`, current-period return labels are withheld/null. Use provided safe lag columns such as `return_lag1` and `residual_return_lag1` for one-bar return features on prediction rows; do not depend on shifting null prediction-period labels.

## Requirements

Implement the `build_signal` function in `starter_code/code.py`:

```python
def build_signal(data_path: str) -> pl.DataFrame:
    """
    Cross-Venue Signal: Upbit (Korea) vs Binance (Global)

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


-     Kimchi premium as a retail sentiment / capital-controls proxy:
    - Treat the kimchi premium as a proxy for Korean retail demand pressure and local constraints (on/off-ramp frictions, capital controls, latency of arbitrage).
    - Hypothesis: premium widens when Korean retail is risk-on; the key question is whether that predicts future global (Binance) moves or represents local-only dislocation.
    - Separate “informational” premium (predictive) from “frictional” premium (pure arbitrage wedge).
    

-     Premium mean reversion vs premium momentum:
    - Test whether extreme premium levels predict convergence (mean reversion), consistent with arbitrage catching up.
    - Test whether changes in premium predict continuation in Binance returns (premium momentum), consistent with Korea being an early demand signal.
    - Evaluate asymmetry: widening premium may behave differently than shrinking premium.
    

-     Decompose premium into level, change, and acceleration:
    - Use the premium level (current dislocation), first difference (hourly change), and second difference (acceleration).
    - Hypothesis: acceleration captures “retail impulse” more cleanly than levels, which may be sticky due to constraints.
    - Normalize these components to compare across tokens and regimes.
    

-     Cross-venue return lead–lag:
    - Compute Upbit USD returns vs Binance returns and test lead–lag relationships at multiple lags.
    - Hypothesis: Upbit may lead during certain times (Korean daytime) or in retail-dominated tokens.
    - Explicitly test whether lead–lag holds after controlling for market-wide moves.
    

-     Volume divergence and attention shocks:
    - Use Upbit volume spikes relative to Binance volume as a proxy for Korea-specific attention or retail bursts.
    - Compare “Upbit-only” volume surges vs “Binance-only” surges; they may imply different subsequent Binance outcomes.
    - Consider whether volume divergence predicts volatility expansion vs directional returns.
    

-     Volume-weighted premium and liquidity relevance:
    - Weight premium features by volume_ratio or by deviations of Upbit volume from its own baseline.
    - Hypothesis: a premium supported by unusually high Upbit volume is more informative than a premium on thin trading.
    - Downweight or filter low-liquidity hours/tokens where premium is noisy.
    

-     Cross-sectional ranking within the Upbit–Binance universe:
    - Rank tokens by kimchi_premium_pct and by premium change each hour; use cross-sectional z-scores/ranks.
    - Hypothesis: the cross-sectional tails (most abnormal premiums vs peers) are where the signal concentrates.
    - Test sector/cluster effects implicitly by whether certain token types systematically carry premium.
    

-     Time-series normalization and anomaly detection:
    - Compute rolling z-scores / percentile ranks of premium and volume_ratio per token to detect “unusual vs history” states.
    - Compare short lookbacks (fast retail shifts) vs longer lookbacks (structural premium baseline).
    - Clip / winsorize extreme values beyond the provided [-10, 10] clipping to reduce sensitivity to microstructure noise.
    

-     Interaction effects and conditional logic:
    - Condition premium signals on volume_ratio (premium with high Upbit share may differ from premium with low Upbit share).
    - Condition on market trend / volatility state: premium may behave differently in bull vs bear vs chop.
    - Hypothesis: in strong bull trends, premium may be momentum-like; in sideways/high-vol markets, it may mean-revert.
    

-     Time-of-day and calendar structure:
    - Test whether predictive relationships strengthen during Korean waking hours vs non-Korean hours.
    - Hypothesis: Upbit retail flows are time-localized, so “when” premium changes happen matters as much as “how much.”
    - Evaluate whether signals degrade when restricting to certain hours (robustness) vs relying on a narrow window (fragility).
    

-     Event-like behavior and structural breaks:
    - Identify sudden step-changes in premium or volume_ratio and treat them as event signals (shock indicators).
    - Hypothesis: sharp premium jumps can reflect local news / social propagation that later diffuses globally.
    - Detect and downweight structural breaks from listing changes, fee changes, or data artifacts.
    

-     Diagnostic separation: informational vs mechanical arbitrage:
    - If premium mean reversion exists without directional Binance predictability, it may be primarily an arbitrage phenomenon.
    - If premium changes predict Binance returns even when premium is small, it suggests informational content.
    - Explicitly test whether the signal is just “betting on convergence” vs “forecasting global direction.”
    

-     Signal composition and ensembling:
    - Combine premium level, premium change, Upbit lead–lag return features, and volume divergence into an ensemble.
    - Allow regime-dependent weights (e.g., more weight on premium momentum in trends; more on premium reversion in high vol).
    - Prefer simple, stable components first; treat more complex interactions as incremental additions.
    

-     Noise reduction and data quality checks:
    - Validate KRW/USD conversion alignment; FX timing mismatches can create false premium signals.
    - Ensure volume units match (base asset) and handle missing/zero volumes robustly.
    - Apply smoothing (EMA) to regime indicators and avoid overreacting to single-hour prints.
    

-     Turnover and horizon alignment considerations:
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
