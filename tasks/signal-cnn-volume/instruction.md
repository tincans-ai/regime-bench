# Time Series CNN Volume Signal

## Objective

Evaluate whether time-series CNNs can identify exploitable structure in residual returns that is not captured by linear factor models or simple time-series statistics.

The core hypothesis is that, even after removing common market components, asset-level returns may contain localized temporal patterns (e.g., bursts, shape motifs, volatility clustering asymmetries, lead–lag effects) that are difficult to model with linear regressions or summary features, but are naturally captured by convolutional filters operating over time.

Economically, this task tests whether residual returns are truly noise, or whether they reflect delayed information diffusion, heterogeneous agent reaction speeds, market microstructure frictions, or nonlinear interactions between volatility, volume, and past returns.

CNNs are treated as a hypothesis-testing tool rather than an assumed superior predictor. The research question is whether any detected structure is statistically stable out-of-sample, robust across assets and regimes, and economically meaningful after transaction costs.

Failure modes include performance collapsing after normalization changes, strong regime dependence, or the signal reducing to known effects (momentum, reversal, volatility timing) under simpler transformations.


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

**Important**: You may use `volume`, `close`, `return`, `residual_return`, and `timestamp_agg` to construct your signal.
**Important**: You may use lagging versions of `return` or `residual_return` but NOT the current period OR future period returns - that is cheating. On rows marked `is_prediction_period = True`, current-period return labels are withheld/null. Use provided safe lag columns such as `return_lag1` and `residual_return_lag1` for one-bar return features on prediction rows; do not depend on shifting null prediction-period labels.

## Requirements

Implement the `build_signal` function in `starter_code/code.py`:

```python
def build_signal(data_path: str) -> pl.DataFrame:
    """
    Time Series CNN Volume Signal

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


-     Target definition and residualization variants:
    - Test different normalizatios of residual returns (with vol, time series z-scored, etc)
    - Compare predicting next-period residual return vs aggregated forward residual returns over multiple horizons.
    - Examine whether predictability exists at specific horizons, distinguishing microstructure-driven effects from behavioral ones.
    

-     Temporal feature representations:
    - Use raw returns, clipped returns, volatility-scaled returns, and signed volatility as alternative inputs.
    - Include volume, turnover, or market-cap–scaled activity to allow detection of interaction patterns (e.g., return–volume motifs).
    - Compare overlapping vs non-overlapping windows to assess whether structure arises from autocorrelation artifacts.
    

-     Normalization and stabilization:
    - Apply time-series normalization (rolling z-score, rank transform, volatility scaling) to prevent sensitivity to level shifts.
    - Compare cross-sectional normalization at each timestamp vs purely time-series normalization.
    - Winsorize or clip extreme returns and volumes to reduce sensitivity to data errors and liquidation spikes.
    

-     Windowing and locality assumptions:
    - Vary input window lengths to test where predictive structure emerges.
    - Short windows may capture microstructure or order-flow effects; longer windows may collapse to trend or reversal signals.
    - Examine whether signal strength decays smoothly with window length, indicating genuine local structure.
    

-     Overlap with known effects:
    - Measure correlations between CNN-derived signals and momentum, mean reversion, volatility, and liquidity signals.
    - Test whether residual-return predictability acts as a conditional enhancer rather than a standalone alpha.
    - Determine whether CNNs primarily act as nonlinear filters over known effects.
    

-     Regime conditioning:
    - Test performance conditional on market volatility, trend strength, and drawdown regimes.
    - Hypothesis: residual structure strengthens during stressed or high-volatility periods when linear models degrade.
    - Evaluate regime-conditioned performance rather than unconditional averages.
    

-     Cross-asset and cross-time robustness:
    - Test whether learned temporal patterns generalize across assets or remain asset-specific.
    - Evaluate stability across market eras (bull, bear, sideways) to rule out regime overfitting.
    

-     Noise and data-quality considerations:
    - Assess sensitivity to missing data, stale prices, and forward-filled market capitalization.
    - Verify that predictability does not arise from data leakage or normalization artifacts.
    

-     Composability and ensembling:
    - Treat CNN-derived outputs as components that can be combined with linear or rule-based signals.
    - Explore whether CNNs are more effective as feature extractors feeding downstream signal logic rather than direct predictors.
    

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
