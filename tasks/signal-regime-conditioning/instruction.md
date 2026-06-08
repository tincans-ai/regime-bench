# Regime-Conditioned Momentum/Mean-Reversion Signal

## Objective

Build a signal that **adaptively switches between momentum and mean reversion** based on detected market regime.

The hypothesis:
- In trending regimes → momentum works (follow the trend)
- In ranging/volatile regimes → mean reversion works (fade extremes)

Your goal: Detect market regime and blend momentum vs mean reversion signals accordingly.


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
    Regime-Conditioned Momentum/Mean-Reversion Signal

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


-     Regime detection:
    - Hypothesis: high-volatility regimes favor mean reversion due to forced deleveraging, liquidation cascades, and short-term overreaction, while low-volatility regimes favor momentum due to gradual information diffusion.
    - Define volatility regimes using rolling realized volatility, range-based measures, or volatility-of-volatility.
    - Test hard regime splits vs smooth transitions (continuous regime scores).
    - Also look at using market-wide returns or volume profiles 
    

-     Hidden Markov Models and latent regime structure:
    - Use HMM-style regime inference to identify latent market states (e.g., trending, mean-reverting, chaotic).
    - Compare regimes inferred from market-level returns vs cross-sectional dispersion vs volatility dynamics.
    - Evaluate regime persistence and transition probabilities to assess trade stability and turnover risk.
    - can also just use simpler continous scores
    

-     Trend vs range detection via correlation structure:
    - Use rolling autocorrelation of returns or rolling correlation with lagged market returns to distinguish trending vs ranging behavior.
    - Hypothesis: rising positive autocorrelation indicates momentum-dominated regimes, while near-zero or negative autocorrelation favors reversion.
    - Examine whether correlation signals are more informative at the market level or asset-specific level.
    

-     Trend strength indicators (ADX-like logic):
    - Construct ADX-style measures using directional movement, range expansion, or cumulative return consistency.
    - Compare classical technical definitions vs statistically motivated variants (e.g., signal-to-noise ratios of returns).
    - Use trend strength as a continuous conditioner rather than a binary filter.
    

-     Regime-dependent blending of signals:
    - Combine momentum and mean reversion signals using weights that depend on inferred regime probabilities.
    - Test linear blends vs nonlinear gating (e.g., only activate MR above a volatility threshold).
    - Evaluate whether blended signals reduce drawdowns relative to pure momentum or pure MR strategies.
    

-     Market-wide vs asset-specific conditioning:
    - Use market-level regime indicators to gate or scale individual asset signals.
    - Hypothesis: individual asset momentum is more reliable when the broader market exhibits coherent trends.
    - Compare conditioning on global market signals vs sector/cluster-level aggregates.
    

-     Momentum signal variants:
    - Define momentum using different horizons (short-term, intermediate-term) and return transformations (raw, volatility-scaled).
    - Test whether momentum strength depends on recent volatility contraction or expansion.
    - Examine cross-sectional vs time-series momentum formulations under different regimes.
    

-     Mean reversion signal variants:
    - Define MR using extreme return thresholds, distance from rolling mean, or rank-based deviations.
    - Compare fast MR (1–5 bars) vs slow MR (multi-day) and their interaction with volatility regimes.
    - Test asymmetric behavior: downside shocks may revert differently than upside spikes.
    

-     ML-assisted regime detection:
    - Use scikit-learn–available models (e.g., clustering, PCA on market features) to infer regime states without labels.
    - Compare ML-derived regimes to hand-crafted indicators for interpretability and stability.
    - Treat ML regimes as soft conditioners rather than direct trade signals.
    

-     Turnover and horizon alignment considerations:
    - Monitor turnover explicitly; high-frequency regime switching may erase alpha after costs.
    - If 1-period-ahead prediction is too noisy, fit signals on longer-horizon forward returns.
    - When using a forward horizon of K, ensure all rolling features at time T only use data up to T−K to avoid leakage.
    

-     Signal smoothing and noise reduction:
    - Apply temporal smoothing (EMA, rolling averages) to regime probabilities and signal weights to reduce churn.
    - Clip or winsorize extreme feature values that can cause spurious regime flips.
    - Require minimum regime confidence before activating aggressive signal weights.
    

-     Diagnostics and overlap analysis:
    - Measure how often regime logic simply reproduces volatility timing or beta exposure.
    - Test whether regime-conditioned signals add incremental value beyond unconditional momentum or MR.
    - Evaluate robustness by perturbing thresholds, lookbacks, and regime definitions.
    



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
