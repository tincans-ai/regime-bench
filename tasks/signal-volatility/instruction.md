# Low Volatility Anomaly Signal Task

## Objective

Build a cross-sectional signal that exploits the low volatility anomaly in crypto: assets with lower realized volatility may deliver superior risk-adjusted returns (and potentially higher absolute returns) than high-volatility assets, contrary to the simple “higher risk = higher return” intuition.

The core hypothesis is that crypto market participants systematically overpay for high-volatility “lottery-like” exposure (high upside narratives, convexity, attention-driven pumps), while more stable assets are under-owned because they are perceived as less exciting or offer less headline return potential. If leverage or portfolio construction constraints prevent investors from scaling low-vol exposure to reach target returns, mispricing can persist, creating a cross-sectional premium for low-vol assets.

The research question is whether this effect is present out-of-sample in crypto and whether it survives practical considerations such as heavy-tailed returns, liquidity differences, and strong regime dependence. The intended signal direction is:
- Long low-volatility assets (more stable realized return paths)
- Short high-volatility assets (more erratic realized return paths)
so that the signal is negatively correlated with realized volatility.

Key failure modes include the signal merely capturing liquidity/size effects (low vol = large, liquid assets), dependence on a small subset of regimes (e.g., only risk-off periods), or apparent performance driven by extreme events and estimation noise rather than persistent mispricing.


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

**Important**: You may use `return`, `close`, `volume`, `residual_return`, and `timestamp_agg` to construct your signal.
**Important**: You may use lagging versions of `return` or `residual_return` but NOT the current period OR future period returns - that is cheating. On rows marked `is_prediction_period = True`, current-period return labels are withheld/null. Use provided safe lag columns such as `return_lag1` and `residual_return_lag1` for one-bar return features on prediction rows; do not depend on shifting null prediction-period labels.

## Requirements

Implement the `build_signal` function in `starter_code/code.py`:

```python
def build_signal(data_path: str) -> pl.DataFrame:
    """
    Low Volatility Anomaly Signal Task

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


-     Baseline realized volatility signal:
    - Start with simple realized volatility as rolling std of returns; signal = -realized_vol.
    - Test multiple lookbacks (short vs medium) to see whether the anomaly is driven by microstructure noise or persistent risk preferences.
    - Compare non-overlapping vs overlapping return sampling to understand any induced autocorrelation.
    - Also try using: residual returns to calculate realized vol, normalizing returns by trailing vol then calculating standard deviations
    

-     Cross-sectional normalization for regime robustness:
    - Use cross-sectional z-scores or ranks of realized volatility per timestamp so the signal is relative (low vol vs peers) rather than absolute.
    - Hypothesis: the anomaly is primarily cross-sectional (relative preference for “lottery vol” assets) rather than time-series timing of market vol.
    - Check whether the effect concentrates in the tails (lowest vol decile vs highest vol decile).
    

-     Exponentially weighted volatility and responsiveness:
    - Use EWMA/EWM volatility to respond more quickly to volatility shifts while reducing window-edge artifacts.
    - Compare fast vs slow decay to test whether the premium is linked to recent stability or long-run stability.
    - Consider blending fast and slow vol estimates (term-structure of volatility) as an additional feature.
    

-     Robust volatility measures under heavy tails:
    - Crypto returns are heavy-tailed; test robust dispersion measures (e.g., median absolute deviation on returns) as alternatives to std.
    - Winsorize or clip returns before volatility estimation to reduce sensitivity to single liquidation candles.
    - Compare “raw vol” vs “clipped vol” to diagnose whether results are driven by extreme events.
    

-     Downside / asymmetry-aware volatility:
    - Compute downside volatility (std of negative returns only) and compare to total volatility.
    - Hypothesis: markets may overpay for upside lottery exposure while underpricing downside stability, producing stronger anomalies when downside risk is emphasized.
    - Construct skew-sensitive variants by combining volatility with the sign balance of returns.
    

-     Volatility-of-volatility and stability-of-stability:
    - Compute vol-of-vol: rolling volatility of the volatility estimate itself.
    - Hypothesis: assets with not just low volatility but stable volatility command a premium due to risk management constraints and leverage targeting.
    - Test whether low vol-of-vol predicts returns even after controlling for vol level.
    

-     Relative volatility vs market / common component:
    - Normalize asset volatility by market volatility (cross-asset average) to isolate idiosyncratic stability.
    - Compare signals built from absolute vol vs relative vol to determine whether the anomaly is “low risk overall” or “low risk relative to peers.”
    - If available, use volume as a proxy for liquidity and control for liquidity-linked volatility effects.
    

-     Volatility–liquidity interaction using volume:
    - Create volatility-per-unit-volume style measures (e.g., vol divided by rolling volume or its log), to separate “high vol because illiquid” from “high vol because speculative.”
    - Hypothesis: the anomaly may be strongest among more liquid assets where volatility reflects risk preference, not microstructure.
    - Filter or downweight extremely low-volume assets to reduce noise-driven “fake low vol.”
    

-     Regime conditioning and state dependence:
    - Condition the low-vol signal on market regime: trend strength, market volatility, or drawdown state.
    - Hypothesis: in strong bull trends, high-vol assets may dominate (speculative upside), while in risk-off regimes low-vol assets outperform.
    - Implement regime as a continuous conditioner (scale exposure) rather than a hard on/off switch.
    

-     Interaction with momentum / mean reversion:
    - Test whether low-vol works best when combined with momentum (e.g., long low-vol assets that also have positive trend).
    - Alternatively, test whether high-vol assets exhibit mean reversion that can be harvested separately, implying the low-vol signal should avoid certain short-horizon horizons.
    - Explicitly measure overlap/correlation with basic momentum and reversal signals to ensure the anomaly is distinct.
    

-     Cross-sectional construction and portfolio mechanics:
    - Consider constructing a market-neutral long/short portfolio via ranks (e.g., long bottom quantile vol, short top quantile vol).
    - Test sector/cluster neutrality proxies (if no sector labels, use coarse buckets like market-cap ranks or volume ranks) to avoid concentration.
    - Evaluate whether the premium is driven by a few recurring “meme-like” high-vol names.
    

-     Stability and falsification checks:
    - Evaluate out-of-sample stability across time (bull/bear/chop) and across asset subsets (top volume vs lower volume).
    - Stress test sensitivity to lookback choice, clipping thresholds, and normalization approach; robust effects should degrade smoothly.
    - Ensure the signal does not unintentionally time the market (e.g., becoming net short in high-vol market regimes).
    

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
