"""
Shared instruction components for signal tasks.

Centralizes data column descriptions, default metadata, and common templates
that are reused across signal task configurations.
"""

# =============================================================================
# Default Metadata
# =============================================================================

DEFAULT_CATEGORY = "quant-research"
DEFAULT_AUTHOR_NAME = "Alpha Team"
DEFAULT_AUTHOR_EMAIL = "team@alpha.ai"

# =============================================================================
# Default Resources
# =============================================================================

DEFAULT_RESOURCES = {
    "verifier_timeout_sec": 300.0,
    "agent_timeout_sec": 1800.0,
    "build_timeout_sec": 600.0,
    "memory_mb": 8192,
    "cpus": 4,
    "gpus": 0,
    "extra_pip_packages": [],
}

# =============================================================================
# Data Column Descriptions
# =============================================================================

DATA_COLUMNS_TABLE = """\
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
| `residual_return_lag1` | Previous-period market-adjusted return, precomputed safely for prediction rows |\
"""

LOOKAHEAD_WARNING = """\
**Important**: You may use lagging versions of `return` or `residual_return` \
but NOT the current period OR future period returns - that is cheating. \
On rows marked `is_prediction_period = True`, current-period return labels are \
withheld/null. Use provided safe lag columns such as `return_lag1` and \
`residual_return_lag1` for one-bar return features on prediction rows; do not \
depend on shifting null prediction-period labels.\
"""


def data_columns_section(
    allowed_features: list[str] | None = None,
    extra_rows: str | None = None,
) -> str:
    """
    Build the data_columns instruction section.

    Args:
        allowed_features: List of feature names that can be used directly
            (e.g., ["funding_rate", "volume", "close"]). If None or empty,
            uses the generic lookahead warning without feature restrictions.
        extra_rows: Additional table rows to append (e.g., "| `...` | desc |")

    Returns:
        Full data_columns markdown string for the config.
    """
    # Build the table
    table = DATA_COLUMNS_TABLE
    if extra_rows:
        table = table + "\n" + extra_rows

    # Build usage note
    if allowed_features:
        features_str = ", ".join(f"`{f}`" for f in allowed_features)
        usage_note = f"**Important**: You may use {features_str}, and `timestamp_agg` to construct your signal.\n"
    else:
        usage_note = ""

    return f"{table}\n\n{usage_note}{LOOKAHEAD_WARNING}"


# =============================================================================
# Starter Docstring Template
# =============================================================================

STARTER_DOCSTRING_TEMPLATE = """\
{brief}

Args:
    data_path: Path to parquet file with columns: timestamp_agg, ticker,
               close, volume, return, return_lag1, funding_rate,
               residual_return, residual_return_lag1, is_prediction_period

Returns:
    DataFrame with columns: timestamp_agg, ticker, signal
    - signal should be a float representing your prediction
    - higher signal = expect higher returns\
"""


def starter_docstring(brief: str, extra: str | None = None) -> str:
    """
    Generate a standard docstring from a brief description.

    Args:
        brief: One-line description (e.g., "Build a funding rate contrarian signal.")
        extra: Additional notes to append (e.g., "Signal should be NEGATIVELY correlated to funding rate")

    Returns:
        Full docstring for the build_signal function.
    """
    result = STARTER_DOCSTRING_TEMPLATE.format(brief=brief)
    if extra:
        result = result + "\n    - " + extra
    return result


def starter_docstring_from_title(title: str) -> str:
    """
    Auto-generate a standard docstring from the task title.

    This is used when starter_brief is not provided, allowing simplified configs
    with just: title, objective, allowed_features, ideas.

    Args:
        title: The task title (e.g., "Time Series CNN Volume Signal")

    Returns:
        Full docstring for the build_signal function.
    """
    return STARTER_DOCSTRING_TEMPLATE.format(brief=title)


# =============================================================================
# Starter Hints Template
# =============================================================================

STARTER_HINTS_COMMON = """\
# You are a signal researcher trying to build a signal based on first principals thinking.
# Units:
# - You are predicting returns residualized to market, size and momentum factors in crypto
# - The target returns are normalized by their trailing exponentially weighted volatility
# - Your signal is automatically cross-sectionally z-scored and clipped at ±4σ before evaluation
# - The z-scored signal is used directly as portfolio weights in the backtest
# Research Approach:
# - You should start by listing a concrete hypothesis about the signal you are trying to build, and then test it using the data available to you.
# -     Be open to the data suggesting that your hypothesis is wrong, and adjust your hypothesis accordingly.
# - Do not make changes to the signal construction that is not motivated by the data or your hypothesis.
# Iteration can be thought of in 2 parts:
# - Find an idea that generally works:
#     - This is the "core hypothesis" that you are trying to test, and usually takes the form of an underlying feature that is expected to predict returns.
#     - Start out with a simple idea + a couple standard normalizaiton steps, and refine as you go.
# - Refine the idea based on the data:
#     - Once we have a core hypothesis that generally works, we should improve the signal shaping
#     - This includes:
#         - Time series smoothing of the underlying feature
#         - Time series and cross sectionally normalization of the signal to ensure it is maximally statistically robust
#         - normalizing the underlying feature with volatilty or volume or marketcap if that helps back up the hypothesis
#         - conditioning based on the volume, volatility, marketcap, or event market-wide perfromance if that helps back up the hypothesis
#         - You should do some cross sectional normalization to even out risk taking and prevent the mean prediction vs the mean residual from being a factor in the portfolio returns
# Outputs for every step:
#     - Latex formula for your previous and new signals for this iteration
#     - Code snippet for your previous and new signals for this iteration
#     - A brief explanation of the changes you made to the signal and why you made them
#     - How your core hypothesis has evolved over this iteration

# Remember, you are getting evaluated on how the signal performs on a holdout set that you will not have access to. The purpose of the in-sample evaluation is to help you iterate on your signal construction and hypothesis, not to hill-climb on this specific dataset.

"""


def starter_hints(task_specific: str) -> str:
    """
    Combine common hints with task-specific hints.

    Args:
        task_specific: Task-specific hints/instructions

    Returns:
        Combined starter hints string.
    """
    return STARTER_HINTS_COMMON + task_specific
