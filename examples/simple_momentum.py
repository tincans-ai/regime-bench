"""Small example RegimeBench candidate.

This is intentionally simple: it ranks recent ticker-level price momentum within
each timestamp. It is meant as an API smoke example, not a strong baseline.
"""

from __future__ import annotations

import polars as pl


def build_signal(data_path):
    df = pl.read_parquet(data_path).sort(["ticker", "timestamp_agg"])
    signal = (
        pl.col("close")
        .pct_change(24)
        .over("ticker")
        .fill_null(0.0)
        .fill_nan(0.0)
        .alias("signal")
    )
    return df.select("timestamp_agg", "ticker", signal)
