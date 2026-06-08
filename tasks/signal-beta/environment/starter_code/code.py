"""Low Beta Anomaly Signal Task

Run this script to generate your signal output:
    python starter_code/code.py
"""

import polars as pl

EPS = 1e-8


def build_signal(data_path: str) -> pl.DataFrame:
    """
    Low Beta Anomaly Signal Task

    Args:
        data_path: Path to parquet file with columns: timestamp_agg, ticker,
                   close, volume, return, return_lag1, funding_rate,
                   residual_return, residual_return_lag1, is_prediction_period

    Returns:
        DataFrame with columns: timestamp_agg, ticker, signal
        - signal should be a float representing your prediction
        - higher signal = expect higher returns
    """
    # TODO: Implement your signal logic here


    raise NotImplementedError("Implement your signal logic here!")


def main():
    """Load data, build signal, and save output."""
    data_path = "/app/data/train.parquet"
    output_path = "/app/output/signal.parquet"

    print(f"Building signal from {data_path}...")
    signal_df = build_signal(data_path)

    # Validate output schema
    required_cols = {"timestamp_agg", "ticker", "signal"}
    missing = required_cols - set(signal_df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Filter invalid values
    valid_df = signal_df.filter(
        pl.col("signal").is_not_null()
        & pl.col("signal").is_not_nan()
        & pl.col("signal").is_finite()
    )

    print(f"Valid signals: {len(valid_df):,} / {len(signal_df):,}")
    print(f"Signal mean: {valid_df['signal'].mean():.6f}")
    print(f"Signal std:  {valid_df['signal'].std():.6f}")
    print(f"Signal min:  {valid_df['signal'].min():.6f}")
    print(f"Signal max:  {valid_df['signal'].max():.6f}")

    signal_df.write_parquet(output_path)
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()
