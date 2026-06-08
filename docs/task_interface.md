# Task Interface

Each RegimeBench task asks for one function:

```python
def build_signal(data_path):
    ...
```

`data_path` points to a parquet file. The input includes `timestamp_agg`,
`ticker`, market features, and safe lagged return columns. Hidden/OOS prediction
rows have `is_prediction_period = True`; current hidden return labels are null.

Return a Polars-compatible DataFrame with:

- `timestamp_agg`
- `ticker`
- `signal`

Signals should be finite floats. The verifier normalizes the signal
cross-sectionally at each timestamp before scoring, so the important content is
relative ranking and direction, not raw scale.
