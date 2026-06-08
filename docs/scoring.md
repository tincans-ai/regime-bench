# Scoring

The RegimeBench public scoring path is deterministic. The default public
verifier is the repair-style saved-code replay in
`verifier/repair_verifier.py`, exposed through the stable CLI
`verifier/pnl_verifier.py`.

1. Load visible train data and hidden/OOS test data for a task.
2. Build a feature frame where visible rows keep labels and hidden prediction
   rows have current return labels withheld.
3. Run `build_signal(data_path)` from the submitted code.
4. Cross-sectionally z-score and clip the signal at each timestamp.
5. Apply the bar-level causality gate.
6. Backtest the prediction-period signal against hidden residual returns.
7. Write daily PnL, signal output, `results.json`, and
   `pnl_verifier_manifest.json`.

The default replay mode is `strict`: hidden/OOS prediction rows withhold the
current return label before `build_signal(data_path)` is called. The bar-level
causality gate is enabled by default and perturbs current-bar labels and
future rows before final replay. The LLM judge used during internal
experimentation is not part of the canonical public score. `pnl_verifier.py`
is the public CLI source of truth.

```bash
uv run python verifier/pnl_verifier.py score-code \
  --task-dir tasks/signal-volume \
  --code-path my_candidate.py \
  --output-dir verifier_runs/signal-volume
```
