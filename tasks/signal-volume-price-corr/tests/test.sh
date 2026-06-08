#!/bin/bash
# Price-Volume Correlation Signal Task - Test Script

# Ensure we always write a reward file, even on unexpected errors
trap 'if [ ! -f /logs/verifier/reward.txt ]; then echo "-1" > /logs/verifier/reward.txt; fi' EXIT

echo "=================================================="
echo "SIGNAL-VOLUME-PRICE-CORR EVALUATION"
echo "=================================================="

# Create output directories
mkdir -p /app/output/plots
mkdir -p /logs/artifacts
mkdir -p /logs/verifier

# Clear old outputs
rm -rf /app/output/*
mkdir -p /app/output

# Check signal code exists
if [ ! -f /app/starter_code/code.py ]; then
    echo "ERROR: No code.py found in /app/starter_code/"
    echo "0.0" > /logs/verifier/reward.txt
    exit 1
fi

# Check GEMINI_API_KEY is set
if [ -z "$GEMINI_API_KEY" ]; then
    echo "WARNING: GEMINI_API_KEY not set - LLM judge will be skipped"
fi

# Debug: print all environment variables (except a few common system ones)
# echo ""
# echo "===== DEBUG: Environment Variables ====="
# printenv | grep -vE '^(PWD|HOME|SHLVL|PATH|_)=|OLDPWD=' | sort
# echo "========================================"
# echo ""

# Run evaluation (DEBUG_MODE=1 generates plots on test data)
echo ""
echo "Running evaluation..."
cd /tests
# stdbuf may not exist on all platforms (e.g. Modal) — fall back to unbuffered python
if command -v stdbuf &> /dev/null; then
    DEBUG_MODE=1 stdbuf -oL python -u evaluate.py 2>&1
else
    DEBUG_MODE=1 python -u evaluate.py 2>&1
fi
EVAL_EXIT_CODE=$?

# Note: exit code 1 means "signal failed quality thresholds" (not a crash)
# We still want to extract the actual score from results.json
if [ $EVAL_EXIT_CODE -ne 0 ]; then
    echo ""
    echo "evaluate.py exited with code $EVAL_EXIT_CODE (signal did not pass thresholds)"
fi

echo "Evaluation complete"

# Replay saved checker snapshots on hidden OOS only after the agent has
# finished. This preserves analysis artifacts without exposing hidden OOS during
# live /check calls.
if [ "${QR_EVAL_REPLAY_HIDDEN_OOS_CHECKS:-1}" != "0" ]; then
    echo ""
    echo "Replaying hidden-OOS checker snapshots..."
    if command -v stdbuf &> /dev/null; then
        stdbuf -oL python -u evaluate.py --replay-hidden-oos-checks 2>&1 | tee /logs/verifier/hidden_oos_replay_stdout.txt || true
    else
        python -u evaluate.py --replay-hidden-oos-checks 2>&1 | tee /logs/verifier/hidden_oos_replay_stdout.txt || true
    fi
fi

# Copy artifacts to logs for viewer
if [ -d /app/output/plots ]; then
    PLOT_COUNT=$(ls -1 /app/output/plots/*.png 2>/dev/null | wc -l)
    echo "Copying $PLOT_COUNT plots to /logs/verifier/"
    cp -f /app/output/plots/*.png /logs/artifacts/ 2>/dev/null || true
    cp -f /app/output/plots/*.png /logs/verifier/ 2>/dev/null || true
    ls -la /logs/verifier/*.png 2>/dev/null || echo "No plots in /logs/verifier/"
else
    echo "WARNING: No plots directory found at /app/output/plots"
fi
if [ -f /app/output/judge_result.json ]; then
    cp -f /app/output/judge_result.json /logs/artifacts/judge_result.json
    cp -f /app/output/judge_result.json /logs/verifier/judge_result.json
fi
if [ -f /app/output/results.json ]; then
    cp -f /app/output/results.json /logs/artifacts/results.json
    cp -f /app/output/results.json /logs/verifier/results.json
fi
# Save agent's signal code for debugging
if [ -f /app/starter_code/code.py ]; then
    cp -f /app/starter_code/code.py /logs/artifacts/code.py
    cp -f /app/starter_code/code.py /logs/verifier/code.py
fi
# Copy debug parquets if they exist
if [ -d /app/output/debug ]; then
    cp -rf /app/output/debug /logs/artifacts/debug
    cp -rf /app/output/debug /logs/verifier/debug
    echo "Copied debug parquets to /logs/artifacts/debug/"
fi

# Extract score from results
if [ -f /app/output/results.json ]; then
    SCORE=$(python3 - <<'PYCODE'
import json, math
try:
    with open("/app/output/results.json") as f:
        r = json.load(f)
    score = r.get("score", None)
    score = float(score)
    if not math.isfinite(score):
        raise ValueError("non-finite score")
    print(score)
except Exception:
    print("INVALID")
PYCODE
)
    if [ "$SCORE" = "INVALID" ]; then
        echo "-1.0" > /logs/verifier/reward.txt
        echo "Score is invalid (NaN/inf/missing)"
    else
        echo "$SCORE" > /logs/verifier/reward.txt
        echo ""
        echo "=================================================="
        echo "SCORE: $SCORE"
        echo "=================================================="
    fi
else
    echo "-1.0" > /logs/verifier/reward.txt
    echo "No results.json found"
fi
