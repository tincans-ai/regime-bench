FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    tmux \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir \
    polars==1.36.1 \
    numpy==2.3.1 \
    scipy==1.16.3 \
    pyarrow==22.0.0 \
    asciinema \
    matplotlib==3.9.0 \
    statsmodels==0.14.4 \
    google-genai==1.57.0 \
    scikit-learn==1.8.0


# Create directories
RUN mkdir -p /app/data /app/output /app/starter_code /logs/verifier /logs/artifacts /logs/checker /logs/checker_val

# Harbor installs /check at runtime. /check-val is task-owned because Harbor's
# native checker exposes a single command.
RUN cat > /check-val <<'EOF' && chmod +x /check-val
#!/usr/bin/env bash
set -u

mkdir -p /logs/checker_val /logs/artifacts/checker_val /app/output
COUNT_FILE=/logs/checker_val/check_count.txt
COUNT="$(cat "$COUNT_FILE" 2>/dev/null || echo 0)"
NEXT=$((COUNT + 1))
MAX_VAL_CHECKS="${MAX_VAL_CHECKS:-1}"

if [ "$NEXT" -gt "$MAX_VAL_CHECKS" ]; then
  cat <<JSON
{"score": 0, "message": "Validation check budget exhausted ($COUNT/$MAX_VAL_CHECKS). Use prior validation feedback and submit your final solution."}
JSON
  exit 0
fi

if [ ! -f /tests/val.parquet ]; then
  cat <<JSON
{"score": 0, "message": "Validation data is unavailable for this task."}
JSON
  exit 0
fi

echo "$NEXT" > "$COUNT_FILE"
export CHECK_NUMBER="$NEXT"
export CHECK_SPLIT="val"
export CHECK_OUTPUT_PATH="/logs/checker_val/result_${NEXT}.json"
LOG_PATH="/logs/checker_val/output_${NEXT}.log"
rm -f /app/output/check_val_result.json
rm -rf "/app/output/checker_val_plots_${NEXT}" "/app/output/debug_${NEXT}"

cd /tests
if command -v stdbuf >/dev/null 2>&1; then
  DEBUG_MODE=1 stdbuf -oL python -u evaluate.py --checker --checker-split val >"$LOG_PATH" 2>&1
else
  DEBUG_MODE=1 python -u evaluate.py --checker --checker-split val >"$LOG_PATH" 2>&1
fi
STATUS=$?

if [ -f /app/output/check_val_result.json ]; then
  cp /app/output/check_val_result.json "$CHECK_OUTPUT_PATH"
  cp /app/output/check_val_result.json "/logs/artifacts/checker_val/result_${NEXT}.json"
fi

CHECK_NUMBER_PADDED=$(printf "%03d" "$NEXT")
if [ -f /app/starter_code/code.py ]; then
  cp /app/starter_code/code.py "/logs/checker_val/code_${CHECK_NUMBER_PADDED}.py"
  cp /app/starter_code/code.py "/logs/artifacts/checker_val/code_${CHECK_NUMBER_PADDED}.py"
fi

PLOTS_DIR="/app/output/checker_val_plots_${NEXT}"
if [ -d "$PLOTS_DIR" ]; then
  for f in "$PLOTS_DIR"/*.png; do
    if [ -f "$f" ]; then
      cp "$f" "/logs/checker_val/check_${NEXT}_$(basename "$f")"
      cp "$f" "/logs/artifacts/checker_val/check_${NEXT}_$(basename "$f")"
    fi
  done
fi

DEBUG_DIR="/app/output/debug_${NEXT}"
if [ -d "$DEBUG_DIR" ]; then
  mkdir -p "/logs/checker_val/debug_${NEXT}" "/logs/artifacts/checker_val/debug_${NEXT}"
  cp -f "$DEBUG_DIR"/*.parquet "/logs/checker_val/debug_${NEXT}/" 2>/dev/null || true
  cp -f "$DEBUG_DIR"/*.parquet "/logs/artifacts/checker_val/debug_${NEXT}/" 2>/dev/null || true
fi

if [ -s "$LOG_PATH" ]; then
  echo "[/check-val log tail: full log at $LOG_PATH]"
  tail -n 80 "$LOG_PATH"
fi

if [ -f "$CHECK_OUTPUT_PATH" ]; then
  echo
  cat "$CHECK_OUTPUT_PATH"
fi

exit "$STATUS"
EOF

# Copy data (will be populated by data prep script)
COPY data/ /app/data/

# Copy starter code
COPY starter_code/ /app/starter_code/

# Set Python path
ENV PYTHONPATH=/app

# Note: /check is set up by Harbor at runtime - do NOT create a symlink here.
# A symlink from /check -> /tests/check.sh causes Harbor's wrapper to
# overwrite check.sh through the symlink, creating an infinite recursion.
