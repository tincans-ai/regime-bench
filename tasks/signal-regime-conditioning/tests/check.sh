#!/bin/bash
# Regime-Conditioned Momentum/Mean-Reversion Signal - Checker Script
# Provides intermediate feedback on training data

echo "[CHECK $CHECK_NUMBER] Running evaluation on training data..."

# Create output directories (including logs for Harbor)
mkdir -p /app/output
mkdir -p /logs/checker

# Clear old outputs to avoid stale data from previous runs
# rm -rf /app/output/*

# Check signal code exists
if [ ! -f /app/starter_code/code.py ]; then
    echo "ERROR: No code.py found in /app/starter_code/"
    cat > "$CHECK_OUTPUT_PATH" << EOF
{
    "score": 0,
    "message": "No code.py found. Please implement the build_signal function."
}
EOF
    exit 0
fi

# Enforce optional harness metadata policy before spending a checker run.
HARNESS_POLICY_PATH="/logs/agent/harness_policy.json"
HARNESS_CANDIDATE_FILE="/app/starter_code/code.py"
if [ -f "$HARNESS_POLICY_PATH" ]; then
    python3 - "$HARNESS_POLICY_PATH" "$HARNESS_CANDIDATE_FILE" <<'PY'
import json
import re
import sys
from pathlib import Path

policy_path = Path(sys.argv[1])
code_path = Path(sys.argv[2])
try:
    policy = json.loads(policy_path.read_text())
except Exception as exc:
    print(f"[harness] WARNING: could not read harness metadata policy: {exc}")
    sys.exit(0)

mode = str(policy.get("metadata_enforcement") or "off").lower()
if mode in {"", "off", "none", "false"}:
    sys.exit(0)

require_candidate = bool(policy.get("require_candidate_metadata"))
require_validation = bool(policy.get("require_validation_metadata"))
if not require_candidate and not require_validation:
    sys.exit(0)

try:
    text = code_path.read_text(errors="replace")
except Exception as exc:
    print(f"[harness] WARNING: could not read {code_path}: {exc}")
    sys.exit(0)

patterns = {
    "HARNESS_CANDIDATE": re.compile(r"^\s*#\s*HARNESS_CANDIDATE:\s*(\{.*\})\s*$", re.MULTILINE),
    "HARNESS_VALIDATION": re.compile(r"^\s*#\s*HARNESS_VALIDATION:\s*(\{.*\})\s*$", re.MULTILINE),
}
allowed_mutations = set(policy.get("allowed_mutation_types") or [])
allowed_islands = set(policy.get("allowed_island_labels") or [])
allowed_verdicts = set(policy.get("allowed_validation_verdicts") or [])
search_policy = str(policy.get("search_policy") or "linear")
issues = []


def parse_metadata(kind: str):
    matches = patterns[kind].findall(text)
    if not matches:
        issues.append(f"missing {kind} comment")
        return None
    if len(matches) > 1:
        issues.append(f"multiple {kind} comments; keep exactly one")
    try:
        parsed = json.loads(matches[0])
    except json.JSONDecodeError as exc:
        issues.append(f"invalid {kind} JSON: {exc.msg}")
        return None
    if not isinstance(parsed, dict):
        issues.append(f"{kind} JSON must be an object")
        return None
    return parsed

if require_candidate:
    candidate = parse_metadata("HARNESS_CANDIDATE")
    if candidate is not None:
        if not str(candidate.get("candidate_id") or "").strip():
            issues.append("HARNESS_CANDIDATE.candidate_id is required")
        mutation_type = str(candidate.get("mutation_type") or "").strip()
        if allowed_mutations and mutation_type not in allowed_mutations:
            issues.append("HARNESS_CANDIDATE.mutation_type is missing or invalid")
        island_label = str(candidate.get("island_label") or "").strip()
        if search_policy == "island" and (not allowed_islands or island_label not in allowed_islands):
            issues.append("HARNESS_CANDIDATE.island_label is required for island search")

if require_validation:
    validation = parse_metadata("HARNESS_VALIDATION")
    if validation is not None:
        if not str(validation.get("candidate_id") or "").strip():
            issues.append("HARNESS_VALIDATION.candidate_id is required")
        if not str(validation.get("claim") or "").strip():
            issues.append("HARNESS_VALIDATION.claim is required")
        tests = validation.get("tests")
        if not isinstance(tests, list) or not tests:
            issues.append("HARNESS_VALIDATION.tests must be a non-empty list")
        thresholds = validation.get("thresholds")
        if not isinstance(thresholds, list) or not thresholds:
            issues.append("HARNESS_VALIDATION.thresholds must be a non-empty list")
        verdict = str(validation.get("verdict") or "").strip()
        if allowed_verdicts and verdict not in allowed_verdicts:
            issues.append("HARNESS_VALIDATION.verdict is missing or invalid")

if issues:
    label = "FAILED" if mode == "fail" else "WARNING"
    print(f"[harness] Metadata policy {label}: " + "; ".join(issues))
    if allowed_mutations:
        print("[harness] Allowed mutation_type values: " + ", ".join(sorted(allowed_mutations)))
    if allowed_islands:
        print("[harness] Allowed island_label values: " + ", ".join(sorted(allowed_islands)))
    print("[harness] mutation_type is separate from island_label; e.g. baseline_refine uses local edit, robustness_simplify uses simplify or robustify, validation_protocol uses validation change.")
    print("[harness] Add exactly one line inside /app/starter_code/code.py like:")
    print('[harness] # HARNESS_CANDIDATE: {"candidate_id":"c001","parent_candidate_id":null,"mutation_type":"new hypothesis","island_label":"new_hypothesis","hypothesis":"short rationale"}')
    if require_validation:
        print('[harness] # HARNESS_VALIDATION: {"candidate_id":"c001","claim":"...","tests":["..."],"thresholds":["..."],"expected_failures":["..."],"change_mind_if":"...","verdict":"pass"}')
    sys.exit(2 if mode == "fail" else 0)

print("[harness] Metadata policy passed.")
PY
    HARNESS_METADATA_RC=$?
    if [ "$HARNESS_METADATA_RC" -eq 2 ]; then
        cat > "$CHECK_OUTPUT_PATH" << EOF
{
    "score": 0,
    "message": "Harness metadata policy failed. Use an allowed mutation_type enum, add the required HARNESS_CANDIDATE/HARNESS_VALIDATION comment inside /app/starter_code/code.py, and rerun /check."
}
EOF
        CHECK_NUMBER_PADDED=$(printf "%03d" "$CHECK_NUMBER")
        cp "$CHECK_OUTPUT_PATH" /logs/checker/result_${CHECK_NUMBER}.json 2>/dev/null || true
        if [ -f "/app/starter_code/code.py" ]; then
            cp /app/starter_code/code.py /logs/checker/code_${CHECK_NUMBER_PADDED}.py
        fi
        echo ""
        echo "[CHECK $CHECK_NUMBER] Score: 0.00"
        echo "Harness metadata policy failed before evaluation. Use an allowed mutation_type enum, add the required harness metadata comment inside /app/starter_code/code.py, and rerun /check."
        exit 0
    fi
fi

# Run checker evaluation (uses training data, no LLM judge)
# Set DEBUG_MODE=1 to generate training plots
# Use stdbuf for line-buffered output (live streaming to Harbor viewer)
cd /tests
# stdbuf may not exist on all platforms (e.g. Modal) — fall back to unbuffered python
if command -v stdbuf &> /dev/null; then
    DEBUG_MODE=1 stdbuf -oL python -u evaluate.py --checker 2>&1 | tee /logs/checker/output_${CHECK_NUMBER}.log
else
    DEBUG_MODE=1 python -u evaluate.py --checker 2>&1 | tee /logs/checker/output_${CHECK_NUMBER}.log
fi

# Move result to CHECK_OUTPUT_PATH and print to stdout
if [ -f /app/output/check_result.json ]; then
    cp /app/output/check_result.json "$CHECK_OUTPUT_PATH"

    # Also save to logs/checker for persistence
    cp /app/output/check_result.json /logs/checker/result_${CHECK_NUMBER}.json
    CHECK_NUMBER_PADDED=$(printf "%03d" "$CHECK_NUMBER")
    if [ -f /app/starter_code/code.py ]; then
        cp /app/starter_code/code.py /logs/checker/code_${CHECK_NUMBER_PADDED}.py
    fi

    # Copy intermediate plots to logs so they persist (with check number prefix)
    # Checker writes to /app/output/checker_plots_${CHECK_NUMBER}/
    PLOTS_DIR="/app/output/checker_plots_${CHECK_NUMBER}"
    if [ -d "$PLOTS_DIR" ]; then
        for f in "$PLOTS_DIR"/*.png; do
            [ -f "$f" ] && cp "$f" "/logs/checker/check_${CHECK_NUMBER}_$(basename "$f")"
        done
        echo "[check] Copied plots from $PLOTS_DIR to /logs/checker/"
    fi

    # Copy debug parquets if they exist
    DEBUG_DIR="/app/output/debug_${CHECK_NUMBER}"
    if [ -d "$DEBUG_DIR" ]; then
        mkdir -p "/logs/checker/debug_${CHECK_NUMBER}"
        cp -f "$DEBUG_DIR"/*.parquet "/logs/checker/debug_${CHECK_NUMBER}/" 2>/dev/null || true
        echo "[check] Copied debug parquets from $DEBUG_DIR to /logs/checker/"
    fi

    # Extract and print score and message for agent to see
    SCORE=$(python3 -c "import json; r=json.load(open('/app/output/check_result.json')); print(f\"{r.get('score', 0):.2f}\")")
    MESSAGE=$(python3 -c "import json; r=json.load(open('/app/output/check_result.json')); print(r.get('message', ''))")

    echo ""
    echo "========================================"
    echo "[CHECK $CHECK_NUMBER] Score: $SCORE"
    echo "========================================"
    echo ""
    echo "$MESSAGE"
    echo ""
else
    # Fallback if evaluation failed
    cat > "$CHECK_OUTPUT_PATH" << EOF
{
    "score": 0,
    "message": "Evaluation failed - check your code for errors. Run your code manually to see error messages."
}
EOF
    echo ""
    echo "[CHECK $CHECK_NUMBER] Score: 0.00"
    echo "Evaluation failed - check your code for errors."
fi
