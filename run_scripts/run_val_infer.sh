#!/usr/bin/env bash
set -euo pipefail

# Simple inference script for running evaluation on a saved validation JSONL
# using a specific trained checkpoint (e.g., checkpoint-2000).
# Adjust variables below as needed or override via environment variables.

# -----------------------------------------------------------------------------
# Configurable variables (can be overridden: VAR=value ./run_val_infer.sh)
# -----------------------------------------------------------------------------
REPO_HOME="${REPO_HOME:-$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )}"
VAL_JSONL="${VAL_JSONL:-${REPO_HOME}/output/val_split.jsonl}"  # Path to validation JSONL
EXP_NAME="${EXP_NAME:-Qwen2.5-VL-3B-Instruct-rec}"              # Training experiment name
CHECKPOINT_STEP="${CHECKPOINT_STEP:-2000}"                     # Which checkpoint number to load
BATCH_SIZE="${BATCH_SIZE:-2}"                                  # Per inference batch size
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-256}"                        # Max generated tokens
DTYPE="${DTYPE:-bfloat16}"                                     # float32|float16|bfloat16
DEVICE="${DEVICE:-cuda}"                                       # cuda or cpu
LIMIT="${LIMIT:-}"                                             # Optional: limit number of samples
OUT_ROOT="${OUT_ROOT:-${REPO_HOME}/runs/infer/${EXP_NAME}}"
TIMESTAMP=$(date +%Y%m%d-%H%M%S)
OUT_DIR="${OUT_ROOT}/ckpt-${CHECKPOINT_STEP}-${TIMESTAMP}"
RESULT_JSON="${OUT_DIR}/val_infer_results.json"
SAMPLES_DIR="${OUT_DIR}/samples"
MODEL_PATH="${MODEL_PATH:-${REPO_HOME}/output/CLEVR_test/checkpoint-${CHECKPOINT_STEP}}"
PYTHON_BIN="${PYTHON_BIN:-python}"

# -----------------------------------------------------------------------------
# Sanity checks
# -----------------------------------------------------------------------------
if [ ! -f "${VAL_JSONL}" ]; then
  echo "[ERROR] Validation JSONL not found: ${VAL_JSONL}" >&2
  exit 1
fi
if [ ! -d "${MODEL_PATH}" ]; then
  echo "[WARN] Model checkpoint directory not found: ${MODEL_PATH}" >&2
  echo "       Check CHECKPOINT_STEP / EXP_NAME / MODEL_PATH variables." >&2
fi
mkdir -p "${OUT_DIR}" || true

# Record config
cat > "${OUT_DIR}/config_run.txt" <<EOF
REPO_HOME=${REPO_HOME}
VAL_JSONL=${VAL_JSONL}
EXP_NAME=${EXP_NAME}
CHECKPOINT_STEP=${CHECKPOINT_STEP}
MODEL_PATH=${MODEL_PATH}
BATCH_SIZE=${BATCH_SIZE}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS}
DTYPE=${DTYPE}
DEVICE=${DEVICE}
LIMIT=${LIMIT}
EOF

echo "[INFO] Running inference"
echo "  JSONL:        ${VAL_JSONL}"
echo "  MODEL_PATH:   ${MODEL_PATH}"
echo "  OUTPUT DIR:   ${OUT_DIR}"
echo "  BATCH_SIZE:   ${BATCH_SIZE}"
echo "  MAX_NEW_TOKENS: ${MAX_NEW_TOKENS}"

LIMIT_ARG=""
if [ -n "${LIMIT}" ]; then
  LIMIT_ARG="--limit ${LIMIT}"
fi

set -x
${PYTHON_BIN} ${REPO_HOME}/src/eval/test_jsonl_infer.py \
  --jsonl "${VAL_JSONL}" \
  --model_path "${MODEL_PATH}" \
  --output_json "${RESULT_JSON}" \
  --output_samples_dir "${SAMPLES_DIR}" \
  --batch_size "${BATCH_SIZE}" \
  --max_new_tokens "${MAX_NEW_TOKENS}" \
  --dtype "${DTYPE}" \
  --device "${DEVICE}" \
  --first100
  ${LIMIT_ARG}
set +x

echo "[INFO] Inference complete. Results: ${RESULT_JSON}"
