#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR"/../.. && pwd)"
source "$REPO_ROOT/scripts/lib/asset_paths.sh"

BIN="$SCRIPT_DIR/build/hfrl_infer_sim"
MODEL_DIR="${QWEN_MODEL_DIR:-}"
OUT_DIR="${OUT_DIR:-$REPO_ROOT/runs/qwen3_hfrl_infer_sim}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-4}"
TRAIN_STEPS="${TRAIN_STEPS:-8}"
PROMPT_LIMIT="${PROMPT_LIMIT:-0}"
TEMPERATURE="${TEMPERATURE:-0}"
TOP_P="${TOP_P:-1.0}"
REPETITION_PENALTY="${REPETITION_PENALTY:-1.08}"
CHAT_TEMPLATE="${CHAT_TEMPLATE:-1}"
THINKING="${THINKING:-0}"
USE_KV_CACHE="${USE_KV_CACHE:-1}"
ECHO_PROMPT="${ECHO_PROMPT:-0}"
BUILD_TYPE="${BUILD_TYPE:-Release}"

if [ "${REBUILD:-0}" = "1" ] || [ ! -x "$BIN" ]; then
  cmake -S "$SCRIPT_DIR" -B "$SCRIPT_DIR/build" -DUSE_BLAS=ON -DCMAKE_BUILD_TYPE="$BUILD_TYPE"
  cmake --build "$SCRIPT_DIR/build" --target hfrl_infer_sim -j
fi

MODEL_DIR="${MODEL_DIR:-$(mf_resolve_model_dir qwen "$SCRIPT_DIR/pretrained")}"
if [ ! -d "$MODEL_DIR" ]; then
  echo "Could not find Qwen3 model directory: $MODEL_DIR" >&2
  exit 1
fi

"$BIN" \
  --model_dir "$MODEL_DIR" \
  --out_dir "$OUT_DIR" \
  --max_new_tokens "$MAX_NEW_TOKENS" \
  --train_steps "$TRAIN_STEPS" \
  --prompt_limit "$PROMPT_LIMIT" \
  --temperature "$TEMPERATURE" \
  --top_p "$TOP_P" \
  --repetition_penalty "$REPETITION_PENALTY" \
  --chat_template "$CHAT_TEMPLATE" \
  --thinking "$THINKING" \
  --use_kv_cache "$USE_KV_CACHE" \
  --echo_prompt "$ECHO_PROMPT"
