#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR"/../.. && pwd)"
source "$REPO_ROOT/scripts/lib/asset_paths.sh"

BIN="$SCRIPT_DIR/build/infer_qwen"
MODEL_DIR="${QWEN_MODEL_DIR:-}"
PROMPT="${PROMPT:-How can I maintain my fitness routine while traveling for work?}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-64}"
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
  cmake --build "$SCRIPT_DIR/build" --target infer_qwen -j
fi

MODEL_DIR="${MODEL_DIR:-$(mf_resolve_model_dir qwen "$SCRIPT_DIR/pretrained")}"
if [ ! -d "$MODEL_DIR" ]; then
  echo "Could not find Qwen model directory: $MODEL_DIR" >&2
  exit 1
fi

"$BIN" \
  --model_dir "$MODEL_DIR" \
  --prompt "$PROMPT" \
  --max_new_tokens "$MAX_NEW_TOKENS" \
  --temperature "$TEMPERATURE" \
  --top_p "$TOP_P" \
  --repetition_penalty "$REPETITION_PENALTY" \
  --chat_template "$CHAT_TEMPLATE" \
  --thinking "$THINKING" \
  --use_kv_cache "$USE_KV_CACHE" \
  --echo_prompt "$ECHO_PROMPT"
