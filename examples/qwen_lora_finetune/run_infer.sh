#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR"/../.. && pwd)"
source "$REPO_ROOT/scripts/lib/asset_paths.sh"

BIN="$SCRIPT_DIR/build/infer_qwen"
MODEL_DIR="${QWEN_MODEL_DIR:-}"
PROMPT="${PROMPT:-How can I maintain my fitness routine while traveling for work?}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-64}"
ECHO_PROMPT="${ECHO_PROMPT:-0}"

if [ "${REBUILD:-0}" = "1" ] || [ ! -x "$BIN" ]; then
  cmake -S "$SCRIPT_DIR" -B "$SCRIPT_DIR/build"
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
  --echo_prompt "$ECHO_PROMPT"
