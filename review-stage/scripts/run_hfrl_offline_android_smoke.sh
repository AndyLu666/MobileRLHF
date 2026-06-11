#!/bin/bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PYTHON="${PYTHON:-$ROOT/pt_env/bin/python}"
BUILD_DIR="${BUILD_DIR:-$ROOT/gpt2_small_lora_finetune/build}"
MODEL_DIR="${MODEL_DIR:-$ROOT/gpt2_small_lora_finetune/pretrained/gpt2}"
OUT_DIR_INPUT="${OUT_DIR:-$ROOT/runs/hfrl_offline_android_smoke}"
SEQ_LEN="${SEQ_LEN:-128}"
TRAIN_SIZE="${TRAIN_SIZE:-16}"
EVAL_SIZE="${EVAL_SIZE:-8}"
STEPS="${STEPS:-2}"
BATCH_SIZE="${BATCH_SIZE:-1}"
LR="${LR:-5e-4}"
RANK="${RANK:-8}"
ALPHA="${ALPHA:-16}"
SEED="${SEED:-42}"
DPO_BETA="${DPO_BETA:-0.1}"
EVAL_INTERVAL="${EVAL_INTERVAL:-0}"
EVAL_BATCHES="${EVAL_BATCHES:-1}"

case "$OUT_DIR_INPUT" in
  /*) OUT_DIR="$OUT_DIR_INPUT" ;;
  *) OUT_DIR="$ROOT/$OUT_DIR_INPUT" ;;
esac

mkdir -p "$OUT_DIR"

TRAIN_PREF="$OUT_DIR/fake_human_feedback_train.jsonl"
EVAL_PREF="$OUT_DIR/fake_human_feedback_eval.jsonl"
LORA_OUT="$OUT_DIR/hfrl_adapter_native.safetensors"
METRICS_OUT="$OUT_DIR/metrics.ndjson"
TRAIN_LOG="$OUT_DIR/train.log"
RUN_INFO="$OUT_DIR/RUN_INFO.md"

echo "[1/4] Build fake human-feedback preference subset"
"$PYTHON" "$ROOT/review-stage/scripts/build_real_mmlu_short_pref_subset.py" \
  --pretrained-dir "$MODEL_DIR" \
  --seq-len "$SEQ_LEN" \
  --train-size "$TRAIN_SIZE" \
  --eval-size "$EVAL_SIZE" \
  --train-output "$TRAIN_PREF" \
  --eval-output "$EVAL_PREF"

echo "[2/4] Ensure native GPT-2 train binary is built"
cmake --build "$BUILD_DIR" --target train -j2

echo "[3/4] Run offline HFRL smoke through native DPO"
cd "$BUILD_DIR"
/usr/bin/time -p ./train \
  --train_mode dpo \
  --pretrained_dir "$MODEL_DIR" \
  --preference_train "$TRAIN_PREF" \
  --preference_valid "$EVAL_PREF" \
  --lora_out "$LORA_OUT" \
  --metrics_out "$METRICS_OUT" \
  --eval_out "$OUT_DIR/eval_metrics.jsonl" \
  --seq_len "$SEQ_LEN" \
  --batch_size "$BATCH_SIZE" \
  --steps "$STEPS" \
  --rank "$RANK" \
  --alpha "$ALPHA" \
  --lr "$LR" \
  --log_interval 1 \
  --eval_interval "$EVAL_INTERVAL" \
  --eval_batches "$EVAL_BATCHES" \
  --dpo_beta "$DPO_BETA" \
  --seed "$SEED" \
  > "$TRAIN_LOG" 2>&1

cd "$ROOT"

echo "[4/4] Write Android run-folder metadata"
cat > "$RUN_INFO" <<EOF
# Offline HFRL Android Smoke

This run uses preference JSONL as simulated human feedback and trains native GPT-2 LoRA with DPO.

Files:

- \`train.log\`: native trainer text log, parsed by Android visualizer.
- \`metrics.ndjson\`: per-step HFRL metrics, parsed by Android visualizer.
- \`fake_human_feedback_train.jsonl\`: simulated human-feedback train pairs.
- \`fake_human_feedback_eval.jsonl\`: simulated human-feedback eval pairs.
- \`hfrl_adapter_native.safetensors\`: native LoRA adapter output.

To view on Android, push or expose the parent \`runs/\` folder and select it in the visualizer.
EOF

echo "Offline HFRL Android smoke finished."
echo "Run folder: $OUT_DIR"
