#!/bin/bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PYTHON="${PYTHON:-$ROOT/pt_env/bin/python}"
BUILD_DIR="${BUILD_DIR:-$ROOT/gpt2_small_lora_finetune/build_release}"
MODEL_DIR="${MODEL_DIR:-$ROOT/gpt2_small_lora_finetune/pretrained/gpt2}"
SEQ_LEN="${SEQ_LEN:-128}"
TRAIN_SIZE="${TRAIN_SIZE:-16}"
EVAL_SIZE="${EVAL_SIZE:-8}"
STEPS="${STEPS:-1}"
BATCH_SIZE="${BATCH_SIZE:-1}"
SEED="${SEED:-42}"

TRAIN_PREF="$ROOT/review-stage/data/real_mmlu_pref_short_train.jsonl"
EVAL_PREF="$ROOT/review-stage/data/real_mmlu_pref_short_eval.jsonl"

echo "[1/3] Build short real-MMLU preference subset"
"$PYTHON" "$ROOT/review-stage/scripts/build_real_mmlu_short_pref_subset.py" \
  --pretrained-dir "$MODEL_DIR" \
  --seq-len "$SEQ_LEN" \
  --train-size "$TRAIN_SIZE" \
  --eval-size "$EVAL_SIZE" \
  --train-output "$TRAIN_PREF" \
  --eval-output "$EVAL_PREF"

echo "[2/3] Ensure native train binary is built"
cmake --build "$BUILD_DIR" --target train -j2

echo "[3/3] Run native GPT-2 DPO short real-data smoke"
cd "$BUILD_DIR"
/usr/bin/time -p ./train \
  --train_mode dpo \
  --pretrained_dir "$MODEL_DIR" \
  --preference_train "$TRAIN_PREF" \
  --preference_valid "$EVAL_PREF" \
  --seq_len "$SEQ_LEN" \
  --batch_size "$BATCH_SIZE" \
  --steps "$STEPS" \
  --log_interval 1 \
  --eval_interval 0 \
  --seed "$SEED"
