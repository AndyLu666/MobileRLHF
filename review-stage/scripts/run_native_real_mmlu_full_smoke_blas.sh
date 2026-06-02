#!/bin/bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
BUILD_DIR="${BUILD_DIR:-$ROOT/gpt2_small_lora_finetune/build_release_blas}"
MODEL_DIR="${MODEL_DIR:-$ROOT/gpt2_small_lora_finetune/pretrained/gpt2}"
TRAIN_PREF="${TRAIN_PREF:-$ROOT/review-stage/data/real_mmlu_pref_train.jsonl}"
EVAL_PREF="${EVAL_PREF:-$ROOT/review-stage/data/real_mmlu_pref_eval.jsonl}"
SEQ_LEN="${SEQ_LEN:-384}"
STEPS="${STEPS:-1}"
BATCH_SIZE="${BATCH_SIZE:-1}"
SEED="${SEED:-42}"

echo "[1/2] Ensure BLAS-enabled native train binary is built"
cmake --build "$BUILD_DIR" --target train -j2

echo "[2/2] Run BLAS-enabled native GPT-2 DPO full real-MMLU smoke"
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
