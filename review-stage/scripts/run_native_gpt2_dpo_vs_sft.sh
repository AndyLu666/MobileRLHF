#!/bin/bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PYTHON="${PYTHON:-python3}"
BUILD_DIR="${BUILD_DIR:-$ROOT/gpt2_small_lora_finetune/build_release_blas}"
MODEL_DIR="${MODEL_DIR:-$ROOT/gpt2_small_lora_finetune/pretrained/gpt2}"
TRAIN_PREF_INPUT="${TRAIN_PREF:-$ROOT/review-stage/data/real_mmlu_pref_train.jsonl}"
EVAL_PREF_INPUT="${EVAL_PREF:-$ROOT/review-stage/data/real_mmlu_pref_eval.jsonl}"
OUT_DIR_INPUT="${OUT_DIR:-$ROOT/runs/native_gpt2_dpo_vs_sft_real_mmlu_s384}"
SEQ_LEN="${SEQ_LEN:-384}"
STEPS="${STEPS:-60}"
BATCH_SIZE="${BATCH_SIZE:-1}"
GRAD_ACCUM="${GRAD_ACCUM:-1}"
LR="${LR:-5e-4}"
RANK="${RANK:-8}"
ALPHA="${ALPHA:-16}"
SEED="${SEED:-42}"
DPO_BETA="${DPO_BETA:-0.1}"

case "$OUT_DIR_INPUT" in
  /*) OUT_DIR="$OUT_DIR_INPUT" ;;
  *) OUT_DIR="$ROOT/$OUT_DIR_INPUT" ;;
esac

case "$MODEL_DIR" in
  /*) ;;
  *) MODEL_DIR="$ROOT/$MODEL_DIR" ;;
esac

case "$TRAIN_PREF_INPUT" in
  /*) TRAIN_PREF="$TRAIN_PREF_INPUT" ;;
  *) TRAIN_PREF="$ROOT/$TRAIN_PREF_INPUT" ;;
esac

case "$EVAL_PREF_INPUT" in
  /*) EVAL_PREF="$EVAL_PREF_INPUT" ;;
  *) EVAL_PREF="$ROOT/$EVAL_PREF_INPUT" ;;
esac

mkdir -p "$OUT_DIR"

SFT_TRAIN_JSONL="$OUT_DIR/train_chosen_only.jsonl"
SFT_EVAL_JSONL="$OUT_DIR/eval_chosen_only.jsonl"
DPO_LORA="$OUT_DIR/dpo_adapter_native.safetensors"
SFT_LORA="$OUT_DIR/sft_adapter_native.safetensors"
DPO_PEFT_DIR="$OUT_DIR/dpo_adapter_peft"
SFT_PEFT_DIR="$OUT_DIR/sft_adapter_peft"

echo "[1/10] Ensure BLAS-enabled native train binary is built"
cmake --build "$BUILD_DIR" --target train -j2

echo "[2/10] Convert preference train set to chosen-only masked JSONL"
"$PYTHON" "$ROOT/review-stage/scripts/preference_to_masked_jsonl.py" \
  --input "$TRAIN_PREF" \
  --output "$SFT_TRAIN_JSONL" \
  --tokenizer_dir "$MODEL_DIR" \
  --seq_len "$SEQ_LEN" \
  | tee "$OUT_DIR/sft_train_data_report.json"

echo "[3/10] Convert preference eval set to chosen-only masked JSONL"
"$PYTHON" "$ROOT/review-stage/scripts/preference_to_masked_jsonl.py" \
  --input "$EVAL_PREF" \
  --output "$SFT_EVAL_JSONL" \
  --tokenizer_dir "$MODEL_DIR" \
  --seq_len "$SEQ_LEN" \
  | tee "$OUT_DIR/sft_eval_data_report.json"

echo "[4/10] Train native GPT-2 LoRA with DPO"
cd "$BUILD_DIR"
/usr/bin/time -p ./train \
  --train_mode dpo \
  --pretrained_dir "$MODEL_DIR" \
  --preference_train "$TRAIN_PREF" \
  --preference_valid "$EVAL_PREF" \
  --lora_out "$DPO_LORA" \
  --eval_out "$OUT_DIR/dpo_eval_metrics.jsonl" \
  --steps "$STEPS" \
  --batch_size "$BATCH_SIZE" \
  --grad_accum_steps "$GRAD_ACCUM" \
  --seq_len "$SEQ_LEN" \
  --rank "$RANK" \
  --alpha "$ALPHA" \
  --lr "$LR" \
  --log_interval 10 \
  --eval_interval "$STEPS" \
  --eval_batches 50 \
  --eval_batch_size "$BATCH_SIZE" \
  --dpo_beta "$DPO_BETA" \
  --seed "$SEED" \
  > "$OUT_DIR/dpo_train.log" 2>&1

echo "[5/10] Train native GPT-2 LoRA with chosen-only SFT"
/usr/bin/time -p ./train \
  --train_mode sft \
  --pretrained_dir "$MODEL_DIR" \
  --jsonl_train "$SFT_TRAIN_JSONL" \
  --jsonl_valid "$SFT_EVAL_JSONL" \
  --lora_out "$SFT_LORA" \
  --eval_out "$OUT_DIR/sft_eval_metrics.jsonl" \
  --steps "$STEPS" \
  --batch_size "$BATCH_SIZE" \
  --grad_accum_steps "$GRAD_ACCUM" \
  --seq_len "$SEQ_LEN" \
  --rank "$RANK" \
  --alpha "$ALPHA" \
  --lr "$LR" \
  --log_interval 10 \
  --eval_interval "$STEPS" \
  --eval_batches 50 \
  --eval_batch_size "$BATCH_SIZE" \
  --seed "$SEED" \
  > "$OUT_DIR/sft_train.log" 2>&1

cd "$ROOT"

echo "[6/10] Convert native DPO LoRA to PEFT adapter directory"
PYTHONPATH="$ROOT" "$PYTHON" "$ROOT/review-stage/scripts/native_lora_to_peft_adapter.py" \
  --native_lora "$DPO_LORA" \
  --base_model_dir "$MODEL_DIR" \
  --output_dir "$DPO_PEFT_DIR" \
  > "$OUT_DIR/dpo_peft_convert.json"

echo "[7/10] Convert native SFT LoRA to PEFT adapter directory"
PYTHONPATH="$ROOT" "$PYTHON" "$ROOT/review-stage/scripts/native_lora_to_peft_adapter.py" \
  --native_lora "$SFT_LORA" \
  --base_model_dir "$MODEL_DIR" \
  --output_dir "$SFT_PEFT_DIR" \
  > "$OUT_DIR/sft_peft_convert.json"

echo "[8/10] Evaluate DPO adapter on held-out preference set"
PYTHONPATH="$ROOT" "$PYTHON" "$ROOT/review-stage/scripts/eval_preference_model.py" \
  --base_model_dir "$MODEL_DIR" \
  --adapter_dir "$DPO_PEFT_DIR" \
  --preference_jsonl "$EVAL_PREF" \
  --seq_len "$SEQ_LEN" \
  > "$OUT_DIR/dpo_eval.json"

echo "[9/10] Evaluate SFT adapter on held-out preference set"
PYTHONPATH="$ROOT" "$PYTHON" "$ROOT/review-stage/scripts/eval_preference_model.py" \
  --base_model_dir "$MODEL_DIR" \
  --adapter_dir "$SFT_PEFT_DIR" \
  --preference_jsonl "$EVAL_PREF" \
  --seq_len "$SEQ_LEN" \
  > "$OUT_DIR/sft_eval.json"

echo "[10/10] Finished"
echo "Outputs in $OUT_DIR"
