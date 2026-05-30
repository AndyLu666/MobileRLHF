#!/bin/bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="${PYTHON:-python3}"
MODEL_DIR="${MODEL_DIR:-$ROOT/gpt2_small_lora_finetune/pretrained/gpt2}"
TRAIN_PREF="${TRAIN_PREF:-$ROOT/review-stage/data/synthetic_pref_from_sft_train.jsonl}"
EVAL_PREF="${EVAL_PREF:-$ROOT/review-stage/data/synthetic_pref_from_sft_eval.jsonl}"
OUT_DIR="${OUT_DIR:-$ROOT/runs/gpt2_dpo_vs_sft}"
SEQ_LEN="${SEQ_LEN:-128}"
STEPS="${STEPS:-60}"
BATCH_SIZE="${BATCH_SIZE:-2}"
GRAD_ACCUM="${GRAD_ACCUM:-1}"
LR="${LR:-5e-4}"

mkdir -p "$OUT_DIR"

SFT_JSONL="$OUT_DIR/train_chosen_only.jsonl"

echo "[1/5] Convert preference train set to chosen-only masked JSONL"
$PYTHON "$ROOT/review-stage/scripts/preference_to_masked_jsonl.py" \
  --input "$TRAIN_PREF" \
  --output "$SFT_JSONL" \
  --tokenizer_dir "$MODEL_DIR" \
  --seq_len "$SEQ_LEN"

echo "[2/5] Train GPT-2 LoRA with DPO"
PYTHONPATH="$ROOT" $PYTHON "$ROOT/pytorch_alignment/gpt2_lora_finetune.py" \
  --data_dir "$ROOT/data/wikitext2/wikitext-2-raw" \
  --pretrained_dir "$MODEL_DIR" \
  --train_mode dpo \
  --preference_jsonl "$TRAIN_PREF" \
  --preference_valid_jsonl "$EVAL_PREF" \
  --lora_out "$OUT_DIR/dpo_adapter" \
  --steps "$STEPS" \
  --batch_size "$BATCH_SIZE" \
  --grad_accum_steps "$GRAD_ACCUM" \
  --seq_len "$SEQ_LEN" \
  --rank 8 \
  --alpha 16 \
  --lr "$LR" \
  --log_interval 10 \
  --eval_interval "$STEPS" \
  --eval_batches 50 \
  --eval_batch_size "$BATCH_SIZE" \
  --dpo_beta 0.1 \
  > "$OUT_DIR/dpo_train.log" 2>&1

echo "[3/5] Train GPT-2 LoRA with chosen-only SFT"
PYTHONPATH="$ROOT" $PYTHON "$ROOT/pytorch_alignment/gpt2_lora_finetune.py" \
  --data_dir "$ROOT/data/wikitext2/wikitext-2-raw" \
  --pretrained_dir "$MODEL_DIR" \
  --jsonl_train "$SFT_JSONL" \
  --jsonl_valid "$SFT_JSONL" \
  --lora_out "$OUT_DIR/sft_adapter" \
  --steps "$STEPS" \
  --batch_size "$BATCH_SIZE" \
  --grad_accum_steps "$GRAD_ACCUM" \
  --seq_len "$SEQ_LEN" \
  --rank 8 \
  --alpha 16 \
  --lr "$LR" \
  --log_interval 10 \
  --eval_interval "$STEPS" \
  --eval_batches 50 \
  --eval_batch_size "$BATCH_SIZE" \
  > "$OUT_DIR/sft_train.log" 2>&1

echo "[4/5] Evaluate DPO adapter on held-out preference set"
PYTHONPATH="$ROOT" $PYTHON "$ROOT/review-stage/scripts/eval_preference_model.py" \
  --base_model_dir "$MODEL_DIR" \
  --adapter_dir "$OUT_DIR/dpo_adapter" \
  --preference_jsonl "$EVAL_PREF" \
  --seq_len "$SEQ_LEN" \
  > "$OUT_DIR/dpo_eval.json"

echo "[5/5] Evaluate SFT adapter on held-out preference set"
PYTHONPATH="$ROOT" $PYTHON "$ROOT/review-stage/scripts/eval_preference_model.py" \
  --base_model_dir "$MODEL_DIR" \
  --adapter_dir "$OUT_DIR/sft_adapter" \
  --preference_jsonl "$EVAL_PREF" \
  --seq_len "$SEQ_LEN" \
  > "$OUT_DIR/sft_eval.json"

echo "Done. Outputs in $OUT_DIR"
