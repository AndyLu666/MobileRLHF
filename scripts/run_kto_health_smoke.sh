#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$ROOT_DIR/pt_env/bin/python3}"
export PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONPYCACHEPREFIX="${PYTHONPYCACHEPREFIX:-$ROOT_DIR/.pycache}"

TRAIN_DPO="$ROOT_DIR/review-stage/data/health_dpo_curated_train.jsonl"
EVAL_DPO="$ROOT_DIR/review-stage/data/health_dpo_curated_eval.jsonl"
TRAIN_KTO="$ROOT_DIR/review-stage/data/health_kto_train.jsonl"
EVAL_KTO="$ROOT_DIR/review-stage/data/health_kto_eval.jsonl"
RUN_ROOT="$ROOT_DIR/runs/kto_health_smoke"

mkdir -p "$RUN_ROOT"

echo "[1/3] Convert curated DPO pairs to KTO binary feedback"
"$PYTHON_BIN" "$ROOT_DIR/scripts/convert_dpo_to_kto_jsonl.py" \
  --input "$TRAIN_DPO" \
  --output "$TRAIN_KTO" \
  --stats "$RUN_ROOT/convert_train_stats.json"

"$PYTHON_BIN" "$ROOT_DIR/scripts/convert_dpo_to_kto_jsonl.py" \
  --input "$EVAL_DPO" \
  --output "$EVAL_KTO" \
  --stats "$RUN_ROOT/convert_eval_stats.json"

echo "[2/3] Run KTO smoke: batch_size=4, max_steps=5"
"$PYTHON_BIN" -m pytorch_alignment.kto_cli \
  --backend qwen \
  --kto_jsonl "$TRAIN_KTO" \
  --eval_jsonl "$EVAL_KTO" \
  --out_dir "$ROOT_DIR/runs/kto_qwen3_0_6b_health_b4_s5" \
  --max_steps 5 \
  --batch_size 4 \
  --seq_len 128 \
  --beta 0.1 \
  --lora_rank 2 \
  --lora_alpha 4

echo "[3/3] Run KTO smoke: batch_size=2, max_steps=10"
"$PYTHON_BIN" -m pytorch_alignment.kto_cli \
  --backend qwen \
  --kto_jsonl "$TRAIN_KTO" \
  --eval_jsonl "$EVAL_KTO" \
  --out_dir "$ROOT_DIR/runs/kto_qwen3_0_6b_health_b2_s10" \
  --max_steps 10 \
  --batch_size 2 \
  --seq_len 128 \
  --beta 0.1 \
  --lora_rank 2 \
  --lora_alpha 4

echo "KTO health smoke complete."
