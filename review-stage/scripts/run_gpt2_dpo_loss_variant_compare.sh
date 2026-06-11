#!/bin/bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PYTHON="${PYTHON:-$ROOT/pt_env/bin/python}"
MODEL_DIR="${MODEL_DIR:-$ROOT/gpt2_small_lora_finetune/pretrained/gpt2}"
OUT_ROOT_INPUT="${OUT_ROOT:-$ROOT/runs/gpt2_dpo_loss_variant_compare}"
SEQ_LEN="${SEQ_LEN:-192}"
TRAIN_SIZE="${TRAIN_SIZE:-128}"
EVAL_SIZE="${EVAL_SIZE:-64}"
STEPS="${STEPS:-40}"
BATCH_SIZE="${BATCH_SIZE:-1}"
LR="${LR:-5e-4}"
RANK="${RANK:-8}"
ALPHA="${ALPHA:-16}"
SEED="${SEED:-20260608}"
DPO_BETA="${DPO_BETA:-0.1}"

case "$OUT_ROOT_INPUT" in
  /*) OUT_ROOT="$OUT_ROOT_INPUT" ;;
  *) OUT_ROOT="$ROOT/$OUT_ROOT_INPUT" ;;
esac

mkdir -p "$OUT_ROOT"

TRAIN_PREF="$OUT_ROOT/health_ab_pref_train.jsonl"
EVAL_PREF="$OUT_ROOT/health_ab_pref_eval.jsonl"

echo "[1/4] Build health A/B preference dataset"
"$PYTHON" "$ROOT/review-stage/scripts/build_health_ab_preference_dataset.py" \
  --tokenizer-dir "$MODEL_DIR" \
  --train-output "$TRAIN_PREF" \
  --eval-output "$EVAL_PREF" \
  --train-size "$TRAIN_SIZE" \
  --eval-size "$EVAL_SIZE" \
  --seq-len "$SEQ_LEN" \
  --seed "$SEED" \
  | tee "$OUT_ROOT/dataset_report.json"

variants=("sigmoid" "robust" "ipo" "sigmoid_norm")
summary="$OUT_ROOT/SUMMARY.jsonl"
rm -f "$summary"

for variant in "${variants[@]}"; do
  run_dir="$OUT_ROOT/$variant"
  mkdir -p "$run_dir"
  echo "[3/4] Train variant=$variant"
  PYTHONPATH="$ROOT" /usr/bin/time -p "$PYTHON" "$ROOT/pytorch_alignment/gpt2_lora_finetune.py" \
    --data_dir "$ROOT/gpt2_small_lora_finetune/pretrained/gpt2" \
    --pretrained_dir "$MODEL_DIR" \
    --train_mode dpo \
    --preference_jsonl "$TRAIN_PREF" \
    --preference_valid_jsonl "$EVAL_PREF" \
    --lora_out "$run_dir" \
    --eval_out "$run_dir/eval_metrics.jsonl" \
    --steps "$STEPS" \
    --batch_size "$BATCH_SIZE" \
    --seq_len "$SEQ_LEN" \
    --rank "$RANK" \
    --alpha "$ALPHA" \
    --lr "$LR" \
    --log_interval 1 \
    --eval_interval "$STEPS" \
    --eval_batches 32 \
    --eval_batch_size "$BATCH_SIZE" \
    --dpo_beta "$DPO_BETA" \
    --seed "$SEED" \
    --dpo_loss_type "$variant" \
    > "$run_dir/train.log" 2>&1

  "$PYTHON" - "$run_dir" "$variant" "$summary" <<'PY'
import json
import re
import sys
from pathlib import Path

run_dir = Path(sys.argv[1])
variant = sys.argv[2]
summary = Path(sys.argv[3])

def read_ndjson(path):
    if not path.exists():
        return []
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]

def find_float(pattern, text, default=None):
    m = re.search(pattern, text, re.MULTILINE)
    return float(m.group(1)) if m else default

metrics = read_ndjson(run_dir / "metrics.ndjson")
evals = read_ndjson(run_dir / "eval_metrics.jsonl")
log_text = (run_dir / "train.log").read_text()
train_match = re.findall(
    r"\[Train\].*?loss\s+([-+0-9.eE]+).*?reward_acc\s+([-+0-9.eE]+).*?reward_margin\s+([-+0-9.eE]+)",
    log_text,
)
last = metrics[-1] if metrics else {}
last_eval = evals[-1] if evals else {}
train_loss = float(train_match[-1][0]) if train_match else last.get("loss")
train_reward_accuracy = float(train_match[-1][1]) if train_match else last.get("reward_accuracy")
train_reward_margin = float(train_match[-1][2]) if train_match else last.get("reward_margin")
row = {
    "variant": variant,
    "steps_logged": len(metrics),
    "train_loss": train_loss,
    "train_reward_accuracy": train_reward_accuracy,
    "train_reward_margin": train_reward_margin,
    "eval_step": last_eval.get("step"),
    "eval_loss": last_eval.get("loss"),
    "eval_reward_accuracy": last_eval.get("reward_acc"),
    "eval_reward_margin": last_eval.get("reward_margin"),
    "train_loop_seconds": find_float(r"\[Timing\]\s+train_loop_seconds\s+([0-9.]+)", log_text),
}
with summary.open("a", encoding="utf-8") as f:
    f.write(json.dumps(row) + "\n")
PY
done

echo "[4/4] Done"
cat "$summary"
