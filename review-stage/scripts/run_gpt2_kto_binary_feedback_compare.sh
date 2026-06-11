#!/bin/bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PYTHON="${PYTHON:-$ROOT/pt_env/bin/python}"
MODEL_DIR="${MODEL_DIR:-$ROOT/gpt2_small_lora_finetune/pretrained/gpt2}"
OUT_ROOT_INPUT="${OUT_ROOT:-$ROOT/runs/gpt2_kto_binary_feedback_compare}"
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
KTO_DESIRABLE_WEIGHT="${KTO_DESIRABLE_WEIGHT:-1.0}"
KTO_UNDESIRABLE_WEIGHT="${KTO_UNDESIRABLE_WEIGHT:-1.0}"

case "$OUT_ROOT_INPUT" in
  /*) OUT_ROOT="$OUT_ROOT_INPUT" ;;
  *) OUT_ROOT="$ROOT/$OUT_ROOT_INPUT" ;;
esac

mkdir -p "$OUT_ROOT"

TRAIN_PREF="$OUT_ROOT/health_ab_pref_train.jsonl"
EVAL_PREF="$OUT_ROOT/health_ab_pref_eval.jsonl"
TRAIN_BINARY="$OUT_ROOT/health_ab_binary_train.jsonl"
EVAL_BINARY="$OUT_ROOT/health_ab_binary_eval.jsonl"
RUN_DIR="$OUT_ROOT/kto"
SUMMARY="$OUT_ROOT/SUMMARY.json"

echo "[1/4] Build health A/B preference dataset"
"$PYTHON" "$ROOT/review-stage/scripts/build_health_ab_preference_dataset.py" \
  --tokenizer-dir "$MODEL_DIR" \
  --train-output "$TRAIN_PREF" \
  --eval-output "$EVAL_PREF" \
  --train-size "$TRAIN_SIZE" \
  --eval-size "$EVAL_SIZE" \
  --seq-len "$SEQ_LEN" \
  --seed "$SEED" \
  > "$OUT_ROOT/dataset_report.json"

echo "[2/4] Convert pairwise preferences to binary feedback rows"
"$PYTHON" "$ROOT/review-stage/scripts/preference_to_binary_feedback_jsonl.py" \
  --input "$TRAIN_PREF" \
  --output "$TRAIN_BINARY" \
  > "$OUT_ROOT/train_binary_report.json"
"$PYTHON" "$ROOT/review-stage/scripts/preference_to_binary_feedback_jsonl.py" \
  --input "$EVAL_PREF" \
  --output "$EVAL_BINARY" \
  > "$OUT_ROOT/eval_binary_report.json"

mkdir -p "$RUN_DIR"
echo "[3/4] Train KTO-style binary feedback model"
PYTHONPATH="$ROOT" /usr/bin/time -p "$PYTHON" "$ROOT/pytorch_alignment/gpt2_lora_finetune.py" \
  --data_dir "$MODEL_DIR" \
  --pretrained_dir "$MODEL_DIR" \
  --train_mode kto \
  --binary_feedback_jsonl "$TRAIN_BINARY" \
  --binary_feedback_valid_jsonl "$EVAL_BINARY" \
  --lora_out "$RUN_DIR" \
  --eval_out "$RUN_DIR/eval_metrics.jsonl" \
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
  --kto_desirable_weight "$KTO_DESIRABLE_WEIGHT" \
  --kto_undesirable_weight "$KTO_UNDESIRABLE_WEIGHT" \
  --seed "$SEED" \
  > "$RUN_DIR/train.log" 2>&1

"$PYTHON" - "$OUT_ROOT" "$SUMMARY" <<'PY'
import json
import re
import sys
from pathlib import Path

out_root = Path(sys.argv[1])
summary_path = Path(sys.argv[2])
run_dir = out_root / "kto"

def read_json(path):
    return json.loads(path.read_text())

def read_ndjson(path):
    if not path.exists():
        return []
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]

def find_float(pattern, text, default=None):
    m = re.search(pattern, text, re.MULTILINE)
    return float(m.group(1)) if m else default

log_text = (run_dir / "train.log").read_text()
train_match = re.findall(
    r"\[Train\].*?loss\s+([-+0-9.eE]+).*?reward_acc\s+([-+0-9.eE]+).*?reward_margin\s+([-+0-9.eE]+)",
    log_text,
)
last_eval = read_ndjson(run_dir / "eval_metrics.jsonl")[-1]
dataset_report = read_json(out_root / "dataset_report.json")
train_binary_report = read_json(out_root / "train_binary_report.json")
eval_binary_report = read_json(out_root / "eval_binary_report.json")

summary = {
    "method": "kto_binary_feedback",
    "seed": dataset_report["seed"],
    "seq_len": dataset_report["seq_len"],
    "preference_train_pairs": dataset_report["train_size"],
    "preference_eval_pairs": dataset_report["eval_size"],
    "binary_train_rows": train_binary_report["binary_rows"],
    "binary_eval_rows": eval_binary_report["binary_rows"],
    "steps": last_eval["step"],
    "train_loss": float(train_match[-1][0]) if train_match else None,
    "train_reward_accuracy": float(train_match[-1][1]) if train_match else None,
    "train_reward_margin": float(train_match[-1][2]) if train_match else None,
    "eval_loss": last_eval["loss"],
    "eval_reward_accuracy": last_eval["reward_acc"],
    "eval_reward_margin": last_eval["reward_margin"],
    "eval_chosen_reward": last_eval["chosen_reward"],
    "eval_rejected_reward": last_eval["rejected_reward"],
    "train_loop_seconds": find_float(r"real\s+([0-9.]+)", log_text),
}
summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
print(json.dumps(summary, indent=2))
PY

echo "[4/4] Done"
cat "$SUMMARY"
