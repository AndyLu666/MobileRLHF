#!/bin/bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PYTHON="${PYTHON:-$ROOT/pt_env/bin/python}"
BUILD_DIR="${BUILD_DIR:-$ROOT/gpt2_small_lora_finetune/build_release_blas}"
MODEL_DIR="${MODEL_DIR:-$ROOT/gpt2_small_lora_finetune/pretrained/gpt2}"
OUT_DIR_INPUT="${OUT_DIR:-$ROOT/runs/hfrl_sparse_feedback_dpo_s220}"
SEQ_LEN="${SEQ_LEN:-192}"
STEPS="${STEPS:-220}"
BATCH_SIZE="${BATCH_SIZE:-1}"
GRAD_ACCUM="${GRAD_ACCUM:-1}"
LR="${LR:-5e-4}"
RANK="${RANK:-8}"
ALPHA="${ALPHA:-16}"
SEED="${SEED:-20260608}"
DPO_BETA="${DPO_BETA:-0.1}"
CACHE_REF_LOGPS="${CACHE_REF_LOGPS:-0}"
DPO_PAIR_FORWARD="${DPO_PAIR_FORWARD:-1}"
EVAL_INTERVAL="${EVAL_INTERVAL:-55}"
EVAL_BATCHES="${EVAL_BATCHES:-32}"
LOG_INTERVAL="${LOG_INTERVAL:-1}"

SPARSE_TRAIN_SIZE="${SPARSE_TRAIN_SIZE:-768}"
SPARSE_EVAL_SIZE="${SPARSE_EVAL_SIZE:-128}"
SPARSE_SEED_CLICKS="${SPARSE_SEED_CLICKS:-12}"
SPARSE_AMPLIFICATION="${SPARSE_AMPLIFICATION:-64}"
SPARSE_EVAL_SEED_CLICKS="${SPARSE_EVAL_SEED_CLICKS:-4}"
SPARSE_EVAL_AMPLIFICATION="${SPARSE_EVAL_AMPLIFICATION:-32}"

case "$OUT_DIR_INPUT" in
  /*) OUT_DIR="$OUT_DIR_INPUT" ;;
  *) OUT_DIR="$ROOT/$OUT_DIR_INPUT" ;;
esac

mkdir -p "$OUT_DIR"

TRAIN_PREF="$OUT_DIR/sparse_health_ab_pref_train.jsonl"
EVAL_PREF="$OUT_DIR/sparse_health_ab_pref_eval.jsonl"
DPO_LORA="$OUT_DIR/dpo_adapter_native.safetensors"
DPO_PEFT_DIR="$OUT_DIR/dpo_adapter_peft"
METRICS_OUT="$OUT_DIR/metrics.ndjson"
EVAL_OUT="$OUT_DIR/eval_metrics.jsonl"
DPO_LOG="$OUT_DIR/dpo_train.log"
RUN_INFO="$OUT_DIR/RUN_INFO.md"

echo "[1/7] Build sparse-feedback-amplified health preference dataset"
"$PYTHON" "$ROOT/review-stage/scripts/build_sparse_feedback_amplified_dataset.py" \
  --tokenizer-dir "$MODEL_DIR" \
  --train-output "$TRAIN_PREF" \
  --eval-output "$EVAL_PREF" \
  --seed-clicks "$SPARSE_SEED_CLICKS" \
  --amplification "$SPARSE_AMPLIFICATION" \
  --eval-seed-clicks "$SPARSE_EVAL_SEED_CLICKS" \
  --eval-amplification "$SPARSE_EVAL_AMPLIFICATION" \
  --seq-len "$SEQ_LEN" \
  --seed "$SEED" \
  | tee "$OUT_DIR/dataset_report.json"

echo "[2/7] Build native GPT-2 train binary"
cmake --build "$BUILD_DIR" --target train -j2

echo "[3/7] Run native DPO on sparse-feedback amplified data"
cd "$BUILD_DIR"
/usr/bin/time -p ./train \
  --train_mode dpo \
  --pretrained_dir "$MODEL_DIR" \
  --preference_train "$TRAIN_PREF" \
  --preference_valid "$EVAL_PREF" \
  --lora_out "$DPO_LORA" \
  --metrics_out "$METRICS_OUT" \
  --eval_out "$EVAL_OUT" \
  --steps "$STEPS" \
  --batch_size "$BATCH_SIZE" \
  --grad_accum_steps "$GRAD_ACCUM" \
  --seq_len "$SEQ_LEN" \
  --rank "$RANK" \
  --alpha "$ALPHA" \
  --lr "$LR" \
  --log_interval "$LOG_INTERVAL" \
  --eval_interval "$EVAL_INTERVAL" \
  --eval_batches "$EVAL_BATCHES" \
  --eval_batch_size "$BATCH_SIZE" \
  --dpo_beta "$DPO_BETA" \
  --dpo_pair_forward "$DPO_PAIR_FORWARD" \
  --cache_ref_logps "$CACHE_REF_LOGPS" \
  --seed "$SEED" \
  > "$DPO_LOG" 2>&1
cd "$ROOT"

echo "[4/7] Convert native DPO LoRA to PEFT adapter directory"
PYTHONPATH="$ROOT" "$PYTHON" "$ROOT/review-stage/scripts/native_lora_to_peft_adapter.py" \
  --native_lora "$DPO_LORA" \
  --base_model_dir "$MODEL_DIR" \
  --output_dir "$DPO_PEFT_DIR" \
  > "$OUT_DIR/dpo_peft_convert.json"

echo "[5/7] Evaluate base and adapter with length-normalized preference metrics"
"$PYTHON" "$ROOT/review-stage/scripts/eval_preference_model.py" \
  --base_model_dir "$MODEL_DIR" \
  --preference_jsonl "$EVAL_PREF" \
  --seq_len "$SEQ_LEN" \
  | tee "$OUT_DIR/base_eval.json"

"$PYTHON" "$ROOT/review-stage/scripts/eval_preference_model.py" \
  --base_model_dir "$MODEL_DIR" \
  --adapter_dir "$DPO_PEFT_DIR" \
  --preference_jsonl "$EVAL_PREF" \
  --seq_len "$SEQ_LEN" \
  | tee "$OUT_DIR/dpo_eval_length_norm.json"

echo "[6/7] Summarize run metrics"
"$PYTHON" - "$OUT_DIR" "$EVAL_PREF" <<'PY'
import json
import re
import sys
from pathlib import Path

out_dir = Path(sys.argv[1])
eval_pref = Path(sys.argv[2])

def read_ndjson(path):
    if not path.exists():
        return []
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]

def find_float(pattern, text, default=None):
    m = re.search(pattern, text, re.MULTILINE)
    return float(m.group(1)) if m else default

metrics = read_ndjson(out_dir / "metrics.ndjson")
evals = read_ndjson(out_dir / "eval_metrics.jsonl")
log_text = (out_dir / "dpo_train.log").read_text() if (out_dir / "dpo_train.log").exists() else ""
base_eval = json.loads((out_dir / "base_eval.json").read_text()) if (out_dir / "base_eval.json").exists() else {}
dpo_eval = json.loads((out_dir / "dpo_eval_length_norm.json").read_text()) if (out_dir / "dpo_eval_length_norm.json").exists() else {}
last = metrics[-1] if metrics else {}
last_eval = evals[-1] if evals else {}
summary = {
    "status": "ok" if metrics else "missing_metrics",
    "steps_logged": len(metrics),
    "final_train_loss": last.get("loss"),
    "final_train_reward_accuracy": last.get("reward_accuracy"),
    "final_train_reward_margin": last.get("reward_margin"),
    "final_eval_step": last_eval.get("step"),
    "final_eval_loss": last_eval.get("loss"),
    "final_eval_reward_accuracy": last_eval.get("reward_accuracy"),
    "final_eval_reward_margin": last_eval.get("reward_margin"),
    "train_loop_seconds": find_float(r"\[Timing\]\s+train_loop_seconds\s+([0-9.]+)", log_text),
    "ref_cache_seconds": find_float(r"\[Timing\]\s+ref_cache_seconds\s+([0-9.]+)", log_text, 0.0),
    "base_length_norm_accuracy": base_eval.get("length_normalized_pair_accuracy"),
    "base_length_norm_margin": base_eval.get("mean_length_normalized_pair_margin"),
    "dpo_length_norm_accuracy": dpo_eval.get("length_normalized_pair_accuracy"),
    "dpo_length_norm_margin": dpo_eval.get("mean_length_normalized_pair_margin"),
}
(out_dir / "SUMMARY.json").write_text(json.dumps(summary, indent=2) + "\n")

lines = [
    "# HFRL Sparse-Feedback DPO 220-Step Baseline",
    "",
    f"- status: `{summary['status']}`",
    f"- steps_logged: `{summary['steps_logged']}`",
    f"- final_train_loss: `{summary['final_train_loss']}`",
    f"- final_train_reward_accuracy: `{summary['final_train_reward_accuracy']}`",
    f"- final_train_reward_margin: `{summary['final_train_reward_margin']}`",
    f"- final_eval_step: `{summary['final_eval_step']}`",
    f"- final_eval_loss: `{summary['final_eval_loss']}`",
    f"- final_eval_reward_accuracy: `{summary['final_eval_reward_accuracy']}`",
    f"- final_eval_reward_margin: `{summary['final_eval_reward_margin']}`",
    f"- base_length_norm_accuracy: `{summary['base_length_norm_accuracy']}`",
    f"- base_length_norm_margin: `{summary['base_length_norm_margin']}`",
    f"- dpo_length_norm_accuracy: `{summary['dpo_length_norm_accuracy']}`",
    f"- dpo_length_norm_margin: `{summary['dpo_length_norm_margin']}`",
    f"- train_loop_seconds: `{summary['train_loop_seconds']}`",
    f"- ref_cache_seconds: `{summary['ref_cache_seconds']}`",
    "",
    "This run uses sparse simulated user clicks expanded into a larger training set.",
]
(out_dir / "SUMMARY.md").write_text("\n".join(lines) + "\n")
print((out_dir / "SUMMARY.md").read_text())
print(json.dumps(summary, indent=2))
PY

echo "[7/7] Write run metadata"
cat > "$RUN_INFO" <<EOF
# HFRL Sparse-Feedback DPO 220-Step Baseline

This run trains native GPT-2 LoRA with DPO on sparse-feedback-amplified synthetic health A/B data.

Key files:

- \`sparse_health_ab_pref_train.jsonl\`: amplified train pairs built from few simulated clicks.
- \`sparse_health_ab_pref_eval.jsonl\`: amplified eval pairs built from few simulated clicks.
- \`dpo_train.log\`: native trainer log.
- \`metrics.ndjson\`: Android-readable per-step HFRL metrics.
- \`eval_metrics.jsonl\`: periodic DPO eval metrics.
- \`base_eval.json\`: length-normalized eval for the frozen base model.
- \`dpo_eval_length_norm.json\`: length-normalized eval for the trained adapter.
- \`dpo_adapter_native.safetensors\`: native LoRA output.
- \`dpo_adapter_peft/\`: PEFT adapter directory for Python evaluation.
- \`SUMMARY.md\`: compact result summary.

Configuration:

- steps: \`$STEPS\`
- seq_len: \`$SEQ_LEN\`
- sparse_seed_clicks: \`$SPARSE_SEED_CLICKS\`
- sparse_amplification: \`$SPARSE_AMPLIFICATION\`
- sparse_eval_seed_clicks: \`$SPARSE_EVAL_SEED_CLICKS\`
- sparse_eval_amplification: \`$SPARSE_EVAL_AMPLIFICATION\`
- dpo_pair_forward: \`$DPO_PAIR_FORWARD\`
- cache_ref_logps: \`$CACHE_REF_LOGPS\`
- log_interval: \`$LOG_INTERVAL\`
EOF

echo "[7/7] Outputs in $OUT_DIR"
