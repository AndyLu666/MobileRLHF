#!/bin/bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PYTHON="${PYTHON:-$ROOT/pt_env/bin/python}"
BUILD_DIR="${BUILD_DIR:-$ROOT/gpt2_small_lora_finetune/build_release_blas}"
MODEL_DIR="${MODEL_DIR:-$ROOT/gpt2_small_lora_finetune/pretrained/gpt2}"
OUT_DIR_INPUT="${OUT_DIR:-$ROOT/runs/hfrl_health_ab_dpo_s220}"
SEQ_LEN="${SEQ_LEN:-192}"
TRAIN_SIZE="${TRAIN_SIZE:-512}"
EVAL_SIZE="${EVAL_SIZE:-128}"
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
RUN_SFT="${RUN_SFT:-0}"

case "$OUT_DIR_INPUT" in
  /*) OUT_DIR="$OUT_DIR_INPUT" ;;
  *) OUT_DIR="$ROOT/$OUT_DIR_INPUT" ;;
esac

mkdir -p "$OUT_DIR"

TRAIN_PREF="$OUT_DIR/health_ab_pref_train.jsonl"
EVAL_PREF="$OUT_DIR/health_ab_pref_eval.jsonl"
DPO_LORA="$OUT_DIR/dpo_adapter_native.safetensors"
METRICS_OUT="$OUT_DIR/metrics.ndjson"
EVAL_OUT="$OUT_DIR/eval_metrics.jsonl"
DPO_LOG="$OUT_DIR/dpo_train.log"
RUN_INFO="$OUT_DIR/RUN_INFO.md"

echo "[1/6] Build A/B-style health preference dataset"
"$PYTHON" "$ROOT/review-stage/scripts/build_health_ab_preference_dataset.py" \
  --tokenizer-dir "$MODEL_DIR" \
  --train-output "$TRAIN_PREF" \
  --eval-output "$EVAL_PREF" \
  --train-size "$TRAIN_SIZE" \
  --eval-size "$EVAL_SIZE" \
  --seq-len "$SEQ_LEN" \
  --seed "$SEED" \
  | tee "$OUT_DIR/dataset_report.json"

echo "[2/6] Build native GPT-2 train binary"
cmake --build "$BUILD_DIR" --target train -j2

echo "[3/6] Run native DPO 220-step health A/B baseline"
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

SFT_STATUS="skipped"
if [[ "$RUN_SFT" != "0" ]]; then
  echo "[4/6] Optional SFT baseline requested"
  SFT_JSONL="$OUT_DIR/train_chosen_only.jsonl"
  SFT_LORA="$OUT_DIR/sft_adapter_native.safetensors"
  "$PYTHON" "$ROOT/review-stage/scripts/preference_to_masked_jsonl.py" \
    --input "$TRAIN_PREF" \
    --output "$SFT_JSONL" \
    --tokenizer_dir "$MODEL_DIR" \
    --seq_len "$SEQ_LEN" \
    | tee "$OUT_DIR/sft_data_report.json"
  cd "$BUILD_DIR"
  /usr/bin/time -p ./train \
    --train_mode sft \
    --pretrained_dir "$MODEL_DIR" \
    --jsonl_train "$SFT_JSONL" \
    --jsonl_valid "$SFT_JSONL" \
    --lora_out "$SFT_LORA" \
    --eval_out "$OUT_DIR/sft_eval_metrics.jsonl" \
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
    --seed "$SEED" \
    > "$OUT_DIR/sft_train.log" 2>&1
  cd "$ROOT"
  SFT_STATUS="completed"
else
  echo "[4/6] Optional SFT baseline skipped"
fi

echo "[5/6] Summarize run metrics"
"$PYTHON" - "$OUT_DIR" "$SFT_STATUS" <<'PY'
import json
import re
import sys
from pathlib import Path

out_dir = Path(sys.argv[1])
sft_status = sys.argv[2]

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
    "sft_status": sft_status,
}
(out_dir / "SUMMARY.json").write_text(json.dumps(summary, indent=2) + "\n")

lines = [
    "# HFRL Health A/B DPO 220-Step Baseline",
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
    f"- train_loop_seconds: `{summary['train_loop_seconds']}`",
    f"- ref_cache_seconds: `{summary['ref_cache_seconds']}`",
    f"- sft_status: `{summary['sft_status']}`",
    "",
    "This run uses deterministic synthetic A/B health preference data as simulated human feedback.",
]
(out_dir / "SUMMARY.md").write_text("\n".join(lines) + "\n")
print((out_dir / "SUMMARY.md").read_text())
PY

echo "[6/6] Write run metadata"
cat > "$RUN_INFO" <<EOF
# HFRL Health A/B DPO 220-Step Baseline

This run trains native GPT-2 LoRA with DPO on deterministic synthetic A/B health preference data.

Key files:

- \`health_ab_pref_train.jsonl\`: simulated A/B feedback train pairs.
- \`health_ab_pref_eval.jsonl\`: simulated A/B feedback eval pairs.
- \`dpo_train.log\`: native trainer log.
- \`metrics.ndjson\`: Android-readable per-step HFRL metrics.
- \`eval_metrics.jsonl\`: periodic DPO eval metrics.
- \`dpo_adapter_native.safetensors\`: native LoRA output.
- \`SUMMARY.md\`: compact result summary.

Configuration:

- steps: \`$STEPS\`
- seq_len: \`$SEQ_LEN\`
- train_size/eval_size: \`$TRAIN_SIZE/$EVAL_SIZE\`
- dpo_pair_forward: \`$DPO_PAIR_FORWARD\`
- cache_ref_logps: \`$CACHE_REF_LOGPS\`
- log_interval: \`$LOG_INTERVAL\`
- sft_status: \`$SFT_STATUS\`
EOF

echo "Outputs in $OUT_DIR"
