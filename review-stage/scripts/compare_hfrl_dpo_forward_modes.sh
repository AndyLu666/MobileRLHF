#!/bin/bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PYTHON="${PYTHON:-$ROOT/pt_env/bin/python}"
BUILD_DIR="${BUILD_DIR:-$ROOT/gpt2_small_lora_finetune/build_release_blas}"
MODEL_DIR="${MODEL_DIR:-$ROOT/gpt2_small_lora_finetune/pretrained/gpt2}"
OUT_ROOT_INPUT="${OUT_ROOT:-$ROOT/runs/hfrl_dpo_forward_mode_compare}"
SEQ_LEN="${SEQ_LEN:-128}"
TRAIN_SIZE="${TRAIN_SIZE:-16}"
EVAL_SIZE="${EVAL_SIZE:-8}"
STEPS="${STEPS:-4}"
BATCH_SIZE="${BATCH_SIZE:-1}"
LR="${LR:-5e-4}"
RANK="${RANK:-8}"
ALPHA="${ALPHA:-16}"
SEED="${SEED:-42}"
DPO_BETA="${DPO_BETA:-0.1}"
RUN_BASELINE="${RUN_BASELINE:-1}"
RUN_PAIRED="${RUN_PAIRED:-1}"
RUN_CACHED="${RUN_CACHED:-1}"

case "$OUT_ROOT_INPUT" in
  /*) OUT_ROOT="$OUT_ROOT_INPUT" ;;
  *) OUT_ROOT="$ROOT/$OUT_ROOT_INPUT" ;;
esac

mkdir -p "$OUT_ROOT"

TRAIN_PREF="$OUT_ROOT/fake_human_feedback_train.jsonl"
EVAL_PREF="$OUT_ROOT/fake_human_feedback_eval.jsonl"
SUMMARY_JSONL="$OUT_ROOT/summary.jsonl"
SUMMARY_MD="$OUT_ROOT/SUMMARY.md"

echo "[1/4] Build shared fake human-feedback preference subset"
"$PYTHON" "$ROOT/review-stage/scripts/build_real_mmlu_short_pref_subset.py" \
  --pretrained-dir "$MODEL_DIR" \
  --seq-len "$SEQ_LEN" \
  --train-size "$TRAIN_SIZE" \
  --eval-size "$EVAL_SIZE" \
  --train-output "$TRAIN_PREF" \
  --eval-output "$EVAL_PREF"

echo "[2/4] Build native train binary"
cmake --build "$BUILD_DIR" --target train -j2

rm -f "$SUMMARY_JSONL"

run_mode() {
  local name="$1"
  local pair_forward="$2"
  local cache_ref="$3"
  local run_dir="$OUT_ROOT/$name"
  mkdir -p "$run_dir"

  echo "[3/4] Run $name (pair_forward=$pair_forward cache_ref=$cache_ref)"
  cd "$BUILD_DIR"
  /usr/bin/time -p ./train \
    --train_mode dpo \
    --pretrained_dir "$MODEL_DIR" \
    --preference_train "$TRAIN_PREF" \
    --preference_valid "$EVAL_PREF" \
    --lora_out "$run_dir/hfrl_adapter_native.safetensors" \
    --metrics_out "$run_dir/metrics.ndjson" \
    --eval_out "$run_dir/eval_metrics.jsonl" \
    --seq_len "$SEQ_LEN" \
    --batch_size "$BATCH_SIZE" \
    --steps "$STEPS" \
    --rank "$RANK" \
    --alpha "$ALPHA" \
    --lr "$LR" \
    --log_interval 1 \
    --eval_interval 0 \
    --eval_batches 1 \
    --dpo_beta "$DPO_BETA" \
    --dpo_pair_forward "$pair_forward" \
    --cache_ref_logps "$cache_ref" \
    --seed "$SEED" \
    > "$run_dir/train.log" 2> "$run_dir/time.txt"
  cd "$ROOT"

  "$PYTHON" - "$name" "$run_dir" "$SUMMARY_JSONL" <<'PY'
import json
import re
import sys
from pathlib import Path

name = sys.argv[1]
run_dir = Path(sys.argv[2])
summary_path = Path(sys.argv[3])
train_log = (run_dir / "train.log").read_text()
time_txt = (run_dir / "time.txt").read_text()

def find_float(pattern, text, default=None):
    m = re.search(pattern, text)
    return float(m.group(1)) if m else default

def find_int(pattern, text, default=None):
    m = re.search(pattern, text)
    return int(m.group(1)) if m else default

metrics = []
metrics_path = run_dir / "metrics.ndjson"
if metrics_path.exists():
    for line in metrics_path.read_text().splitlines():
        line = line.strip()
        if line:
            metrics.append(json.loads(line))
last = metrics[-1] if metrics else {}

row = {
    "mode": name,
    "real_seconds": find_float(r"^real\s+([0-9.]+)", time_txt, None),
    "user_seconds": find_float(r"^user\s+([0-9.]+)", time_txt, None),
    "sys_seconds": find_float(r"^sys\s+([0-9.]+)", time_txt, None),
    "ref_cache_seconds": find_float(r"\[Timing\]\s+ref_cache_seconds\s+([0-9.]+)", train_log, 0.0),
    "train_loop_seconds": find_float(r"\[Timing\]\s+train_loop_seconds\s+([0-9.]+)", train_log, None),
    "steps": len(metrics),
    "final_loss": last.get("loss"),
    "final_reward_accuracy": last.get("reward_accuracy"),
    "final_reward_margin": last.get("reward_margin"),
    "final_grad_norm": last.get("grad_norm"),
    "nonzero_grad_tensors": find_int(r"nonzero_grad_tensors\s+([0-9]+)", train_log, None),
    "peak_rss_mb": last.get("peak_rss_mb") or last.get("rss_mb"),
}
with summary_path.open("a") as f:
    f.write(json.dumps(row, sort_keys=True) + "\n")
PY
}

if [[ "$RUN_BASELINE" != "0" ]]; then
  run_mode "baseline_4forward" 0 0
fi
if [[ "$RUN_PAIRED" != "0" ]]; then
  run_mode "paired_2forward" 1 0
fi
if [[ "$RUN_CACHED" != "0" ]]; then
  run_mode "cached_ref_1policy_forward" 1 1
fi

echo "[4/4] Write summary"
"$PYTHON" - "$SUMMARY_JSONL" "$SUMMARY_MD" <<'PY'
import json
import sys
from pathlib import Path

rows = [json.loads(x) for x in Path(sys.argv[1]).read_text().splitlines() if x.strip()]
out = ["# HFRL DPO Forward-Mode Performance Compare", ""]
out.append("| Mode | real_s | train_loop_s | ref_cache_s | final_loss | reward_acc | reward_margin | grad_norm | nonzero_grad | peak_rss_mb |")
out.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
for r in rows:
    out.append(
        f"| {r['mode']} | {r.get('real_seconds')} | {r.get('train_loop_seconds')} | "
        f"{r.get('ref_cache_seconds')} | {r.get('final_loss')} | {r.get('final_reward_accuracy')} | "
        f"{r.get('final_reward_margin')} | {r.get('final_grad_norm')} | "
        f"{r.get('nonzero_grad_tensors')} | {r.get('peak_rss_mb')} |"
    )
Path(sys.argv[2]).write_text("\n".join(out) + "\n")
print(Path(sys.argv[2]).read_text())
PY

echo "Outputs in $OUT_ROOT"
