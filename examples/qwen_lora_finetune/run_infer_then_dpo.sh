#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR"/../.. && pwd)"
source "$REPO_ROOT/scripts/lib/asset_paths.sh"

RUN_ID="${RUN_ID:-qwen3_infer_then_gpt2_dpo_smoke}"
OUT_DIR_INPUT="${OUT_DIR:-$REPO_ROOT/runs/$RUN_ID}"
case "$OUT_DIR_INPUT" in
  /*) OUT_DIR="$OUT_DIR_INPUT" ;;
  *) OUT_DIR="$REPO_ROOT/$OUT_DIR_INPUT" ;;
esac

QWEN_OUT="$OUT_DIR/qwen_inference"
DPO_OUT="$OUT_DIR/dpo_training"
mkdir -p "$QWEN_OUT" "$DPO_OUT"

MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-4}"
PROMPT_LIMIT="${PROMPT_LIMIT:-2}"
INFER_TRAIN_STEPS="${INFER_TRAIN_STEPS:-0}"
TEMPERATURE="${TEMPERATURE:-0.7}"
TOP_P="${TOP_P:-0.9}"

GPT2_BUILD_DIR="${GPT2_BUILD_DIR:-$REPO_ROOT/gpt2_small_lora_finetune/build_release_blas}"
GPT2_MODEL_DIR="${GPT2_MODEL_DIR:-$REPO_ROOT/gpt2_small_lora_finetune/pretrained/gpt2}"
GPT2_BIN="$GPT2_BUILD_DIR/train"

DPO_STEPS="${DPO_STEPS:-2}"
DPO_SEQ_LEN="${DPO_SEQ_LEN:-128}"
DPO_LR="${DPO_LR:-0.001}"
DPO_RANK="${DPO_RANK:-1}"
DPO_ALPHA="${DPO_ALPHA:-2}"
DPO_EVAL_INTERVAL="${DPO_EVAL_INTERVAL:-1}"
DPO_EVAL_BATCHES="${DPO_EVAL_BATCHES:-1}"
DPO_PAIR_FORWARD="${DPO_PAIR_FORWARD:-0}"
DPO_CACHE_REF_LOGPS="${DPO_CACHE_REF_LOGPS:-0}"
DPO_SEED="${DPO_SEED:-42}"

echo "[1/3] Generate Qwen A/B feedback JSONL"
OUT_DIR="$QWEN_OUT" \
MAX_NEW_TOKENS="$MAX_NEW_TOKENS" \
PROMPT_LIMIT="$PROMPT_LIMIT" \
TRAIN_STEPS="$INFER_TRAIN_STEPS" \
TEMPERATURE="$TEMPERATURE" \
TOP_P="$TOP_P" \
"$SCRIPT_DIR/run_hfrl_infer_sim.sh"

TRAIN_PREF="$QWEN_OUT/inference_feedback.jsonl"
EVAL_PREF="${EVAL_PREF:-$REPO_ROOT/review-stage/data/hf_fitness_pref_eval.jsonl}"
if [ ! -s "$TRAIN_PREF" ]; then
  echo "No inference feedback was generated at $TRAIN_PREF" >&2
  exit 1
fi
if [ ! -f "$EVAL_PREF" ]; then
  echo "Missing eval preference JSONL: $EVAL_PREF" >&2
  exit 1
fi

cp "$TRAIN_PREF" "$DPO_OUT/inference_feedback.jsonl"

echo "[2/3] Build native GPT-2 DPO trainer if needed"
if [ ! -x "$GPT2_BIN" ]; then
  cmake -S "$REPO_ROOT/examples/gpt2_small_lora_finetune" -B "$GPT2_BUILD_DIR" -DUSE_BLAS=ON -DCMAKE_BUILD_TYPE=Release
  cmake --build "$GPT2_BUILD_DIR" --target train -j2
fi

echo "[3/3] Train DPO from Qwen-generated feedback"
(
  cd "$DPO_OUT"
  "$GPT2_BIN" \
    --train_mode dpo \
    --pretrained_dir "$GPT2_MODEL_DIR" \
    --preference_train "$TRAIN_PREF" \
    --preference_valid "$EVAL_PREF" \
    --lora_out "$DPO_OUT/dpo_from_qwen_feedback.safetensors" \
    --metrics_out "$DPO_OUT/metrics.ndjson" \
    --eval_out "$DPO_OUT/eval_metrics.jsonl" \
    --steps "$DPO_STEPS" \
    --batch_size 1 \
    --grad_accum_steps 1 \
    --seq_len "$DPO_SEQ_LEN" \
    --rank "$DPO_RANK" \
    --alpha "$DPO_ALPHA" \
    --lr "$DPO_LR" \
    --log_interval 1 \
    --eval_interval "$DPO_EVAL_INTERVAL" \
    --eval_batches "$DPO_EVAL_BATCHES" \
    --eval_batch_size 1 \
    --dpo_pair_forward "$DPO_PAIR_FORWARD" \
    --cache_ref_logps "$DPO_CACHE_REF_LOGPS" \
    --seed "$DPO_SEED" \
    > "$DPO_OUT/train.log" 2>&1
)

cat > "$OUT_DIR/SUMMARY.md" <<EOF
# Qwen Inference to DPO Smoke

- Qwen feedback: \`$TRAIN_PREF\`
- DPO run dir: \`$DPO_OUT\`
- DPO steps: \`$DPO_STEPS\`
- DPO trainer: GPT-2 native DPO
- Eval data: \`$EVAL_PREF\`

This proves the current local bridge: Qwen generates A/B candidates, the selected
pair format is saved as DPO-compatible JSONL, and native DPO training consumes
that JSONL as \`--preference_train\`.
EOF

echo "[Done] $OUT_DIR"
echo "  feedback: $TRAIN_PREF"
echo "  dpo log : $DPO_OUT/train.log"
