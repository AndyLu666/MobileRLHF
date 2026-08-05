#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")"/../.. && pwd)"

OBJECTIVE="${OBJECTIVE:-dpo}"
ROUND="${ROUND:-001}"
WEIGHT_KEY="${WEIGHT_KEY:-auto}"
OUT_ROOT="${OUT_ROOT:-$ROOT/runs/fedrl_round_${ROUND}_${OBJECTIVE}}"

case "$OBJECTIVE" in
  dpo|kto) ;;
  *)
    echo "OBJECTIVE must be 'dpo' or 'kto'; got '$OBJECTIVE'" >&2
    exit 1
    ;;
esac

echo "=== Federated Round $ROUND ($OBJECTIVE) ==="

echo "--- Step 1: Collect adapters from phones ---"
ROUND="$ROUND" OBJECTIVE="$OBJECTIVE" OUT_ROOT="$OUT_ROOT" \
  bash "$ROOT/scripts/android/collect_fedrl_round.sh"

echo "--- Step 2: FedAvg aggregation ---"
GLOBAL_ADAPTER="$OUT_ROOT/global_${OBJECTIVE}_lora_adapter.jsonl"
INPUTS=()
while IFS= read -r adapter; do
  INPUTS+=(--input "$adapter")
done < <(find "$OUT_ROOT" -name "qwen_${OBJECTIVE}_lora_adapter.jsonl" -print | sort)

if [ "${#INPUTS[@]}" -eq 0 ]; then
  echo "[Error] No adapters found to aggregate in $OUT_ROOT" >&2
  exit 1
fi

python3 "$ROOT/scripts/federated/aggregate_lora_jsonl.py" \
  "${INPUTS[@]}" \
  --output "$GLOBAL_ADAPTER" \
  --weight-key "$WEIGHT_KEY"

echo "[FedRL] global adapter: $GLOBAL_ADAPTER"

echo "--- Step 3: Distribute global adapter to phones ---"
bash "$ROOT/scripts/android/distribute_global_adapter.sh" "$GLOBAL_ADAPTER"

NEXT_ROUND="$(printf '%03d' $((10#$ROUND + 1)))"
echo "=== Round $ROUND complete. Phones are ready for round $NEXT_ROUND. ==="
