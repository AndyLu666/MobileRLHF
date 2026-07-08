#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")"/../.. && pwd)"
source "$ROOT/scripts/android/android_env.sh"

ADB="$(mf_android_resolve_adb)"
PKG="${PKG:-com.mobilefinetuner.visualizer}"
OBJECTIVE="${OBJECTIVE:-dpo}"
ROUND="${ROUND:-001}"
OUT_ROOT="${OUT_ROOT:-$ROOT/runs/fedrl_round_${ROUND}_${OBJECTIVE}}"

case "$OBJECTIVE" in
  dpo|kto) ;;
  *)
    echo "OBJECTIVE must be 'dpo' or 'kto'; got '$OBJECTIVE'" >&2
    exit 1
    ;;
esac

SERIALS=()
if [ -n "${FEDRL_SERIALS:-}" ]; then
  # shellcheck disable=SC2206
  SERIALS=($FEDRL_SERIALS)
else
  while IFS= read -r serial; do
    SERIALS+=("$serial")
  done < <("$ADB" devices | awk 'NR > 1 && $2 == "device" {print $1}')
fi

if [ "${#SERIALS[@]}" -eq 0 ]; then
  echo "No adb devices found. Set FEDRL_SERIALS=\"serial1 serial2 serial3\" or connect phones." >&2
  exit 1
fi

mkdir -p "$OUT_ROOT"
echo "[FedRL] objective=$OBJECTIVE round=$ROUND out=$OUT_ROOT"

latest_run_name() {
  local serial="$1"
  "$ADB" -s "$serial" shell \
    "run-as '$PKG' sh -c 'if [ -d files/mft_runs ]; then cd files/mft_runs && ls -td *${OBJECTIVE}* 2>/dev/null | head -1; fi'" \
    | tr -d '\r' || true
}

for serial in "${SERIALS[@]}"; do
  echo "[FedRL] collecting from $serial"
  device_dir="$OUT_ROOT/$serial"
  mkdir -p "$device_dir"
  run_name="$(latest_run_name "$serial")"
  if [ -z "$run_name" ]; then
    echo "[FedRL] $serial has no private run matching objective '$OBJECTIVE'" >&2
    exit 2
  fi
  echo "[FedRL] latest run on $serial: $run_name"
  "$ADB" -s "$serial" exec-out run-as "$PKG" tar -C files/mft_runs -cf - "$run_name" \
    | tar -C "$device_dir" -xf -
  find "$device_dir/$run_name" -maxdepth 1 -type f \
    \( -name 'report.json' -o -name 'metrics.ndjson' -o -name 'reference_logps.jsonl' -o -name 'qwen_*_lora_adapter.jsonl' \) \
    -print | sort
done

echo "[FedRL] collection complete"
echo "[FedRL] aggregate example:"
echo "python3 $ROOT/scripts/federated/aggregate_lora_jsonl.py \\"
find "$OUT_ROOT" -name "qwen_${OBJECTIVE}_lora_adapter.jsonl" -print | sort | sed 's/^/  --input /; s/$/ \\/'
echo "  --output $OUT_ROOT/global_${OBJECTIVE}_lora_adapter.jsonl"
