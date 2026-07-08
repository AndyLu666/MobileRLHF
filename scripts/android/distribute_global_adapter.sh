#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")"/../.. && pwd)"
source "$ROOT/scripts/android/android_env.sh"

ADB="$(mf_android_resolve_adb)"
PKG="${PKG:-com.mobilefinetuner.visualizer}"
ADAPTER="${1:?Usage: distribute_global_adapter.sh <global_adapter.jsonl>}"

if [ ! -f "$ADAPTER" ]; then
  echo "[Error] Adapter file not found: $ADAPTER" >&2
  exit 1
fi

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

DEVICE_PATH="files/federated/global_lora_adapter.jsonl"
LOCAL_SIZE="$(wc -c < "$ADAPTER" | tr -d ' ')"
TMP_PATH="/data/local/tmp/fedrl_global_lora_adapter.jsonl"

for serial in "${SERIALS[@]}"; do
  echo "[FedRL] distributing to $serial"
  "$ADB" -s "$serial" push "$ADAPTER" "$TMP_PATH" >/dev/null
  "$ADB" -s "$serial" shell chmod 644 "$TMP_PATH"
  "$ADB" -s "$serial" shell "run-as '$PKG' sh -c 'mkdir -p files/federated && cp $TMP_PATH $DEVICE_PATH'"
  remote_size="$("$ADB" -s "$serial" shell "run-as '$PKG' sh -c 'wc -c < $DEVICE_PATH'" | tr -d '\r ')"
  echo "[FedRL] $serial: pushed $LOCAL_SIZE bytes → device reports $remote_size bytes"
done

echo "[FedRL] distribution complete (${#SERIALS[@]} devices)"
