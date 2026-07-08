#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")"/../.. && pwd)"
source "$ROOT/scripts/android/android_env.sh"

ADB="$(mf_android_resolve_adb)"
PKG="${PKG:-com.mobilefinetuner.visualizer}"

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
  echo "No adb devices found." >&2
  exit 1
fi

for serial in "${SERIALS[@]}"; do
  "$ADB" -s "$serial" shell run-as "$PKG" rm -f files/federated/global_lora_adapter.jsonl
  echo "[FedRL] cleared global adapter on $serial"
done

echo "[FedRL] all devices reset to standalone training mode"
