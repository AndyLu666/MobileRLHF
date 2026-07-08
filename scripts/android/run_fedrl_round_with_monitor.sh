#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")"/../.. && pwd)"
source "$ROOT/scripts/android/android_env.sh"

ADB="$(mf_android_resolve_adb)"
PKG="${PKG:-com.mobilefinetuner.visualizer}"
OBJECTIVE="${OBJECTIVE:-dpo}"
ROUND="${ROUND:-001}"
STEPS="${STEPS:-10}"
SEQ_LEN="${SEQ_LEN:-64}"
RANK="${RANK:-2}"
ALPHA="${ALPHA:-4}"
LR="${LR:-0.001}"
EVAL_INTERVAL="${EVAL_INTERVAL:-4}"
EVAL_BATCHES="${EVAL_BATCHES:-1}"
CACHE_REF_LOGPS="${CACHE_REF_LOGPS:-true}"
POLL_INTERVAL_SECONDS="${POLL_INTERVAL_SECONDS:-15}"
TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-7200}"
OUT_ROOT="${OUT_ROOT:-$ROOT/runs/fedrl_round_${ROUND}_${OBJECTIVE}}"
PREVIOUS_GLOBAL_ADAPTER="${PREVIOUS_GLOBAL_ADAPTER:-}"
DEVICE_GLOBAL_ADAPTER="/data/user/0/$PKG/files/federated/global_lora_adapter.jsonl"

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

sanitize_serial() {
  printf '%s' "$1" | tr ':/' '__'
}

epoch_ms() {
  python3 -c 'import time; print(int(time.time() * 1000))'
}

latest_run_name() {
  local serial="$1"
  "$ADB" -s "$serial" shell \
    "run-as '$PKG' sh -c 'if [ -d files/mft_runs ]; then cd files/mft_runs && ls -td *${OBJECTIVE}* 2>/dev/null | head -1; fi'" \
    | tr -d '\r' || true
}

report_status() {
  local serial="$1"
  local run_name="$2"
  "$ADB" -s "$serial" shell \
    "run-as '$PKG' sh -c 'test -f files/mft_runs/$run_name/report.json && sed -n \"s/.*\\\"status\\\"[[:space:]]*:[[:space:]]*\\\"\\([^\\\"]*\\)\\\".*/\\1/p\" files/mft_runs/$run_name/report.json | head -1'" \
    | tr -d '\r' || true
}

metric_field() {
  local json="$1"
  local field="$2"
  python3 -c 'import json,sys
raw=sys.stdin.read().strip()
field=sys.argv[1]
if not raw:
    print("")
else:
    try:
        value=json.loads(raw).get(field, "")
        print(value)
    except Exception:
        print("")
' "$field" <<< "$json"
}

monitor_device() {
  local serial="$1"
  local monitor_file="$2"
  local start_epoch_ms
  start_epoch_ms="$(epoch_ms)"
  printf 'timestamp_utc,epoch_ms,elapsed_s,serial,pid,vmrss_kb,vmhwm_kb,latest_run,metrics_rows,latest_step,latest_loss,latest_reward_accuracy,latest_reward_margin,report_exists,report_status\n' > "$monitor_file"
  while true; do
    local epoch_ms elapsed_s timestamp pid status_text vmrss vmhwm run_name metrics_rows last_metric latest_step latest_loss latest_acc latest_margin report_exists status
    epoch_ms="$(epoch_ms)"
    elapsed_s="$(( (epoch_ms - start_epoch_ms) / 1000 ))"
    timestamp="$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
    pid="$("$ADB" -s "$serial" shell "pidof '$PKG' 2>/dev/null | awk '{print \$1}'" | tr -d '\r' || true)"
    vmrss=""
    vmhwm=""
    if [ -n "$pid" ]; then
      status_text="$("$ADB" -s "$serial" shell "cat /proc/$pid/status 2>/dev/null | grep -E 'VmRSS|VmHWM'" | tr -d '\r' || true)"
      vmrss="$(awk '/VmRSS/ {print $2}' <<< "$status_text")"
      vmhwm="$(awk '/VmHWM/ {print $2}' <<< "$status_text")"
    fi
    run_name="$(latest_run_name "$serial")"
    metrics_rows=""
    last_metric=""
    report_exists="0"
    status=""
    if [ -n "$run_name" ]; then
      metrics_rows="$("$ADB" -s "$serial" shell "run-as '$PKG' sh -c 'wc -l < files/mft_runs/$run_name/metrics.ndjson 2>/dev/null || true'" | tr -d '\r ' || true)"
      last_metric="$("$ADB" -s "$serial" shell "run-as '$PKG' sh -c 'tail -1 files/mft_runs/$run_name/metrics.ndjson 2>/dev/null || true'" | tr -d '\r' || true)"
      if "$ADB" -s "$serial" shell "run-as '$PKG' test -f files/mft_runs/$run_name/report.json" >/dev/null 2>&1; then
        report_exists="1"
        status="$(report_status "$serial" "$run_name")"
      fi
    fi
    latest_step="$(metric_field "$last_metric" step)"
    latest_loss="$(metric_field "$last_metric" loss)"
    latest_acc="$(metric_field "$last_metric" reward_accuracy)"
    latest_margin="$(metric_field "$last_metric" reward_margin)"
    printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
      "$timestamp" "$epoch_ms" "$elapsed_s" "$serial" "$pid" "$vmrss" "$vmhwm" \
      "$run_name" "$metrics_rows" "$latest_step" "$latest_loss" "$latest_acc" "$latest_margin" \
      "$report_exists" "$status" >> "$monitor_file"
    sleep "$POLL_INTERVAL_SECONDS"
  done
}

mkdir -p "$OUT_ROOT/monitor"
echo "[FedRL] objective=$OBJECTIVE round=$ROUND devices=${#SERIALS[@]} out=$OUT_ROOT"

PREVIOUS_RUNS=()
RUN_NAMES=()
for serial in "${SERIALS[@]}"; do
  previous_run="$(latest_run_name "$serial")"
  PREVIOUS_RUNS+=("$previous_run")
  RUN_NAMES+=("")
  echo "[FedRL] $serial previous_latest=${previous_run:-none}"
done

if [ -n "$PREVIOUS_GLOBAL_ADAPTER" ]; then
  echo "[FedRL] distributing previous global adapter: $PREVIOUS_GLOBAL_ADAPTER"
  FEDRL_SERIALS="${SERIALS[*]}" "$ROOT/scripts/android/distribute_global_adapter.sh" "$PREVIOUS_GLOBAL_ADAPTER"
else
  echo "[FedRL] no previous global adapter supplied; clearing device global adapter"
  FEDRL_SERIALS="${SERIALS[*]}" "$ROOT/scripts/android/clear_global_adapter.sh"
fi

MONITOR_PIDS=()
cleanup() {
  for pid in "${MONITOR_PIDS[@]:-}"; do
    kill "$pid" >/dev/null 2>&1 || true
  done
}
trap cleanup EXIT

objective_upper="$(printf '%s' "$OBJECTIVE" | tr '[:lower:]' '[:upper:]')"
for serial in "${SERIALS[@]}"; do
  monitor_file="$OUT_ROOT/monitor/$(sanitize_serial "$serial").csv"
  monitor_device "$serial" "$monitor_file" &
  MONITOR_PIDS+=("$!")
done

for serial in "${SERIALS[@]}"; do
  echo "[FedRL] starting $OBJECTIVE round $ROUND on $serial"
  "$ADB" -s "$serial" shell am force-stop "$PKG"
  "$ADB" -s "$serial" shell am start -n "$PKG/.MainActivity" \
    --es fedrl_objective "$OBJECTIVE" \
    --ez fedrl_start_native_demo true \
    --es fedrl_preset_name "Fed${objective_upper}-${ROUND}" \
    --ei fedrl_steps "$STEPS" \
    --ei fedrl_seq_len "$SEQ_LEN" \
    --ei fedrl_rank "$RANK" \
    --ei fedrl_alpha "$ALPHA" \
    --ef fedrl_lr "$LR" \
    --ei fedrl_eval_interval "$EVAL_INTERVAL" \
    --ei fedrl_eval_batches "$EVAL_BATCHES" \
    --ez fedrl_cache_ref_logps "$CACHE_REF_LOGPS" \
    --es fedrl_global_adapter_path "$DEVICE_GLOBAL_ADAPTER" >/dev/null
done

start_s="$(date +%s)"
while true; do
  complete=0
  failed=0
  for idx in "${!SERIALS[@]}"; do
    serial="${SERIALS[$idx]}"
    previous_run="${PREVIOUS_RUNS[$idx]}"
    run_name="$(latest_run_name "$serial")"
    status=""
    if [ -n "$run_name" ] && [ "$run_name" != "$previous_run" ]; then
      if [ -z "${RUN_NAMES[$idx]}" ]; then
        RUN_NAMES[$idx]="$run_name"
      fi
      status="$(report_status "$serial" "$run_name")"
    fi
    if [ "$status" = "ok" ]; then
      complete="$((complete + 1))"
    elif [ "$status" = "failed" ] || [ "$status" = "error" ]; then
      failed="$((failed + 1))"
    fi
    echo "[FedRL] $serial latest=${run_name:-none} status=${status:-starting_or_running}"
  done
  if [ "$failed" -gt 0 ]; then
    echo "[FedRL] at least one client failed" >&2
    exit 2
  fi
  if [ "$complete" -eq "${#SERIALS[@]}" ]; then
    break
  fi
  now_s="$(date +%s)"
  if [ "$((now_s - start_s))" -gt "$TIMEOUT_SECONDS" ]; then
    echo "[FedRL] timeout after ${TIMEOUT_SECONDS}s" >&2
    exit 3
  fi
  sleep 30
done

cleanup
trap - EXIT

echo "[FedRL] collecting exact run artifacts"
for idx in "${!SERIALS[@]}"; do
  serial="${SERIALS[$idx]}"
  run_name="${RUN_NAMES[$idx]}"
  if [ -z "$run_name" ]; then
    echo "[FedRL] missing run name for $serial" >&2
    exit 4
  fi
  device_dir="$OUT_ROOT/$serial/$run_name"
  mkdir -p "$device_dir"
  for file in report.json metrics.ndjson reference_logps.jsonl "qwen_${OBJECTIVE}_lora_adapter.jsonl"; do
    echo "[FedRL] collecting $serial $run_name $file"
    "$ADB" -s "$serial" exec-out run-as "$PKG" cat "files/mft_runs/$run_name/$file" > "$device_dir/$file"
  done
done

adapters=()
while IFS= read -r adapter; do
  adapters+=("$adapter")
done < <(find "$OUT_ROOT" -name "qwen_${OBJECTIVE}_lora_adapter.jsonl" -print | sort)
if [ "${#adapters[@]}" -ne "${#SERIALS[@]}" ]; then
  echo "[FedRL] expected ${#SERIALS[@]} adapters, found ${#adapters[@]}" >&2
  exit 4
fi
aggregate_args=()
for adapter in "${adapters[@]}"; do
  aggregate_args+=(--input "$adapter")
done
python3 "$ROOT/scripts/federated/aggregate_lora_jsonl.py" \
  "${aggregate_args[@]}" \
  --output "$OUT_ROOT/global_${OBJECTIVE}_lora_adapter.jsonl"
python3 "$ROOT/scripts/federated/summarize_fedrl_round.py" --round-dir "$OUT_ROOT"

echo "[FedRL] round complete"
echo "[FedRL] global adapter: $OUT_ROOT/global_${OBJECTIVE}_lora_adapter.jsonl"
echo "[FedRL] summary: $OUT_ROOT/device_summary.csv"
