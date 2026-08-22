#!/usr/bin/env bash
set -euo pipefail

show_help() {
  cat <<'EOF'
Write atomic health snapshots for one Linux/WSL long-running GPU command.

Required environment:
  HEARTBEAT_PATH=/path/to/heartbeat.tsv
  WATCH_PID=<runner pid>

Optional environment:
  TRAINING_LOG=/path/to/training_log.csv
  HEARTBEAT_INTERVAL_SECONDS=60
  HANG_TIMEOUT_SECONDS=1800
  DISK_PATH=/path/to/output-parent
  MIN_FREE_GIB=20

Usage:
  bash scripts/run_long_run_heartbeat.sh [--once]
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  show_help
  exit 0
fi

: "${HEARTBEAT_PATH:?Set HEARTBEAT_PATH.}"
: "${WATCH_PID:?Set WATCH_PID.}"

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=scripts/long_run_harness.sh
source "$repo_root/scripts/long_run_harness.sh"

interval="${HEARTBEAT_INTERVAL_SECONDS:-60}"
hang_timeout="${HANG_TIMEOUT_SECONDS:-1800}"
disk_path="${DISK_PATH:-$(dirname "$HEARTBEAT_PATH")}"
minimum_gib="${MIN_FREE_GIB:-20}"
training_log="${TRAINING_LOG:-}"
once="${1:-}"

write_snapshot() {
  local now_epoch log_age health gpu_values tmp_path free_bytes minimum_bytes
  now_epoch="$(date +%s)"
  log_age=""
  health="healthy"
  if [[ -n "$training_log" && -s "$training_log" ]]; then
    log_age="$((now_epoch - $(stat -c %Y "$training_log")))"
    if (( log_age > hang_timeout )); then
      health="stalled"
    fi
  fi
  free_bytes="$(harness_free_bytes "$disk_path")"
  minimum_bytes="$((minimum_gib * 1024 * 1024 * 1024))"
  if (( free_bytes < minimum_bytes )); then
    health="low_disk"
  fi
  gpu_values="unavailable"
  if command -v nvidia-smi >/dev/null 2>&1; then
    gpu_values="$(nvidia-smi --query-gpu=temperature.gpu,memory.used,memory.free,utilization.gpu --format=csv,noheader,nounits 2>/dev/null | head -n 1 | tr -d '\r' || true)"
  fi
  tmp_path="${HEARTBEAT_PATH}.tmp.$$"
  {
    printf 'key\tvalue\n'
    printf 'health\t%s\n' "$health"
    printf 'watch_pid\t%s\n' "$WATCH_PID"
    printf 'watch_process_alive\t%s\n' "$(harness_process_is_alive "$WATCH_PID" && echo true || echo false)"
    printf 'boot_id\t%s\n' "$(harness_boot_id)"
    printf 'updated_at\t%s\n' "$(harness_now)"
    printf 'training_log_age_seconds\t%s\n' "$log_age"
    printf 'free_bytes\t%s\n' "$free_bytes"
    printf 'gpu_temperature_memory_used_free_utilization\t%s\n' "${gpu_values//$'\t'/ }"
  } > "$tmp_path"
  mv "$tmp_path" "$HEARTBEAT_PATH"
}

while true; do
  write_snapshot
  [[ "$once" == "--once" ]] && exit 0
  harness_process_is_alive "$WATCH_PID" || exit 0
  sleep "$interval"
done
