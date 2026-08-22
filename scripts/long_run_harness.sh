#!/usr/bin/env bash

# Shared, side-effect-light helpers for the Linux/WSL GPU runners.  This file is
# sourced by the entrypoint scripts; it must not enable shell options or mutate
# the caller's working directory.

harness_boot_id() {
  if [[ -r /proc/sys/kernel/random/boot_id ]]; then
    tr -d '\r\n' < /proc/sys/kernel/random/boot_id
  else
    printf 'unknown'
  fi
}

harness_now() {
  # GNU and BSD date disagree on --iso-8601; this UTC form is portable and
  # remains lexically sortable in status/provenance files.
  date -u '+%Y-%m-%dT%H:%M:%SZ'
}

harness_command_line() {
  local rendered=()
  local item
  for item in "$@"; do
    printf -v item '%q' "$item"
    rendered+=("$item")
  done
  printf '%s' "${rendered[*]}"
}

harness_sha256() {
  sha256sum "$1" | awk '{print $1}'
}

harness_free_bytes() {
  # df requires an existing path. Walk upward without creating anything so a
  # preflight failure cannot leave a half-created experiment directory.
  local path="$1"
  while [[ ! -e "$path" && "$path" != "/" && "$path" != "." ]]; do
    path="$(dirname "$path")"
  done
  df -Pk "$path" | awk 'NR == 2 { printf "%.0f\n", $4 * 1024 }'
}

harness_require_free_space() {
  local path="$1"
  local minimum_gib="$2"
  local minimum_bytes_override="${3:-}"
  local free_bytes minimum_bytes
  free_bytes="$(harness_free_bytes "$path")"
  if [[ -n "$minimum_bytes_override" ]]; then
    minimum_bytes="$minimum_bytes_override"
  else
    minimum_bytes="$((minimum_gib * 1024 * 1024 * 1024))"
  fi
  if (( free_bytes < minimum_bytes )); then
    printf 'Insufficient free space: path=%s free_bytes=%s required_bytes=%s\n' \
      "$path" "$free_bytes" "$minimum_bytes" >&2
    return 1
  fi
}

harness_gpu_memory_mib() {
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    return 1
  fi
  nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits \
    | head -n 1 | tr -d '[:space:]'
}

harness_require_batch_hardware() {
  local batch_size="$1"
  local minimum_batch8_mib="${2:-12288}"
  if (( batch_size < 8 )); then
    return 0
  fi
  local gpu_memory_mib
  gpu_memory_mib="$(harness_gpu_memory_mib)" || {
    echo 'blocked_hardware: nvidia-smi could not report GPU memory.' >&2
    return 3
  }
  if (( gpu_memory_mib < minimum_batch8_mib )); then
    printf 'blocked_hardware: batch size %s requires at least %s MiB; detected %s MiB. No fallback batch size was selected.\n' \
      "$batch_size" "$minimum_batch8_mib" "$gpu_memory_mib" >&2
    return 3
  fi
}

harness_process_is_alive() {
  local pid="$1"
  [[ "$pid" =~ ^[0-9]+$ ]] && kill -0 "$pid" 2>/dev/null
}

harness_atomic_write_status() {
  local path="$1"
  local status="$2"
  local exit_code="${3:-}"
  local pid="${4:-$$}"
  local started_at="${5:-}"
  local tmp_path="${path}.tmp.$$"
  {
    printf 'key\tvalue\n'
    printf 'status\t%s\n' "$status"
    printf 'exit_code\t%s\n' "$exit_code"
    printf 'pid\t%s\n' "$pid"
    printf 'boot_id\t%s\n' "$(harness_boot_id)"
    printf 'started_at\t%s\n' "$started_at"
    printf 'updated_at\t%s\n' "$(harness_now)"
  } > "$tmp_path"
  mv "$tmp_path" "$path"
}

harness_status_value() {
  local path="$1"
  local key="$2"
  awk -F '\t' -v key="$key" '$1 == key { print $2; exit }' "$path"
}

harness_running_state() {
  # Return live, stale, or terminal for an existing status file. A different
  # boot ID makes an old `running` state stale even if Linux has reused its PID.
  local path="$1"
  if [[ ! -s "$path" ]]; then
    printf 'missing'
    return 0
  fi
  local status pid recorded_boot
  status="$(harness_status_value "$path" status)"
  if [[ "$status" != "running" ]]; then
    printf 'terminal'
    return 0
  fi
  pid="$(harness_status_value "$path" pid)"
  recorded_boot="$(harness_status_value "$path" boot_id)"
  if [[ "$recorded_boot" == "$(harness_boot_id)" ]] && harness_process_is_alive "$pid"; then
    printf 'live'
  else
    printf 'stale'
  fi
}

harness_write_provenance() {
  local path="$1"
  local python_executable="$2"
  local command_line="$3"
  local free_space_path="$4"
  local tmp_path="${path}.tmp.$$"
  local gpu_summary="unavailable"
  local tensorflow_version="not_probed"
  if command -v nvidia-smi >/dev/null 2>&1; then
    gpu_summary="$(nvidia-smi --query-gpu=name,uuid,driver_version,memory.total --format=csv,noheader 2>/dev/null | paste -sd ';' - || true)"
  fi
  if [[ "${HARNESS_SKIP_TENSORFLOW_PROBE:-0}" != "1" ]]; then
    tensorflow_version="$($python_executable -c 'import tensorflow as tf; print(tf.__version__)' 2>/dev/null || printf unavailable)"
  fi
  {
    printf 'key\tvalue\n'
    printf 'recorded_at\t%s\n' "$(harness_now)"
    printf 'boot_id\t%s\n' "$(harness_boot_id)"
    printf 'kernel\t%s\n' "$(uname -srvmo | tr '\t' ' ')"
    printf 'python_executable\t%s\n' "$python_executable"
    printf 'python_version\t%s\n' "$($python_executable --version 2>&1 | tr '\t' ' ')"
    printf 'tensorflow_version\t%s\n' "$tensorflow_version"
    printf 'gpu\t%s\n' "${gpu_summary//$'\t'/ }"
    printf 'free_space_path\t%s\n' "$free_space_path"
    printf 'free_bytes_at_start\t%s\n' "$(harness_free_bytes "$free_space_path")"
    printf 'command\t%s\n' "${command_line//$'\t'/ }"
  } > "$tmp_path"
  mv "$tmp_path" "$path"
}
