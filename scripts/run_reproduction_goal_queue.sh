#!/usr/bin/env bash
set -euo pipefail

show_help() {
  cat <<'EOF'
Run paper-reproduction presets sequentially on a single GPU.

Usage:
  bash scripts/run_reproduction_goal_queue.sh <preset> [<preset> ...]

Example:
  bash scripts/run_reproduction_goal_queue.sh t2_space_4cls090_cap1000

Environment overrides:
  RUN_ROOT=$HOME/lumbarseg_runs
  CAMPAIGN_ID=goal_YYYYmmdd_HHMMSS
  CAMPAIGN_ROOT=$RUN_ROOT/$CAMPAIGN_ID
  RUNNER=scripts/run_reproduction_experiment.sh
  CHECKER=scripts/check_reproduction_target.py
  PYTHON_BIN=python
  GPU_LOCK_DIR=$RUN_ROOT/.lumbarseg_gpu_campaign.lock
  ALLOW_NO_TMUX=0|1             Test-only escape hatch; long runs require tmux.
  SKIP_ACTIVE_TRAIN_CHECK=0|1   Test-only escape hatch; do not use for GPU runs.
  RESUME_CAMPAIGN=0|1           Resume only between completed experiments.
  EXPECTED_TRAIN_COUNT=730       Baseline-only check; filtering candidates may differ.
  EXPECTED_VALIDATION_COUNT=270
  MIN_FREE_GIB=20
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" || $# -lt 1 ]]; then
  show_help
  [[ $# -lt 1 ]] && exit 2
  exit 0
fi

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
run_root="${RUN_ROOT:-$HOME/lumbarseg_runs}"
campaign_id="${CAMPAIGN_ID:-goal_$(date +%Y%m%d_%H%M%S)}"
campaign_root="${CAMPAIGN_ROOT:-$run_root/$campaign_id}"
runner="${RUNNER:-$repo_root/scripts/run_reproduction_experiment.sh}"
checker="${CHECKER:-$repo_root/scripts/check_reproduction_target.py}"
python_bin="${PYTHON_BIN:-}"
skip_active_train_check="${SKIP_ACTIVE_TRAIN_CHECK:-0}"
allow_no_tmux="${ALLOW_NO_TMUX:-0}"
gpu_lock_dir="${GPU_LOCK_DIR:-$run_root/.lumbarseg_gpu_campaign.lock}"
resume_campaign="${RESUME_CAMPAIGN:-0}"
expected_train_count="${EXPECTED_TRAIN_COUNT:-730}"
expected_validation_count="${EXPECTED_VALIDATION_COUNT:-270}"
min_free_gib="${MIN_FREE_GIB:-20}"
min_free_bytes="${MIN_FREE_BYTES:-}"
disable_heartbeat="${DISABLE_HEARTBEAT:-0}"
heartbeat_interval_seconds="${HEARTBEAT_INTERVAL_SECONDS:-60}"
hang_timeout_seconds="${HANG_TIMEOUT_SECONDS:-1800}"

if [[ -z "$python_bin" ]]; then
  if command -v python >/dev/null 2>&1; then
    python_bin="python"
  else
    python_bin="python3"
  fi
fi
if ! command -v "$python_bin" >/dev/null 2>&1; then
  echo "PYTHON_BIN is not executable: $python_bin" >&2
  exit 2
fi

# shellcheck source=scripts/long_run_harness.sh
source "$repo_root/scripts/long_run_harness.sh"

if [[ -e "$campaign_root" ]]; then
  if [[ "$resume_campaign" != "1" ]]; then
    echo "Campaign root already exists; refusing to mix or overwrite results: $campaign_root" >&2
    exit 2
  fi
  queue_state="$(harness_running_state "$campaign_root/queue_status.tsv")"
  if [[ "$queue_state" == "live" ]]; then
    echo "Campaign is still active; refusing a second queue: $campaign_root" >&2
    exit 2
  fi
fi
if [[ ! -f "$runner" || ! -f "$checker" ]]; then
  echo "Runner or checker is missing: runner=$runner checker=$checker" >&2
  exit 2
fi
if [[ "$allow_no_tmux" != "1" && -z "${TMUX:-}" ]]; then
  echo "Long GPU campaigns must run inside tmux. Start a detached tmux session first." >&2
  exit 2
fi
for preset in "$@"; do
  if [[ "$preset" != t2_space_* ]]; then
    echo "Goal campaigns accept only T2 SPACE presets; rejected: $preset" >&2
    exit 2
  fi
done
if [[ "$1" != "t2_space_4cls090_cap1000" ]]; then
  echo "The first preset must be the canonical T2 SPACE baseline: t2_space_4cls090_cap1000" >&2
  exit 2
fi
harness_require_free_space "$campaign_root" "$min_free_gib" "$min_free_bytes" || exit 2

mkdir -p "$run_root"

# mkdir is the portable atomic lock here: two campaigns cannot both create the
# same directory. We intentionally leave a stale lock after an unclean crash so
# a human must inspect GPU/process state before deciding it is safe to remove.
if ! mkdir "$gpu_lock_dir" 2>/dev/null; then
  lock_owner="$gpu_lock_dir/owner.tsv"
  lock_state="$(harness_running_state "$lock_owner")"
  if [[ "$resume_campaign" == "1" && "$lock_state" == "stale" ]]; then
    stale_lock="${gpu_lock_dir}.stale_$(date +%Y%m%d_%H%M%S)"
    mv "$gpu_lock_dir" "$stale_lock"
    echo "Preserved stale GPU lock evidence at: $stale_lock"
    if ! mkdir "$gpu_lock_dir" 2>/dev/null; then
      echo "GPU campaign lock could not be reacquired: $gpu_lock_dir" >&2
      exit 2
    fi
  else
    echo "GPU campaign lock already exists or cannot be proven stale: $gpu_lock_dir" >&2
    exit 2
  fi
fi
lock_owned=1
queue_completed=0
heartbeat_pid=""
started_at="$(harness_now)"
harness_atomic_write_status "$gpu_lock_dir/owner.tsv" "running" "" "$$" "$started_at"
release_lock() {
  local exit_code=$?
  if [[ -n "$heartbeat_pid" ]]; then
    kill "$heartbeat_pid" 2>/dev/null || true
    wait "$heartbeat_pid" 2>/dev/null || true
  fi
  if [[ -d "$campaign_root" && "$queue_completed" != "1" ]]; then
    harness_atomic_write_status "$campaign_root/queue_status.tsv" "failed_or_interrupted" "$exit_code" "$$" "$started_at"
  fi
  if [[ "${lock_owned:-0}" == "1" ]]; then
    rm -f "$gpu_lock_dir/owner.tsv"
    rmdir "$gpu_lock_dir" 2>/dev/null || true
    lock_owned=0
  fi
}
handle_int() { exit 130; }
handle_term() { exit 143; }
handle_hup() { exit 129; }
trap release_lock EXIT
trap handle_int INT
trap handle_term TERM
trap handle_hup HUP

if [[ "$skip_active_train_check" != "1" ]] && pgrep -af '[p]ython(3)? .*train.py' >/dev/null; then
  echo "A training process is already active. Refusing to start a second GPU campaign." >&2
  exit 2
fi

mkdir -p "$campaign_root/logs"
if [[ "$resume_campaign" == "1" && ! -s "$campaign_root/campaign_status.tsv" ]]; then
  echo "Cannot resume without campaign_status.tsv: $campaign_root" >&2
  exit 2
fi
fixed_validation_list="$campaign_root/fixed_validation_files.txt"
fixed_cohort_manifest="$campaign_root/fixed_validation_cohort.tsv"
status_path="$campaign_root/campaign_status.tsv"
if [[ "$resume_campaign" != "1" ]]; then
  printf 'index\tpreset\tstatus\toutput_root\ttarget_check\n' > "$status_path"
fi
harness_atomic_write_status "$campaign_root/queue_status.tsv" "running" "" "$$" "$started_at"
exec > >(tee -a "$campaign_root/logs/gpu-queue.log") 2>&1
if [[ "$disable_heartbeat" != "1" ]]; then
  HEARTBEAT_PATH="$campaign_root/heartbeat.tsv" \
  WATCH_PID="$$" \
  HEARTBEAT_INTERVAL_SECONDS="$heartbeat_interval_seconds" \
  HANG_TIMEOUT_SECONDS="$hang_timeout_seconds" \
  DISK_PATH="$campaign_root" \
  MIN_FREE_GIB="$min_free_gib" \
    bash "$repo_root/scripts/run_long_run_heartbeat.sh" &
  heartbeat_pid=$!
fi

echo "Campaign: $campaign_id"
echo "Campaign root: $campaign_root"
echo "Presets: $*"

index=0
for preset in "$@"; do
  index=$((index + 1))
  safe_preset="$(printf '%s' "$preset" | tr -c 'A-Za-z0-9_-' '_')"
  experiment_root="$(printf '%s/%02d_%s' "$campaign_root" "$index" "$safe_preset")"
  target_path="$experiment_root/target_check.json"

  existing_status="$(awk -F '\t' -v wanted_index="$index" -v wanted_preset="$preset" 'NR > 1 && $1 == wanted_index && $2 == wanted_preset {print $3; exit}' "$status_path")"
  if [[ -n "$existing_status" ]]; then
    case "$existing_status" in
      target_met)
        echo "[$index/$#] Previously completed with target_met: $preset"
        queue_completed=1
        harness_atomic_write_status "$campaign_root/queue_status.tsv" "completed_goal_met" "0" "$$" "$started_at"
        exit 0
        ;;
      target_missed)
        echo "[$index/$#] Reusing completed target_missed evidence: $preset"
        continue
        ;;
      *)
        echo "Campaign contains non-resumable status $existing_status for $preset; use a new campaign root." >&2
        exit 2
        ;;
    esac
  fi
  if [[ -e "$experiment_root" ]]; then
    echo "Partial experiment output exists without a completed status; refusing to overwrite: $experiment_root" >&2
    exit 2
  fi

  echo
  echo "[$index/$#] Starting preset: $preset"
  if [[ "$index" -eq 1 ]]; then
    run_status=0
    (
      cd "$repo_root"
      OUTPUT_ROOT="$experiment_root" \
      LOG_DIR="$campaign_root/logs" \
      RECORD_TO_DOCS=0 \
      ALLOW_NO_TMUX="$allow_no_tmux" \
      MIN_FREE_GIB="$min_free_gib" \
      MIN_FREE_BYTES="$min_free_bytes" \
      PYTHON_BIN="$python_bin" \
      bash "$runner" "$preset"
    ) || run_status=$?
  else
    run_status=0
    (
      cd "$repo_root"
      OUTPUT_ROOT="$experiment_root" \
      LOG_DIR="$campaign_root/logs" \
      RECORD_TO_DOCS=0 \
      PROCESSED_ROOT="$campaign_root/01_t2_space_4cls090_cap1000" \
      EVAL_FILE_LIST="$fixed_validation_list" \
      EVAL_COHORT_MANIFEST="$fixed_cohort_manifest" \
      ALLOW_NO_TMUX="$allow_no_tmux" \
      MIN_FREE_GIB="$min_free_gib" \
      MIN_FREE_BYTES="$min_free_bytes" \
      PYTHON_BIN="$python_bin" \
      bash "$runner" "$preset"
    ) || run_status=$?
  fi
  if [[ "$run_status" -ne 0 ]]; then
    printf '%s\t%s\texecution_failed\t%s\t%s\n' \
      "$index" "$preset" "$experiment_root" "$target_path" >> "$status_path"
    echo "Experiment failed; queue stopped: $preset" >&2
    exit 2
  fi

  # Every candidate may select a different training cohort, but its actual
  # list and image/mask content hashes must exist before it can be evaluated.
  if [[ ! -s "$experiment_root/train_files.txt" || ! -s "$experiment_root/training_cohort.tsv" ]]; then
    echo "Training cohort evidence is missing or empty; queue stopped: $preset" >&2
    exit 2
  fi

  if [[ "$index" -eq 1 ]]; then
    # Freeze the first baseline's exact validation slices. Later candidates may
    # alter training selection, but cannot improve by dropping difficult cases.
    if [[ ! -s "$experiment_root/validation_files.txt" ]]; then
      echo "Baseline validation file list is missing or empty; queue stopped." >&2
      exit 2
    fi
    if [[ ! -s "$experiment_root/train_files.txt" ]]; then
      echo "Baseline training file list is missing or empty; queue stopped." >&2
      exit 2
    fi
    train_count="$(awk 'NF {count += 1} END {print count + 0}' "$experiment_root/train_files.txt")"
    validation_count="$(awk 'NF {count += 1} END {print count + 0}' "$experiment_root/validation_files.txt")"
    if [[ "$train_count" -ne "$expected_train_count" || "$validation_count" -ne "$expected_validation_count" ]]; then
      echo "Frozen cohort count mismatch: train=$train_count/$expected_train_count validation=$validation_count/$expected_validation_count" >&2
      exit 2
    fi
    cp "$experiment_root/validation_files.txt" "$fixed_validation_list"
    if [[ ! -s "$experiment_root/validation_cohort.tsv" ]]; then
      echo "Baseline validation cohort manifest is missing or empty; queue stopped." >&2
      exit 2
    fi
    cp "$experiment_root/validation_cohort.tsv" "$fixed_cohort_manifest"
  fi

  set +e
  "$python_bin" "$checker" \
    "$experiment_root/validation_metrics.csv" \
    --output "$target_path" \
    --run-config "$experiment_root/run_config.tsv" \
    --require-sequence T2_SPACE
  check_status=$?
  set -e

  case "$check_status" in
    0)
      printf '%s\t%s\ttarget_met\t%s\t%s\n' \
        "$index" "$preset" "$experiment_root" "$target_path" >> "$status_path"
      echo "Score target met; remaining presets will not run."
      queue_completed=1
      harness_atomic_write_status "$campaign_root/queue_status.tsv" "completed_goal_met" "0" "$$" "$started_at"
      exit 0
      ;;
    1)
      printf '%s\t%s\ttarget_missed\t%s\t%s\n' \
        "$index" "$preset" "$experiment_root" "$target_path" >> "$status_path"
      echo "Score target missed; continuing to the next queued preset."
      ;;
    *)
      printf '%s\t%s\tinvalid_evidence\t%s\t%s\n' \
        "$index" "$preset" "$experiment_root" "$target_path" >> "$status_path"
      echo "Target evidence was invalid; queue stopped: $preset" >&2
      exit 2
      ;;
  esac
done

echo "Queue finished without meeting the target."
queue_completed=1
harness_atomic_write_status "$campaign_root/queue_status.tsv" "completed_goal_miss" "1" "$$" "$started_at"
exit 1
