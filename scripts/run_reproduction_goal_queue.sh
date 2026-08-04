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
python_bin="${PYTHON_BIN:-python}"
skip_active_train_check="${SKIP_ACTIVE_TRAIN_CHECK:-0}"
allow_no_tmux="${ALLOW_NO_TMUX:-0}"
gpu_lock_dir="${GPU_LOCK_DIR:-$run_root/.lumbarseg_gpu_campaign.lock}"

if [[ -e "$campaign_root" ]]; then
  echo "Campaign root already exists; refusing to mix or overwrite results: $campaign_root" >&2
  exit 2
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

mkdir -p "$run_root"

# mkdir is the portable atomic lock here: two campaigns cannot both create the
# same directory. We intentionally leave a stale lock after an unclean crash so
# a human must inspect GPU/process state before deciding it is safe to remove.
if ! mkdir "$gpu_lock_dir" 2>/dev/null; then
  echo "GPU campaign lock already exists: $gpu_lock_dir" >&2
  exit 2
fi
lock_owned=1
release_lock() {
  if [[ "${lock_owned:-0}" == "1" ]]; then
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
fixed_validation_list="$campaign_root/fixed_validation_files.txt"
fixed_cohort_manifest="$campaign_root/fixed_validation_cohort.tsv"
status_path="$campaign_root/campaign_status.tsv"
printf 'index\tpreset\tstatus\toutput_root\ttarget_check\n' > "$status_path"

echo "Campaign: $campaign_id"
echo "Campaign root: $campaign_root"
echo "Presets: $*"

index=0
for preset in "$@"; do
  index=$((index + 1))
  safe_preset="$(printf '%s' "$preset" | tr -c 'A-Za-z0-9_-' '_')"
  experiment_root="$(printf '%s/%02d_%s' "$campaign_root" "$index" "$safe_preset")"
  target_path="$experiment_root/target_check.json"

  echo
  echo "[$index/$#] Starting preset: $preset"
  if [[ "$index" -eq 1 ]]; then
    run_status=0
    (
      cd "$repo_root"
      OUTPUT_ROOT="$experiment_root" \
      LOG_DIR="$campaign_root/logs" \
      RECORD_TO_DOCS=0 \
      bash "$runner" "$preset"
    ) || run_status=$?
  else
    run_status=0
    (
      cd "$repo_root"
      OUTPUT_ROOT="$experiment_root" \
      LOG_DIR="$campaign_root/logs" \
      RECORD_TO_DOCS=0 \
      EVAL_FILE_LIST="$fixed_validation_list" \
      EVAL_COHORT_MANIFEST="$fixed_cohort_manifest" \
      bash "$runner" "$preset"
    ) || run_status=$?
  fi
  if [[ "$run_status" -ne 0 ]]; then
    printf '%s\t%s\texecution_failed\t%s\t%s\n' \
      "$index" "$preset" "$experiment_root" "$target_path" >> "$status_path"
    echo "Experiment failed; queue stopped: $preset" >&2
    exit 2
  fi

  if [[ "$index" -eq 1 ]]; then
    # Freeze the first baseline's exact validation slices. Later candidates may
    # alter training selection, but cannot improve by dropping difficult cases.
    if [[ ! -s "$experiment_root/validation_files.txt" ]]; then
      echo "Baseline validation file list is missing or empty; queue stopped." >&2
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
exit 1
