#!/usr/bin/env bash
set -euo pipefail

show_help() {
  cat <<'EOF'
Run one paper-reproduction experiment from a named preset.

Usage:
  bash scripts/run_reproduction_experiment.sh <preset>

Presets:
  all_4cls090_cap1000        T1/T2/T2_SPACE combined, relaxed imbalance, 1000 cap per sequence.
  t2_space_4cls090_cap1000   T2 SPACE only, same condition as the 2026-07-11 recorded run.
  t2_space_4cls055_cap1000   T2 SPACE only, paper-interpreted 55% imbalance threshold.
  t2_4cls090_cap1000         T2 only, diagnostic comparison against the earlier relaxed T2 run.
  paper_strict_all_4cls055   T1/T2/T2_SPACE combined, paper-style strict imbalance threshold.

Environment overrides:
  DATA_ROOT=/mnt/c/Users/ctlab/somomma/dataset
  RUN_ROOT=$HOME/lumbarseg_runs
  OUTPUT_ROOT=$RUN_ROOT/<preset-output-name>
  LOG_DIR=logs
  BATCH_SIZE=2
  EPOCHS=100
  SEED=42
  FORCE_REPROCESS=0|1
  RECORD_TO_DOCS=1|0
  EVAL_FILE_LIST=/path/to/fixed_validation_files.txt
  TRAIN_FILE_LIST=/path/to/fixed_train_files.txt
  TRAIN_COHORT_MANIFEST=/path/to/fixed_train_cohort.tsv
  EVAL_COHORT_MANIFEST=/path/to/fixed_validation_cohort.tsv
  PROCESSED_ROOT=/path/to/read-only/processed-cache
  COHORT_DISJOINT_MODE=strict_series|author_diagnostic_slice
  SPLIT_CONFIG=/path/to/author-diagnostic/split_config.tsv
  ALLOW_DIRTY_RUN=0|1
  ALLOW_NO_TMUX=0|1
  MIN_FREE_GIB=20
  ORIENTATION_MODE=legacy|metadata|manifest
  ORIENTATION_MANIFEST=/path/to/reviewed_orientation.csv
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" || $# -lt 1 ]]; then
  show_help
  exit 0
fi

preset="$1"
data_root="${DATA_ROOT:-/mnt/c/Users/ctlab/somomma/dataset}"
run_root="${RUN_ROOT:-$HOME/lumbarseg_runs}"
log_dir="${LOG_DIR:-logs}"
batch_size="${BATCH_SIZE:-2}"
epochs="${EPOCHS:-100}"
seed="${SEED:-42}"
python_bin="${PYTHON_BIN:-}"
force_reprocess="${FORCE_REPROCESS:-0}"
record_to_docs="${RECORD_TO_DOCS:-1}"
allow_dirty_run="${ALLOW_DIRTY_RUN:-0}"
allow_no_tmux="${ALLOW_NO_TMUX:-0}"
min_free_gib="${MIN_FREE_GIB:-20}"
min_free_bytes="${MIN_FREE_BYTES:-}"
heartbeat_interval_seconds="${HEARTBEAT_INTERVAL_SECONDS:-60}"
hang_timeout_seconds="${HANG_TIMEOUT_SECONDS:-1800}"
disable_heartbeat="${DISABLE_HEARTBEAT:-0}"
orientation_mode="${ORIENTATION_MODE:-legacy}"
orientation_manifest="${ORIENTATION_MANIFEST:-}"
cohort_disjoint_mode="${COHORT_DISJOINT_MODE:-strict_series}"
split_config="${SPLIT_CONFIG:-}"

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

sequences=""
min_classes=4
imbalance_threshold="0.90"
max_slices_per_sequence=1000

case "$preset" in
  all_4cls090_cap1000)
    # This is the next main reproduction candidate because the paper-level
    # result is unlikely to be represented by T2_SPACE-only training. Keeping
    # the relaxed 0.90 filter preserves enough slices for a stable comparison.
    output_name="all_sequences_reproduction_4cls090_cap1000"
    ;;
  t2_space_4cls090_cap1000)
    sequences="T2_SPACE"
    output_name="t2_space_reproduction_4cls090_cap1000"
    ;;
  t2_space_4cls055_cap1000)
    sequences="T2_SPACE"
    imbalance_threshold="0.55"
    output_name="t2_space_reproduction_4cls055_cap1000"
    ;;
  t2_4cls090_cap1000)
    sequences="T2"
    output_name="t2_reproduction_4cls090_cap1000"
    ;;
  paper_strict_all_4cls055)
    # This preset is intentionally kept even though earlier audits showed very
    # few kept slices. It gives us a repeatable way to demonstrate whether the
    # paper-style strict filter is too aggressive for this implementation.
    imbalance_threshold="0.55"
    output_name="all_sequences_paper_strict_4cls055_cap1000"
    ;;
  *)
    echo "Unknown preset: $preset" >&2
    show_help >&2
    exit 2
    ;;
esac

# A campaign must isolate every candidate so preprocessed slices, checkpoints,
# and metrics cannot silently overwrite another method. Keeping the old path as
# the fallback preserves all existing one-preset commands and documentation.
output_root="${OUTPUT_ROOT:-${run_root}/${output_name}}"
processed_root="${PROCESSED_ROOT:-$output_root}"
train_file_list_override="${TRAIN_FILE_LIST:-}"
train_file_list="${train_file_list_override:-$output_root/train_files.txt}"
train_cohort_manifest="${TRAIN_COHORT_MANIFEST:-}"
eval_file_list_override="${EVAL_FILE_LIST:-}"
eval_file_list="${eval_file_list_override:-$output_root/validation_files.txt}"
eval_cohort_manifest="${EVAL_COHORT_MANIFEST:-}"

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=scripts/long_run_harness.sh
source "$repo_root/scripts/long_run_harness.sh"

if [[ "$allow_no_tmux" != "1" && -z "${TMUX:-}" ]]; then
  echo "Long GPU experiments must run inside tmux so an SSH disconnect does not terminate training." >&2
  exit 2
fi
fixed_train_mode=0
fixed_validation_mode=0
if [[ -n "$train_file_list_override" || -n "$train_cohort_manifest" ]]; then
  if [[ -z "$train_file_list_override" || ! -s "$train_file_list_override" || -z "$train_cohort_manifest" || ! -s "$train_cohort_manifest" ]]; then
    echo "A fixed training cohort requires both a non-empty TRAIN_FILE_LIST and TRAIN_COHORT_MANIFEST." >&2
    exit 2
  fi
  fixed_train_mode=1
fi
if [[ -n "$eval_file_list_override" || -n "$eval_cohort_manifest" ]]; then
  if [[ -z "$eval_file_list_override" || ! -s "$eval_file_list_override" || -z "$eval_cohort_manifest" || ! -s "$eval_cohort_manifest" ]]; then
    echo "A fixed validation cohort requires both a non-empty EVAL_FILE_LIST and EVAL_COHORT_MANIFEST." >&2
    exit 2
  fi
  fixed_validation_mode=1
fi
if [[ "$fixed_train_mode" == "1" && "$fixed_validation_mode" != "1" ]]; then
  echo "A fixed training cohort also requires a fixed validation cohort." >&2
  exit 2
fi
if [[ "$fixed_train_mode" == "1" || "$fixed_validation_mode" == "1" ]]; then
  if [[ "$processed_root" == "$output_root" ]]; then
    echo "Fixed-cohort modes require PROCESSED_ROOT to differ from OUTPUT_ROOT." >&2
    exit 2
  fi
  if [[ ! -d "$processed_root/images" || ! -d "$processed_root/masks" ]]; then
    echo "PROCESSED_ROOT must contain images/ and masks/: $processed_root" >&2
    exit 2
  fi
  if [[ "$force_reprocess" == "1" ]]; then
    echo "Fixed-cohort mode is read-only and cannot use FORCE_REPROCESS=1." >&2
    exit 2
  fi
fi
case "$cohort_disjoint_mode" in
  strict_series)
    split_mode="spider_overview_series"
    final_generalization_evidence="true"
    ;;
  author_diagnostic_slice)
    if [[ ! -s "$split_config" ]]; then
      echo "Author diagnostic mode requires SPLIT_CONFIG from create_author_diagnostic_split.py." >&2
      exit 2
    fi
    split_mode="$(awk -F '\t' '$1 == "split_mode" {print $2; exit}' "$split_config")"
    split_final_evidence="$(awk -F '\t' '$1 == "final_generalization_evidence" {print $2; exit}' "$split_config")"
    if [[ "$split_mode" != author_diagnostic_* || "$split_final_evidence" != "false" ]]; then
      echo "SPLIT_CONFIG does not describe a diagnostic-only author split." >&2
      exit 2
    fi
    if [[ "$fixed_train_mode" != "1" || "$fixed_validation_mode" != "1" ]]; then
      echo "Author diagnostic mode requires explicit frozen train/validation lists and manifests." >&2
      exit 2
    fi
    final_generalization_evidence="false"
    ;;
  *)
    echo "Unknown COHORT_DISJOINT_MODE: $cohort_disjoint_mode" >&2
    exit 2
    ;;
esac
harness_require_free_space "$output_root" "$min_free_gib" "$min_free_bytes" || exit 2
harness_require_batch_hardware "$batch_size" || exit $?

case "$orientation_mode" in
  legacy|metadata) ;;
  manifest)
    if [[ -z "$orientation_manifest" || ! -f "$orientation_manifest" ]]; then
      echo "ORIENTATION_MANIFEST must name an existing reviewed CSV in manifest mode." >&2
      exit 2
    fi
    ;;
  *)
    echo "Unknown ORIENTATION_MODE: $orientation_mode" >&2
    exit 2
    ;;
esac

# A result tied to uncommitted code cannot be recreated from its Git revision.
# Keep an explicit escape hatch for local debugging, but reject it by default.
if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  echo "Experiment runner must execute inside a Git worktree." >&2
  exit 2
fi
git_revision="$(git rev-parse HEAD)"
if [[ "$allow_dirty_run" != "1" ]] && [[ -n "$(git status --porcelain --untracked-files=no)" ]]; then
  echo "Repository has tracked changes; refusing an auditable experiment. Unrelated untracked files are preserved." >&2
  exit 2
fi
mkdir -p "$log_dir" "$output_root"
log_path="${log_dir}/${output_name}_$(date +%Y%m%d_%H%M%S).log"
status_path="$output_root/run_status.tsv"
started_at="$(harness_now)"
heartbeat_pid=""
cleanup() {
  local exit_code=$?
  if [[ -n "$heartbeat_pid" ]]; then
    kill "$heartbeat_pid" 2>/dev/null || true
    wait "$heartbeat_pid" 2>/dev/null || true
  fi
  if [[ "$exit_code" -ne 0 && -d "$output_root" ]]; then
    harness_atomic_write_status "$status_path" "failed" "$exit_code" "$$" "$started_at"
  fi
}
trap cleanup EXIT

# The metrics CSV has no sequence or revision metadata of its own. Persisting a
# small manifest beside it prevents a mixed-sequence result from being accepted
# as T2 SPACE evidence and makes a detached tmux run auditable after reconnect.
python_executable="$(command -v "$python_bin")"
python_version="$("$python_bin" --version 2>&1)"
platform="$(uname -sr)"
{
  printf 'key\tvalue\n'
  printf 'git_revision\t%s\n' "$git_revision"
  printf 'preset\t%s\n' "$preset"
  printf 'sequences\t%s\n' "${sequences:-ALL}"
  printf 'data_root\t%s\n' "$data_root"
  printf 'output_root\t%s\n' "$output_root"
  printf 'processed_root\t%s\n' "$processed_root"
  printf 'min_classes\t%s\n' "$min_classes"
  printf 'imbalance_threshold\t%s\n' "$imbalance_threshold"
  # The public paper versions disagree on the ratio formula and the author code
  # has no 55% removal step. Persist "proxy" in the evidence instead of allowing
  # this experiment to be mistaken for an exact reconstruction later.
  printf 'filter_definition\t%s\n' "paper_filter_proxy_dominant_foreground_fraction"
  printf 'imbalance_mode\t%s\n' "foreground_max_fraction"
  printf 'max_slices_per_sequence\t%s\n' "$max_slices_per_sequence"
  printf 'batch_size\t%s\n' "$batch_size"
  printf 'orientation_mode\t%s\n' "$orientation_mode"
  printf 'orientation_manifest\t%s\n' "$orientation_manifest"
  if [[ -n "$orientation_manifest" ]]; then
    printf 'orientation_manifest_sha256\t%s\n' "$(sha256sum "$orientation_manifest" | awk '{print $1}')"
  fi
  printf 'epochs\t%s\n' "$epochs"
  printf 'seed\t%s\n' "$seed"
  printf 'evaluation_file_list\t%s\n' "$eval_file_list"
  printf 'training_file_list\t%s\n' "$train_file_list"
  printf 'cohort_disjoint_mode\t%s\n' "$cohort_disjoint_mode"
  printf 'split_mode\t%s\n' "$split_mode"
  printf 'split_config\t%s\n' "$split_config"
  if [[ -n "$split_config" ]]; then
    printf 'split_config_sha256\t%s\n' "$(harness_sha256 "$split_config")"
  fi
  printf 'final_generalization_evidence\t%s\n' "$final_generalization_evidence"
  # These are evidence states, not optimistic claims. Public descriptions are
  # insufficient to verify the unpublished filtering and final evaluation
  # details, so every run explains exactly why protocol verification is blocked.
  printf 'paper_protocol_preprocessing_status\tunverified\n'
  printf 'paper_protocol_preprocessing_evidence\tpublic paper and code disagree on image dimensions and orientation details\n'
  printf 'paper_protocol_filtering_status\tblocked\n'
  printf 'paper_protocol_filtering_evidence\texact 55 percent rule and selected 1000 slices were not published\n'
  printf 'paper_protocol_split_status\tunverified\n'
  printf 'paper_protocol_split_evidence\t%s\n' "$split_mode is recorded but the exact author cohort is unavailable"
  printf 'paper_protocol_training_status\tunverified\n'
  printf 'paper_protocol_training_evidence\tpublic paper and code report conflicting batch and loss settings\n'
  printf 'paper_protocol_evaluation_status\tunverified\n'
  printf 'paper_protocol_evaluation_evidence\thard and probability Dice are both recorded because the paper aggregation is not fully specified\n'
  printf 'python_executable\t%s\n' "$python_executable"
  printf 'python_version\t%s\n' "$python_version"
  printf 'platform\t%s\n' "$platform"
  printf 'git_tracked_clean\t%s\n' "$([[ "$allow_dirty_run" == "1" ]] && echo unchecked || echo true)"
  printf 'untracked_file_count\t%s\n' "$(git status --porcelain --untracked-files=normal | awk '$1 == "??" {count += 1} END {print count + 0}')"
  printf 'boot_id\t%s\n' "$(harness_boot_id)"
  printf 'started_at\t%s\n' "$(harness_now)"
} > "$output_root/run_config.tsv"
harness_write_provenance \
  "$output_root/environment.tsv" \
  "$python_executable" \
  "$(harness_command_line "$0" "$@")" \
  "$output_root"
harness_atomic_write_status "$status_path" "running" "" "$$" "$started_at"

sequence_args=()
if [[ -n "$sequences" ]]; then
  sequence_args=(--sequences "$sequences")
fi

force_args=()
if [[ "$force_reprocess" == "1" ]]; then
  force_args=(--force_reprocess)
fi

fixed_training_args=()
if [[ "$fixed_train_mode" == "1" ]]; then
  fixed_training_args=(
    --run_output_root "$output_root"
    --reuse_processed_only
    --train_file_list "$train_file_list"
    --validation_file_list "$eval_file_list"
    --cohort_disjoint_mode "$cohort_disjoint_mode"
  )
elif [[ "$fixed_validation_mode" == "1" ]]; then
  # Filtering candidates must derive their own training cohort; freezing the
  # baseline training list would make a 0.55-vs-0.90 comparison a mislabeled
  # no-op. Reuse only the immutable processed arrays and fixed validation set.
  fixed_training_args=(
    --run_output_root "$output_root"
    --reuse_processed_cache
    --validation_file_list "$eval_file_list"
    --cohort_disjoint_mode "$cohort_disjoint_mode"
  )
fi

orientation_args=(--orientation_mode "$orientation_mode")
if [[ -n "$orientation_manifest" ]]; then
  orientation_args+=(--orientation_manifest "$orientation_manifest")
fi

echo "Preset: $preset"
echo "Output root: $output_root"
echo "Log: $log_path"
echo
echo "Run this inside tmux if the session may disconnect:"
echo "  bash scripts/run_reproduction_experiment.sh $preset"
echo

if [[ "$disable_heartbeat" != "1" ]]; then
  HEARTBEAT_PATH="$output_root/heartbeat.tsv" \
  WATCH_PID="$$" \
  TRAINING_LOG="$output_root/checkpoints/training_log.csv" \
  HEARTBEAT_INTERVAL_SECONDS="$heartbeat_interval_seconds" \
  HANG_TIMEOUT_SECONDS="$hang_timeout_seconds" \
  DISK_PATH="$output_root" \
  MIN_FREE_GIB="$min_free_gib" \
    bash "$repo_root/scripts/run_long_run_heartbeat.sh" &
  heartbeat_pid=$!
fi

{
  echo "Started: $(harness_now)"
  echo "Preset: $preset"
  echo "Data root: $data_root"
  echo "Output root: $output_root"
  echo "Sequences: ${sequences:-ALL}"
  echo "Min classes: $min_classes"
  echo "Imbalance threshold: $imbalance_threshold"
  echo "Max slices per sequence: $max_slices_per_sequence"
  echo "Batch size: $batch_size"
  echo "Epochs: $epochs"
  echo "Seed: $seed"
  echo "Git revision: $git_revision"
  echo "Force reprocess: $force_reprocess"
  echo "Record to docs: $record_to_docs"
  echo

  "$python_bin" repair_processed_slices.py \
    --output_root "$processed_root" \
    --dry_run || true

  "$python_bin" train.py \
    --data_root "$data_root" \
    --output_root "$processed_root" \
    "${sequence_args[@]}" \
    --batch_size "$batch_size" \
    --epochs "$epochs" \
    --seed "$seed" \
    --min_classes "$min_classes" \
    --imbalance_threshold "$imbalance_threshold" \
    --max_slices_per_sequence "$max_slices_per_sequence" \
    "${orientation_args[@]}" \
    "${fixed_training_args[@]}" \
    "${force_args[@]}"

  if [[ ! -s "$eval_file_list" ]]; then
    echo "Validation file list is missing or empty: $eval_file_list" >&2
    exit 2
  fi

  local_train_manifest="$output_root/training_cohort.tsv"
  local_cohort_manifest="$output_root/validation_cohort.tsv"
  if [[ "$fixed_train_mode" == "1" ]]; then
    "$python_bin" scripts/hash_validation_cohort.py \
      --output-root "$processed_root" \
      --file-list "$train_file_list" \
      --verify "$train_cohort_manifest"
    cp "$train_cohort_manifest" "$local_train_manifest"
  else
    train_file_list="$output_root/train_files.txt"
    "$python_bin" scripts/hash_validation_cohort.py \
      --output-root "$processed_root" \
      --file-list "$train_file_list" \
      --write "$local_train_manifest"
  fi
  if [[ "$fixed_validation_mode" == "1" ]]; then
    "$python_bin" scripts/hash_validation_cohort.py \
      --output-root "$processed_root" \
      --file-list "$eval_file_list" \
      --verify "$eval_cohort_manifest"
    cp "$eval_cohort_manifest" "$local_cohort_manifest"
  else
    "$python_bin" scripts/hash_validation_cohort.py \
      --output-root "$processed_root" \
      --file-list "$eval_file_list" \
      --write "$local_cohort_manifest"
  fi

  printf 'train_slices\t%s\n' "$(wc -l < "$train_file_list" | tr -d ' ')" >> "$output_root/run_config.tsv"
  printf 'validation_slices\t%s\n' "$(wc -l < "$eval_file_list" | tr -d ' ')" >> "$output_root/run_config.tsv"
  if [[ -s "$output_root/filtered_files.txt" ]]; then
    printf 'filtered_slices\t%s\n' "$(wc -l < "$output_root/filtered_files.txt" | tr -d ' ')" >> "$output_root/run_config.tsv"
  fi
  printf 'train_file_list_sha256\t%s\n' "$(sha256sum "$train_file_list" | awk '{print $1}')" >> "$output_root/run_config.tsv"
  printf 'validation_file_list_sha256\t%s\n' "$(sha256sum "$eval_file_list" | awk '{print $1}')" >> "$output_root/run_config.tsv"
  printf 'train_cohort_sha256\t%s\n' "$(sha256sum "$local_train_manifest" | awk '{print $1}')" >> "$output_root/run_config.tsv"
  printf 'validation_cohort_sha256\t%s\n' "$(sha256sum "$local_cohort_manifest" | awk '{print $1}')" >> "$output_root/run_config.tsv"

  "$python_bin" evaluate.py \
    --data_root "$data_root" \
    --output_root "$processed_root" \
    --evaluation_output_root "$output_root" \
    --min_classes "$min_classes" \
    --imbalance_threshold "$imbalance_threshold" \
    --max_slices_per_sequence "$max_slices_per_sequence" \
    --file_list "$eval_file_list" \
    --model_path "$output_root/checkpoints/best_model.keras"

  echo
  echo "Finished: $(harness_now)"
  echo "Metrics: $output_root/validation_metrics.csv"
  echo "Training log: $output_root/checkpoints/training_log.csv"

  if [[ "$record_to_docs" == "1" ]]; then
    # The heavy artifacts stay under RUN_ROOT, but the small CSV files are the
    # evidence we need for commits and paper-reproduction decisions. Recording
    # them here prevents the common failure mode where a long GPU run succeeds
    # but the exact metric CSV is never copied back into the repository.
    record_dir="docs/experiments/${output_name}_$(date +%Y%m%d)"
    mkdir -p "$record_dir"
    cp "$output_root/validation_metrics.csv" "$record_dir/"
    cp "$output_root/checkpoints/training_log.csv" "$record_dir/"
    cp "$output_root/run_config.tsv" "$record_dir/"
    "$python_bin" scripts/summarize_reproduction_results.py
    echo "Recorded small experiment artifacts to: $record_dir"
  fi
} 2>&1 | tee "$log_path"
harness_atomic_write_status "$status_path" "completed" "0" "$$" "$started_at"
