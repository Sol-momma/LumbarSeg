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
  EVAL_COHORT_MANIFEST=/path/to/fixed_validation_cohort.tsv
  ALLOW_DIRTY_RUN=0|1
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
force_reprocess="${FORCE_REPROCESS:-0}"
record_to_docs="${RECORD_TO_DOCS:-1}"
allow_dirty_run="${ALLOW_DIRTY_RUN:-0}"
orientation_mode="${ORIENTATION_MODE:-legacy}"
orientation_manifest="${ORIENTATION_MANIFEST:-}"

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
eval_file_list_override="${EVAL_FILE_LIST:-}"
eval_file_list="${eval_file_list_override:-$output_root/validation_files.txt}"
eval_cohort_manifest="${EVAL_COHORT_MANIFEST:-}"

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
if [[ "$allow_dirty_run" != "1" ]] && [[ -n "$(git status --porcelain)" ]]; then
  echo "Repository has uncommitted or untracked changes; refusing an auditable experiment." >&2
  exit 2
fi
mkdir -p "$log_dir" "$output_root"
log_path="${log_dir}/${output_name}_$(date +%Y%m%d_%H%M%S).log"

# The metrics CSV has no sequence or revision metadata of its own. Persisting a
# small manifest beside it prevents a mixed-sequence result from being accepted
# as T2 SPACE evidence and makes a detached tmux run auditable after reconnect.
python_executable="$(command -v python)"
python_version="$(python --version 2>&1)"
platform="$(uname -sr)"
{
  printf 'key\tvalue\n'
  printf 'git_revision\t%s\n' "$git_revision"
  printf 'preset\t%s\n' "$preset"
  printf 'sequences\t%s\n' "${sequences:-ALL}"
  printf 'data_root\t%s\n' "$data_root"
  printf 'output_root\t%s\n' "$output_root"
  printf 'min_classes\t%s\n' "$min_classes"
  printf 'imbalance_threshold\t%s\n' "$imbalance_threshold"
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
  printf 'python_executable\t%s\n' "$python_executable"
  printf 'python_version\t%s\n' "$python_version"
  printf 'platform\t%s\n' "$platform"
  printf 'git_clean\t%s\n' "$([[ "$allow_dirty_run" == "1" ]] && echo unchecked || echo true)"
  printf 'started_at\t%s\n' "$(date -Is)"
} > "$output_root/run_config.tsv"

sequence_args=()
if [[ -n "$sequences" ]]; then
  sequence_args=(--sequences "$sequences")
fi

force_args=()
if [[ "$force_reprocess" == "1" ]]; then
  force_args=(--force_reprocess)
fi

validation_args=()
if [[ -n "$eval_file_list_override" ]]; then
  validation_args=(--validation_file_list "$eval_file_list")
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

{
  echo "Started: $(date -Is)"
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

  python repair_processed_slices.py \
    --output_root "$output_root" \
    --dry_run || true

  python train.py \
    --data_root "$data_root" \
    --output_root "$output_root" \
    "${sequence_args[@]}" \
    --batch_size "$batch_size" \
    --epochs "$epochs" \
    --seed "$seed" \
    --min_classes "$min_classes" \
    --imbalance_threshold "$imbalance_threshold" \
    --max_slices_per_sequence "$max_slices_per_sequence" \
    "${orientation_args[@]}" \
    "${validation_args[@]}" \
    "${force_args[@]}"

  if [[ ! -s "$eval_file_list" ]]; then
    echo "Validation file list is missing or empty: $eval_file_list" >&2
    exit 2
  fi

  local_cohort_manifest="$output_root/validation_cohort.tsv"
  if [[ -n "$eval_cohort_manifest" ]]; then
    python scripts/hash_validation_cohort.py \
      --output-root "$output_root" \
      --file-list "$eval_file_list" \
      --verify "$eval_cohort_manifest"
    cp "$eval_cohort_manifest" "$local_cohort_manifest"
  else
    python scripts/hash_validation_cohort.py \
      --output-root "$output_root" \
      --file-list "$eval_file_list" \
      --write "$local_cohort_manifest"
  fi

  printf 'train_slices\t%s\n' "$(wc -l < "$output_root/train_files.txt" | tr -d ' ')" >> "$output_root/run_config.tsv"
  printf 'validation_slices\t%s\n' "$(wc -l < "$eval_file_list" | tr -d ' ')" >> "$output_root/run_config.tsv"
  printf 'filtered_slices\t%s\n' "$(wc -l < "$output_root/filtered_files.txt" | tr -d ' ')" >> "$output_root/run_config.tsv"
  printf 'validation_file_list_sha256\t%s\n' "$(sha256sum "$eval_file_list" | awk '{print $1}')" >> "$output_root/run_config.tsv"
  printf 'validation_cohort_sha256\t%s\n' "$(sha256sum "$local_cohort_manifest" | awk '{print $1}')" >> "$output_root/run_config.tsv"

  python evaluate.py \
    --data_root "$data_root" \
    --output_root "$output_root" \
    --min_classes "$min_classes" \
    --imbalance_threshold "$imbalance_threshold" \
    --max_slices_per_sequence "$max_slices_per_sequence" \
    --file_list "$eval_file_list" \
    --model_path "$output_root/checkpoints/best_model.keras"

  echo
  echo "Finished: $(date -Is)"
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
    python scripts/summarize_reproduction_results.py
    echo "Recorded small experiment artifacts to: $record_dir"
  fi
} 2>&1 | tee "$log_path"
