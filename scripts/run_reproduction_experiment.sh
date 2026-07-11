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
  t2_4cls090_cap1000         T2 only, diagnostic comparison against the earlier relaxed T2 run.
  paper_strict_all_4cls055   T1/T2/T2_SPACE combined, paper-style strict imbalance threshold.

Environment overrides:
  DATA_ROOT=/mnt/c/Users/ctlab/somomma/dataset
  RUN_ROOT=$HOME/lumbarseg_runs
  LOG_DIR=logs
  BATCH_SIZE=2
  EPOCHS=100
  FORCE_REPROCESS=0|1
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
force_reprocess="${FORCE_REPROCESS:-0}"

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

output_root="${run_root}/${output_name}"
mkdir -p "$log_dir" "$output_root"
log_path="${log_dir}/${output_name}_$(date +%Y%m%d_%H%M%S).log"

sequence_args=()
if [[ -n "$sequences" ]]; then
  sequence_args=(--sequences "$sequences")
fi

force_args=()
if [[ "$force_reprocess" == "1" ]]; then
  force_args=(--force_reprocess)
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
  echo "Force reprocess: $force_reprocess"
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
    --min_classes "$min_classes" \
    --imbalance_threshold "$imbalance_threshold" \
    --max_slices_per_sequence "$max_slices_per_sequence" \
    "${force_args[@]}"

  python evaluate.py \
    --data_root "$data_root" \
    --output_root "$output_root" \
    --min_classes "$min_classes" \
    --imbalance_threshold "$imbalance_threshold" \
    --max_slices_per_sequence "$max_slices_per_sequence" \
    --model_path "$output_root/checkpoints/best_model.keras"

  echo
  echo "Finished: $(date -Is)"
  echo "Metrics: $output_root/validation_metrics.csv"
  echo "Training log: $output_root/checkpoints/training_log.csv"
} 2>&1 | tee "$log_path"
