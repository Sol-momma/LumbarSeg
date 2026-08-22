#!/usr/bin/env bash
set -euo pipefail

# This probe answers only whether the approved fixed cohort can complete one
# train/validation step at batch size 8. It deliberately reuses the processed
# cache while placing every new model/log artifact in a separate directory.
: "${DATA_ROOT:?Set DATA_ROOT to the SPIDER DataSet directory.}"
: "${PROCESSED_ROOT:?Set PROCESSED_ROOT to the existing processed baseline directory.}"
: "${RUN_OUTPUT_ROOT:?Set RUN_OUTPUT_ROOT to a new directory for this probe.}"
: "${TRAIN_FILE_LIST:?Set TRAIN_FILE_LIST to the fixed training cohort file.}"
: "${VALIDATION_FILE_LIST:?Set VALIDATION_FILE_LIST to the fixed validation cohort file.}"
: "${TRAIN_COHORT_MANIFEST:?Set TRAIN_COHORT_MANIFEST to the fixed training content hashes.}"
: "${VALIDATION_COHORT_MANIFEST:?Set VALIDATION_COHORT_MANIFEST to the fixed validation content hashes.}"

BATCH_SIZE="${BATCH_SIZE:-8}"
SEQUENCES="${SEQUENCES:-T2_SPACE}"
PYTHON_EXECUTABLE="${PYTHON_EXECUTABLE:-python}"
MIN_FREE_GIB="${MIN_FREE_GIB:-5}"
MIN_FREE_BYTES="${MIN_FREE_BYTES:-}"
MIN_BATCH8_GPU_MEMORY_MIB="${MIN_BATCH8_GPU_MEMORY_MIB:-12288}"

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"
# shellcheck source=scripts/long_run_harness.sh
source "$repo_root/scripts/long_run_harness.sh"

if [[ ! -f "$TRAIN_FILE_LIST" || ! -s "$TRAIN_FILE_LIST" ]]; then
  echo "Training cohort file is missing or empty: $TRAIN_FILE_LIST" >&2
  exit 1
fi
if [[ ! -f "$VALIDATION_FILE_LIST" || ! -s "$VALIDATION_FILE_LIST" ]]; then
  echo "Validation cohort file is missing or empty: $VALIDATION_FILE_LIST" >&2
  exit 1
fi
for manifest in "$TRAIN_COHORT_MANIFEST" "$VALIDATION_COHORT_MANIFEST"; do
  if [[ ! -s "$manifest" ]]; then
    echo "Cohort content manifest is missing or empty: $manifest" >&2
    exit 1
  fi
done

train_count="$(awk 'NF { count += 1 } END { print count + 0 }' "$TRAIN_FILE_LIST")"
validation_count="$(awk 'NF { count += 1 } END { print count + 0 }' "$VALIDATION_FILE_LIST")"
if (( BATCH_SIZE != 8 )); then
  echo "This probe is fixed to batch size 8; received $BATCH_SIZE." >&2
  exit 1
fi
if (( train_count != BATCH_SIZE )); then
  echo "TRAIN_FILE_LIST must contain exactly $BATCH_SIZE entries; found $train_count." >&2
  exit 1
fi
if (( validation_count != BATCH_SIZE )); then
  echo "VALIDATION_FILE_LIST must contain exactly $BATCH_SIZE entries; found $validation_count." >&2
  exit 1
fi

processed_real="$($PYTHON_EXECUTABLE -c 'import os,sys; print(os.path.realpath(sys.argv[1]))' "$PROCESSED_ROOT")"
run_real="$($PYTHON_EXECUTABLE -c 'import os,sys; print(os.path.realpath(sys.argv[1]))' "$RUN_OUTPUT_ROOT")"
if [[ "$processed_real" == "$run_real" ]]; then
  echo "RUN_OUTPUT_ROOT must differ from PROCESSED_ROOT." >&2
  exit 1
fi
if [[ -e "$RUN_OUTPUT_ROOT" ]]; then
  echo "RUN_OUTPUT_ROOT must not already exist: $RUN_OUTPUT_ROOT" >&2
  exit 1
fi
harness_require_free_space "$RUN_OUTPUT_ROOT" "$MIN_FREE_GIB" "$MIN_FREE_BYTES" || exit 2
case "$run_real/" in
  "$processed_real/"*)
  echo "RUN_OUTPUT_ROOT must not be inside PROCESSED_ROOT." >&2
  exit 1
  ;;
esac

mkdir -p "$RUN_OUTPUT_ROOT"
started_at="$(harness_now)"
status_path="$RUN_OUTPUT_ROOT/probe_status.tsv"
harness_atomic_write_status "$status_path" "preflight" "" "$$" "$started_at"
HARNESS_SKIP_TENSORFLOW_PROBE=1 harness_write_provenance \
  "$RUN_OUTPUT_ROOT/environment.tsv" \
  "$PYTHON_EXECUTABLE" \
  "$(harness_command_line "$0" "$@")" \
  "$RUN_OUTPUT_ROOT"
exec > >(tee -a "$RUN_OUTPUT_ROOT/probe.log") 2>&1
probe_terminal=0
cleanup() {
  local exit_code=$?
  if [[ "$probe_terminal" != "1" && "$exit_code" -ne 0 ]]; then
    harness_atomic_write_status "$status_path" "failed" "$exit_code" "$$" "$started_at"
  fi
}
trap cleanup EXIT
{
  printf 'purpose\tbatch_size_compatibility_probe\n'
  printf 'batch_size\t%s\n' "$BATCH_SIZE"
  printf 'minimum_batch8_gpu_memory_mib\t%s\n' "$MIN_BATCH8_GPU_MEMORY_MIB"
  printf 'train_count\t%s\n' "$train_count"
  printf 'validation_count\t%s\n' "$validation_count"
  printf 'processed_root\t%s\n' "$processed_real"
  printf 'train_file_list_sha256\t%s\n' "$(harness_sha256 "$TRAIN_FILE_LIST")"
  printf 'validation_file_list_sha256\t%s\n' "$(harness_sha256 "$VALIDATION_FILE_LIST")"
  printf 'train_cohort_sha256\t%s\n' "$(harness_sha256 "$TRAIN_COHORT_MANIFEST")"
  printf 'validation_cohort_sha256\t%s\n' "$(harness_sha256 "$VALIDATION_COHORT_MANIFEST")"
} > "$RUN_OUTPUT_ROOT/probe_config.tsv"

set +e
harness_require_batch_hardware "$BATCH_SIZE" "$MIN_BATCH8_GPU_MEMORY_MIB"
hardware_status=$?
set -e
if [[ "$hardware_status" -ne 0 ]]; then
  harness_atomic_write_status "$status_path" "blocked_hardware" "$hardware_status" "$$" "$started_at"
  probe_terminal=1
  exit "$hardware_status"
fi

"$PYTHON_EXECUTABLE" -c \
  "import tensorflow as tf; g=tf.config.list_physical_devices('GPU'); print('GPUs:', g); raise SystemExit(0 if g else 'No GPU detected by TensorFlow.')"

"$PYTHON_EXECUTABLE" "$repo_root/scripts/hash_validation_cohort.py" \
  --output-root "$PROCESSED_ROOT" \
  --file-list "$TRAIN_FILE_LIST" \
  --verify "$TRAIN_COHORT_MANIFEST"
"$PYTHON_EXECUTABLE" "$repo_root/scripts/hash_validation_cohort.py" \
  --output-root "$PROCESSED_ROOT" \
  --file-list "$VALIDATION_FILE_LIST" \
  --verify "$VALIDATION_COHORT_MANIFEST"
"$PYTHON_EXECUTABLE" train.py \
  --data_root "$DATA_ROOT" \
  --output_root "$PROCESSED_ROOT" \
  --run_output_root "$RUN_OUTPUT_ROOT" \
  --sequences "$SEQUENCES" \
  --epochs 1 \
  --batch_size "$BATCH_SIZE" \
  --reuse_processed_only \
  --train_file_list "$TRAIN_FILE_LIST" \
  --validation_file_list "$VALIDATION_FILE_LIST"
harness_atomic_write_status "$status_path" "completed" "0" "$$" "$started_at"
probe_terminal=1
