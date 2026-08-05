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

BATCH_SIZE="${BATCH_SIZE:-8}"
SEQUENCES="${SEQUENCES:-T2_SPACE}"
PYTHON_EXECUTABLE="${PYTHON_EXECUTABLE:-python}"

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

if [[ ! -f "$TRAIN_FILE_LIST" || ! -s "$TRAIN_FILE_LIST" ]]; then
  echo "Training cohort file is missing or empty: $TRAIN_FILE_LIST" >&2
  exit 1
fi
if [[ ! -f "$VALIDATION_FILE_LIST" || ! -s "$VALIDATION_FILE_LIST" ]]; then
  echo "Validation cohort file is missing or empty: $VALIDATION_FILE_LIST" >&2
  exit 1
fi

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
case "$run_real/" in
  "$processed_real/"*)
  echo "RUN_OUTPUT_ROOT must not be inside PROCESSED_ROOT." >&2
  exit 1
  ;;
esac

"$PYTHON_EXECUTABLE" -c \
  "import tensorflow as tf; g=tf.config.list_physical_devices('GPU'); print('GPUs:', g); raise SystemExit(0 if g else 'No GPU detected by TensorFlow.')"

mkdir -p "$RUN_OUTPUT_ROOT"
{
  printf 'purpose\tbatch_size_compatibility_probe\n'
  printf 'batch_size\t%s\n' "$BATCH_SIZE"
  printf 'train_count\t%s\n' "$train_count"
  printf 'validation_count\t%s\n' "$validation_count"
  printf 'processed_root\t%s\n' "$processed_real"
} > "$RUN_OUTPUT_ROOT/probe_config.tsv"

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
