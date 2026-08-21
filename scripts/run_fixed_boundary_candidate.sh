#!/usr/bin/env bash
set -euo pipefail

# Run one controlled boundary-loss candidate against the already frozen T2
# SPACE cohorts.  The processed cache is an input only; every new model, log,
# metric, and status file is written below a new RUN_OUTPUT_ROOT.
: "${DATA_ROOT:?Set DATA_ROOT to the SPIDER DataSet directory.}"
: "${PROCESSED_ROOT:?Set PROCESSED_ROOT to the existing processed cache.}"
: "${RUN_OUTPUT_ROOT:?Set RUN_OUTPUT_ROOT to a new WSL-side result directory.}"
: "${TRAIN_FILE_LIST:?Set TRAIN_FILE_LIST to the frozen 730-slice training list.}"
: "${VALIDATION_FILE_LIST:?Set VALIDATION_FILE_LIST to the frozen 270-slice validation list.}"
: "${VALIDATION_COHORT_MANIFEST:?Set VALIDATION_COHORT_MANIFEST to the frozen validation hashes.}"

PYTHON_EXECUTABLE="${PYTHON_EXECUTABLE:-python}"
BATCH_SIZE="${BATCH_SIZE:-2}"
EPOCHS="${EPOCHS:-100}"
SEED="${SEED:-42}"
SEQUENCES="${SEQUENCES:-T2_SPACE}"
CANAL_BOUNDARY_BOOST="${CANAL_BOUNDARY_BOOST:-2.0}"
EXPECTED_TRAIN_COUNT="${EXPECTED_TRAIN_COUNT:-730}"
EXPECTED_VALIDATION_COUNT="${EXPECTED_VALIDATION_COUNT:-270}"
GPU_LOCK_DIR="${GPU_LOCK_DIR:-/tmp/lumbarseg-fixed-candidate-gpu.lock}"

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
checker="$repo_root/scripts/check_reproduction_target.py"
hasher="$repo_root/scripts/hash_validation_cohort.py"
status_path=""
lock_owned=0

write_status() {
  local status="$1"
  local exit_code="$2"
  local tmp_path="${status_path}.tmp"
  {
    printf 'key\tvalue\n'
    printf 'status\t%s\n' "$status"
    printf 'exit_code\t%s\n' "$exit_code"
    printf 'updated_at\t%s\n' "$(date --iso-8601=seconds)"
    printf 'run_output_root\t%s\n' "$RUN_OUTPUT_ROOT"
  } > "$tmp_path"
  mv "$tmp_path" "$status_path"
}

cleanup() {
  local exit_code=$?
  if [[ "$lock_owned" -eq 1 ]]; then
    rmdir "$GPU_LOCK_DIR" 2>/dev/null || true
  fi
  if [[ "$exit_code" -ne 0 && -n "$status_path" && -d "$RUN_OUTPUT_ROOT" ]]; then
    write_status "failed" "$exit_code"
  fi
}
trap cleanup EXIT

cd "$repo_root"

for path in "$TRAIN_FILE_LIST" "$VALIDATION_FILE_LIST" "$VALIDATION_COHORT_MANIFEST"; do
  if [[ ! -s "$path" ]]; then
    echo "Required fixed-cohort evidence is missing or empty: $path" >&2
    exit 2
  fi
done
if [[ ! -d "$PROCESSED_ROOT/images" || ! -d "$PROCESSED_ROOT/masks" ]]; then
  echo "Processed cache must contain images/ and masks/: $PROCESSED_ROOT" >&2
  exit 2
fi
if [[ "$SEQUENCES" != "T2_SPACE" ]]; then
  echo "This controlled candidate is fixed to T2_SPACE; received $SEQUENCES." >&2
  exit 2
fi
if [[ "$BATCH_SIZE" -ne 2 ]]; then
  echo "This controlled candidate is fixed to batch size 2; received $BATCH_SIZE." >&2
  exit 2
fi

train_count="$(awk 'NF { count += 1 } END { print count + 0 }' "$TRAIN_FILE_LIST")"
validation_count="$(awk 'NF { count += 1 } END { print count + 0 }' "$VALIDATION_FILE_LIST")"
if [[ "$train_count" -ne "$EXPECTED_TRAIN_COUNT" ]]; then
  echo "Expected $EXPECTED_TRAIN_COUNT training slices; found $train_count." >&2
  exit 2
fi
if [[ "$validation_count" -ne "$EXPECTED_VALIDATION_COUNT" ]]; then
  echo "Expected $EXPECTED_VALIDATION_COUNT validation slices; found $validation_count." >&2
  exit 2
fi

processed_real="$($PYTHON_EXECUTABLE -c 'import os,sys; print(os.path.realpath(sys.argv[1]))' "$PROCESSED_ROOT")"
run_real="$($PYTHON_EXECUTABLE -c 'import os,sys; print(os.path.realpath(sys.argv[1]))' "$RUN_OUTPUT_ROOT")"
if [[ "$processed_real" == "$run_real" ]]; then
  echo "RUN_OUTPUT_ROOT must differ from PROCESSED_ROOT." >&2
  exit 2
fi
if [[ -e "$RUN_OUTPUT_ROOT" ]]; then
  echo "RUN_OUTPUT_ROOT must not already exist: $RUN_OUTPUT_ROOT" >&2
  exit 2
fi
case "$run_real/" in
  "$processed_real/"*)
    echo "RUN_OUTPUT_ROOT must not be inside PROCESSED_ROOT." >&2
    exit 2
    ;;
esac
# The frozen manifest proves that validation did not silently change since the
# baseline.  This reads every selected image/mask but never rewrites the cache.
"$PYTHON_EXECUTABLE" "$hasher" \
  --output-root "$PROCESSED_ROOT" \
  --file-list "$VALIDATION_FILE_LIST" \
  --verify "$VALIDATION_COHORT_MANIFEST"

if [[ -n "$(git status --porcelain --untracked-files=no)" ]]; then
  echo "Repository has tracked changes; refusing an unrecorded experiment." >&2
  exit 2
fi
if pgrep -af '[p]ython(3)? .*(train.py|evaluate.py)' >/dev/null; then
  echo "Another LumbarSeg training or evaluation process is active." >&2
  exit 2
fi
if ! mkdir "$GPU_LOCK_DIR" 2>/dev/null; then
  echo "GPU experiment lock already exists: $GPU_LOCK_DIR" >&2
  exit 2
fi
lock_owned=1

"$PYTHON_EXECUTABLE" -c \
  "import tensorflow as tf; g=tf.config.list_physical_devices('GPU'); print('GPUs:', g); raise SystemExit(0 if g else 'No GPU detected by TensorFlow.')"

mkdir -p "$RUN_OUTPUT_ROOT/inputs"
status_path="$RUN_OUTPUT_ROOT/campaign_status.tsv"
cp "$TRAIN_FILE_LIST" "$RUN_OUTPUT_ROOT/inputs/fixed_train_files.txt"
cp "$VALIDATION_FILE_LIST" "$RUN_OUTPUT_ROOT/inputs/fixed_validation_files.txt"
cp "$VALIDATION_COHORT_MANIFEST" "$RUN_OUTPUT_ROOT/inputs/fixed_validation_cohort.tsv"

git_revision="$(git rev-parse HEAD)"
train_sha="$(sha256sum "$TRAIN_FILE_LIST" | awk '{print $1}')"
validation_sha="$(sha256sum "$VALIDATION_FILE_LIST" | awk '{print $1}')"
manifest_sha="$(sha256sum "$VALIDATION_COHORT_MANIFEST" | awk '{print $1}')"
{
  printf 'key\tvalue\n'
  printf 'preset\tt2_space_canal_boundary_focal\n'
  printf 'git_revision\t%s\n' "$git_revision"
  printf 'sequences\t%s\n' "$SEQUENCES"
  printf 'batch_size\t%s\n' "$BATCH_SIZE"
  printf 'epochs\t%s\n' "$EPOCHS"
  printf 'seed\t%s\n' "$SEED"
  printf 'focal_class_weight_mode\tnone\n'
  printf 'focal_canal_boundary_boost\t%s\n' "$CANAL_BOUNDARY_BOOST"
  printf 'train_slices\t%s\n' "$train_count"
  printf 'validation_slices\t%s\n' "$validation_count"
  printf 'train_file_list_sha256\t%s\n' "$train_sha"
  printf 'validation_file_list_sha256\t%s\n' "$validation_sha"
  printf 'validation_cohort_sha256\t%s\n' "$manifest_sha"
  printf 'processed_root\t%s\n' "$processed_real"
} > "$RUN_OUTPUT_ROOT/run_config.tsv"
write_status "running" ""

"$PYTHON_EXECUTABLE" train.py \
  --data_root "$DATA_ROOT" \
  --output_root "$PROCESSED_ROOT" \
  --run_output_root "$RUN_OUTPUT_ROOT" \
  --sequences "$SEQUENCES" \
  --batch_size "$BATCH_SIZE" \
  --epochs "$EPOCHS" \
  --seed "$SEED" \
  --focal_class_weight_mode none \
  --focal_canal_boundary_boost "$CANAL_BOUNDARY_BOOST" \
  --reuse_processed_only \
  --train_file_list "$TRAIN_FILE_LIST" \
  --validation_file_list "$VALIDATION_FILE_LIST"

"$PYTHON_EXECUTABLE" evaluate.py \
  --data_root "$DATA_ROOT" \
  --output_root "$PROCESSED_ROOT" \
  --evaluation_output_root "$RUN_OUTPUT_ROOT" \
  --sequences "$SEQUENCES" \
  --batch_size "$BATCH_SIZE" \
  --focal_canal_boundary_boost "$CANAL_BOUNDARY_BOOST" \
  --model_path "$RUN_OUTPUT_ROOT/checkpoints/best_model.keras" \
  --file_list "$VALIDATION_FILE_LIST"

set +e
"$PYTHON_EXECUTABLE" "$checker" \
  "$RUN_OUTPUT_ROOT/validation_metrics.csv" \
  --output "$RUN_OUTPUT_ROOT/target_check.json" \
  --run-config "$RUN_OUTPUT_ROOT/run_config.tsv" \
  --require-sequence T2_SPACE
check_status=$?
set -e

case "$check_status" in
  0)
    write_status "completed_goal_met" "0"
    ;;
  1)
    # A score miss is a valid completed experiment, not an execution failure.
    write_status "completed_goal_miss" "0"
    ;;
  *)
    echo "Target evidence is invalid." >&2
    exit 2
    ;;
esac
