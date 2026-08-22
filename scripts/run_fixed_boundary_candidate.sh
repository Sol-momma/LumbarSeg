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
: "${TRAIN_COHORT_MANIFEST:?Set TRAIN_COHORT_MANIFEST to the frozen training hashes.}"
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
ALLOW_NO_TMUX="${ALLOW_NO_TMUX:-0}"
MIN_FREE_GIB="${MIN_FREE_GIB:-20}"
MIN_FREE_BYTES="${MIN_FREE_BYTES:-}"
HEARTBEAT_INTERVAL_SECONDS="${HEARTBEAT_INTERVAL_SECONDS:-60}"
HANG_TIMEOUT_SECONDS="${HANG_TIMEOUT_SECONDS:-1800}"
DISABLE_HEARTBEAT="${DISABLE_HEARTBEAT:-0}"
RESUME_RUN="${RESUME_RUN:-0}"

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
checker="$repo_root/scripts/check_reproduction_target.py"
hasher="$repo_root/scripts/hash_validation_cohort.py"
heartbeat_runner="$repo_root/scripts/run_long_run_heartbeat.sh"
# shellcheck source=scripts/long_run_harness.sh
source "$repo_root/scripts/long_run_harness.sh"
status_path=""
lock_owned=0
heartbeat_pid=""
started_at="$(harness_now)"
resume_phase="train"
resume_training=0
initialize_run=1

write_status() {
  local status="$1"
  local exit_code="$2"
  local tmp_path="${status_path}.tmp"
  {
    printf 'key\tvalue\n'
    printf 'status\t%s\n' "$status"
    printf 'exit_code\t%s\n' "$exit_code"
    printf 'pid\t%s\n' "$$"
    printf 'boot_id\t%s\n' "$(harness_boot_id)"
    printf 'started_at\t%s\n' "$started_at"
    printf 'updated_at\t%s\n' "$(harness_now)"
    printf 'run_output_root\t%s\n' "$RUN_OUTPUT_ROOT"
  } > "$tmp_path"
  mv "$tmp_path" "$status_path"
}

cleanup() {
  local exit_code=$?
  if [[ -n "$heartbeat_pid" ]]; then
    kill "$heartbeat_pid" 2>/dev/null || true
    wait "$heartbeat_pid" 2>/dev/null || true
  fi
  if [[ "$lock_owned" -eq 1 ]]; then
    rm -f "$GPU_LOCK_DIR/owner.tsv"
    rmdir "$GPU_LOCK_DIR" 2>/dev/null || true
  fi
  if [[ "$exit_code" -ne 0 && -n "$status_path" && -d "$RUN_OUTPUT_ROOT" ]]; then
    current_status="$(harness_status_value "$status_path" status 2>/dev/null || true)"
    if [[ "$current_status" != "stale_interrupted_restart_required" ]]; then
      write_status "failed" "$exit_code"
    fi
  fi
}
trap cleanup EXIT

cd "$repo_root"

if [[ "$ALLOW_NO_TMUX" != "1" && -z "${TMUX:-}" ]]; then
  echo "Long GPU candidates must run inside tmux so an SSH disconnect does not terminate training." >&2
  exit 2
fi

for path in "$TRAIN_FILE_LIST" "$VALIDATION_FILE_LIST" "$TRAIN_COHORT_MANIFEST" "$VALIDATION_COHORT_MANIFEST"; do
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
harness_require_free_space "$RUN_OUTPUT_ROOT" "$MIN_FREE_GIB" "$MIN_FREE_BYTES" || exit 2
harness_require_batch_hardware "$BATCH_SIZE" || exit $?

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
  if [[ "$RESUME_RUN" != "1" ]]; then
    echo "RUN_OUTPUT_ROOT must not already exist: $RUN_OUTPUT_ROOT" >&2
    exit 2
  fi
  status_path="$RUN_OUTPUT_ROOT/campaign_status.tsv"
  state="$(harness_running_state "$status_path")"
  if [[ "$state" == "live" ]]; then
    echo "RUN_OUTPUT_ROOT belongs to a live process; refusing a second runner: $RUN_OUTPUT_ROOT" >&2
    exit 2
  fi
  for pair in \
    "$TRAIN_FILE_LIST:$RUN_OUTPUT_ROOT/inputs/fixed_train_files.txt" \
    "$VALIDATION_FILE_LIST:$RUN_OUTPUT_ROOT/inputs/fixed_validation_files.txt" \
    "$TRAIN_COHORT_MANIFEST:$RUN_OUTPUT_ROOT/inputs/fixed_train_cohort.tsv" \
    "$VALIDATION_COHORT_MANIFEST:$RUN_OUTPUT_ROOT/inputs/fixed_validation_cohort.tsv"; do
    source_path="${pair%%:*}"
    saved_path="${pair#*:}"
    if [[ ! -s "$saved_path" || "$(harness_sha256 "$source_path")" != "$(harness_sha256 "$saved_path")" ]]; then
      echo "Resume inputs differ from the frozen copies; refusing to reuse output: $saved_path" >&2
      exit 2
    fi
  done
  if [[ -s "$RUN_OUTPUT_ROOT/checkpoints/final_model.keras" ]]; then
    resume_phase="evaluate"
    initialize_run=0
  elif [[ -d "$RUN_OUTPUT_ROOT/checkpoints/training_backup" ]]; then
    # train.py validates the saved train/validation lists again before Keras
    # restores optimizer and epoch state. This is the only permitted partial
    # training resume; an arbitrary checkpoint is not sufficient evidence.
    resume_phase="train"
    resume_training=1
    initialize_run=0
  else
    write_status "stale_interrupted_restart_required" "75"
    echo "The previous process is stale but training did not finish. Epoch-level overwrite is unsafe; use a new RUN_OUTPUT_ROOT." >&2
    exit 75
  fi
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
  --file-list "$TRAIN_FILE_LIST" \
  --verify "$TRAIN_COHORT_MANIFEST"
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
  lock_state="$(harness_running_state "$GPU_LOCK_DIR/owner.tsv")"
  if [[ "$RESUME_RUN" == "1" && "$lock_state" == "stale" ]]; then
    stale_lock="${GPU_LOCK_DIR}.stale_$(date +%Y%m%d_%H%M%S)"
    mv "$GPU_LOCK_DIR" "$stale_lock"
    echo "Preserved stale GPU lock evidence at: $stale_lock"
    mkdir "$GPU_LOCK_DIR"
  else
    echo "GPU experiment lock already exists or cannot be proven stale: $GPU_LOCK_DIR" >&2
    exit 2
  fi
fi
lock_owned=1
harness_atomic_write_status "$GPU_LOCK_DIR/owner.tsv" "running" "" "$$" "$started_at"

"$PYTHON_EXECUTABLE" -c \
  "import tensorflow as tf; g=tf.config.list_physical_devices('GPU'); print('GPUs:', g); raise SystemExit(0 if g else 'No GPU detected by TensorFlow.')"

if [[ "$initialize_run" == "1" ]]; then
  mkdir -p "$RUN_OUTPUT_ROOT/inputs"
  status_path="$RUN_OUTPUT_ROOT/campaign_status.tsv"
  cp "$TRAIN_FILE_LIST" "$RUN_OUTPUT_ROOT/inputs/fixed_train_files.txt"
  cp "$VALIDATION_FILE_LIST" "$RUN_OUTPUT_ROOT/inputs/fixed_validation_files.txt"
  cp "$TRAIN_COHORT_MANIFEST" "$RUN_OUTPUT_ROOT/inputs/fixed_train_cohort.tsv"
  cp "$VALIDATION_COHORT_MANIFEST" "$RUN_OUTPUT_ROOT/inputs/fixed_validation_cohort.tsv"
fi

git_revision="$(git rev-parse HEAD)"
train_sha="$(sha256sum "$TRAIN_FILE_LIST" | awk '{print $1}')"
validation_sha="$(sha256sum "$VALIDATION_FILE_LIST" | awk '{print $1}')"
manifest_sha="$(sha256sum "$VALIDATION_COHORT_MANIFEST" | awk '{print $1}')"
train_manifest_sha="$(sha256sum "$TRAIN_COHORT_MANIFEST" | awk '{print $1}')"
if [[ "$initialize_run" == "1" ]]; then
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
  printf 'train_cohort_sha256\t%s\n' "$train_manifest_sha"
  printf 'validation_cohort_sha256\t%s\n' "$manifest_sha"
  printf 'processed_root\t%s\n' "$processed_real"
  printf 'cohort_disjoint_mode\tstrict_series\n'
  printf 'split_mode\tspider_overview_series_fixed_cohort\n'
  printf 'final_generalization_evidence\ttrue\n'
  printf 'paper_protocol_preprocessing_status\tunverified\n'
  printf 'paper_protocol_preprocessing_evidence\tpublic paper and code disagree on image dimensions and orientation details\n'
  printf 'paper_protocol_filtering_status\tblocked\n'
  printf 'paper_protocol_filtering_evidence\texact 55 percent rule and selected 1000 slices were not published\n'
  printf 'paper_protocol_split_status\tunverified\n'
  printf 'paper_protocol_split_evidence\tfixed SPIDER series cohorts are auditable but the exact author cohort is unavailable\n'
  printf 'paper_protocol_training_status\tunverified\n'
  printf 'paper_protocol_training_evidence\tpublic paper and code report conflicting batch and loss settings\n'
  printf 'paper_protocol_evaluation_status\tunverified\n'
  printf 'paper_protocol_evaluation_evidence\thard and probability Dice are recorded because the paper aggregation is not fully specified\n'
} > "$RUN_OUTPUT_ROOT/run_config.tsv"
  harness_write_provenance \
    "$RUN_OUTPUT_ROOT/environment.tsv" \
    "$PYTHON_EXECUTABLE" \
    "$(harness_command_line "$0" "$@")" \
    "$RUN_OUTPUT_ROOT"
fi
write_status "running" ""

log_path="$RUN_OUTPUT_ROOT/gpu-queue.log"
exec > >(tee -a "$log_path") 2>&1
echo "Run phase: $resume_phase"
echo "Resume training backup: $resume_training"
if [[ "$DISABLE_HEARTBEAT" != "1" ]]; then
  HEARTBEAT_PATH="$RUN_OUTPUT_ROOT/heartbeat.tsv" \
  WATCH_PID="$$" \
  TRAINING_LOG="$RUN_OUTPUT_ROOT/checkpoints/training_log.csv" \
  HEARTBEAT_INTERVAL_SECONDS="$HEARTBEAT_INTERVAL_SECONDS" \
  HANG_TIMEOUT_SECONDS="$HANG_TIMEOUT_SECONDS" \
  DISK_PATH="$RUN_OUTPUT_ROOT" \
  MIN_FREE_GIB="$MIN_FREE_GIB" \
    bash "$heartbeat_runner" &
  heartbeat_pid=$!
fi

if [[ "$resume_phase" == "train" ]]; then
resume_training_args=()
if [[ "$resume_training" == "1" ]]; then
  resume_training_args=(--resume_training)
fi
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
  --cohort_disjoint_mode strict_series \
  --reuse_processed_only \
  "${resume_training_args[@]}" \
  --train_file_list "$TRAIN_FILE_LIST" \
  --validation_file_list "$VALIDATION_FILE_LIST"
fi

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
  --training-resume "$RUN_OUTPUT_ROOT/training_resume.tsv" \
  --require-sequence T2_SPACE
check_status=$?
set -e

case "$check_status" in
  0)
    if [[ "$resume_training" == "1" ]]; then
      # A resumed result is useful recovery evidence, but callback patience is
      # reset by Keras. Do not let it masquerade as the canonical uninterrupted
      # comparison even when its numeric score reaches the campaign target.
      write_status "completed_goal_met_noncanonical_resumed" "0"
    else
      write_status "completed_goal_met" "0"
    fi
    ;;
  1)
    # A score miss is a valid completed experiment, not an execution failure.
    if [[ "$resume_training" == "1" ]]; then
      write_status "completed_goal_miss_noncanonical_resumed" "0"
    else
      write_status "completed_goal_miss" "0"
    fi
    ;;
  *)
    echo "Target evidence is invalid." >&2
    exit 2
    ;;
esac
