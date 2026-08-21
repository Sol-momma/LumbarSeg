from pathlib import Path


SCRIPT_PATH = Path(__file__).parents[1] / "scripts" / "run_fixed_boundary_candidate.sh"


def test_candidate_reuses_fixed_cohorts_without_writing_processed_cache():
    script = SCRIPT_PATH.read_text(encoding="utf-8")

    assert ': "${TRAIN_FILE_LIST:?' in script
    assert ': "${VALIDATION_FILE_LIST:?' in script
    assert ': "${VALIDATION_COHORT_MANIFEST:?' in script
    assert 'EXPECTED_TRAIN_COUNT="${EXPECTED_TRAIN_COUNT:-730}"' in script
    assert 'EXPECTED_VALIDATION_COUNT="${EXPECTED_VALIDATION_COUNT:-270}"' in script
    assert '--verify "$VALIDATION_COHORT_MANIFEST"' in script
    assert '--run_output_root "$RUN_OUTPUT_ROOT"' in script
    assert '--reuse_processed_only' in script
    assert '--train_file_list "$TRAIN_FILE_LIST"' in script
    assert '--validation_file_list "$VALIDATION_FILE_LIST"' in script
    assert 'RUN_OUTPUT_ROOT must not already exist' in script
    assert 'RUN_OUTPUT_ROOT must not be inside PROCESSED_ROOT' in script


def test_candidate_changes_only_the_boundary_loss_and_requires_gpu():
    script = SCRIPT_PATH.read_text(encoding="utf-8")

    assert 'CANAL_BOUNDARY_BOOST="${CANAL_BOUNDARY_BOOST:-2.0}"' in script
    assert '--focal_class_weight_mode none' in script
    assert '--focal_canal_boundary_boost "$CANAL_BOUNDARY_BOOST"' in script
    assert "tf.config.list_physical_devices('GPU')" in script
    assert 'BATCH_SIZE="${BATCH_SIZE:-2}"' in script
    assert 'SEED="${SEED:-42}"' in script


def test_training_disables_xla_only_for_the_boundary_candidate():
    train_source = (SCRIPT_PATH.parents[1] / "train.py").read_text(encoding="utf-8")

    assert 'jit_compile = False if opt.focal_canal_boundary_boost > 0.0 else "auto"' in train_source
    assert "jit_compile=jit_compile" in train_source


def test_candidate_evaluates_and_records_terminal_status():
    script = SCRIPT_PATH.read_text(encoding="utf-8")

    assert '--evaluation_output_root "$RUN_OUTPUT_ROOT"' in script
    assert 'validation_metrics.csv' in script
    assert 'target_check.json' in script
    assert 'write_status "completed_goal_met" "0"' in script
    assert 'write_status "completed_goal_miss" "0"' in script
    assert 'write_status "failed" "$exit_code"' in script
