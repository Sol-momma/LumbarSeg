from pathlib import Path


SCRIPT_PATH = Path(__file__).parents[1] / "scripts" / "run_wsl_batch_probe.sh"


def test_wsl_probe_is_fixed_cohort_and_output_isolated():
    script = SCRIPT_PATH.read_text(encoding="utf-8")

    assert 'BATCH_SIZE="${BATCH_SIZE:-8}"' in script
    assert ': "${TRAIN_FILE_LIST:?' in script
    assert ': "${VALIDATION_FILE_LIST:?' in script
    assert '--train_file_list "$TRAIN_FILE_LIST"' in script
    assert '--validation_file_list "$VALIDATION_FILE_LIST"' in script
    assert '--run_output_root "$RUN_OUTPUT_ROOT"' in script
    assert '--reuse_processed_only' in script
    assert 'RUN_OUTPUT_ROOT must differ from PROCESSED_ROOT' in script
    assert 'RUN_OUTPUT_ROOT must not already exist' in script
    assert 'train_count != BATCH_SIZE' in script
    assert 'validation_count != BATCH_SIZE' in script
    assert 'BATCH_SIZE != 8' in script


def test_wsl_probe_requires_tensorflow_gpu():
    script = SCRIPT_PATH.read_text(encoding="utf-8")

    assert "tf.config.list_physical_devices('GPU')" in script
    assert "No GPU detected by TensorFlow" in script
