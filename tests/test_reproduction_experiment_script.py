from pathlib import Path


SCRIPT_PATH = Path(__file__).parents[1] / "scripts" / "run_reproduction_experiment.sh"


def test_runner_records_protocol_and_generalization_evidence() -> None:
    script = SCRIPT_PATH.read_text(encoding="utf-8")

    assert 'COHORT_DISJOINT_MODE=strict_series|author_diagnostic_slice' in script
    assert "paper_protocol_filtering_status\\tblocked" in script
    assert "paper_protocol_evaluation_status\\tunverified" in script
    assert "final_generalization_evidence" in script
    assert "split_config_sha256" in script


def test_author_diagnostic_mode_requires_frozen_split_evidence() -> None:
    script = SCRIPT_PATH.read_text(encoding="utf-8")

    assert "Author diagnostic mode requires SPLIT_CONFIG" in script
    assert 'split_mode" != author_diagnostic_*' in script
    assert "requires explicit frozen train/validation lists and manifests" in script
    assert '--cohort_disjoint_mode "$cohort_disjoint_mode"' in script


def test_fixed_validation_can_reselect_training_without_mutating_cache() -> None:
    script = SCRIPT_PATH.read_text(encoding="utf-8")

    assert "A fixed validation cohort requires both" in script
    assert "baseline training list would make a 0.55-vs-0.90 comparison" in script
    assert "--reuse_processed_cache" in script
    assert 'train_file_list="$output_root/train_files.txt"' in script
    assert '--write "$local_train_manifest"' in script


def test_runner_uses_portable_timestamps() -> None:
    script = SCRIPT_PATH.read_text(encoding="utf-8")

    assert "date -Is" not in script
    assert 'Started: $(harness_now)' in script


def test_runner_uses_one_resolved_python_for_every_stage() -> None:
    script = SCRIPT_PATH.read_text(encoding="utf-8")

    assert 'python_bin="${PYTHON_BIN:-}"' in script
    assert 'python_bin="python3"' in script
    assert '"$python_bin" train.py' in script
    assert '"$python_bin" evaluate.py' in script
    assert "\n  python train.py" not in script
