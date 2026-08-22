from pathlib import Path

import pytest

from spine_baseline.training_resume import (
    read_resume_best_metric,
    validate_training_resume_state,
    write_training_resume_evidence,
)


TRAIN = ["001_t2_SPACE_s001.npz", "001_t2_SPACE_s002.npz"]
VALIDATION = ["002_t2_SPACE_s001.npz"]


def _write_cohorts(root: Path) -> None:
    (root / "train_files.txt").write_text("\n".join(TRAIN) + "\n", encoding="utf-8")
    (root / "validation_files.txt").write_text(
        "\n".join(VALIDATION) + "\n",
        encoding="utf-8",
    )


def test_fresh_run_allows_shell_created_empty_output_root(tmp_path: Path) -> None:
    (tmp_path / "inputs").mkdir()
    (tmp_path / "run_config.tsv").write_text("key\tvalue\n", encoding="utf-8")

    paths = validate_training_resume_state(
        tmp_path,
        resume_requested=False,
        train_files=TRAIN,
        validation_files=VALIDATION,
    )

    assert paths.backup_dir == tmp_path / "checkpoints" / "training_backup"


@pytest.mark.parametrize("artifact", ["best_model.keras", "final_model.keras", "training_log.csv"])
def test_fresh_run_rejects_existing_training_artifact(tmp_path: Path, artifact: str) -> None:
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    (checkpoint_dir / artifact).write_text("old", encoding="utf-8")

    with pytest.raises(ValueError, match="Training artifacts already exist"):
        validate_training_resume_state(
            tmp_path,
            resume_requested=False,
            train_files=TRAIN,
            validation_files=VALIDATION,
        )


def test_resume_requires_backup_and_exact_saved_cohorts(tmp_path: Path) -> None:
    _write_cohorts(tmp_path)

    with pytest.raises(ValueError, match="No interrupted-training backup"):
        validate_training_resume_state(
            tmp_path,
            resume_requested=True,
            train_files=TRAIN,
            validation_files=VALIDATION,
        )

    (tmp_path / "checkpoints" / "training_backup").mkdir(parents=True)
    paths = validate_training_resume_state(
        tmp_path,
        resume_requested=True,
        train_files=TRAIN,
        validation_files=VALIDATION,
    )
    assert paths.backup_dir.is_dir()

    with pytest.raises(ValueError, match="training cohort differs"):
        validate_training_resume_state(
            tmp_path,
            resume_requested=True,
            train_files=[*TRAIN, "003_t2_SPACE_s001.npz"],
            validation_files=VALIDATION,
        )


def test_resume_rejects_completed_run(tmp_path: Path) -> None:
    _write_cohorts(tmp_path)
    checkpoint_dir = tmp_path / "checkpoints"
    (checkpoint_dir / "training_backup").mkdir(parents=True)
    (checkpoint_dir / "final_model.keras").write_text("complete", encoding="utf-8")

    with pytest.raises(ValueError, match="already complete"):
        validate_training_resume_state(
            tmp_path,
            resume_requested=True,
            train_files=TRAIN,
            validation_files=VALIDATION,
        )


def test_resume_recovers_best_validation_metric_from_existing_log(tmp_path: Path) -> None:
    training_log = tmp_path / "training_log.csv"
    training_log.write_text(
        "epoch,loss,val_mean_iou\n0,0.8,0.41\n1,0.7,0.53\n2,0.6,0.49\n",
        encoding="utf-8",
    )

    assert read_resume_best_metric(training_log, "val_mean_iou") == 0.53


def test_resume_evidence_marks_recovered_run_noncanonical(tmp_path: Path) -> None:
    evidence = tmp_path / "training_resume.tsv"

    write_training_resume_evidence(
        evidence,
        resume_requested=True,
        historical_best=0.53,
    )

    contents = evidence.read_text(encoding="utf-8")
    assert "restored_state\tmodel_optimizer_completed_epoch" in contents
    assert "historical_best_val_mean_iou\t0.53000000000000003" in contents
    assert "callback_patience_state\tearly_stopping_and_lr_patience_reset" in contents
    assert "equivalent_to_uninterrupted_run\tfalse" in contents
    assert "final_comparison_eligibility\trequires_fresh_uninterrupted_rerun" in contents


def test_fresh_run_evidence_remains_eligible_for_protocol_gates(tmp_path: Path) -> None:
    evidence = tmp_path / "training_resume.tsv"

    write_training_resume_evidence(
        evidence,
        resume_requested=False,
        historical_best=None,
    )

    contents = evidence.read_text(encoding="utf-8")
    assert "restored_state\tnone_fresh_run" in contents
    assert "historical_best_val_mean_iou\tnone" in contents
    assert "equivalent_to_uninterrupted_run\ttrue" in contents
    assert "final_comparison_eligibility\teligible_if_all_other_protocol_gates_pass" in contents


@pytest.mark.parametrize(
    ("contents", "message"),
    [
        ("epoch,loss\n0,0.8\n", "does not contain"),
        ("epoch,val_mean_iou\n", "no completed epochs"),
        ("epoch,val_mean_iou\n0,nan\n", "Non-finite"),
    ],
)
def test_resume_rejects_unusable_training_log(
    tmp_path: Path,
    contents: str,
    message: str,
) -> None:
    training_log = tmp_path / "training_log.csv"
    training_log.write_text(contents, encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        read_resume_best_metric(training_log, "val_mean_iou")
