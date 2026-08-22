from __future__ import annotations

import csv
from dataclasses import dataclass
import math
from pathlib import Path


@dataclass(frozen=True)
class TrainingResumePaths:
    """Paths whose presence distinguishes a fresh run from an interrupted run."""

    checkpoint_dir: Path
    backup_dir: Path
    final_model: Path


def write_training_resume_evidence(
    path: Path,
    *,
    resume_requested: bool,
    historical_best: float | None,
) -> None:
    """Record what an interrupted Keras run does and does not restore.

    BackupAndRestore recovers the model, optimizer, and completed epoch, but
    callback counters such as EarlyStopping patience are process-local.  A
    resumed run is therefore safe for recovering work and inspecting a
    candidate, but it must not silently become the canonical final comparison
    against a fresh uninterrupted baseline.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    if resume_requested:
        restored_state = "model_optimizer_completed_epoch"
        checkpoint_guard = "historical_best_restored_from_training_log"
        callback_state = "early_stopping_and_lr_patience_reset"
        uninterrupted_equivalence = "false"
        final_comparison_eligibility = "requires_fresh_uninterrupted_rerun"
    else:
        restored_state = "none_fresh_run"
        checkpoint_guard = "fresh_callback_state"
        callback_state = "fresh_callback_state"
        uninterrupted_equivalence = "true"
        final_comparison_eligibility = "eligible_if_all_other_protocol_gates_pass"

    best_value = "none" if historical_best is None else format(historical_best, ".17g")
    rows = [
        ("resume_requested", str(resume_requested).lower()),
        ("restored_state", restored_state),
        ("historical_best_val_mean_iou", best_value),
        ("best_checkpoint_guard", checkpoint_guard),
        ("callback_patience_state", callback_state),
        ("equivalent_to_uninterrupted_run", uninterrupted_equivalence),
        ("final_comparison_eligibility", final_comparison_eligibility),
    ]
    path.write_text(
        "key\tvalue\n" + "".join(f"{key}\t{value}\n" for key, value in rows),
        encoding="utf-8",
    )


def read_resume_best_metric(training_log: Path, monitor: str) -> float:
    """Recover the historical best so resume cannot overwrite a better model.

    ``BackupAndRestore`` restores model, optimizer, and epoch but not another
    callback's in-memory ``best`` field. ModelCheckpoint would otherwise reset
    that field to negative infinity and accept the first resumed epoch even
    when it regressed from the pre-interruption best.
    """
    if not training_log.is_file():
        raise ValueError(f"Resume requires the existing training log: {training_log}")
    values: list[float] = []
    with training_log.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or monitor not in reader.fieldnames:
            raise ValueError(f"Training log does not contain resume monitor {monitor!r}: {training_log}")
        for row in reader:
            raw_value = (row.get(monitor) or "").strip()
            try:
                value = float(raw_value)
            except ValueError as exc:
                raise ValueError(f"Invalid {monitor} in training log: {raw_value!r}") from exc
            if not math.isfinite(value):
                raise ValueError(f"Non-finite {monitor} in training log: {raw_value!r}")
            values.append(value)
    if not values:
        raise ValueError(f"Training log has no completed epochs: {training_log}")
    return max(values)


def _read_saved_cohort(path: Path) -> list[str]:
    if not path.is_file():
        raise ValueError(f"Resume requires the saved cohort file: {path}")
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def validate_training_resume_state(
    run_output_root: Path,
    *,
    resume_requested: bool,
    train_files: list[str],
    validation_files: list[str],
) -> TrainingResumePaths:
    """Reject ambiguous checkpoint reuse before Keras mutates a run directory.

    Keras ``BackupAndRestore`` automatically resumes whenever its backup exists.
    That convenience is dangerous for audited experiments: a mistyped output
    path could silently continue another candidate.  The caller therefore has
    to opt in explicitly, and the exact saved cohorts must still match before
    TensorFlow is allowed to restore optimizer or epoch state.
    """
    checkpoint_dir = run_output_root / "checkpoints"
    backup_dir = checkpoint_dir / "training_backup"
    final_model = checkpoint_dir / "final_model.keras"
    paths = TrainingResumePaths(checkpoint_dir, backup_dir, final_model)

    if resume_requested:
        if final_model.exists():
            raise ValueError(
                f"Training is already complete; refusing to resume over the final model: {final_model}"
            )
        if not backup_dir.is_dir():
            raise ValueError(f"No interrupted-training backup exists: {backup_dir}")

        saved_train = _read_saved_cohort(run_output_root / "train_files.txt")
        saved_validation = _read_saved_cohort(run_output_root / "validation_files.txt")
        if saved_train != train_files:
            raise ValueError("Resume training cohort differs from the saved train_files.txt")
        if saved_validation != validation_files:
            raise ValueError("Resume validation cohort differs from the saved validation_files.txt")
        return paths

    # A fresh audited run must never inherit model or optimizer state.  The
    # output root may already contain immutable inputs and a status file created
    # by the shell harness, so only training artifacts are rejected here.
    existing_artifacts = [
        backup_dir,
        checkpoint_dir / "best_model.keras",
        checkpoint_dir / "final_model.keras",
        checkpoint_dir / "training_log.csv",
    ]
    existing = [path for path in existing_artifacts if path.exists()]
    if existing:
        preview = ", ".join(str(path) for path in existing)
        raise ValueError(
            "Training artifacts already exist. Use an explicit resume only for an "
            f"interrupted run with identical cohorts: {preview}"
        )
    return paths
