from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path
import sys

import numpy as np
import pandas as pd

# Running this file as ``python scripts/...`` puts only ``scripts`` on
# sys.path. Add the repository root explicitly so the documented command works
# without requiring an editable package install in the Windows/WSL environment.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from spine_baseline.filtering import FILTER_DEFINITION, evaluate_slice_filter


def classify_sequence(filename: str) -> str:
    """Classify names without importing the OpenCV-backed extraction module."""

    name = filename.lower()
    if "t2_space" in name or "t2_sag_space" in name or "space" in name:
        return "T2_SPACE"
    if "t2" in name:
        return "T2"
    if "t1" in name:
        return "T1"
    return "Unknown"


def get_series_id(slice_filename: str) -> str:
    return slice_filename.removesuffix(".npz").rsplit("_s", 1)[0]


def split_train_val(data_root: Path, files: list[str]) -> tuple[list[str], list[str], list[str]]:
    overview = pd.read_csv(data_root / "SPIDER Lumbar Spine Segmentation Overview.csv")
    train_ids = set(overview.loc[overview["subset"] == "training", "new_file_name"].astype(str))
    validation_ids = set(overview.loc[overview["subset"] == "validation", "new_file_name"].astype(str))
    train = [name for name in files if get_series_id(name) in train_ids]
    validation = [name for name in files if get_series_id(name) in validation_ids]
    unmatched = [
        name for name in files
        if get_series_id(name) not in train_ids and get_series_id(name) not in validation_ids
    ]
    return train, validation, unmatched


def parse_thresholds(raw: str) -> list[float]:
    thresholds = [float(item.strip()) for item in raw.split(",") if item.strip()]
    if not thresholds:
        raise ValueError("At least one threshold is required")
    if len(set(thresholds)) != len(thresholds):
        raise ValueError("Thresholds must be unique")
    if any(not 0.0 <= value <= 1.0 for value in thresholds):
        raise ValueError("Thresholds must be within [0, 1]")
    return thresholds


def cap_files_by_sequence(files: list[str], max_per_sequence: int | None) -> tuple[list[str], int]:
    grouped: dict[str, list[str]] = {}
    for filename in files:
        grouped.setdefault(classify_sequence(get_series_id(filename)), []).append(filename)

    selected: list[str] = []
    removed = 0
    for sequence_files in grouped.values():
        if max_per_sequence is None or len(sequence_files) <= max_per_sequence:
            selected.extend(sequence_files)
            continue
        indices = np.linspace(0, len(sequence_files) - 1, max_per_sequence, dtype=int)
        selected.extend(sequence_files[int(index)] for index in indices)
        removed += len(sequence_files) - len(indices)
    return selected, removed


def audit_threshold(
    data_root: Path,
    processed_root: Path,
    threshold: float,
    min_classes: int,
    max_per_sequence: int | None,
) -> tuple[list[dict], list[dict]]:
    decisions = []
    eligible_files = []
    for mask_path in sorted((processed_root / "masks").glob("*.npz")):
        with np.load(mask_path) as sample:
            mask = sample["mask"]
        decision = evaluate_slice_filter(mask, min_classes, threshold)
        sequence = classify_sequence(get_series_id(mask_path.name))
        decisions.append({
            "filter_definition": FILTER_DEFINITION,
            "threshold": threshold,
            "file": mask_path.name,
            "sequence": sequence,
            "keep_before_cap": decision.keep,
            "reason": decision.reason,
            "unique_class_count": decision.unique_class_count,
            "dominant_foreground_fraction": decision.dominant_foreground_fraction,
        })
        if decision.keep:
            eligible_files.append(mask_path.name)

    selected_files, removed_cap = cap_files_by_sequence(eligible_files, max_per_sequence)
    selected_set = set(selected_files)
    for row in decisions:
        row["selected_after_cap"] = row["file"] in selected_set

    train_files, validation_files, unmatched_files = split_train_val(data_root, selected_files)
    summary = []
    sequences = sorted({row["sequence"] for row in decisions} | {"ALL"})
    for sequence in sequences:
        scoped = decisions if sequence == "ALL" else [row for row in decisions if row["sequence"] == sequence]
        scoped_selected = selected_files if sequence == "ALL" else [
            name for name in selected_files if classify_sequence(get_series_id(name)) == sequence
        ]
        scoped_train = train_files if sequence == "ALL" else [
            name for name in train_files if classify_sequence(get_series_id(name)) == sequence
        ]
        scoped_validation = validation_files if sequence == "ALL" else [
            name for name in validation_files if classify_sequence(get_series_id(name)) == sequence
        ]
        scoped_unmatched = unmatched_files if sequence == "ALL" else [
            name for name in unmatched_files if classify_sequence(get_series_id(name)) == sequence
        ]
        summary.append({
            "filter_definition": FILTER_DEFINITION,
            "threshold": threshold,
            "sequence": sequence,
            "before_filtering": len(scoped),
            "removed_class_count": sum(row["reason"] == "fewer_than_min_classes" for row in scoped),
            "removed_imbalance": sum(
                row["reason"] == "dominant_foreground_above_threshold" for row in scoped
            ),
            "eligible_before_cap": sum(row["keep_before_cap"] for row in scoped),
            "selected_after_cap": len(scoped_selected),
            "train_slices": len(scoped_train),
            "validation_slices": len(scoped_validation),
            "unmatched_slices": len(scoped_unmatched),
            "removed_sequence_cap": removed_cap if sequence == "ALL" else (
                sum(row["keep_before_cap"] for row in scoped) - len(scoped_selected)
            ),
        })
    return decisions, summary


def main() -> None:
    parser = ArgumentParser(description="Audit paper-threshold sensitivity without changing filtered files.")
    parser.add_argument("--data_root", type=Path, required=True)
    parser.add_argument("--processed_root", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--thresholds", default="0.55,0.90,1.0")
    parser.add_argument("--min_classes", type=int, default=4)
    parser.add_argument("--max_slices_per_sequence", type=int, default=1000)
    args = parser.parse_args()

    thresholds = parse_thresholds(args.thresholds)
    max_per_sequence = args.max_slices_per_sequence if args.max_slices_per_sequence > 0 else None
    all_decisions = []
    all_summaries = []
    for threshold in thresholds:
        decisions, summary = audit_threshold(
            args.data_root,
            args.processed_root,
            threshold,
            args.min_classes,
            max_per_sequence,
        )
        all_decisions.extend(decisions)
        all_summaries.extend(summary)

    # The audit intentionally writes only to its dedicated output directory.
    # Existing filtered_files.txt and experiment checkpoints remain untouched.
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(all_summaries).to_csv(args.output_dir / "filter_sensitivity.csv", index=False)
    pd.DataFrame(all_decisions).to_csv(args.output_dir / "filter_sensitivity_files.csv", index=False)
    print(pd.DataFrame(all_summaries).to_string(index=False))


if __name__ == "__main__":
    main()
