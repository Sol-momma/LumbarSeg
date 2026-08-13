from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path
import sys

import cv2
import numpy as np
import pandas as pd
import SimpleITK as sitk

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from spine_baseline.apta import APTA_AUDIT_DEFINITION, reconstruct_public_apta
from spine_baseline.orientation import apply_in_plane_transform, resolve_orientation
from spine_baseline.preprocessing import classify_sequence, map_labels


GEOMETRY_MODES = {"metadata_512x640", "author_x_620x512"}
CLASS_NAMES = ("Background", "Vertebrae", "Spinal Canal", "IVDs")


def select_indices(mask: np.ndarray, axis: int, max_slices: int) -> list[int]:
    eligible = []
    for index in range(mask.shape[axis]):
        direct = map_labels(np.take(mask, index, axis=axis))
        if np.unique(direct).size == 4:
            eligible.append(index)
    if max_slices <= 0 or len(eligible) <= max_slices:
        return eligible
    positions = np.linspace(0, len(eligible) - 1, max_slices, dtype=int)
    return [eligible[int(position)] for position in positions]


def confusion_matrix(reference: np.ndarray, candidate: np.ndarray) -> np.ndarray:
    if reference.shape != candidate.shape:
        raise ValueError(f"Comparison shape mismatch: {reference.shape} vs {candidate.shape}")
    return np.bincount(reference.ravel() * 4 + candidate.ravel(), minlength=16).reshape(4, 4)


def dice_from_confusion(confusion: np.ndarray, class_id: int) -> float:
    true_pixels = int(confusion[class_id, :].sum())
    predicted_pixels = int(confusion[:, class_id].sum())
    denominator = true_pixels + predicted_pixels
    return float(2 * confusion[class_id, class_id] / denominator) if denominator else 1.0


def audit_series(
    mask_path: Path,
    geometry_mode: str,
    max_slices_per_series: int,
) -> tuple[list[dict], np.ndarray]:
    image = sitk.ReadImage(str(mask_path))
    mask = sitk.GetArrayFromImage(image).astype(np.int16)
    series_id = mask_path.stem
    if geometry_mode == "metadata_512x640":
        transform = resolve_orientation(
            mode="metadata",
            array_shape=mask.shape,
            direction=image.GetDirection(),
            series_id=series_id,
        )
        axis = transform.sagittal_axis
        target_height, target_width = 512, 640
        author_flip = False
    else:
        # ``image[x,:,:]`` in SimpleITK corresponds to NumPy axis 2. This mode
        # intentionally diagnoses the public script as written, even for series
        # whose direction metadata says the physical sagittal axis differs.
        transform = None
        axis = 2
        target_height, target_width = 620, 512
        author_flip = True

    rows = []
    pooled = np.zeros((4, 4), dtype=np.int64)
    for index in select_indices(mask, axis, max_slices_per_series):
        raw_slice = np.take(mask, index, axis=axis)
        if transform is not None:
            raw_slice = apply_in_plane_transform(raw_slice, transform)
        direct = cv2.resize(
            map_labels(raw_slice),
            (target_width, target_height),
            interpolation=cv2.INTER_NEAREST,
        )
        apta = reconstruct_public_apta(
            raw_slice,
            target_height=target_height,
            target_width=target_width,
            apply_author_flip=author_flip,
        )
        confusion = confusion_matrix(direct, apta)
        pooled += confusion
        row = {
            "apta_definition": APTA_AUDIT_DEFINITION,
            "geometry_mode": geometry_mode,
            "series_id": series_id,
            "sequence": classify_sequence(series_id),
            "sagittal_axis": axis,
            "slice_index": index,
            "height": target_height,
            "width": target_width,
            "changed_pixel_fraction": float(1.0 - np.trace(confusion) / confusion.sum()),
        }
        for class_id, class_name in enumerate(CLASS_NAMES):
            key = class_name.lower().replace(" ", "_")
            row[f"{key}_dice_vs_direct"] = dice_from_confusion(confusion, class_id)
            row[f"{key}_direct_pixels"] = int(confusion[class_id, :].sum())
            row[f"{key}_apta_pixels"] = int(confusion[:, class_id].sum())
        rows.append(row)
    return rows, pooled


def main() -> None:
    parser = ArgumentParser(
        description="Compare public APTA reconstruction with numeric label mapping without changing data."
    )
    parser.add_argument("--data_root", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--sequences", default="T2_SPACE")
    parser.add_argument("--geometry_mode", choices=sorted(GEOMETRY_MODES), default="metadata_512x640")
    parser.add_argument("--max_slices_per_series", type=int, default=1)
    args = parser.parse_args()

    if args.output_dir.exists():
        raise ValueError(f"Output directory already exists: {args.output_dir}")
    allowed_sequences = {item.strip().upper() for item in args.sequences.split(",") if item.strip()}
    mask_paths = [
        path for path in sorted((args.data_root / "masks").glob("*.mha"))
        if classify_sequence(path.stem) in allowed_sequences
    ]
    if not mask_paths:
        raise ValueError(f"No matching mask MHA files under {args.data_root / 'masks'}")

    all_rows = []
    pooled = np.zeros((4, 4), dtype=np.int64)
    for mask_path in mask_paths:
        rows, confusion = audit_series(mask_path, args.geometry_mode, args.max_slices_per_series)
        all_rows.extend(rows)
        pooled += confusion
    if not all_rows:
        raise ValueError("No four-class slices were available for the APTA audit")

    summary = []
    for class_id, class_name in enumerate(CLASS_NAMES):
        summary.append({
            "apta_definition": APTA_AUDIT_DEFINITION,
            "geometry_mode": args.geometry_mode,
            "series_count": len(mask_paths),
            "slice_count": len(all_rows),
            "class_id": class_id,
            "class_name": class_name,
            "dice_vs_direct": dice_from_confusion(pooled, class_id),
            "direct_pixels": int(pooled[class_id, :].sum()),
            "apta_pixels": int(pooled[:, class_id].sum()),
        })

    args.output_dir.mkdir(parents=True)
    pd.DataFrame(all_rows).to_csv(args.output_dir / "apta_slice_comparison.csv", index=False)
    pd.DataFrame(summary).to_csv(args.output_dir / "apta_summary.csv", index=False)
    pd.DataFrame(
        pooled,
        index=[f"direct_{name}" for name in CLASS_NAMES],
        columns=[f"apta_{name}" for name in CLASS_NAMES],
    ).to_csv(args.output_dir / "apta_confusion.csv")
    print(pd.DataFrame(summary).to_string(index=False))


if __name__ == "__main__":
    main()
