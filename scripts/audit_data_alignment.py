from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path
import sys

import numpy as np
import pandas as pd

# Keep the standalone WSL command independent from package installation. The
# audit must be runnable from a normal repository checkout before any training.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from spine_baseline.orientation import OrientationError, infer_sagittal_axis_from_direction


def classify_sequence(filename: str) -> str:
    name = filename.lower()
    if "t2_space" in name or "t2_sag_space" in name or "space" in name:
        return "T2_SPACE"
    if "t2" in name:
        return "T2"
    if "t1" in name:
        return "T1"
    return "Unknown"


def map_numeric_labels(mask: np.ndarray) -> np.ndarray:
    """Mirror the lossless integer-label consolidation without image I/O dependencies."""

    mapped = np.zeros_like(mask, dtype=np.uint8)
    mapped[(mask >= 1) & (mask <= 99)] = 1
    mapped[mask == 100] = 2
    mapped[mask >= 200] = 3
    return mapped


def summarize_labels(mask: np.ndarray, sagittal_axis: int) -> dict[str, object]:
    """Describe whether direct numeric mapping preserves the three structures.

    Ahmed et al. do not publish APTA color thresholds. This audit therefore
    measures observable defects before any cleanup is proposed, rather than
    inventing morphology that could erase the small spinal-canal class.
    """

    raw_labels = np.unique(mask)
    mapped = map_numeric_labels(mask)
    unexpected = raw_labels[(raw_labels < 0) | ((raw_labels > 100) & (raw_labels < 200))]
    slice_stack = np.moveaxis(mapped, sagittal_axis, 0)
    slice_class_counts = np.asarray([np.unique(slice_mask).size for slice_mask in slice_stack])
    mapped_counts = np.bincount(mapped.ravel(), minlength=4)
    return {
        "raw_labels": ";".join(str(int(value)) for value in raw_labels),
        "unexpected_raw_labels": ";".join(str(int(value)) for value in unexpected),
        "background_voxels": int(mapped_counts[0]),
        "vertebrae_voxels": int(mapped_counts[1]),
        "canal_voxels": int(mapped_counts[2]),
        "ivd_voxels": int(mapped_counts[3]),
        "slices_total": int(slice_stack.shape[0]),
        "slices_with_4_classes": int(np.count_nonzero(slice_class_counts == 4)),
        "slices_with_fewer_than_4_classes": int(np.count_nonzero(slice_class_counts < 4)),
        "volume_has_all_foreground_classes": bool(np.all(mapped_counts[1:] > 0)),
        "direct_mapping_review": "needs_review" if unexpected.size else "numeric_ranges_valid",
    }


def _normalized_u8(image: np.ndarray) -> np.ndarray:
    finite = image[np.isfinite(image)]
    if finite.size == 0 or finite.max() <= finite.min():
        return np.zeros(image.shape, dtype=np.uint8)
    normalized = (image - finite.min()) / (finite.max() - finite.min())
    return np.clip(normalized * 255.0, 0, 255).astype(np.uint8)


def _write_preview(path: Path, image_slice: np.ndarray, mapped_mask: np.ndarray) -> None:
    # Import OpenCV only for the system audit path. Pure summary functions stay
    # usable in the lightweight Mac test environment.
    import cv2

    base = cv2.cvtColor(_normalized_u8(image_slice), cv2.COLOR_GRAY2BGR)
    colors = np.zeros_like(base)
    colors[mapped_mask == 1] = (0, 0, 255)
    colors[mapped_mask == 2] = (0, 255, 0)
    colors[mapped_mask == 3] = (255, 0, 0)
    foreground = mapped_mask > 0
    overlay = base.copy()
    overlay[foreground] = cv2.addWeighted(base, 0.35, colors, 0.65, 0)[foreground]
    side_by_side = np.concatenate([base, overlay], axis=1)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), side_by_side):
        raise OSError(f"Failed to write preview: {path}")


def audit_dataset(data_root: Path, output_dir: Path, expected_series: int) -> None:
    import SimpleITK as sitk

    image_paths = sorted((data_root / "images").glob("*.mha"))
    if len(image_paths) != expected_series:
        raise ValueError(
            f"Expected {expected_series} image series but found {len(image_paths)} under {data_root / 'images'}"
        )

    orientation_rows = []
    label_rows = []
    manifest_rows = []
    preview_dir = output_dir / "orientation_previews"
    for image_path in image_paths:
        series_id = image_path.stem
        mask_path = data_root / "masks" / image_path.name
        if not mask_path.is_file():
            raise ValueError(f"Missing mask for {image_path.name}")

        image_itk = sitk.ReadImage(str(image_path))
        mask_itk = sitk.ReadImage(str(mask_path))
        image = sitk.GetArrayFromImage(image_itk).astype(np.float32)
        mask = sitk.GetArrayFromImage(mask_itk).astype(np.int16)
        if image.shape != mask.shape:
            raise ValueError(f"{series_id}: image/mask shape mismatch {image.shape} vs {mask.shape}")

        legacy_axis = int(np.argmin(image.shape))
        metadata_error = ""
        try:
            metadata_axis = infer_sagittal_axis_from_direction(image_itk.GetDirection())
        except OrientationError as exc:
            metadata_axis = None
            metadata_error = str(exc)
        audit_axis = metadata_axis if metadata_axis is not None else legacy_axis
        directions_match = bool(
            np.allclose(image_itk.GetDirection(), mask_itk.GetDirection(), atol=1e-6, rtol=0.0)
        )
        spacings_match = bool(
            np.allclose(image_itk.GetSpacing(), mask_itk.GetSpacing(), atol=1e-6, rtol=0.0)
        )
        origins_match = bool(
            np.allclose(image_itk.GetOrigin(), mask_itk.GetOrigin(), atol=1e-6, rtol=0.0)
        )
        needs_review = (
            metadata_axis is None
            or metadata_axis != legacy_axis
            or not directions_match
            or not spacings_match
            or not origins_match
        )
        review_reason = (
            "metadata_error" if metadata_axis is None else
            "legacy_metadata_axis_mismatch" if metadata_axis != legacy_axis else
            "image_mask_direction_mismatch" if not directions_match else
            "image_mask_spacing_mismatch" if not spacings_match else
            "image_mask_origin_mismatch" if not origins_match else
            "legacy_metadata_axis_agree"
        )

        image_stack = np.moveaxis(image, audit_axis, 0)
        mask_stack = np.moveaxis(mask, audit_axis, 0)
        center = image_stack.shape[0] // 2
        mapped_center = map_numeric_labels(mask_stack[center])
        preview_path = preview_dir / f"{series_id}.png"
        _write_preview(preview_path, image_stack[center], mapped_center)

        orientation_rows.append({
            "series_id": series_id,
            "sequence": classify_sequence(series_id),
            "array_shape": "x".join(str(value) for value in image.shape),
            "legacy_axis": legacy_axis,
            "metadata_axis": "" if metadata_axis is None else metadata_axis,
            "axis_mismatch": metadata_axis is not None and metadata_axis != legacy_axis,
            "image_mask_directions_match": directions_match,
            "image_mask_spacings_match": spacings_match,
            "image_mask_origins_match": origins_match,
            "metadata_error": metadata_error,
            "preview_path": str(preview_path),
            "review_status": "needs_review" if needs_review else "auto_consistent_needs_visual_review",
        })
        label_rows.append({
            "series_id": series_id,
            "sequence": classify_sequence(series_id),
            "audit_axis": audit_axis,
            **summarize_labels(mask, audit_axis),
        })
        manifest_rows.append({
            "series_id": series_id,
            "sagittal_axis": audit_axis,
            "rotate_k": 0,
            "flip_lr": False,
            "flip_ud": False,
            "reason": review_reason,
            # The training loader rejects this template until every row has
            # been inspected and explicitly changed to "reviewed".
            "review_status": "needs_review",
        })

    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(orientation_rows).to_csv(output_dir / "orientation_audit.csv", index=False)
    pd.DataFrame(label_rows).to_csv(output_dir / "label_audit.csv", index=False)
    pd.DataFrame(manifest_rows).to_csv(output_dir / "orientation_manifest_template.csv", index=False)
    print(f"Audited {len(image_paths)} series")
    print(f"Orientation audit: {output_dir / 'orientation_audit.csv'}")
    print(f"Label audit: {output_dir / 'label_audit.csv'}")
    print(f"Review template: {output_dir / 'orientation_manifest_template.csv'}")


def main() -> None:
    parser = ArgumentParser(description="Audit all SPIDER series before paper-aligned preprocessing.")
    parser.add_argument("--data_root", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--expected_series", type=int, default=447)
    args = parser.parse_args()
    audit_dataset(args.data_root, args.output_dir, args.expected_series)


if __name__ == "__main__":
    main()
