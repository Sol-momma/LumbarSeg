from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import SimpleITK as sitk
from tqdm import tqdm

from spine_baseline.filtering import FILTER_DEFINITION, evaluate_slice_filter
from spine_baseline.orientation import (
    ORIENTATION_MODES,
    OrientationTransform,
    load_orientation_manifest,
    resolve_orientation,
    transform_pair,
)


def classify_sequence(filename: str) -> str:
    name = filename.lower()
    if "t2_space" in name or "t2_sag_space" in name or "space" in name:
        return "T2_SPACE"
    if "t2" in name:
        return "T2"
    if "t1" in name:
        return "T1"
    return "Unknown"


def parse_sequences(sequences: str | None) -> set[str] | None:
    if not sequences:
        return None
    return {item.strip().upper() for item in sequences.split(",") if item.strip()}


def map_labels(mask: np.ndarray) -> np.ndarray:
    """Map SPIDER labels to Background, Vertebrae, Spinal Canal, and IVDs."""
    new_mask = np.zeros_like(mask, dtype=np.uint8)
    new_mask[(mask >= 1) & (mask <= 99)] = 1
    new_mask[mask == 100] = 2
    new_mask[mask >= 200] = 3
    return new_mask


def array_axis_spacings(image: sitk.Image) -> tuple[float, float, float]:
    """Return physical spacings in NumPy array axis order: z, y, x."""
    spacing_x, spacing_y, spacing_z = image.GetSpacing()
    return float(spacing_z), float(spacing_y), float(spacing_x)


def resized_slice_spacing(
    original_shape: tuple[int, int],
    sagittal_axis: int,
    axis_spacings: tuple[float, float, float],
    target_height: int,
    target_width: int,
) -> np.ndarray:
    remaining_axes = [axis for axis in range(3) if axis != sagittal_axis]
    row_spacing = axis_spacings[remaining_axes[0]] * (original_shape[0] / target_height)
    col_spacing = axis_spacings[remaining_axes[1]] * (original_shape[1] / target_width)
    return np.array([row_spacing, col_spacing], dtype=np.float32)


def infer_sagittal_axis(array_shape: tuple[int, ...]) -> int:
    """Infer the sagittal stack axis for SPIDER volumes.

    SPIDER MHA arrays are returned by SimpleITK as (z, y, x), but the sagittal
    slice dimension is not consistently axis 0 across files. The sagittal stack
    is generally the smallest dimension, so this baseline uses the smallest axis.
    """
    return int(np.argmin(array_shape))


def iter_sagittal_slices(volume: np.ndarray, mask: np.ndarray, sagittal_axis: int | None = None):
    # ``None`` deliberately retains the original smallest-dimension heuristic.
    # Callers opting into metadata or a reviewed manifest pass an explicit axis.
    axis = infer_sagittal_axis(volume.shape) if sagittal_axis is None else sagittal_axis
    volume_slices = np.moveaxis(volume, axis, 0)
    mask_slices = np.moveaxis(mask, axis, 0)
    for index in range(volume_slices.shape[0]):
        yield index, axis, volume_slices[index], mask_slices[index]


def transformed_resized_slice_spacing(
    original_shape: tuple[int, int],
    sagittal_axis: int,
    axis_spacings: tuple[float, float, float],
    transform: OrientationTransform,
    target_height: int,
    target_width: int,
) -> np.ndarray:
    """Calculate output spacing after the reviewed in-plane rotation."""
    remaining_axes = [axis for axis in range(3) if axis != sagittal_axis]
    row_spacing = axis_spacings[remaining_axes[0]]
    col_spacing = axis_spacings[remaining_axes[1]]
    transformed_shape = original_shape
    if transform.rotate_k % 2:
        # A quarter turn swaps both pixel dimensions and their physical spacing.
        # Flips alter direction but not spacing, so they need no special case.
        row_spacing, col_spacing = col_spacing, row_spacing
        transformed_shape = (original_shape[1], original_shape[0])
    return np.array(
        [
            row_spacing * (transformed_shape[0] / target_height),
            col_spacing * (transformed_shape[1] / target_width),
        ],
        dtype=np.float32,
    )


def _saved_orientation_matches(
    paths: list[Path],
    *,
    mode: str,
    transform: OrientationTransform,
    expected_shape: tuple[int, int],
    series_id: str,
) -> bool:
    """Avoid reusing slices produced with a different orientation policy."""
    for path in paths:
        try:
            with np.load(path) as sample:
                array_key = "image" if path.parent.name == "images" else "mask"
                if array_key not in sample or sample[array_key].shape != expected_shape:
                    return False
                if "series_id" not in sample or str(sample["series_id"].item()) != series_id:
                    return False
                if str(sample["orientation_mode"].item()) != mode:
                    return False
                if int(sample["sagittal_axis"].item()) != transform.sagittal_axis:
                    return False
                if int(sample["rotate_k"].item()) != transform.rotate_k:
                    return False
                if bool(sample["flip_lr"].item()) != transform.flip_lr:
                    return False
                if bool(sample["flip_ud"].item()) != transform.flip_ud:
                    return False
        except (KeyError, OSError, ValueError):
            return False
    return True


def extract_slices(data_root: Path, output_root: Path, target_height: int, target_width: int,
                   sequences: str | None = None, force: bool = False,
                   selected_files: set[str] | None = None,
                   orientation_mode: str = "legacy",
                   orientation_manifest: Path | None = None) -> dict:
    image_dir = data_root / "images"
    mask_dir = data_root / "masks"
    output_img_dir = output_root / "images"
    output_mask_dir = output_root / "masks"
    output_img_dir.mkdir(parents=True, exist_ok=True)
    output_mask_dir.mkdir(parents=True, exist_ok=True)

    allowed_sequences = parse_sequences(sequences)
    if orientation_mode not in ORIENTATION_MODES:
        raise ValueError(
            f"Unknown orientation_mode {orientation_mode!r}; expected one of {sorted(ORIENTATION_MODES)}"
        )
    if orientation_mode != "legacy" and force:
        # Overwriting in place is unsafe when the corrected stack has fewer
        # slices than the legacy stack: stale tail slices would survive and mix
        # two orientation policies. Use a new output root for every corrected
        # campaign instead of deleting or mutating a previous experiment.
        raise ValueError(
            "force reprocessing is disabled for orientation-aware modes; use a new output_root"
        )
    if orientation_mode == "manifest" and orientation_manifest is None:
        raise ValueError("orientation_manifest is required when orientation_mode='manifest'")
    manifest = (
        load_orientation_manifest(orientation_manifest)
        if orientation_mode == "manifest" and orientation_manifest is not None
        else None
    )
    mha_files = sorted(file.name for file in image_dir.glob("*.mha"))
    if allowed_sequences is not None:
        mha_files = [name for name in mha_files if classify_sequence(name) in allowed_sequences]

    selected_by_series: dict[str, set[str]] | None = None
    if selected_files is not None:
        # Failure analysis only needs a few hundred validation slices. Grouping
        # the requested names by series lets us read each source MHA once while
        # avoiding the roughly 10 GB full preprocessing output.
        selected_by_series = {}
        for slice_name in selected_files:
            selected_by_series.setdefault(get_series_id(slice_name), set()).add(slice_name)
        mha_files = [
            name for name in mha_files
            if name.removesuffix(".mha") in selected_by_series
        ]

    stats = {"total_slices": 0, "files_processed": 0, "skipped_existing": 0, "axes": {}, "errors": []}

    for filename in tqdm(mha_files, desc="Extracting sagittal slices"):
        mask_path = mask_dir / filename
        if not mask_path.exists():
            stats["errors"].append(f"No mask for {filename}")
            continue

        base_name = filename.removesuffix(".mha")
        requested_for_series = selected_by_series.get(base_name) if selected_by_series is not None else None
        existing = list(output_img_dir.glob(f"{base_name}_s*.npz"))
        existing_masks = list(output_mask_dir.glob(f"{base_name}_s*.npz"))
        if orientation_mode != "legacy" and {path.name for path in existing} != {
            path.name for path in existing_masks
        }:
            raise RuntimeError(
                f"{base_name}: existing image/mask slice sets differ; use a new output_root"
            )
        # Keep the baseline's early skip byte-for-byte compatible. New modes
        # inspect saved provenance after resolving the requested orientation so
        # they cannot accidentally reuse legacy slices from the same directory.
        if existing and not force and orientation_mode == "legacy":
            if requested_for_series is None:
                stats["skipped_existing"] += 1
                stats["total_slices"] += len(existing)
                continue
            existing_names = {path.name for path in existing}
            if requested_for_series.issubset(existing_names):
                stats["skipped_existing"] += 1
                stats["total_slices"] += len(requested_for_series)
                continue

        try:
            image_itk = sitk.ReadImage(str(image_dir / filename))
            mask_itk = sitk.ReadImage(str(mask_path))
            image = sitk.GetArrayFromImage(image_itk).astype(np.float32)
            mask = sitk.GetArrayFromImage(mask_itk).astype(np.int16)
            if image.shape != mask.shape:
                stats["errors"].append(f"{filename}: shape mismatch {image.shape} vs {mask.shape}")
                continue

            if orientation_mode != "legacy" and not np.allclose(
                image_itk.GetDirection(), mask_itk.GetDirection(), atol=1e-6, rtol=0.0
            ):
                raise ValueError("image and mask direction metadata differ")
            if orientation_mode != "legacy" and not np.allclose(
                image_itk.GetSpacing(), mask_itk.GetSpacing(), atol=1e-6, rtol=0.0
            ):
                raise ValueError("image and mask spacing metadata differ")
            if orientation_mode != "legacy" and not np.allclose(
                image_itk.GetOrigin(), mask_itk.GetOrigin(), atol=1e-6, rtol=0.0
            ):
                raise ValueError("image and mask origin metadata differ")
            transform = resolve_orientation(
                mode=orientation_mode,
                array_shape=image.shape,
                direction=image_itk.GetDirection(),
                series_id=base_name,
                manifest=manifest,
            )
            axis = transform.sagittal_axis

            if existing and not force and orientation_mode != "legacy":
                relevant_existing = existing
                if requested_for_series is not None:
                    relevant_existing = [path for path in existing if path.name in requested_for_series]
                provenance_paths = relevant_existing + [
                    output_mask_dir / path.name for path in relevant_existing
                ]
                if relevant_existing and _saved_orientation_matches(
                    provenance_paths,
                    mode=orientation_mode,
                    transform=transform,
                    expected_shape=(target_height, target_width),
                    series_id=base_name,
                ):
                    if requested_for_series is None or requested_for_series.issubset(
                        {path.name for path in relevant_existing}
                    ):
                        stats["skipped_existing"] += 1
                        stats["total_slices"] += len(relevant_existing)
                        continue
                if relevant_existing:
                    # ``force=False`` historically promises not to overwrite
                    # existing slices. A separate output root is preferable for
                    # a new orientation campaign; ``force`` remains the explicit
                    # opt-in when replacement is intentional.
                    raise ValueError(
                        "existing slices have different or incomplete orientation provenance; "
                        "use a separate output root or force reprocessing"
                    )

            sequence = classify_sequence(filename)
            spacings = array_axis_spacings(image_itk)
            stats["axes"][axis] = stats["axes"].get(axis, 0) + 1

            for slice_index, _, image_slice, mask_slice in iter_sagittal_slices(image, mask, axis):
                if image_slice.max() == image_slice.min():
                    continue

                original_shape = image_slice.shape
                image_slice, mask_slice = transform_pair(image_slice, mask_slice, transform)
                mask_4class = map_labels(mask_slice)
                slice_name = f"{base_name}_s{slice_index:03d}.npz"
                if requested_for_series is not None and slice_name not in requested_for_series:
                    continue
                image_resized = cv2.resize(image_slice, (target_width, target_height), interpolation=cv2.INTER_LINEAR)
                mask_resized = cv2.resize(mask_4class, (target_width, target_height), interpolation=cv2.INTER_NEAREST)
                spacing = transformed_resized_slice_spacing(
                    original_shape,
                    axis,
                    spacings,
                    transform,
                    target_height,
                    target_width,
                )

                np.savez_compressed(
                    output_img_dir / slice_name,
                    image=image_resized.astype(np.float32),
                    spacing=spacing,
                    sequence=sequence,
                    series_id=base_name,
                    sagittal_axis=axis,
                    slice_index=slice_index,
                    orientation_mode=orientation_mode,
                    rotate_k=transform.rotate_k,
                    flip_lr=transform.flip_lr,
                    flip_ud=transform.flip_ud,
                    orientation_reason=transform.reason,
                    orientation_review_status=transform.review_status,
                )
                np.savez_compressed(
                    output_mask_dir / slice_name,
                    mask=mask_resized.astype(np.uint8),
                    spacing=spacing,
                    sequence=sequence,
                    series_id=base_name,
                    sagittal_axis=axis,
                    slice_index=slice_index,
                    orientation_mode=orientation_mode,
                    rotate_k=transform.rotate_k,
                    flip_lr=transform.flip_lr,
                    flip_ud=transform.flip_ud,
                    orientation_reason=transform.reason,
                    orientation_review_status=transform.review_status,
                )
                stats["total_slices"] += 1

            stats["files_processed"] += 1
        except Exception as exc:
            stats["errors"].append(f"{filename}: {exc}")

    if selected_files is not None:
        written_images = {path.name for path in output_img_dir.glob("*.npz")}
        written_masks = {path.name for path in output_mask_dir.glob("*.npz")}
        missing = sorted(selected_files - (written_images & written_masks))
        if missing:
            stats["errors"].append(
                f"Selected slices missing after extraction: {len(missing)}; first={missing[:5]}"
            )

    if orientation_mode != "legacy" and stats["errors"]:
        # A paper-aligned orientation run must be all-or-nothing. Continuing
        # after one series fails could silently mix legacy and corrected slices
        # in the same training cohort, making the resulting score uninterpretable.
        preview = "; ".join(stats["errors"][:5])
        suffix = "" if len(stats["errors"]) <= 5 else f"; and {len(stats['errors']) - 5} more"
        raise RuntimeError(
            f"Orientation-aware extraction failed for {len(stats['errors'])} item(s): "
            f"{preview}{suffix}"
        )
    if orientation_mode != "legacy":
        all_images = {path.name for path in output_img_dir.glob("*.npz")}
        all_masks = {path.name for path in output_mask_dir.glob("*.npz")}
        if all_images != all_masks:
            raise RuntimeError(
                "Orientation-aware output has different image/mask file sets; use a new output_root"
            )

    return stats


def class_fractions(mask: np.ndarray, num_classes: int = 4) -> dict[int, float]:
    counts = np.bincount(mask.astype(np.int64).ravel(), minlength=num_classes)
    total = counts.sum()
    if total == 0:
        return {class_id: 0.0 for class_id in range(num_classes)}
    return {class_id: float(counts[class_id] / total) for class_id in range(num_classes)}


def dominant_class_fraction(mask: np.ndarray) -> float:
    fractions = class_fractions(mask)
    return max(fractions.values()) if fractions else 1.0


def filter_slices(
    output_root: Path,
    min_classes: int,
    imbalance_threshold: float,
    max_slices_per_sequence: int | None = 1000,
    evidence_root: Path | None = None,
) -> tuple[list[str], dict]:
    """Filter a processed cache and keep each run's evidence separate.

    ``output_root`` owns the masks being read. A campaign may reuse those
    immutable arrays while writing ``filtered_files.txt`` and its statistics to
    a new ``evidence_root``. This prevents one filtering candidate from
    modifying the baseline cache or another candidate's evidence.
    """
    mask_dir = output_root / "masks"
    eligible_rows = []
    eligible_files = []
    removed_class_count = 0
    removed_imbalance = 0
    removed_sequence_cap = 0
    corrupt_files = []

    for mask_file in tqdm(sorted(mask_dir.glob("*.npz")), desc="Filtering slices"):
        try:
            # Interrupted WSL/Windows preprocessing can leave a zero-byte or
            # half-written `.npz`. Treat that as a repairable data artifact
            # instead of crashing the whole training run; the companion repair
            # script can delete the bad image/mask pair before retrying.
            with np.load(mask_file) as sample:
                mask = sample["mask"]
        except Exception as exc:
            corrupt_files.append(f"{mask_file.name}: {exc}")
            continue
        decision = evaluate_slice_filter(mask, min_classes, imbalance_threshold)
        if decision.reason == "fewer_than_min_classes":
            removed_class_count += 1
            continue
        if decision.reason == "dominant_foreground_above_threshold":
            removed_imbalance += 1
            continue

        max_foreground_fraction = decision.dominant_foreground_fraction
        fractions = class_fractions(mask)
        sequence = classify_sequence(get_series_id(mask_file.name))
        eligible_files.append(mask_file.name)
        eligible_rows.append({
            "file": mask_file.name,
            "sequence": sequence,
            "max_foreground_class_fraction": max_foreground_fraction,
            "max_class_fraction": dominant_class_fraction(mask),
            "background_fraction": fractions.get(0, 0.0),
            "vertebrae_fraction": fractions.get(1, 0.0),
            "canal_fraction": fractions.get(2, 0.0),
            "ivd_fraction": fractions.get(3, 0.0),
        })

    grouped: dict[str, list[tuple[str, dict]]] = {}
    for filename, row in zip(eligible_files, eligible_rows):
        grouped.setdefault(row["sequence"], []).append((filename, row))

    kept = []
    rows = []
    kept_by_sequence: dict[str, int] = {}
    for sequence, items in grouped.items():
        if max_slices_per_sequence is None or len(items) <= max_slices_per_sequence:
            selected_indices = set(range(len(items)))
        else:
            selected_indices = set(np.linspace(0, len(items) - 1, max_slices_per_sequence, dtype=int))

        for index, (filename, row) in enumerate(items):
            if index not in selected_indices:
                removed_sequence_cap += 1
                continue
            kept_by_sequence[sequence] = kept_by_sequence.get(sequence, 0) + 1
            kept.append(filename)
            rows.append(row)

    evidence_root = evidence_root or output_root
    evidence_root.mkdir(parents=True, exist_ok=True)
    with (evidence_root / "filtered_files.txt").open("w") as handle:
        for filename in kept:
            handle.write(filename + "\n")
    pd.DataFrame(rows).to_csv(evidence_root / "filtered_slice_stats.csv", index=False)

    stats = {
        "before_filtering": len(list(mask_dir.glob("*.npz"))),
        "removed_class_count": removed_class_count,
        "removed_imbalance": removed_imbalance,
        "imbalance_basis": "foreground_classes_only",
        # Keep the caveat machine-readable so a successful Dice score cannot be
        # relabelled later as an exact paper reproduction by reading only logs.
        "filter_definition": FILTER_DEFINITION,
        "removed_sequence_cap": removed_sequence_cap,
        "corrupt_files": len(corrupt_files),
        "corrupt_file_errors": corrupt_files,
        "kept": len(kept),
        "kept_by_sequence": kept_by_sequence,
    }
    return kept, stats


def get_series_id(slice_filename: str) -> str:
    return slice_filename.removesuffix(".npz").rsplit("_s", 1)[0]


def split_train_val(data_root: Path, kept_files: list[str]) -> tuple[list[str], list[str], list[str]]:
    csv_path = data_root / "SPIDER Lumbar Spine Segmentation Overview.csv"
    overview = pd.read_csv(csv_path)
    train_ids = set(overview.loc[overview["subset"] == "training", "new_file_name"].astype(str))
    val_ids = set(overview.loc[overview["subset"] == "validation", "new_file_name"].astype(str))

    train_files = [name for name in kept_files if get_series_id(name) in train_ids]
    val_files = [name for name in kept_files if get_series_id(name) in val_ids]
    unmatched = [name for name in kept_files if get_series_id(name) not in train_ids and get_series_id(name) not in val_ids]
    return train_files, val_files, unmatched
