from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import SimpleITK as sitk
from tqdm import tqdm


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


def iter_sagittal_slices(volume: np.ndarray, mask: np.ndarray):
    axis = infer_sagittal_axis(volume.shape)
    volume_slices = np.moveaxis(volume, axis, 0)
    mask_slices = np.moveaxis(mask, axis, 0)
    for index in range(volume_slices.shape[0]):
        yield index, axis, volume_slices[index], mask_slices[index]


def extract_slices(data_root: Path, output_root: Path, target_height: int, target_width: int,
                   sequences: str | None = None, force: bool = False,
                   selected_files: set[str] | None = None) -> dict:
    image_dir = data_root / "images"
    mask_dir = data_root / "masks"
    output_img_dir = output_root / "images"
    output_mask_dir = output_root / "masks"
    output_img_dir.mkdir(parents=True, exist_ok=True)
    output_mask_dir.mkdir(parents=True, exist_ok=True)

    allowed_sequences = parse_sequences(sequences)
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
        if existing and not force:
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
            image = sitk.GetArrayFromImage(image_itk).astype(np.float32)
            mask = sitk.GetArrayFromImage(sitk.ReadImage(str(mask_path))).astype(np.int16)
            if image.shape != mask.shape:
                stats["errors"].append(f"{filename}: shape mismatch {image.shape} vs {mask.shape}")
                continue

            axis = infer_sagittal_axis(image.shape)
            sequence = classify_sequence(filename)
            spacings = array_axis_spacings(image_itk)
            stats["axes"][axis] = stats["axes"].get(axis, 0) + 1

            for slice_index, _, image_slice, mask_slice in iter_sagittal_slices(image, mask):
                if image_slice.max() == image_slice.min():
                    continue

                mask_4class = map_labels(mask_slice)
                slice_name = f"{base_name}_s{slice_index:03d}.npz"
                if requested_for_series is not None and slice_name not in requested_for_series:
                    continue
                image_resized = cv2.resize(image_slice, (target_width, target_height), interpolation=cv2.INTER_LINEAR)
                mask_resized = cv2.resize(mask_4class, (target_width, target_height), interpolation=cv2.INTER_NEAREST)
                spacing = resized_slice_spacing(image_slice.shape, axis, spacings, target_height, target_width)

                np.savez_compressed(
                    output_img_dir / slice_name,
                    image=image_resized.astype(np.float32),
                    spacing=spacing,
                    sequence=sequence,
                    series_id=base_name,
                    sagittal_axis=axis,
                    slice_index=slice_index,
                )
                np.savez_compressed(
                    output_mask_dir / slice_name,
                    mask=mask_resized.astype(np.uint8),
                    spacing=spacing,
                    sequence=sequence,
                    series_id=base_name,
                    sagittal_axis=axis,
                    slice_index=slice_index,
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

    return stats


def foreground_class_fractions(mask: np.ndarray) -> dict[int, float]:
    foreground = mask[mask > 0]
    if foreground.size == 0:
        return {}
    unique, counts = np.unique(foreground, return_counts=True)
    total = counts.sum()
    return {int(label): float(count / total) for label, count in zip(unique, counts)}


def dominant_foreground_fraction(mask: np.ndarray) -> float:
    fractions = foreground_class_fractions(mask)
    return max(fractions.values()) if fractions else 1.0


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
) -> tuple[list[str], dict]:
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
        unique_classes = np.unique(mask)
        if len(unique_classes) < min_classes:
            removed_class_count += 1
            continue

        max_foreground_fraction = dominant_foreground_fraction(mask)
        if max_foreground_fraction > imbalance_threshold:
            removed_imbalance += 1
            continue

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

    output_root.mkdir(parents=True, exist_ok=True)
    with (output_root / "filtered_files.txt").open("w") as handle:
        for filename in kept:
            handle.write(filename + "\n")
    pd.DataFrame(rows).to_csv(output_root / "filtered_slice_stats.csv", index=False)

    stats = {
        "before_filtering": len(list(mask_dir.glob("*.npz"))),
        "removed_class_count": removed_class_count,
        "removed_imbalance": removed_imbalance,
        "imbalance_basis": "foreground_classes_only",
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
