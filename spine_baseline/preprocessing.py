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
                   sequences: str | None = None, force: bool = False) -> dict:
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

    stats = {"total_slices": 0, "files_processed": 0, "skipped_existing": 0, "axes": {}, "errors": []}

    for filename in tqdm(mha_files, desc="Extracting sagittal slices"):
        mask_path = mask_dir / filename
        if not mask_path.exists():
            stats["errors"].append(f"No mask for {filename}")
            continue

        base_name = filename.removesuffix(".mha")
        existing = list(output_img_dir.glob(f"{base_name}_s*.npz"))
        if existing and not force:
            stats["skipped_existing"] += 1
            stats["total_slices"] += len(existing)
            continue

        try:
            image = sitk.GetArrayFromImage(sitk.ReadImage(str(image_dir / filename))).astype(np.float32)
            mask = sitk.GetArrayFromImage(sitk.ReadImage(str(mask_path))).astype(np.int16)
            if image.shape != mask.shape:
                stats["errors"].append(f"{filename}: shape mismatch {image.shape} vs {mask.shape}")
                continue

            axis = infer_sagittal_axis(image.shape)
            stats["axes"][axis] = stats["axes"].get(axis, 0) + 1

            for slice_index, _, image_slice, mask_slice in iter_sagittal_slices(image, mask):
                if image_slice.max() == image_slice.min():
                    continue

                mask_4class = map_labels(mask_slice)
                image_resized = cv2.resize(image_slice, (target_width, target_height), interpolation=cv2.INTER_LINEAR)
                mask_resized = cv2.resize(mask_4class, (target_width, target_height), interpolation=cv2.INTER_NEAREST)

                slice_name = f"{base_name}_s{slice_index:03d}.npz"
                np.savez_compressed(output_img_dir / slice_name, image=image_resized.astype(np.float32))
                np.savez_compressed(output_mask_dir / slice_name, mask=mask_resized.astype(np.uint8))
                stats["total_slices"] += 1

            stats["files_processed"] += 1
        except Exception as exc:
            stats["errors"].append(f"{filename}: {exc}")

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


def filter_slices(output_root: Path, min_classes: int, imbalance_threshold: float) -> tuple[list[str], dict]:
    mask_dir = output_root / "masks"
    rows = []
    kept = []
    removed_class_count = 0
    removed_imbalance = 0

    for mask_file in tqdm(sorted(mask_dir.glob("*.npz")), desc="Filtering slices"):
        mask = np.load(mask_file)["mask"]
        unique_classes = np.unique(mask)
        if len(unique_classes) < min_classes:
            removed_class_count += 1
            continue

        max_fraction = dominant_foreground_fraction(mask)
        if max_fraction > imbalance_threshold:
            removed_imbalance += 1
            continue

        fractions = foreground_class_fractions(mask)
        kept.append(mask_file.name)
        rows.append({
            "file": mask_file.name,
            "max_foreground_fraction": max_fraction,
            "vertebrae_fraction": fractions.get(1, 0.0),
            "canal_fraction": fractions.get(2, 0.0),
            "ivd_fraction": fractions.get(3, 0.0),
        })

    output_root.mkdir(parents=True, exist_ok=True)
    with (output_root / "filtered_files.txt").open("w") as handle:
        for filename in kept:
            handle.write(filename + "\n")
    pd.DataFrame(rows).to_csv(output_root / "filtered_slice_stats.csv", index=False)

    stats = {
        "before_filtering": len(list(mask_dir.glob("*.npz"))),
        "removed_class_count": removed_class_count,
        "removed_imbalance": removed_imbalance,
        "kept": len(kept),
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
