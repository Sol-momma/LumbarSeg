from __future__ import annotations

from pathlib import Path

import numpy as np


INVERSE_SQRT_TRAIN = "inverse_sqrt_train"


def count_training_class_pixels(
    files: list[str],
    output_root: Path,
    num_classes: int,
) -> np.ndarray:
    """Count class pixels from the training cohort without reading validation masks."""
    if num_classes < 1:
        raise ValueError("num_classes must be positive")
    if not files:
        raise ValueError("Training cohort is empty")

    counts = np.zeros(num_classes, dtype=np.int64)
    for filename in files:
        mask_path = output_root / "masks" / filename
        with np.load(mask_path, allow_pickle=False) as archive:
            mask = np.asarray(archive["mask"])

        # validate_slice_files normally rejects these values first. Keep this
        # guard here as well because class-weight derivation is an experiment
        # boundary: silently truncating 0.5 or NaN would record false evidence.
        if not np.issubdtype(mask.dtype, np.number) or not np.all(np.isfinite(mask)):
            raise ValueError(f"Training mask contains non-finite or non-numeric labels: {filename}")
        if not np.all(mask == np.floor(mask)):
            raise ValueError(f"Training mask contains non-integer labels: {filename}")

        integer_mask = mask.astype(np.int64, copy=False)
        if integer_mask.size == 0:
            raise ValueError(f"Training mask is empty: {filename}")
        if integer_mask.min() < 0 or integer_mask.max() >= num_classes:
            raise ValueError(
                f"Training mask labels must be in [0, {num_classes - 1}]: {filename}"
            )
        counts += np.bincount(integer_mask.ravel(), minlength=num_classes)[:num_classes]

    return counts


def inverse_sqrt_frequency_weights(class_counts: np.ndarray) -> np.ndarray:
    """Return inverse-square-root weights with average per-pixel weight equal to one."""
    counts = np.asarray(class_counts, dtype=np.float64)
    if counts.ndim != 1 or counts.size == 0:
        raise ValueError("class_counts must be a non-empty one-dimensional array")
    if not np.all(np.isfinite(counts)) or np.any(counts <= 0):
        raise ValueError("Every class must have a positive finite training pixel count")

    frequencies = counts / counts.sum()
    raw_weights = 1.0 / np.sqrt(frequencies)

    # Pure inverse frequency would make the rare spinal-canal/IVD classes
    # roughly forty times stronger than background in this dataset. The square
    # root preserves the intended emphasis while limiting that instability.
    # Normalizing by observed training frequency keeps the expected focal-loss
    # scale unchanged, so learning-rate comparisons remain meaningful.
    weights = raw_weights / np.dot(frequencies, raw_weights)
    return weights.astype(np.float32)


def derive_focal_class_weights(
    mode: str,
    files: list[str],
    output_root: Path,
    num_classes: int,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Resolve an optional weighting mode using only the training cohort."""
    if mode == "none":
        return None, None
    if mode != INVERSE_SQRT_TRAIN:
        raise ValueError(f"Unsupported focal class-weight mode: {mode}")

    counts = count_training_class_pixels(files, output_root, num_classes)
    return inverse_sqrt_frequency_weights(counts), counts


def write_focal_class_weight_evidence(
    path: Path,
    mode: str,
    class_counts: np.ndarray | None,
    class_weights: np.ndarray | None,
) -> None:
    """Record the exact loss input so a candidate run can be audited later."""
    lines = [f"focal_class_weight_mode\t{mode}"]
    if class_counts is not None and class_weights is not None:
        total = int(np.sum(class_counts))
        for class_id, (count, weight) in enumerate(zip(class_counts, class_weights, strict=True)):
            lines.extend(
                (
                    f"class_{class_id}_pixel_count\t{int(count)}",
                    f"class_{class_id}_pixel_frequency\t{float(count) / total:.12g}",
                    f"class_{class_id}_focal_weight\t{float(weight):.12g}",
                )
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
