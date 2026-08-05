from pathlib import Path

import numpy as np


def read_file_list(path: Path) -> list[str]:
    """Read a non-empty, duplicate-free slice list used as experiment evidence."""
    files = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not files:
        raise ValueError(f"File list is empty: {path}")
    # Windows paths are case-insensitive. Reject case-only duplicates on every
    # platform so a cohort behaves the same when prepared on Mac and run in WSL.
    if len(files) != len({filename.casefold() for filename in files}):
        raise ValueError(f"File list contains duplicates: {path}")
    return files


def validate_slice_files(
    files: list[str],
    output_root: Path,
    cohort_name: str,
    allowed_sequences: set[str] | None = None,
    expected_shape: tuple[int, int] | None = None,
) -> None:
    """Reject invalid explicit cohorts before TensorFlow starts allocating GPU memory."""
    if not files:
        raise ValueError(f"{cohort_name} file list is empty")
    if len(files) != len(set(files)):
        raise ValueError(f"{cohort_name} file list contains duplicates")

    missing: list[str] = []
    for filename in files:
        entry = Path(filename)
        if entry.name != filename or entry.suffix.casefold() != ".npz":
            raise ValueError(
                f"{cohort_name} file list must contain .npz basenames only: {filename!r}"
            )
        # Explicit lists intentionally bypass the Overview split, but they must
        # still point to a complete preprocessed image/mask pair. Checking both
        # here makes a smoke probe fail before model construction instead of
        # halfway through a GPU run with a less useful np.load error.
        image_path = output_root / "images" / filename
        mask_path = output_root / "masks" / filename
        if not image_path.is_file() or not mask_path.is_file():
            missing.append(filename)
            continue

        try:
            with np.load(image_path) as image_sample, np.load(mask_path) as mask_sample:
                if "image" not in image_sample or "mask" not in mask_sample:
                    raise ValueError("required image/mask array is missing")
                image = image_sample["image"]
                mask = mask_sample["mask"]
                if image.ndim != 2 or mask.ndim != 2 or image.shape != mask.shape:
                    raise ValueError(f"image/mask shape mismatch: {image.shape} vs {mask.shape}")
                if expected_shape is not None and image.shape != expected_shape:
                    raise ValueError(f"slice shape {image.shape} does not match {expected_shape}")
                if not np.issubdtype(mask.dtype, np.integer):
                    raise ValueError(f"mask dtype must be integer; got {mask.dtype}")
                if mask.size and not np.isin(mask, np.arange(4)).all():
                    raise ValueError("mask contains a class outside 0..3")

                image_sequence = str(image_sample["sequence"].item()) if "sequence" in image_sample else ""
                mask_sequence = str(mask_sample["sequence"].item()) if "sequence" in mask_sample else ""
                if image_sequence != mask_sequence:
                    raise ValueError(
                        f"image/mask sequence mismatch: {image_sequence!r} vs {mask_sequence!r}"
                    )
                if allowed_sequences is not None and image_sequence not in allowed_sequences:
                    raise ValueError(
                        f"sequence {image_sequence!r} is outside {sorted(allowed_sequences)}"
                    )
        except Exception as exc:
            # This is a preflight boundary around user-provided NPZ artifacts.
            # Corrupt ZIP containers raise BadZipFile rather than OSError, so
            # normalize all parsing failures into a cohort-specific error.
            raise ValueError(f"{cohort_name} slice {filename!r} is invalid: {exc}") from exc

    if missing:
        preview = ", ".join(missing[:5])
        suffix = "" if len(missing) <= 5 else f" (and {len(missing) - 5} more)"
        raise ValueError(
            f"{cohort_name} file list contains {len(missing)} slice(s) without both image and mask: "
            f"{preview}{suffix}"
        )


def validate_disjoint_cohorts(train_files: list[str], validation_files: list[str]) -> None:
    """Prevent the same slice from being used for fitting and validation."""
    train_by_normalized = {filename.casefold(): filename for filename in train_files}
    validation_by_normalized = {filename.casefold(): filename for filename in validation_files}
    overlap_keys = sorted(train_by_normalized.keys() & validation_by_normalized.keys())
    overlap = [
        f"{train_by_normalized[key]} / {validation_by_normalized[key]}"
        for key in overlap_keys
    ]
    if overlap:
        preview = ", ".join(overlap[:5])
        suffix = "" if len(overlap) <= 5 else f" (and {len(overlap) - 5} more)"
        raise ValueError(
            f"Train and validation file lists overlap on {len(overlap)} slice(s): {preview}{suffix}"
        )


def exclude_files(derived_files: list[str], explicit_other_cohort: list[str]) -> list[str]:
    """Let an explicit cohort override membership inferred from Overview."""
    excluded = set(explicit_other_cohort)
    return [filename for filename in derived_files if filename not in excluded]
