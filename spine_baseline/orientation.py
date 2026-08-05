"""Orientation policies for converting SPIDER volumes into sagittal slices."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd


ORIENTATION_MODES = {"legacy", "metadata", "manifest"}
MANIFEST_COLUMNS = (
    "series_id",
    "sagittal_axis",
    "rotate_k",
    "flip_lr",
    "flip_ud",
    "reason",
    "review_status",
)


class OrientationError(ValueError):
    """Raised when an orientation policy cannot be applied without guessing."""


@dataclass(frozen=True)
class OrientationTransform:
    """A reproducible transform applied after moving the stack axis to axis 0."""

    sagittal_axis: int
    rotate_k: int = 0
    flip_lr: bool = False
    flip_ud: bool = False
    reason: str = ""
    review_status: str = ""

    def __post_init__(self) -> None:
        if self.sagittal_axis not in (0, 1, 2):
            raise OrientationError(f"sagittal_axis must be 0, 1, or 2; got {self.sagittal_axis}")
        if self.rotate_k not in (0, 1, 2, 3):
            raise OrientationError(f"rotate_k must be 0, 1, 2, or 3; got {self.rotate_k}")


def _parse_bool(value: object, *, field: str, series_id: str) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)) and int(value) in (0, 1):
        return bool(value)
    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "yes"}:
        return True
    if normalized in {"false", "0", "no"}:
        return False
    raise OrientationError(
        f"{series_id}: {field} must be true/false or 1/0; got {value!r}"
    )


def _parse_bounded_int(
    value: object,
    *,
    field: str,
    series_id: str,
    allowed: tuple[int, ...],
) -> int:
    try:
        parsed = int(value)
        numeric_value = float(value)
    except (TypeError, ValueError) as exc:
        raise OrientationError(f"{series_id}: {field} must be an integer") from exc
    if not np.isfinite(numeric_value) or numeric_value != parsed or parsed not in allowed:
        raise OrientationError(f"{series_id}: {field} must be one of {allowed}; got {value!r}")
    return parsed


def load_orientation_manifest(path: Path) -> dict[str, OrientationTransform]:
    """Load and strictly validate the human-reviewed orientation manifest.

    Silent defaults would make a partially filled review sheet look valid. Every
    field is therefore required, even when a series needs no correction.
    """
    if not path.is_file():
        raise OrientationError(f"Orientation manifest does not exist: {path}")

    table = pd.read_csv(path, dtype={"series_id": str, "reason": str, "review_status": str})
    missing_columns = [column for column in MANIFEST_COLUMNS if column not in table.columns]
    if missing_columns:
        raise OrientationError(f"Orientation manifest is missing columns: {missing_columns}")
    if table.empty:
        raise OrientationError("Orientation manifest is empty")
    if table["series_id"].isna().any() or (table["series_id"].str.strip() == "").any():
        raise OrientationError("Orientation manifest contains an empty series_id")
    table["series_id"] = table["series_id"].str.strip()

    duplicate_ids = sorted(table.loc[table["series_id"].duplicated(keep=False), "series_id"].unique())
    if duplicate_ids:
        raise OrientationError(f"Orientation manifest contains duplicate series_id values: {duplicate_ids}")

    transforms: dict[str, OrientationTransform] = {}
    for row in table.loc[:, MANIFEST_COLUMNS].itertuples(index=False):
        series_id = row.series_id.strip()
        sagittal_axis = _parse_bounded_int(
            row.sagittal_axis,
            field="sagittal_axis",
            series_id=series_id,
            allowed=(0, 1, 2),
        )
        rotate_k = _parse_bounded_int(
            row.rotate_k,
            field="rotate_k",
            series_id=series_id,
            allowed=(0, 1, 2, 3),
        )

        # A manifest is the reviewed source of truth. Refusing blank review
        # fields prevents an unreviewed row from silently entering training.
        if pd.isna(row.review_status) or not str(row.review_status).strip():
            raise OrientationError(f"{series_id}: review_status must not be empty")
        if pd.isna(row.reason) or not str(row.reason).strip():
            raise OrientationError(f"{series_id}: reason must not be empty")
        review_status = str(row.review_status).strip()
        reason = str(row.reason).strip()
        if review_status.lower() != "reviewed":
            # The manifest is applied as one versioned experiment input. A
            # partially reviewed table is therefore rejected as a whole, even
            # when the current sequence filter would use only reviewed rows.
            raise OrientationError(
                f"{series_id}: every manifest row must have review_status='reviewed'"
            )
        transforms[series_id] = OrientationTransform(
            sagittal_axis=sagittal_axis,
            rotate_k=rotate_k,
            flip_lr=_parse_bool(row.flip_lr, field="flip_lr", series_id=series_id),
            flip_ud=_parse_bool(row.flip_ud, field="flip_ud", series_id=series_id),
            reason=reason,
            review_status=review_status,
        )
    return transforms


def infer_sagittal_axis_from_direction(
    direction: tuple[float, ...],
    *,
    ambiguity_tolerance: float = 1e-6,
) -> int:
    """Return the NumPy axis whose normal best matches physical left/right.

    SimpleITK direction columns describe the physical direction of the image
    index axes (x, y, z). Its physical x coordinate is the left/right axis in
    LPS space. NumPy reverses the index order to (z, y, x), hence ``2 - axis``.
    Oblique scans are supported, but an equal-best match is rejected because
    choosing either axis would be an undocumented correction.
    """
    if len(direction) != 9:
        raise OrientationError(f"Expected a 3x3 direction matrix; got {len(direction)} values")
    matrix = np.asarray(direction, dtype=np.float64).reshape(3, 3)
    if not np.isfinite(matrix).all():
        raise OrientationError("Direction matrix contains a non-finite value")
    if not np.allclose(matrix.T @ matrix, np.eye(3), atol=1e-4, rtol=0.0):
        raise OrientationError("Direction matrix is not orthonormal")

    left_right_alignment = np.abs(matrix[0, :])
    ranked = np.sort(left_right_alignment)
    if ranked[-1] - ranked[-2] <= ambiguity_tolerance:
        raise OrientationError(
            "Direction metadata is ambiguous: multiple image axes align equally with left/right"
        )
    sitk_index_axis = int(np.argmax(left_right_alignment))
    return 2 - sitk_index_axis


def resolve_orientation(
    *,
    mode: str,
    array_shape: tuple[int, ...],
    direction: tuple[float, ...],
    series_id: str,
    manifest: Mapping[str, OrientationTransform] | None = None,
) -> OrientationTransform:
    if mode not in ORIENTATION_MODES:
        raise OrientationError(f"Unknown orientation mode {mode!r}; expected one of {sorted(ORIENTATION_MODES)}")
    if len(array_shape) != 3:
        raise OrientationError(f"{series_id}: expected a 3D volume; got shape {array_shape}")
    if mode == "legacy":
        return OrientationTransform(
            sagittal_axis=int(np.argmin(array_shape)),
            reason="smallest_array_dimension",
            review_status="legacy_heuristic",
        )
    if mode == "metadata":
        return OrientationTransform(
            sagittal_axis=infer_sagittal_axis_from_direction(direction),
            reason="simpleitk_direction",
            review_status="metadata_inferred",
        )
    if manifest is None:
        raise OrientationError("manifest mode requires a loaded orientation manifest")
    try:
        transform = manifest[series_id]
    except KeyError as exc:
        raise OrientationError(f"{series_id}: no row in orientation manifest") from exc
    if transform.review_status.strip().lower() != "reviewed":
        # The manifest may double as an audit worklist, but only a human-reviewed
        # row is permitted to influence training data.
        raise OrientationError(
            f"{series_id}: review_status must be 'reviewed' before applying the manifest"
        )
    return transform


def apply_in_plane_transform(array: np.ndarray, transform: OrientationTransform) -> np.ndarray:
    """Apply the reviewed 2D operations in their documented, stable order."""
    result = np.rot90(array, k=transform.rotate_k)
    if transform.flip_lr:
        result = np.fliplr(result)
    if transform.flip_ud:
        result = np.flipud(result)
    return result


def transform_pair(
    image_slice: np.ndarray,
    mask_slice: np.ndarray,
    transform: OrientationTransform,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply exactly the same geometric operation to an image and its mask."""
    if image_slice.shape != mask_slice.shape:
        raise OrientationError(
            f"Image/mask slice shape mismatch: {image_slice.shape} vs {mask_slice.shape}"
        )
    return (
        apply_in_plane_transform(image_slice, transform),
        apply_in_plane_transform(mask_slice, transform),
    )
