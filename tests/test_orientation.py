from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from spine_baseline.orientation import (
    OrientationError,
    OrientationTransform,
    infer_sagittal_axis_from_direction,
    load_orientation_manifest,
    resolve_orientation,
    transform_pair,
)


def test_legacy_orientation_keeps_smallest_dimension_heuristic() -> None:
    transform = resolve_orientation(
        mode="legacy",
        array_shape=(7, 3, 5),
        direction=tuple(np.eye(3).ravel()),
        series_id="series",
    )

    assert transform.sagittal_axis == 1
    assert transform.rotate_k == 0
    assert not transform.flip_lr
    assert not transform.flip_ud


@pytest.mark.parametrize(
    ("direction", "expected_numpy_axis"),
    [
        (tuple(np.eye(3).ravel()), 2),
        ((0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0), 0),
    ],
)
def test_metadata_orientation_maps_physical_left_right_to_numpy_axis(
    direction: tuple[float, ...], expected_numpy_axis: int
) -> None:
    assert infer_sagittal_axis_from_direction(direction) == expected_numpy_axis


def test_metadata_orientation_rejects_ambiguous_direction() -> None:
    root_half = np.sqrt(0.5)
    direction = (
        root_half,
        root_half,
        0.0,
        -root_half,
        root_half,
        0.0,
        0.0,
        0.0,
        1.0,
    )

    with pytest.raises(OrientationError, match="ambiguous"):
        infer_sagittal_axis_from_direction(direction)


def test_manifest_loads_reviewed_transform(tmp_path: Path) -> None:
    path = tmp_path / "orientation.csv"
    pd.DataFrame(
        [
            {
                "series_id": "scan_01",
                "sagittal_axis": 1,
                "rotate_k": 3,
                "flip_lr": "yes",
                "flip_ud": 0,
                "reason": "manual overlay review",
                "review_status": "reviewed",
            }
        ]
    ).to_csv(path, index=False)

    transform = load_orientation_manifest(path)["scan_01"]

    assert transform == OrientationTransform(
        sagittal_axis=1,
        rotate_k=3,
        flip_lr=True,
        flip_ud=False,
        reason="manual overlay review",
        review_status="reviewed",
    )


def test_manifest_rejects_duplicate_series(tmp_path: Path) -> None:
    path = tmp_path / "orientation.csv"
    row = {
        "series_id": "scan_01",
        "sagittal_axis": 1,
        "rotate_k": 0,
        "flip_lr": False,
        "flip_ud": False,
        "reason": "reviewed",
        "review_status": "reviewed",
    }
    duplicate_with_whitespace = {**row, "series_id": " scan_01 "}
    pd.DataFrame([row, duplicate_with_whitespace]).to_csv(path, index=False)

    with pytest.raises(OrientationError, match="duplicate"):
        load_orientation_manifest(path)


def test_manifest_rejects_fractional_axis_instead_of_truncating(tmp_path: Path) -> None:
    path = tmp_path / "orientation.csv"
    pd.DataFrame(
        [
            {
                "series_id": "scan_01",
                "sagittal_axis": 1.5,
                "rotate_k": 0,
                "flip_lr": False,
                "flip_ud": False,
                "reason": "reviewed",
                "review_status": "reviewed",
            }
        ]
    ).to_csv(path, index=False)

    with pytest.raises(OrientationError, match="sagittal_axis"):
        load_orientation_manifest(path)


def test_manifest_loader_rejects_any_unreviewed_row(tmp_path: Path) -> None:
    path = tmp_path / "orientation.csv"
    pd.DataFrame(
        [
            {
                "series_id": "scan_reviewed",
                "sagittal_axis": 1,
                "rotate_k": 0,
                "flip_lr": False,
                "flip_ud": False,
                "reason": "checked",
                "review_status": "reviewed",
            },
            {
                "series_id": "scan_pending",
                "sagittal_axis": 1,
                "rotate_k": 0,
                "flip_lr": False,
                "flip_ud": False,
                "reason": "not checked",
                "review_status": "needs_review",
            },
        ]
    ).to_csv(path, index=False)

    with pytest.raises(OrientationError, match="every manifest row"):
        load_orientation_manifest(path)


def test_transform_pair_applies_identical_geometry() -> None:
    image = np.arange(6).reshape(2, 3)
    mask = image + 100
    transform = OrientationTransform(
        sagittal_axis=0,
        rotate_k=1,
        flip_lr=True,
        flip_ud=True,
    )

    transformed_image, transformed_mask = transform_pair(image, mask, transform)

    np.testing.assert_array_equal(transformed_mask - transformed_image, np.full((3, 2), 100))


def test_manifest_mode_requires_series_row() -> None:
    with pytest.raises(OrientationError, match="no row"):
        resolve_orientation(
            mode="manifest",
            array_shape=(2, 3, 4),
            direction=tuple(np.eye(3).ravel()),
            series_id="missing",
            manifest={},
        )


def test_manifest_mode_rejects_unreviewed_row() -> None:
    with pytest.raises(OrientationError, match="must be 'reviewed'"):
        resolve_orientation(
            mode="manifest",
            array_shape=(2, 3, 4),
            direction=tuple(np.eye(3).ravel()),
            series_id="scan",
            manifest={
                "scan": OrientationTransform(
                    sagittal_axis=0,
                    review_status="pending",
                )
            },
        )
