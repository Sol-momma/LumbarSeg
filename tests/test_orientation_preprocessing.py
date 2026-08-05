from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sitk = pytest.importorskip("SimpleITK")

from spine_baseline.orientation import OrientationTransform, transform_pair
from spine_baseline.preprocessing import extract_slices, transformed_resized_slice_spacing


def test_quarter_turn_swaps_physical_spacing_before_resize() -> None:
    spacing = transformed_resized_slice_spacing(
        original_shape=(2, 4),
        sagittal_axis=0,
        axis_spacings=(9.0, 2.0, 3.0),
        transform=OrientationTransform(sagittal_axis=0, rotate_k=1),
        target_height=4,
        target_width=2,
    )

    # Original in-plane spacing is (2, 3). Rotation changes shape to (4, 2)
    # and swaps the spacing to (3, 2), with no further resize scale here.
    np.testing.assert_allclose(spacing, np.array([3.0, 2.0], dtype=np.float32))


def test_extract_slices_records_and_applies_manifest_orientation(tmp_path: Path) -> None:
    data_root = tmp_path / "dataset"
    (data_root / "images").mkdir(parents=True)
    (data_root / "masks").mkdir()
    output_root = tmp_path / "processed"
    series_id = "scan_t2"

    volume = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    mask = np.zeros_like(volume, dtype=np.int16)
    mask[:, 0, :] = 1
    mask[:, 1, :] = 100
    mask[:, 2, :] = 200
    sitk.WriteImage(sitk.GetImageFromArray(volume), str(data_root / "images" / f"{series_id}.mha"))
    sitk.WriteImage(sitk.GetImageFromArray(mask), str(data_root / "masks" / f"{series_id}.mha"))

    manifest_path = tmp_path / "orientation.csv"
    pd.DataFrame(
        [
            {
                "series_id": series_id,
                "sagittal_axis": 0,
                "rotate_k": 1,
                "flip_lr": True,
                "flip_ud": False,
                "reason": "reviewed overlay",
                "review_status": "reviewed",
            }
        ]
    ).to_csv(manifest_path, index=False)

    stats = extract_slices(
        data_root=data_root,
        output_root=output_root,
        target_height=4,
        target_width=3,
        orientation_mode="manifest",
        orientation_manifest=manifest_path,
    )

    assert stats["errors"] == []
    assert stats["total_slices"] == 2
    with np.load(output_root / "images" / f"{series_id}_s000.npz") as image_sample:
        expected_image, _ = transform_pair(
            volume[0], mask[0], OrientationTransform(0, 1, True, False)
        )
        np.testing.assert_array_equal(image_sample["image"], expected_image)
        assert image_sample["orientation_mode"].item() == "manifest"
        assert image_sample["sagittal_axis"].item() == 0
        assert image_sample["rotate_k"].item() == 1
        assert bool(image_sample["flip_lr"].item())
        assert image_sample["orientation_review_status"].item() == "reviewed"
    with np.load(output_root / "masks" / f"{series_id}_s000.npz") as mask_sample:
        _, expected_mask = transform_pair(
            volume[0], mask[0], OrientationTransform(0, 1, True, False)
        )
        # The stored mask is mapped after geometry, so compare to expected class IDs.
        expected_mask = np.where(expected_mask >= 200, 3, np.where(expected_mask == 100, 2, expected_mask))
        np.testing.assert_array_equal(mask_sample["mask"], expected_mask.astype(np.uint8))


def test_orientation_mode_rejects_reusing_legacy_output(tmp_path: Path) -> None:
    data_root = tmp_path / "dataset"
    (data_root / "images").mkdir(parents=True)
    (data_root / "masks").mkdir()
    output_root = tmp_path / "processed"
    series_id = "scan_t2"

    volume = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    mask = np.zeros_like(volume, dtype=np.int16)
    sitk.WriteImage(sitk.GetImageFromArray(volume), str(data_root / "images" / f"{series_id}.mha"))
    sitk.WriteImage(sitk.GetImageFromArray(mask), str(data_root / "masks" / f"{series_id}.mha"))

    legacy_stats = extract_slices(
        data_root=data_root,
        output_root=output_root,
        target_height=4,
        target_width=3,
        orientation_mode="legacy",
    )
    assert legacy_stats["errors"] == []

    with pytest.raises(RuntimeError, match="Orientation-aware extraction failed"):
        extract_slices(
            data_root=data_root,
            output_root=output_root,
            target_height=4,
            target_width=3,
            orientation_mode="metadata",
        )


def test_metadata_mode_rejects_image_mask_spacing_mismatch(tmp_path: Path) -> None:
    data_root = tmp_path / "dataset"
    (data_root / "images").mkdir(parents=True)
    (data_root / "masks").mkdir()
    output_root = tmp_path / "processed"
    series_id = "scan_t2"

    volume = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    image_itk = sitk.GetImageFromArray(volume)
    mask_itk = sitk.GetImageFromArray(np.zeros_like(volume, dtype=np.int16))
    image_itk.SetSpacing((1.0, 1.0, 1.0))
    mask_itk.SetSpacing((1.0, 1.0, 2.0))
    sitk.WriteImage(image_itk, str(data_root / "images" / f"{series_id}.mha"))
    sitk.WriteImage(mask_itk, str(data_root / "masks" / f"{series_id}.mha"))

    with pytest.raises(RuntimeError, match="spacing metadata differ"):
        extract_slices(
            data_root=data_root,
            output_root=output_root,
            target_height=4,
            target_width=3,
            orientation_mode="metadata",
        )


def test_orientation_mode_rejects_force_reprocessing(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="new output_root"):
        extract_slices(
            data_root=tmp_path / "dataset",
            output_root=tmp_path / "processed",
            target_height=4,
            target_width=3,
            force=True,
            orientation_mode="metadata",
        )
