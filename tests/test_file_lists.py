import tempfile
import unittest
from pathlib import Path

import numpy as np

from spine_baseline.file_lists import (
    exclude_files,
    read_file_list,
    validate_disjoint_cohorts,
    validate_slice_files,
)


class FileListValidationTests(unittest.TestCase):
    def test_read_file_list_rejects_empty_and_duplicate_lists(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            empty = root / "empty.txt"
            duplicate = root / "duplicate.txt"
            empty.write_text("\n", encoding="utf-8")
            duplicate.write_text("slice.npz\nslice.npz\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "empty"):
                read_file_list(empty)
            with self.assertRaisesRegex(ValueError, "duplicates"):
                read_file_list(duplicate)

    def test_validate_slice_files_requires_image_and_mask_pairs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "images").mkdir()
            (root / "masks").mkdir()
            np.savez(root / "images" / "complete.npz", image=np.zeros((2, 3)), sequence="T2_SPACE")
            np.savez(
                root / "masks" / "complete.npz",
                mask=np.zeros((2, 3), dtype=np.uint8),
                sequence="T2_SPACE",
            )
            (root / "images" / "missing_mask.npz").touch()

            validate_slice_files(["complete.npz"], root, "Train", {"T2_SPACE"})
            with self.assertRaisesRegex(ValueError, "without both image and mask"):
                validate_slice_files(["missing_mask.npz"], root, "Train")

    def test_validate_slice_files_rejects_paths_and_wrong_sequence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "images").mkdir()
            (root / "masks").mkdir()
            np.savez(root / "images" / "slice.npz", image=np.zeros((2, 3)), sequence="T1")
            np.savez(
                root / "masks" / "slice.npz",
                mask=np.zeros((2, 3), dtype=np.uint8),
                sequence="T1",
            )

            with self.assertRaisesRegex(ValueError, "basenames only"):
                validate_slice_files(["../slice.npz"], root, "Train")
            with self.assertRaisesRegex(ValueError, "outside"):
                validate_slice_files(["slice.npz"], root, "Train", {"T2_SPACE"})

    def test_validate_disjoint_cohorts_rejects_overlap(self) -> None:
        with self.assertRaisesRegex(ValueError, "overlap"):
            validate_disjoint_cohorts(
                ["train_only.npz", "shared.npz"],
                ["validation_only.npz", "shared.npz"],
            )

        with self.assertRaisesRegex(ValueError, "overlap"):
            validate_disjoint_cohorts(["Slice.npz"], ["slice.npz"])

    def test_validate_slice_files_rejects_fractional_mask(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "images").mkdir()
            (root / "masks").mkdir()
            np.savez(root / "images" / "slice.npz", image=np.zeros((2, 3)), sequence="T2_SPACE")
            np.savez(
                root / "masks" / "slice.npz",
                mask=np.full((2, 3), 0.5, dtype=np.float32),
                sequence="T2_SPACE",
            )

            with self.assertRaisesRegex(ValueError, "dtype must be integer"):
                validate_slice_files(["slice.npz"], root, "Train", {"T2_SPACE"})

    def test_explicit_validation_overrides_overview_training_membership(self) -> None:
        derived_train = ["train_only.npz", "explicit_validation.npz"]

        self.assertEqual(
            exclude_files(derived_train, ["explicit_validation.npz"]),
            ["train_only.npz"],
        )


if __name__ == "__main__":
    unittest.main()
