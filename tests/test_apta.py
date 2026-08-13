import unittest
import importlib.util

import numpy as np

from spine_baseline.apta import (
    apply_public_apta_neighborhoods,
    normalize_mask_like_author,
    reconstruct_public_apta,
    threshold_author_colors,
)


class PublicAptaReconstructionTests(unittest.TestCase):
    def test_threshold_boundaries_follow_elif_order(self):
        values = np.asarray([[0, 1, 10, 11, 89, 90, 179, 180, 181, 255]], dtype=np.uint8)

        actual = threshold_author_colors(values)

        np.testing.assert_array_equal(actual, [[0, 1, 1, 0, 0, 2, 2, 2, 3, 3]])

    def test_constant_slice_is_rejected_instead_of_casting_nan(self):
        with self.assertRaisesRegex(ValueError, "constant"):
            normalize_mask_like_author(np.full((3, 3), 100, dtype=np.int16))

    @unittest.skipUnless(importlib.util.find_spec("cv2"), "OpenCV is available in the WSL training environment")
    def test_author_rotate_then_horizontal_flip_is_vertical_flip(self):
        raw = np.asarray([[0, 1, 2], [3, 4, 5]], dtype=np.int16)

        without_flip = reconstruct_public_apta(
            raw,
            target_height=2,
            target_width=3,
            apply_author_flip=False,
        )
        with_flip = reconstruct_public_apta(
            raw,
            target_height=2,
            target_width=3,
            apply_author_flip=True,
        )

        np.testing.assert_array_equal(with_flip, np.flipud(without_flip))

    def test_red_absence_collapses_green_and_blue_to_red(self):
        labels = np.asarray([
            [2, 2, 2, 3],
            [2, 2, 3, 3],
            [2, 3, 3, 3],
            [0, 0, 0, 0],
        ], dtype=np.uint8)

        actual = apply_public_apta_neighborhoods(labels)

        self.assertFalse(np.any(actual == 2))
        self.assertFalse(np.any(actual == 3))
        self.assertTrue(np.any(actual == 1))


if __name__ == "__main__":
    unittest.main()
