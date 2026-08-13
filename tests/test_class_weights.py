from argparse import ArgumentParser
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from arguments import add_optimization_args

try:
    import tensorflow as tf
except ImportError:  # pragma: no cover - WSL is the authoritative training environment.
    tf = None

from spine_baseline.class_weights import (
    count_training_class_pixels,
    derive_focal_class_weights,
    inverse_sqrt_frequency_weights,
    write_focal_class_weight_evidence,
)

if tf is not None:
    from spine_baseline.losses import focal_loss


class ClassWeightTests(unittest.TestCase):
    def test_default_cli_mode_preserves_equal_weight_baseline(self) -> None:
        parser = ArgumentParser()
        add_optimization_args(parser)

        args = parser.parse_args([])

        self.assertEqual(args.focal_class_weight_mode, "none")

    def test_counts_training_masks_and_inverse_sqrt_weights(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "masks").mkdir()
            np.savez_compressed(root / "masks" / "a.npz", mask=np.array([[0, 0], [1, 2]]))
            np.savez_compressed(root / "masks" / "b.npz", mask=np.array([[0, 1], [1, 3]]))

            counts = count_training_class_pixels(["a.npz", "b.npz"], root, 4)
            weights = inverse_sqrt_frequency_weights(counts)

            np.testing.assert_array_equal(counts, [3, 3, 1, 1])
            self.assertGreater(weights[2], weights[0])
            self.assertGreater(weights[3], weights[1])
            frequencies = counts / counts.sum()
            self.assertAlmostEqual(float(np.dot(frequencies, weights)), 1.0, places=6)

    def test_none_mode_does_not_read_training_masks(self) -> None:
        weights, counts = derive_focal_class_weights("none", ["missing.npz"], Path("missing"), 4)
        self.assertIsNone(weights)
        self.assertIsNone(counts)

    def test_rejects_missing_class_in_training_cohort(self) -> None:
        with self.assertRaisesRegex(ValueError, "Every class"):
            inverse_sqrt_frequency_weights(np.array([10, 2, 1, 0]))

    def test_rejects_fractional_mask_before_casting(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "masks").mkdir()
            np.savez_compressed(root / "masks" / "bad.npz", mask=np.array([[0.0, 0.5]]))

            with self.assertRaisesRegex(ValueError, "non-integer"):
                count_training_class_pixels(["bad.npz"], root, 4)

    def test_writes_auditable_evidence(self) -> None:
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "focal_class_weights.tsv"
            write_focal_class_weight_evidence(
                path,
                "inverse_sqrt_train",
                np.array([3, 3, 1, 1]),
                np.array([0.8, 0.8, 1.4, 1.4]),
            )

            contents = path.read_text(encoding="utf-8")
            self.assertIn("focal_class_weight_mode\tinverse_sqrt_train", contents)
            self.assertIn("class_2_pixel_count\t1", contents)
            self.assertIn("class_2_focal_weight\t1.4", contents)

    @unittest.skipIf(tf is None, "TensorFlow is available in the WSL training environment")
    def test_class_weights_change_only_the_target_class_focal_term(self) -> None:
        y_true = tf.constant([[[[0.0, 0.0, 1.0, 0.0]]]])
        y_pred = tf.constant([[[[0.1, 0.1, 0.4, 0.4]]]])

        equal = float(focal_loss(y_true, y_pred, gamma=4.0).numpy())
        weighted = float(
            focal_loss(y_true, y_pred, gamma=4.0, class_weights=[1.0, 1.0, 2.0, 1.0]).numpy()
        )

        self.assertAlmostEqual(weighted, 2.0 * equal, places=6)


if __name__ == "__main__":
    unittest.main()
