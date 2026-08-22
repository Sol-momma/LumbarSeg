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

if tf is not None:
    from spine_baseline.losses import (
        combined_loss,
        dice_loss,
        focal_loss,
        spinal_canal_boundary_mask,
        validate_loss_configuration,
        write_loss_config,
    )


class BoundaryFocalConfigurationTests(unittest.TestCase):
    def test_cli_default_preserves_baseline(self) -> None:
        parser = ArgumentParser()
        add_optimization_args(parser)

        args = parser.parse_args([])

        self.assertEqual(args.focal_canal_boundary_boost, 0.0)

    @unittest.skipIf(tf is None, "TensorFlow is available in the WSL training environment")
    def test_rejects_negative_or_combined_loss_changes(self) -> None:
        with self.assertRaisesRegex(ValueError, "zero or greater"):
            validate_loss_configuration("none", -0.1)
        with self.assertRaisesRegex(ValueError, "one loss change"):
            validate_loss_configuration("inverse_sqrt_train", 2.0)

    @unittest.skipIf(tf is None, "TensorFlow is available in the WSL training environment")
    def test_writes_auditable_loss_evidence(self) -> None:
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "loss_config.tsv"
            write_loss_config(
                path,
                focal_weight=0.6,
                focal_gamma=4.0,
                focal_class_weight_mode="none",
                focal_canal_boundary_boost=2.0,
            )

            contents = path.read_text(encoding="utf-8")
            self.assertIn("combined_focal_weight\t0.6", contents)
            self.assertIn("combined_dice_weight\t0.4", contents)
            self.assertIn("boundary_class_id\t2", contents)
            self.assertIn("focal_canal_boundary_boost\t2.0", contents)
            self.assertIn("keras_jit_compile\tfalse", contents)

    @unittest.skipIf(tf is None, "TensorFlow is available in the WSL training environment")
    def test_baseline_records_the_same_deterministic_non_xla_path(self) -> None:
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "loss_config.tsv"
            write_loss_config(
                path,
                focal_weight=0.6,
                focal_gamma=4.0,
                focal_class_weight_mode="none",
                focal_canal_boundary_boost=0.0,
            )

            self.assertIn("keras_jit_compile\tfalse", path.read_text(encoding="utf-8"))


@unittest.skipIf(tf is None, "TensorFlow is available in the WSL training environment")
class BoundaryFocalTensorTests(unittest.TestCase):
    @staticmethod
    def _one_hot(mask: np.ndarray):
        return tf.one_hot(mask.astype(np.int32), depth=4, dtype=tf.float32)[None, ...]

    def test_boundary_mask_is_only_the_inner_outer_band(self) -> None:
        mask = np.zeros((7, 7), dtype=np.int32)
        mask[2:5, 2:5] = 2
        expected = np.zeros((7, 7), dtype=np.float32)
        expected[1:6, 1:6] = 1.0
        expected[3, 3] = 0.0

        actual = spinal_canal_boundary_mask(self._one_hot(mask)).numpy()[0]

        np.testing.assert_array_equal(actual, expected)

    def test_zero_boost_matches_the_previous_focal_formula(self) -> None:
        y_true = self._one_hot(np.array([[0, 1], [2, 3]], dtype=np.int32))
        y_pred = tf.constant(
            [[[[0.7, 0.1, 0.1, 0.1], [0.1, 0.6, 0.2, 0.1]],
              [[0.1, 0.1, 0.7, 0.1], [0.1, 0.2, 0.1, 0.6]]]],
            dtype=tf.float32,
        )
        clipped = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        previous = tf.reduce_mean(
            tf.reduce_sum(-tf.pow(1.0 - clipped, 4.0) * y_true * tf.math.log(clipped), axis=-1)
        )

        actual = focal_loss(y_true, y_pred, gamma=4.0, canal_boundary_boost=0.0)

        self.assertEqual(float(actual.numpy()), float(previous.numpy()))

    def test_empty_canal_is_safe_and_matches_baseline(self) -> None:
        y_true = self._one_hot(np.zeros((3, 3), dtype=np.int32))
        y_pred = tf.fill((1, 3, 3, 4), 0.25)

        baseline = focal_loss(y_true, y_pred, canal_boundary_boost=0.0)
        boosted = focal_loss(y_true, y_pred, canal_boundary_boost=2.0)

        self.assertTrue(np.isfinite(float(boosted.numpy())))
        self.assertAlmostEqual(float(boosted.numpy()), float(baseline.numpy()), places=7)

    def test_only_equal_boundary_error_receives_extra_weight(self) -> None:
        mask = np.zeros((7, 7), dtype=np.int32)
        mask[2:5, 2:5] = 2
        y_true = self._one_hot(mask)

        def prediction_with_error(row: int, column: int):
            probabilities = tf.one_hot(mask, depth=4, dtype=tf.float32).numpy() * 0.96 + 0.01
            true_class = mask[row, column]
            probabilities[row, column] = 0.5 / 3.0
            probabilities[row, column, true_class] = 0.5
            return tf.constant(probabilities[None, ...], dtype=tf.float32)

        boundary = focal_loss(y_true, prediction_with_error(2, 3), canal_boundary_boost=2.0)
        centre = focal_loss(y_true, prediction_with_error(3, 3), canal_boundary_boost=2.0)
        distant = focal_loss(y_true, prediction_with_error(0, 0), canal_boundary_boost=2.0)

        self.assertGreater(float(boundary.numpy()), float(centre.numpy()))
        self.assertAlmostEqual(float(centre.numpy()), float(distant.numpy()), places=7)

    def test_combined_loss_keeps_the_paper_ratio(self) -> None:
        y_true = self._one_hot(np.array([[0, 2], [2, 3]], dtype=np.int32))
        y_pred = tf.fill((1, 2, 2, 4), 0.25)
        expected = 0.6 * focal_loss(
            y_true,
            y_pred,
            gamma=4.0,
            canal_boundary_boost=2.0,
        ) + 0.4 * dice_loss(y_true, y_pred)

        actual = combined_loss(alpha=0.6, gamma=4.0, canal_boundary_boost=2.0)(y_true, y_pred)

        self.assertAlmostEqual(float(actual.numpy()), float(expected.numpy()), places=7)


if __name__ == "__main__":
    unittest.main()
