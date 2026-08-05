from __future__ import annotations

import importlib.util
import sys
import types
import unittest
from unittest.mock import patch

import numpy as np

# The Mac-side lightweight test environment intentionally omits TensorFlow;
# these tests exercise NumPy evaluation logic only. Keep the stub local to the
# test module and do not hide a real TensorFlow installation on GPU hosts.
if importlib.util.find_spec("tensorflow") is None:
    tensorflow_stub = types.ModuleType("tensorflow")
    tensorflow_stub.data = types.SimpleNamespace(Dataset=object)
    sys.modules["tensorflow"] = tensorflow_stub

from spine_baseline.constants import CLASS_NAMES
from spine_baseline.metrics import (
    aggregate_overlap_metrics,
    evaluate_classwise,
    evaluate_classwise_with_aggregations,
)


class _FakeModel:
    def __init__(self, predictions: dict[int, np.ndarray], num_classes: int) -> None:
        self.predictions = predictions
        self.num_classes = num_classes

    def predict(self, image: np.ndarray, verbose: int = 0) -> np.ndarray:
        marker = int(image[0, 0, 0, 0])
        predicted_classes = self.predictions[marker]
        return np.eye(self.num_classes, dtype=np.float32)[predicted_classes][np.newaxis, ...]


class MetricsAggregationTests(unittest.TestCase):
    def test_aggregations_distinguish_slice_pixel_and_series_weighting(self) -> None:
        # Series A contains two slices while series B contains one. Deliberately
        # unequal error distributions make all three weighting definitions
        # produce different foreground Dice values.
        slice_confusions = [
            np.array([[4, 0], [0, 4]]),
            np.array([[4, 0], [4, 0]]),
            np.array([[0, 1], [0, 7]]),
        ]
        results = aggregate_overlap_metrics(
            slice_confusions,
            ["series_a", "series_a", "series_b"],
            num_classes=2,
        )

        foreground = results.loc[results["scope"] == "foreground_classes"].set_index("aggregation")
        self.assertEqual(set(foreground.index), {"slice_macro", "pixel_pooled", "series_macro"})
        self.assertAlmostEqual(foreground.loc["slice_macro", "dice"], (1.0 + 0.0 + 14.0 / 15.0) / 3.0)
        self.assertAlmostEqual(foreground.loc["pixel_pooled", "dice"], 22.0 / 27.0)
        self.assertAlmostEqual(foreground.loc["series_macro", "dice"], (2.0 / 3.0 + 14.0 / 15.0) / 2.0)
        self.assertTrue((results["slice_count"] == 3).all())
        self.assertTrue((results["series_count"] == 2).all())

    def test_evaluation_keeps_legacy_schema_and_returns_explicit_scopes(self) -> None:
        filenames = ["series_a_s000.npz", "series_b_s000.npz"]
        true_masks = {
            filenames[0]: np.array([[0, 1], [2, 3]], dtype=np.int32),
            filenames[1]: np.array([[0, 1], [2, 3]], dtype=np.int32),
        }
        predictions = {
            1: np.array([[0, 1], [2, 3]], dtype=np.int32),
            2: np.array([[0, 0], [2, 3]], dtype=np.int32),
        }

        def fake_load_sample(filename, output_root, num_classes):
            marker = 1 if filename == filenames[0] else 2
            # This focused loader bypasses image normalization so a marker can
            # select a deterministic fake-model prediction.
            image = np.full((2, 2, 1), marker, dtype=np.float32)
            mask = np.eye(num_classes, dtype=np.float32)[true_masks[filename]]
            return image, mask

        model = _FakeModel(predictions, num_classes=4)
        with (
            patch("spine_baseline.metrics.load_sample", side_effect=fake_load_sample),
            patch("spine_baseline.metrics.load_slice_spacing", return_value=np.array([1.0, 1.0])),
        ):
            legacy = evaluate_classwise(model, filenames, output_root=None, num_classes=4)
            classwise, aggregations = evaluate_classwise_with_aggregations(
                model,
                filenames,
                output_root=None,
                num_classes=4,
            )

        expected_legacy_columns = [
            "class", "dice", "iou", "asd", "nsd", "precision", "recall", "f1"
        ]
        self.assertEqual(list(legacy.columns), expected_legacy_columns)
        self.assertEqual(list(classwise.columns), expected_legacy_columns)
        self.assertEqual(legacy["class"].tolist(), [*CLASS_NAMES, "Mean"])
        np.testing.assert_allclose(
            legacy.drop(columns="class").to_numpy(),
            classwise.drop(columns="class").to_numpy(),
        )

        expected_scopes = {*CLASS_NAMES, "all_classes", "foreground_classes"}
        self.assertEqual(set(aggregations["scope"]), expected_scopes)
        self.assertEqual(
            set(aggregations["aggregation"]),
            {"slice_macro", "pixel_pooled", "series_macro"},
        )
        self.assertEqual(len(aggregations), 3 * (len(CLASS_NAMES) + 2))

        slice_macro = aggregations.loc[aggregations["aggregation"] == "slice_macro"].set_index("scope")
        legacy_by_class = legacy.set_index("class")
        for class_name in CLASS_NAMES:
            self.assertAlmostEqual(
                slice_macro.loc[class_name, "dice"],
                legacy_by_class.loc[class_name, "dice"],
            )
            self.assertAlmostEqual(
                slice_macro.loc[class_name, "iou"],
                legacy_by_class.loc[class_name, "iou"],
            )

        self.assertAlmostEqual(
            slice_macro.loc["all_classes", "dice"],
            slice_macro.loc[CLASS_NAMES, "dice"].mean(),
        )
        self.assertAlmostEqual(
            slice_macro.loc["foreground_classes", "dice"],
            slice_macro.loc[CLASS_NAMES[1:], "dice"].mean(),
        )


if __name__ == "__main__":
    unittest.main()
