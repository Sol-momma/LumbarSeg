import unittest

import numpy as np

from spine_baseline.filtering import FILTER_DEFINITION, evaluate_slice_filter


class SliceFilterDecisionTests(unittest.TestCase):
    @staticmethod
    def mask_with_foreground_counts(vertebrae: int, canal: int, ivds: int) -> np.ndarray:
        values = [0, *([1] * vertebrae), *([2] * canal), *([3] * ivds)]
        return np.asarray(values, dtype=np.uint8)

    def test_exactly_55_percent_is_kept(self):
        mask = self.mask_with_foreground_counts(55, 20, 25)

        decision = evaluate_slice_filter(mask, min_classes=4, imbalance_threshold=0.55)

        self.assertTrue(decision.keep)
        self.assertEqual(decision.reason, "kept")
        self.assertAlmostEqual(decision.dominant_foreground_fraction, 0.55)

    def test_more_than_55_percent_is_removed(self):
        mask = self.mask_with_foreground_counts(56, 19, 25)

        decision = evaluate_slice_filter(mask, min_classes=4, imbalance_threshold=0.55)

        self.assertFalse(decision.keep)
        self.assertEqual(decision.reason, "dominant_foreground_above_threshold")

    def test_class_count_gate_precedes_imbalance_gate(self):
        mask = np.asarray([0, 1, 1, 2], dtype=np.uint8)

        decision = evaluate_slice_filter(mask, min_classes=4, imbalance_threshold=0.55)

        self.assertFalse(decision.keep)
        self.assertEqual(decision.reason, "fewer_than_min_classes")

    def test_threshold_outside_probability_range_is_rejected(self):
        mask = self.mask_with_foreground_counts(1, 1, 1)

        with self.assertRaisesRegex(ValueError, "within"):
            evaluate_slice_filter(mask, min_classes=4, imbalance_threshold=1.1)

    def test_public_paper_examples_match_proxy_decisions(self):
        # These are the exact RGB color counts in Figure 5(a-c) from the public
        # arXiv source package.  Locking only the published keep/remove examples
        # is deliberate: they support this proxy but cannot resolve the paper's
        # contradictory ratio formula or prove an exact author-side algorithm.
        examples = [
            ((282915, 17800, 13269, 3456), True),
            ((288265, 14582, 11239, 3354), True),
            ((302411, 11488, 934, 2607), False),
        ]

        self.assertIn("proxy", FILTER_DEFINITION)
        for counts, expected_keep in examples:
            with self.subTest(counts=counts):
                background, vertebrae, canal, ivds = counts
                mask = np.concatenate([
                    np.full(background, 0, dtype=np.uint8),
                    np.full(vertebrae, 1, dtype=np.uint8),
                    np.full(canal, 2, dtype=np.uint8),
                    np.full(ivds, 3, dtype=np.uint8),
                ])

                decision = evaluate_slice_filter(mask, min_classes=4, imbalance_threshold=0.55)

                self.assertEqual(decision.keep, expected_keep)


if __name__ == "__main__":
    unittest.main()
