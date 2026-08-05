import unittest

import numpy as np

from scripts.audit_data_alignment import map_numeric_labels, summarize_labels


class DataAlignmentAuditTests(unittest.TestCase):
    def test_direct_mapping_preserves_confirmed_spider_groups(self):
        raw = np.asarray([0, 1, 8, 99, 100, 201, 208], dtype=np.int16)

        mapped = map_numeric_labels(raw)

        np.testing.assert_array_equal(mapped, np.asarray([0, 1, 1, 1, 2, 3, 3], dtype=np.uint8))

    def test_summary_flags_unexpected_unmapped_label_range(self):
        mask = np.asarray([[[0, 1], [100, 150]], [[0, 1], [100, 200]]], dtype=np.int16)

        summary = summarize_labels(mask, sagittal_axis=0)

        self.assertEqual(summary["unexpected_raw_labels"], "150")
        self.assertEqual(summary["direct_mapping_review"], "needs_review")


if __name__ == "__main__":
    unittest.main()
