import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.audit_filter_sensitivity import audit_threshold, parse_thresholds


class FilterSensitivityAuditTests(unittest.TestCase):
    def test_parse_thresholds_rejects_duplicates(self):
        with self.assertRaisesRegex(ValueError, "unique"):
            parse_thresholds("0.55,0.55")

    def test_audit_does_not_modify_processed_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_root = root / "data"
            processed_root = root / "processed"
            (processed_root / "masks").mkdir(parents=True)
            data_root.mkdir()
            pd.DataFrame([
                {"subset": "training", "new_file_name": "case_t2_space"},
            ]).to_csv(data_root / "SPIDER Lumbar Spine Segmentation Overview.csv", index=False)
            mask = np.asarray([0, *([1] * 55), *([2] * 20), *([3] * 25)], dtype=np.uint8)
            np.savez_compressed(processed_root / "masks" / "case_t2_space_s000.npz", mask=mask)

            decisions, summary = audit_threshold(
                data_root,
                processed_root,
                threshold=0.55,
                min_classes=4,
                max_per_sequence=1000,
            )

            self.assertTrue(decisions[0]["selected_after_cap"])
            self.assertIn("proxy", decisions[0]["filter_definition"])
            self.assertIn("proxy", summary[-1]["filter_definition"])
            self.assertEqual(summary[-1]["train_slices"], 1)
            self.assertFalse((processed_root / "filtered_files.txt").exists())


if __name__ == "__main__":
    unittest.main()
