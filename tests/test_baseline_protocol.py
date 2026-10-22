"""The fixed baseline protocol loads and matches the paper settings and the Overview CSV test cases."""
from __future__ import annotations

import csv
import json
import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from lumbarseg_cli import config as cfgmod  # noqa: E402

OVERVIEW = REPO / "data" / "SPIDER Lumbar Spine Segmentation Overview.csv"


class BaselineProtocolTest(unittest.TestCase):
    def setUp(self):
        self.cfg = cfgmod.load("baseline_protocol", profile="box")
        self.proto = json.loads((cfgmod.EXPERIMENTS / "baseline_protocol.json").read_text(encoding="utf-8"))["_protocol"]

    def test_training_settings(self):
        t, d, e = self.cfg["train"], self.cfg["data"], self.cfg["eval"]
        self.assertEqual((t["focal_gamma"], t["focal_weight"], t["batch_size"], t["epochs"]), (0.4, 0.6, 8, 100))
        self.assertEqual((d["target_height"], d["target_width"]), (512, 640))
        self.assertEqual(d["max_slices_per_sequence"], 1000)
        self.assertEqual(d["inner_val_cases"], 4)
        self.assertEqual(e["dice_average_unit"], "case")
        self.assertEqual(e["headline_metric"], self.proto["primary_metric"]["eval.headline_metric"])

    def test_same_as_measured_paper_effective(self):
        # PR #13 ran paper_effective with these --set overrides
        ref = cfgmod.load("paper_effective", profile="box", sets=["data.inner_val_cases=4", "eval.dice_average_unit=case"])
        for sec in ("data", "model", "train", "eval"):
            self.assertEqual(self.cfg[sec], ref[sec], sec)

    def test_seeds(self):
        self.assertGreaterEqual(len(self.proto["seeds"]), 3)
        self.assertEqual(self.proto["seeds"], [42, 43, 44])

    def test_test_cases_match_overview_csv(self):
        with OVERVIEW.open(encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        val = {r["new_file_name"] for r in rows if r["subset"] == "validation" and r["new_file_name"].endswith("_t2_SPACE")}
        self.assertEqual(set(self.proto["test_cases"]["t2_SPACE"]), val)
        self.assertEqual(len(val), 11)


if __name__ == "__main__":
    unittest.main()
