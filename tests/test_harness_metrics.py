from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from lumbarseg_cli import config as cfgmod  # noqa: E402
from lumbarseg_cli.metrics import aggregate, slice_metrics  # noqa: E402
from spine_baseline import metrics as legacy  # noqa: E402

EVAL = {"dice_average_unit": "slice", "include_background": False, "smooth": 1e-7, "surface_inf_policy": "propagate", "nsd_tolerance_mm": 1.0}


def _slices(seed=0, n=6):
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        t = np.zeros((32, 40), int)
        t[4:12, 5:30] = 1; t[14:18, 5:30] = 2; t[20:26, 8:20] = 3
        p = t.copy()
        p[rng.integers(0, 32, 40), rng.integers(0, 40, 40)] = rng.integers(0, 4, 40)
        if i == 0:
            p[p == 2] = 0  # canal missed -> inf surface distance
        out.append((f"s{i % 2}_t2_SPACE_s{i:03d}.npz", p, t))
    return out


def _per(slices, ev=EVAL):
    sp = np.array([0.5, 0.6])
    return [{"series": n.rsplit("_s", 1)[0], "spacing": tuple(sp), "shape": t.shape,
             "classes": slice_metrics(p, t, sp, 4, ev["smooth"], ev["nsd_tolerance_mm"])} for n, p, t in slices]


class HarnessMetricTests(unittest.TestCase):
    def test_slice_rules_match_legacy_evaluator(self):
        sl = _slices()
        agg = aggregate(_per(sl), EVAL, 4)
        sp = np.array([0.5, 0.6])
        for c, name in enumerate(["Background", "Vertebrae", "Spinal Canal", "IVDs"]):
            dice = np.mean([(2 * ((p == c) & (t == c)).sum() + 1e-7) / ((p == c).sum() + (t == c).sum() + 1e-7) for _, p, t in sl])
            asd = np.mean([legacy.average_surface_distance(p == c, t == c, sp) for _, p, t in sl])
            nsd = np.mean([legacy.normalized_surface_dice(p == c, t == c, sp, 1.0) for _, p, t in sl])
            self.assertAlmostEqual(agg["classes"][name]["dice"], dice, places=12)
            self.assertAlmostEqual(agg["classes"][name]["nsd"], nsd, places=12)
            if math.isinf(asd):
                self.assertTrue(math.isinf(agg["classes"][name]["asd"]))
            else:
                self.assertAlmostEqual(agg["classes"][name]["asd"], asd, places=9)
        self.assertTrue(math.isinf(agg["classes"]["Spinal Canal"]["asd"]))
        self.assertEqual(agg["classes"]["Spinal Canal"]["asd_n_inf"], 1)

    def test_units_and_policies_differ_as_documented(self):
        per = _per(_slices())
        s = aggregate(per, EVAL, 4)
        px = aggregate(per, {**EVAL, "dice_average_unit": "pixel"}, 4)
        cs = aggregate(per, {**EVAL, "dice_average_unit": "case"}, 4)
        self.assertEqual(cs["n_cases"], 2)
        self.assertNotAlmostEqual(s["classes"]["Spinal Canal"]["dice"], px["classes"]["Spinal Canal"]["dice"], places=6)
        ex = aggregate(per, {**EVAL, "surface_inf_policy": "exclude"}, 4)
        md = aggregate(per, {**EVAL, "surface_inf_policy": "max_distance"}, 4)
        self.assertTrue(math.isfinite(ex["classes"]["Spinal Canal"]["asd"]))
        self.assertGreater(md["classes"]["Spinal Canal"]["asd"], ex["classes"]["Spinal Canal"]["asd"])
        bg = aggregate(per, {**EVAL, "include_background": True}, 4)
        self.assertAlmostEqual(bg["mean_dice"], bg["all_class_mean_dice"])
        self.assertAlmostEqual(s["mean_dice"], s["foreground_macro_dice"])

    def test_configs(self):
        d = cfgmod.load("default", "box")
        p = cfgmod.load("paper", "box")
        self.assertEqual((d["train"]["focal_gamma"], d["train"]["batch_size"], d["data"]["imbalance_threshold"]), (4.0, 2, 0.90))
        self.assertEqual((p["train"]["focal_gamma"], p["train"]["batch_size"], p["data"]["imbalance_threshold"]), (0.4, 8, 0.55))
        self.assertEqual(cfgmod.load("default", "lab")["profile"]["data_root"], "/mnt/c/Users/ctlab/somomma/DataSet")
        with self.assertRaises(cfgmod.ConfigError):
            cfgmod.load("default", "box", ["eval.dice_average_unit=volume"])


class HarnessCliTests(unittest.TestCase):
    def _cli(self, *args, env=None):
        e = {**os.environ, **(env or {})}
        return subprocess.run([sys.executable, "-m", "lumbarseg_cli", *args], cwd=REPO, capture_output=True, text=True, env=e)

    def test_missing_data_gives_next_step(self):
        with tempfile.TemporaryDirectory() as tmp:
            r = self._cli("prepare-data", "--json", env={"LUMBARSEG_DATA_ROOT": tmp, "LUMBARSEG_OUTPUT_ROOT": tmp})
            self.assertEqual(r.returncode, 3)
            out = json.loads(r.stdout)
            self.assertFalse(out["ok"]); self.assertIn("--download", out["next"])

    def test_cleanup_dry_run_keeps_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp) / "default" / "run"; run.mkdir(parents=True); (run / "x").write_text("1")
            r = self._cli("cleanup", "run", "--dry-run", "--json", env={"LUMBARSEG_OUTPUT_ROOT": tmp})
            self.assertEqual(r.returncode, 0, r.stderr)
            self.assertTrue((run / "x").exists())
            r = self._cli("cleanup", "run", "--json", env={"LUMBARSEG_OUTPUT_ROOT": tmp})
            self.assertFalse(run.exists())

    def test_score_csv_reproduces_recorded_headline(self):
        r = self._cli("score-csv", "docs/experiments/t2_space_reproduction_4cls090_cap1000_20260711/validation_metrics.csv", "--json")
        self.assertEqual(round(json.loads(r.stdout)["foreground_macro_dice"], 4), 0.8497)


if __name__ == "__main__":
    unittest.main()
