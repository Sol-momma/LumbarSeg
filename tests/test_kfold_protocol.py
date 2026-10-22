import unittest
from unittest import mock

from lumbarseg_cli import cli, config


def fake_files():
    tr = [f"{c}_t2_SPACE_s{i:03d}.npz" for c in range(1, 31) for i in range(5)]
    te = [f"{c}_t2_SPACE_s{i:03d}.npz" for c in range(100, 111) for i in range(5)]
    return tr, te, []


class KFoldTest(unittest.TestCase):
    def split(self, fold, seed=0, name="paper_effective"):
        cfg = config.load(name, sets=["data.split=patient_kfold", f"data.fold={fold}", f"data.kfold_seed={seed}", "data.inner_val_cases=4"])
        with mock.patch.object(cli, "_split", lambda *a, **k: fake_files()), \
             mock.patch("spine_baseline.preprocessing.get_series_id", lambda f: f.rsplit("_s", 1)[0]), \
             mock.patch("spine_baseline.preprocessing.classify_sequence", lambda s: "T2_SPACE"):
            return cli.split_with_test(cfg, None, None)

    def pats(self, files):
        return {f.split("_", 1)[0] for f in files}

    def test_folds_partition_patients(self):
        tests = [self.pats(self.split(k)[2]) for k in range(5)]
        allp = set().union(*tests)
        self.assertEqual(len(allp), 41)
        self.assertEqual(sum(len(t) for t in tests), 41)  # disjoint
        self.assertLessEqual(max(map(len, tests)) - min(map(len, tests)), 1)

    def test_no_patient_leak_and_deterministic(self):
        tr, va, te, _ = self.split(2)
        self.assertFalse(self.pats(tr) & self.pats(te))
        self.assertFalse(self.pats(va) & self.pats(te))
        self.assertEqual(len(self.pats(va)), 4)
        self.assertEqual(te, self.split(2)[2])
        self.assertNotEqual(te, self.split(2, seed=1)[2])

    def test_bad_fold(self):
        with self.assertRaises(cli.CliError):
            self.split(5)

    def test_ivd_boundary_config(self):
        t = config.load("ivd_boundary")["train"]
        self.assertEqual((t["boundary_weight"], t["boundary_class"], t["batch_size"]), (4.0, 3, 8))
        self.assertEqual(config.load("paper_effective")["train"]["boundary_weight"], 0)


class BoundaryLossTest(unittest.TestCase):
    def test_boundary_band_and_default_unchanged(self):
        try:
            import numpy as np
            import tensorflow as tf
        except ImportError:
            self.skipTest("tensorflow not installed")
        from spine_baseline.losses import boundary_map, combined_loss
        y = np.zeros((1, 16, 16, 4), np.float32); y[..., 0] = 1
        y[0, 4:12, 4:12, 0] = 0; y[0, 4:12, 4:12, 3] = 1
        b = boundary_map(tf.constant(y), 3, 3).numpy()[0, ..., 0]
        self.assertEqual(b[8, 8], 0); self.assertEqual(b[0, 0], 0); self.assertEqual(b[4, 8], 1); self.assertEqual(b[3, 8], 1)
        p = tf.constant(np.full_like(y, 0.25))
        base = float(combined_loss(0.6, 0.4)(tf.constant(y), p))
        self.assertGreater(float(combined_loss(0.6, 0.4, boundary_weight=4.0)(tf.constant(y), p)), base)
        self.assertAlmostEqual(float(combined_loss(0.6, 0.4, boundary_weight=0.0)(tf.constant(y), p)), base, places=6)


if __name__ == "__main__":
    unittest.main()
