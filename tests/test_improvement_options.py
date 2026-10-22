"""Checks for the improvement options: augmentation, class-weighted losses, config validation, evaluate --pred_dir."""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from lumbarseg_cli import config as cfgmod  # noqa: E402


def _sample(h=64, w=80):
    img = np.linspace(0, 1, h * w, dtype=np.float32).reshape(h, w, 1)
    lab = np.zeros((h, w), int); lab[10:30, 10:60] = 1; lab[32:40, 10:60] = 2; lab[45:55, 20:50] = 3
    return img, np.eye(4, dtype=np.float32)[lab]


class AugmentTest(unittest.TestCase):
    def test_strong_keeps_shape_range_and_onehot(self):
        from spine_baseline.dataset import augment_sample
        img, oh = _sample()
        rng = np.random.default_rng(0)
        changed = False
        for _ in range(5):
            ai, ao = augment_sample(img, oh, rng, "strong")
            self.assertEqual(ai.shape, img.shape); self.assertEqual(ao.shape, oh.shape)
            self.assertTrue(0 <= ai.min() and ai.max() <= 1)
            np.testing.assert_array_equal(ao.sum(-1), 1)
            changed |= not np.array_equal(ao, oh)
        self.assertTrue(changed)

    def test_none_is_identity(self):
        from spine_baseline.dataset import augment_sample
        img, oh = _sample()
        ai, ao = augment_sample(img, oh, np.random.default_rng(0), "none")
        self.assertIs(ai, img); self.assertIs(ao, oh)


class LossTest(unittest.TestCase):
    def test_weights_none_equals_original_and_weights_change_loss(self):
        import tensorflow as tf
        from spine_baseline.losses import combined_loss
        _, oh = _sample()
        y = tf.constant(oh[None]); p = tf.nn.softmax(tf.random.stateless_normal(y.shape, (1, 2)))
        base = float(combined_loss(0.6, 4.0)(y, p))
        same = float(combined_loss(0.6, 4.0, None, None)(y, p))
        self.assertAlmostEqual(base, same, places=6)
        uniform = float(combined_loss(0.6, 4.0, [1, 1, 1, 1], [1, 1, 1, 1])(y, p))
        self.assertAlmostEqual(base, uniform, places=5)
        self.assertNotAlmostEqual(base, float(combined_loss(0.6, 4.0, [0.5, 1, 3, 1], [0, 1, 2, 1])(y, p)), places=4)


class ConfigTest(unittest.TestCase):
    def test_new_configs_load(self):
        for name in ("aug_strong", "canal_weighted", "hires_long"):
            cfg = cfgmod.load(name, "box")
            self.assertEqual(cfg["name"], name)
        self.assertEqual(cfgmod.load("default", "box")["train"]["augment"], "none")

    def test_bad_values_rejected(self):
        with self.assertRaises(cfgmod.ConfigError):
            cfgmod.load("default", "box", ["train.augment=heavy"])
        with self.assertRaises(cfgmod.ConfigError):
            cfgmod.load("default", "box", ["train.class_weights=[1,2]"])


class PredDirTest(unittest.TestCase):
    def test_evaluate_pred_dir_perfect_prediction(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d); proc = d / "processed"
            (proc / "images").mkdir(parents=True); (proc / "masks").mkdir(); (d / "pred").mkdir()
            names = []
            for i in range(3):
                img, oh = _sample(); lab = oh.argmax(-1).astype(np.uint8)
                n = f"{i + 1}_t2_SPACE_slice_{i:03d}.npz"; names.append(n)
                np.savez(proc / "images" / n, image=img[..., 0], spacing=np.array([1.0, 1.0]))
                np.savez(proc / "masks" / n, mask=lab)
                np.savez(d / "pred" / n, pred=lab)
            (d / "val.txt").write_text("\n".join(names) + "\n")
            r = subprocess.run([sys.executable, "-m", "lumbarseg_cli", "evaluate", "--profile", "box", "--pred_dir", str(d / "pred"),
                                "--file_list", str(d / "val.txt"), "--processed_root", str(proc), "--no-record", "--json"],
                               cwd=REPO, capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, r.stderr)
            m = json.loads(r.stdout)["metrics"]
            self.assertAlmostEqual(m["foreground_macro_dice"], 1.0, places=6)


if __name__ == "__main__":
    unittest.main()


class InnerValSplitTest(unittest.TestCase):
    def test_inner_val_cases_are_disjoint_and_test_is_fixed(self):
        from lumbarseg_cli import cli
        from spine_baseline.preprocessing import get_series_id
        train = [f"{c}_t2_SPACE_s{i:03d}.npz" for c in range(1, 11) for i in range(3)]
        test = [f"{c}_t2_SPACE_s{i:03d}.npz" for c in range(50, 53) for i in range(3)]
        orig = cli._split
        cli._split = lambda cfg, proc, root, smoke: (list(train), list(test), [])
        try:
            cfg = cfgmod.load("default", "box", ["data.inner_val_cases=3"])
            tr, va, te, _ = cli.split_with_test(cfg, Path("."), Path("."))
            tr2, va2, te2, _ = cli.split_with_test(cfgmod.load("default", "box", ["data.inner_val_cases=3", "train.seed=7"]), Path("."), Path("."))
            base = cli.split_with_test(cfgmod.load("default", "box"), Path("."), Path("."))
        finally:
            cli._split = orig
        self.assertEqual(te, test)
        self.assertEqual(len({get_series_id(f) for f in va}), 3)
        self.assertFalse({get_series_id(f) for f in va} & {get_series_id(f) for f in tr})
        self.assertEqual(sorted(tr + va), sorted(train))
        self.assertEqual((tr, va), (tr2, va2))  # training seed does not move the split
        self.assertEqual(base[1], base[2])  # inner_val_cases=0 keeps val == test
        with self.assertRaises(cfgmod.ConfigError):
            cfgmod.load("default", "box", ["data.inner_val_cases=-1"])


class AllSeriesTest(unittest.TestCase):
    def test_inner_val_is_per_sequence(self):
        from lumbarseg_cli import cli
        from spine_baseline.preprocessing import classify_sequence, get_series_id
        train = [f"{c}_{q}_s{i:03d}.npz" for q in ("t1", "t2", "t2_SPACE") for c in range(1, 11) for i in range(2)]
        orig = cli._split
        cli._split = lambda cfg, proc, root, smoke: (list(train), ["99_t1_s000.npz"], [])
        try:
            tr, va, te, _ = cli.split_with_test(cfgmod.load("baseline_all_series", "box"), Path("."), Path("."))
        finally:
            cli._split = orig
        per = {}
        for f in va:
            per.setdefault(classify_sequence(get_series_id(f)), set()).add(get_series_id(f))
        self.assertEqual({k: len(v) for k, v in per.items()}, {"T1": 4, "T2": 4, "T2_SPACE": 4})
        self.assertFalse(set(map(get_series_id, va)) & set(map(get_series_id, tr)))

    def test_configs(self):
        b = cfgmod.load("baseline_all_series", "box"); a = cfgmod.load("aug_all_series", "box")
        self.assertEqual(b["data"]["sequences"], "T1,T2,T2_SPACE")
        self.assertEqual((b["data"]["imbalance_threshold"], b["data"]["max_slices_per_sequence"]), (0.90, 1000))
        for sec in ("data", "model", "eval"):
            self.assertEqual(a[sec], b[sec])
        self.assertEqual(a["train"]["augment"], "strong")

    def test_parallel_augmentation_is_reproducible(self):
        from spine_baseline.dataset import create_dataset
        with tempfile.TemporaryDirectory() as d:
            d = Path(d); (d / "images").mkdir(); (d / "masks").mkdir(); names = []
            for i in range(6):
                img, oh = _sample(); n = f"{i}_t1_s000.npz"; names.append(n)
                np.savez(d / "images" / n, image=img[..., 0]); np.savez(d / "masks" / n, mask=oh.argmax(-1).astype(np.uint8))
            run = lambda: np.concatenate([x.numpy() for x, _ in create_dataset(names, d, 64, 80, 4, 2, False, "strong", 7, 4)])
            np.testing.assert_array_equal(run(), run())

    def test_evaluate_sequence_filter(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d); proc = d / "p"; (proc / "images").mkdir(parents=True); (proc / "masks").mkdir(); (d / "pred").mkdir()
            names = []
            for q in ("t1", "t2", "t2_SPACE"):
                img, oh = _sample(); lab = oh.argmax(-1).astype(np.uint8); n = f"5_{q}_s000.npz"; names.append(n)
                np.savez(proc / "images" / n, image=img[..., 0], spacing=np.array([1.0, 1.0])); np.savez(proc / "masks" / n, mask=lab)
                np.savez(d / "pred" / n, pred=lab)
            (d / "l.txt").write_text("\n".join(names) + "\n")
            r = subprocess.run([sys.executable, "-m", "lumbarseg_cli", "evaluate", "--profile", "box", "--pred_dir", str(d / "pred"), "--sequence", "T2",
                                "--file_list", str(d / "l.txt"), "--processed_root", str(proc), "--no-record", "--json"], cwd=REPO, capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, r.stderr)
            rec = json.loads(r.stdout)
            self.assertEqual((rec["split"]["n_slices"], rec["split"]["sequence"]), (1, "T2"))
