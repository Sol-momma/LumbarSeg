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
