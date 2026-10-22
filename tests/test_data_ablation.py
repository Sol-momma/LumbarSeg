import unittest
from unittest import mock

from lumbarseg_cli import cli, config


def fake_files():
    tr = [f"{c}_t2_SPACE_s{i:03d}.npz" for c in range(1, 31) for i in range(25)]
    te = [f"{c}_t2_SPACE_s{i:03d}.npz" for c in range(100, 111) for i in range(25)]
    return tr, te, []


class DataAblationTest(unittest.TestCase):
    def run_split(self, name, **sets):
        cfg = config.load(name, sets=[f"{k}={v}" for k, v in sets.items()])
        with mock.patch.object(cli, "_split", lambda *a, **k: fake_files()), \
             mock.patch("spine_baseline.preprocessing.get_series_id", lambda f: f.split("_s")[0]), \
             mock.patch("spine_baseline.preprocessing.classify_sequence", lambda s: "T2_SPACE"):
            return cli.split_with_test(cfg, None, None)

    def test_configs_load(self):
        for n in ("ablation_image_split", "ablation_allseries_627", "ablation_space_t2", "ablation_space_t1"):
            config.load(n)
        self.assertEqual(config.load("ablation_space_t2")["data"]["sequences"], "T2,T2_SPACE")
        self.assertEqual(config.load("ablation_allseries_627")["data"]["max_train_slices"], 627)

    def test_image_split_keeps_sizes_and_mixes_cases(self):
        p_tr, p_va, p_te, _ = self.run_split("baseline_protocol")
        tr, va, te, _ = self.run_split("ablation_image_split")
        self.assertEqual((len(tr), len(va), len(te)), (len(p_tr), len(p_va), len(p_te)))
        self.assertEqual(sorted(tr + va + te), sorted(p_tr + p_va + p_te))
        self.assertFalse(set(tr) & set(te))
        cases = lambda fs: {f.split("_s")[0] for f in fs}
        self.assertTrue(cases(tr) & cases(te))  # same patient on both sides
        self.assertEqual(te, self.run_split("ablation_image_split")[2])  # fixed

    def test_max_train_slices(self):
        tr, va, te, _ = self.run_split("baseline_protocol", **{"data.max_train_slices": 100})
        p_tr, p_va, p_te, _ = self.run_split("baseline_protocol")
        self.assertEqual(len(tr), 100)
        self.assertTrue(set(tr) <= set(p_tr))
        self.assertEqual((va, te), (p_va, p_te))

    def test_bad_values(self):
        with self.assertRaises(config.ConfigError):
            config.load("default", sets=["data.split=foo"])
        with self.assertRaises(config.ConfigError):
            config.load("default", sets=["data.max_train_slices=-1"])


if __name__ == "__main__":
    unittest.main()
