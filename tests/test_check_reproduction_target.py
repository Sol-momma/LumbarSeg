from __future__ import annotations

import csv
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from scripts.check_reproduction_target import (
    FOREGROUND_MACRO_TARGET,
    PAPER_DICE_TARGETS,
    TargetEvidenceError,
    check_metrics,
)


class CheckReproductionTargetTests(unittest.TestCase):
    def write_metrics(self, root: Path, rows: list[tuple[str, object]]) -> Path:
        path = root / "validation_metrics.csv"
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["class", "dice", "iou"])
            writer.writeheader()
            for class_name, dice in rows:
                writer.writerow({"class": class_name, "dice": dice, "iou": 0.0})
        return path

    def passing_rows(self, background: float = 0.1) -> list[tuple[str, object]]:
        return [
            ("Background", background),
            ("IVDs", PAPER_DICE_TARGETS["IVDs"] + 0.0020),
            ("Vertebrae", PAPER_DICE_TARGETS["Vertebrae"]),
            ("Spinal Canal", PAPER_DICE_TARGETS["Spinal Canal"] + 0.0010),
            ("Mean", 0.0),
        ]

    def test_exact_class_floors_and_macro_boundary_pass(self) -> None:
        scores = dict(PAPER_DICE_TARGETS)
        required_macro = FOREGROUND_MACRO_TARGET * 3
        scores["IVDs"] += required_macro - sum(scores.values())
        with tempfile.TemporaryDirectory() as temp_dir:
            path = self.write_metrics(Path(temp_dir), list(scores.items()))
            result = check_metrics(path)
        self.assertAlmostEqual(result.foreground_macro_dice, FOREGROUND_MACRO_TARGET)
        self.assertTrue(result.score_target_met)

    def test_background_and_mean_do_not_change_decision(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            low_background = check_metrics(self.write_metrics(root, self.passing_rows(0.0)))
            high_background = check_metrics(self.write_metrics(root, self.passing_rows(1.0)))
        self.assertEqual(low_background.score_target_met, high_background.score_target_met)
        self.assertEqual(low_background.foreground_macro_dice, high_background.foreground_macro_dice)

    def test_macro_pass_but_one_class_miss_fails(self) -> None:
        rows = [
            ("IVDs", 0.99),
            ("Vertebrae", 0.99),
            ("Spinal Canal", PAPER_DICE_TARGETS["Spinal Canal"] - 0.0001),
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            result = check_metrics(self.write_metrics(Path(temp_dir), rows))
        self.assertGreater(result.foreground_macro_dice, FOREGROUND_MACRO_TARGET)
        self.assertFalse(result.score_target_met)

    def test_missing_duplicate_and_invalid_values_are_invalid_evidence(self) -> None:
        invalid_rows = [
            [("IVDs", 0.98), ("Vertebrae", 0.98)],
            [("IVDs", 0.98), ("IVDs", 0.99), ("Vertebrae", 0.98), ("Spinal Canal", 0.98)],
            [("IVDs", "NaN"), ("Vertebrae", 0.98), ("Spinal Canal", 0.98)],
            [("IVDs", 1.01), ("Vertebrae", 0.98), ("Spinal Canal", 0.98)],
        ]
        for rows in invalid_rows:
            with self.subTest(rows=rows), tempfile.TemporaryDirectory() as temp_dir:
                path = self.write_metrics(Path(temp_dir), rows)
                with self.assertRaises(TargetEvidenceError):
                    check_metrics(path)

    def test_cli_exit_codes_and_json(self) -> None:
        script = Path(__file__).parents[1] / "scripts" / "check_reproduction_target.py"
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            passing = self.write_metrics(root, self.passing_rows())
            output = root / "target_check.json"
            completed = subprocess.run(
                [sys.executable, str(script), str(passing), "--output", str(output)],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertTrue(json.loads(output.read_text(encoding="utf-8"))["score_target_met"])

            missed = self.write_metrics(
                root,
                [("IVDs", 0.8), ("Vertebrae", 0.8), ("Spinal Canal", 0.8)],
            )
            completed = subprocess.run(
                [sys.executable, str(script), str(missed)],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(completed.returncode, 1, completed.stderr)

            invalid = self.write_metrics(root, [("IVDs", 0.98)])
            completed = subprocess.run(
                [sys.executable, str(script), str(invalid)],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(completed.returncode, 2)

    def test_required_sequence_uses_run_config(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            metrics = self.write_metrics(root, self.passing_rows())
            config = root / "run_config.tsv"
            config.write_text("key\tvalue\nsequences\tALL\npreset\tall_4cls090_cap1000\n", encoding="utf-8")
            with self.assertRaises(TargetEvidenceError):
                check_metrics(metrics, run_config_path=config, required_sequence="T2_SPACE")
            config.write_text(
                "key\tvalue\nsequences\tT2_SPACE\npreset\tt2_space_candidate\n",
                encoding="utf-8",
            )
            result = check_metrics(metrics, run_config_path=config, required_sequence="T2_SPACE")
            self.assertEqual(result.run_config["sequences"], "T2_SPACE")

    def test_invalid_macro_target_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            metrics = self.write_metrics(Path(temp_dir), self.passing_rows())
            for target in (float("nan"), -0.1, 1.1):
                with self.subTest(target=target), self.assertRaises(TargetEvidenceError):
                    check_metrics(metrics, foreground_macro_target=target)


if __name__ == "__main__":
    unittest.main()
