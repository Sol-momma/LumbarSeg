from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from spine_baseline.runtime_metrics import (
    PROCESSED_REVIEW_SCOPE,
    build_runtime_report,
    write_runtime_report,
)


class RuntimeMetricsTests(unittest.TestCase):
    def test_report_states_processed_scope_and_keeps_stage_breakdown(self) -> None:
        report = build_runtime_report(
            stage_seconds={"model_load": 1.25, "score_inference": 2.5},
            total_seconds=4.0,
            input_files=["case_a_s000.npz", "case_a_s001.npz", "case_b_s000.npz"],
            scored_slice_count=3,
            rendered_panel_count=2,
            batch_size=2,
            target_height=512,
            target_width=640,
            split="validation",
            model_path=Path("model.keras"),
            processed_root=Path("processed"),
            gpu_devices=["/physical_device:GPU:0"],
        )

        self.assertEqual(report["measurement_scope"], PROCESSED_REVIEW_SCOPE)
        self.assertIn("MHA-to-NPZ preprocessing", report["scope_excludes"])
        self.assertEqual(report["counts"]["input_slices"], 3)
        self.assertEqual(report["counts"]["input_series"], 2)
        self.assertEqual(report["counts"]["rendered_panels"], 2)
        self.assertEqual(report["stage_seconds"]["score_inference"], 2.5)
        self.assertAlmostEqual(report["unattributed_seconds"], 0.25)

    def test_write_report_replaces_json_atomically(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "nested" / "runtime_metrics.json"
            write_runtime_report(path, {"schema_version": 1, "total_seconds": 3.0})

            self.assertEqual(json.loads(path.read_text(encoding="utf-8"))["total_seconds"], 3.0)
            self.assertFalse(path.with_name(f".{path.name}.tmp").exists())


if __name__ == "__main__":
    unittest.main()
