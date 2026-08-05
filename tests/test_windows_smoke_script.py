import unittest
from pathlib import Path


class WindowsSmokeScriptTests(unittest.TestCase):
    def test_batch8_probe_uses_fixed_cohorts_and_separate_run_output(self) -> None:
        script = (
            Path(__file__).resolve().parents[1] / "scripts" / "run_windows_smoke.ps1"
        ).read_text(encoding="utf-8")

        # These assertions protect the operational contract of the Windows
        # entrypoint without requiring PowerShell on the macOS development host.
        self.assertIn('[int]$BatchSize = 8', script)
        self.assertIn('[string]$TrainFileList', script)
        self.assertIn('[string]$ValidationFileList', script)
        self.assertIn('[string]$RunOutputRoot = "outputs\\batch8_smoke"', script)
        self.assertIn('"--run_output_root", $RunOutputRoot', script)
        self.assertIn('"--train_file_list", $TrainFileList', script)
        self.assertIn('"--validation_file_list", $ValidationFileList', script)
        self.assertIn('$TrainCount -ne $BatchSize', script)
        self.assertIn('$ValidationCount -ne $BatchSize', script)
        self.assertIn('$BatchSize -ne 8', script)
        self.assertIn('"--reuse_processed_only"', script)
        self.assertIn('$ProcessedPath -eq $ProbePath', script)
        self.assertIn('RunOutputRoot must not already exist', script)


if __name__ == "__main__":
    unittest.main()
