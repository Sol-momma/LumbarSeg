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
        self.assertIn('[int]$Seed = 42', script)
        self.assertIn('[Int64]$MinimumFreeBytes = 2GB', script)
        self.assertIn('Assert-FreeDiskSpace', script)
        self.assertIn('Assert-BatchHardware', script)
        self.assertIn('blocked_hardware', script)
        self.assertIn('12288', script)
        self.assertIn('"--reuse_processed_only"', script)
        self.assertIn('"--seed", $Seed', script)
        self.assertIn('$ProcessedPath -eq $ProbePath', script)
        self.assertIn('RunOutputRoot must not already exist', script)
        self.assertIn('Assert-NativeSuccess "TensorFlow GPU check"', script)
        self.assertIn('Assert-NativeSuccess "Batch-size smoke probe"', script)
        self.assertIn('environment_provenance.tsv', script)
        self.assertIn('installed-packages.txt', script)
        self.assertIn('windows_smoke.log', script)
        self.assertIn('Smoke probe did not produce a non-empty best model', script)


if __name__ == "__main__":
    unittest.main()
