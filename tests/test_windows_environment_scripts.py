from pathlib import Path
import re
import unittest


REPO_ROOT = Path(__file__).parents[1]


class WindowsEnvironmentRequirementsTests(unittest.TestCase):
    def test_baseline_requirements_are_exact_and_include_pytest(self) -> None:
        requirements = (REPO_ROOT / "requirements-baseline.txt").read_text(encoding="utf-8")
        package_lines = [
            line.strip()
            for line in requirements.splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]

        self.assertTrue(any(line.startswith("tensorflow[and-cuda]==") for line in package_lines))
        self.assertTrue(any(line.startswith("pytest==") for line in package_lines))
        for line in package_lines:
            # Environment markers may follow the exact pin, but compatible-range
            # operators would let a rebuild silently change numerical behavior.
            requirement = line.split(";", 1)[0].strip()
            self.assertRegex(requirement, r"^[A-Za-z0-9_.-]+(?:\[[A-Za-z0-9_.-]+\])?==[^<>=!~]+$")

    def test_native_windows_requirements_preserve_tensorflow_210_contract(self) -> None:
        requirements = (REPO_ROOT / "requirements-windows-native-gpu.txt").read_text(encoding="utf-8")
        self.assertIn("tensorflow==2.10.1", requirements)
        self.assertIn("numpy==1.23.5", requirements)
        self.assertRegex(requirements, r"(?m)^pytest==")
        self.assertNotRegex(requirements, r"(?m)^[A-Za-z0-9_.-]+[<>~]=")


class WindowsSetupScriptTests(unittest.TestCase):
    def test_setup_rejects_reuse_and_checks_every_native_step(self) -> None:
        script = (REPO_ROOT / "scripts" / "setup_windows_native_gpu.ps1").read_text(encoding="utf-8")

        self.assertIn("VenvPath already exists", script)
        self.assertIn("Python 3.10", script)
        self.assertIn("Assert-FreeDiskSpace", script)
        self.assertIn('Assert-NativeSuccess "Virtual environment creation"', script)
        self.assertIn('Assert-NativeSuccess "Pinned requirement installation"', script)
        self.assertIn('Assert-NativeSuccess "pip check"', script)
        self.assertIn('Assert-NativeSuccess "TensorFlow GPU check"', script)
        self.assertIn("environment_provenance.tsv", script)
        self.assertIn("installed-packages.txt", script)
        self.assertIn("windows_setup.log", script)


class WindowsTrainingScriptTests(unittest.TestCase):
    def setUp(self) -> None:
        self.script = (REPO_ROOT / "scripts" / "run_windows_train.ps1").read_text(encoding="utf-8")

    def test_preflight_rejects_unsafe_or_incomplete_runs(self) -> None:
        self.assertIn("OutputRoot must not already exist", self.script)
        self.assertIn("OutputRoot must not be DataRoot", self.script)
        self.assertIn("Assert-FreeDiskSpace", self.script)
        self.assertIn('Assert-NativeSuccess "TensorFlow GPU preflight"', self.script)
        self.assertIn("SPIDER overview CSV is missing", self.script)
        self.assertIn('[int]$BatchSize = 2', self.script)
        self.assertIn("blocked_hardware", self.script)
        self.assertIn("12288", self.script)

    def test_failed_training_cannot_fall_through_to_old_model_evaluation(self) -> None:
        training_check = self.script.index('Assert-NativeSuccess "Training"')
        model_check = self.script.index("Training completed without a new best model")
        evaluation_call = self.script.index("& $PythonExe evaluate.py")
        evaluation_check = self.script.index('Assert-NativeSuccess "Evaluation"')

        self.assertLess(training_check, model_check)
        self.assertLess(model_check, evaluation_call)
        self.assertLess(evaluation_call, evaluation_check)
        self.assertIn("Best model predates this run", self.script)

    def test_run_records_log_and_environment_provenance(self) -> None:
        for evidence_name in (
            "windows_train.log",
            "environment_provenance.tsv",
            "installed-packages.txt",
            "requirements_sha256",
            "git_revision",
            "seed",
        ):
            with self.subTest(evidence_name=evidence_name):
                self.assertIn(evidence_name, self.script)

        self.assertRegex(self.script, re.compile(r'"--seed", \$Seed'))


if __name__ == "__main__":
    unittest.main()
