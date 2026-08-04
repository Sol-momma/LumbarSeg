from __future__ import annotations

import os
from pathlib import Path
import subprocess
import tempfile
import textwrap
import unittest


class ReproductionGoalQueueTests(unittest.TestCase):
    def test_queue_runs_sequentially_and_stops_after_target(self) -> None:
        repo_root = Path(__file__).parents[1]
        queue = repo_root / "scripts" / "run_reproduction_goal_queue.sh"
        checker = repo_root / "scripts" / "check_reproduction_target.py"
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            fake_runner = root / "fake_runner.sh"
            calls = root / "calls.txt"
            fake_runner.write_text(
                textwrap.dedent(
                    """\
                    #!/usr/bin/env bash
                    set -euo pipefail
                    preset="$1"
                    printf '%s|%s\\n' "$preset" "${EVAL_FILE_LIST:-baseline}" >> "$CALLS_FILE"
                    mkdir -p "$OUTPUT_ROOT"
                    printf 'patient_001_slice_001.npz\\n' > "$OUTPUT_ROOT/validation_files.txt"
                    printf 'file\\timage_sha256\\tmask_sha256\\npatient_001_slice_001.npz\\timage\\tmask\\n' \
                      > "$OUTPUT_ROOT/validation_cohort.tsv"
                    if [[ "$preset" == "t2_space_pass" ]]; then
                      ivd=0.9720; vertebrae=0.9730; canal=0.9710
                    else
                      ivd=0.8500; vertebrae=0.9200; canal=0.7900
                    fi
                    printf 'class,dice,iou\\nBackground,0.999,0.99\\nIVDs,%s,0.0\\nVertebrae,%s,0.0\\nSpinal Canal,%s,0.0\\nMean,0.999,0.99\\n' \
                      "$ivd" "$vertebrae" "$canal" > "$OUTPUT_ROOT/validation_metrics.csv"
                    printf 'key\\tvalue\\npreset\\t%s\\nsequences\\tT2_SPACE\\ngit_revision\\ttest\\n' \
                      "$preset" > "$OUTPUT_ROOT/run_config.tsv"
                    """
                ),
                encoding="utf-8",
            )
            env = os.environ.copy()
            env.update(
                {
                    "CAMPAIGN_ROOT": str(root / "campaign"),
                    "RUNNER": str(fake_runner),
                    "CHECKER": str(checker),
                    "CALLS_FILE": str(calls),
                    "SKIP_ACTIVE_TRAIN_CHECK": "1",
                    "ALLOW_NO_TMUX": "1",
                    "GPU_LOCK_DIR": str(root / "gpu.lock"),
                }
            )
            completed = subprocess.run(
                [
                    "bash",
                    str(queue),
                    "t2_space_4cls090_cap1000",
                    "t2_space_pass",
                    "t2_space_must_not_run",
                ],
                cwd=repo_root,
                env=env,
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertEqual(
                calls.read_text(encoding="utf-8").splitlines(),
                [
                    "t2_space_4cls090_cap1000|baseline",
                    f"t2_space_pass|{root / 'campaign' / 'fixed_validation_files.txt'}",
                ],
            )
            self.assertEqual(
                (root / "campaign" / "fixed_validation_files.txt").read_text(encoding="utf-8"),
                "patient_001_slice_001.npz\n",
            )
            status = (root / "campaign" / "campaign_status.tsv").read_text(encoding="utf-8")
            self.assertIn("t2_space_4cls090_cap1000\ttarget_missed", status)
            self.assertIn("t2_space_pass\ttarget_met", status)
            self.assertNotIn("t2_space_must_not_run", status)

    def test_existing_campaign_root_is_not_reused(self) -> None:
        repo_root = Path(__file__).parents[1]
        queue = repo_root / "scripts" / "run_reproduction_goal_queue.sh"
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            campaign = root / "campaign"
            campaign.mkdir()
            env = os.environ.copy()
            env.update(
                {
                    "CAMPAIGN_ROOT": str(campaign),
                    "RUNNER": str(repo_root / "scripts" / "run_reproduction_experiment.sh"),
                    "CHECKER": str(repo_root / "scripts" / "check_reproduction_target.py"),
                    "SKIP_ACTIVE_TRAIN_CHECK": "1",
                    "ALLOW_NO_TMUX": "1",
                    "GPU_LOCK_DIR": str(root / "gpu.lock"),
                }
            )
            completed = subprocess.run(
                ["bash", str(queue), "t2_space_4cls090_cap1000"],
                cwd=repo_root,
                env=env,
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(completed.returncode, 2)
            self.assertIn("refusing to mix or overwrite", completed.stderr)

    def test_existing_gpu_lock_blocks_campaign(self) -> None:
        repo_root = Path(__file__).parents[1]
        queue = repo_root / "scripts" / "run_reproduction_goal_queue.sh"
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            lock = root / "gpu.lock"
            lock.mkdir()
            env = os.environ.copy()
            env.update(
                {
                    "CAMPAIGN_ROOT": str(root / "campaign"),
                    "RUNNER": str(repo_root / "scripts" / "run_reproduction_experiment.sh"),
                    "CHECKER": str(repo_root / "scripts" / "check_reproduction_target.py"),
                    "ALLOW_NO_TMUX": "1",
                    "GPU_LOCK_DIR": str(lock),
                }
            )
            completed = subprocess.run(
                ["bash", str(queue), "t2_space_4cls090_cap1000"],
                cwd=repo_root,
                env=env,
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(completed.returncode, 2)
            self.assertIn("GPU campaign lock already exists", completed.stderr)

    def test_mixed_sequence_preset_is_rejected_before_runner(self) -> None:
        repo_root = Path(__file__).parents[1]
        queue = repo_root / "scripts" / "run_reproduction_goal_queue.sh"
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            env = os.environ.copy()
            env.update(
                {
                    "CAMPAIGN_ROOT": str(root / "campaign"),
                    "RUNNER": str(repo_root / "scripts" / "run_reproduction_experiment.sh"),
                    "CHECKER": str(repo_root / "scripts" / "check_reproduction_target.py"),
                    "ALLOW_NO_TMUX": "1",
                    "GPU_LOCK_DIR": str(root / "gpu.lock"),
                }
            )
            completed = subprocess.run(
                ["bash", str(queue), "all_4cls090_cap1000"],
                cwd=repo_root,
                env=env,
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(completed.returncode, 2)
            self.assertIn("accept only T2 SPACE", completed.stderr)

    def test_runner_failure_stops_queue_and_records_status(self) -> None:
        repo_root = Path(__file__).parents[1]
        queue = repo_root / "scripts" / "run_reproduction_goal_queue.sh"
        checker = repo_root / "scripts" / "check_reproduction_target.py"
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            fake_runner = root / "fail.sh"
            fake_runner.write_text("#!/usr/bin/env bash\nexit 7\n", encoding="utf-8")
            env = os.environ.copy()
            env.update(
                {
                    "CAMPAIGN_ROOT": str(root / "campaign"),
                    "RUNNER": str(fake_runner),
                    "CHECKER": str(checker),
                    "SKIP_ACTIVE_TRAIN_CHECK": "1",
                    "ALLOW_NO_TMUX": "1",
                    "GPU_LOCK_DIR": str(root / "gpu.lock"),
                }
            )
            completed = subprocess.run(
                ["bash", str(queue), "t2_space_4cls090_cap1000"],
                cwd=repo_root,
                env=env,
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(completed.returncode, 2)
            status = (root / "campaign" / "campaign_status.tsv").read_text(encoding="utf-8")
            self.assertIn("t2_space_4cls090_cap1000\texecution_failed", status)
            self.assertFalse((root / "gpu.lock").exists())


if __name__ == "__main__":
    unittest.main()
