from __future__ import annotations

import os
from pathlib import Path
import subprocess
import tempfile


REPO_ROOT = Path(__file__).parents[1]
HARNESS = REPO_ROOT / "scripts" / "long_run_harness.sh"
HEARTBEAT = REPO_ROOT / "scripts" / "run_long_run_heartbeat.sh"


def _fake_nvidia_smi(directory: Path, memory_mib: int = 8192) -> None:
    executable = directory / "nvidia-smi"
    executable.write_text(
        "#!/usr/bin/env bash\n"
        "case \"$*\" in\n"
        f"  *memory.total*) echo '{memory_mib}' ;;\n"
        "  *temperature.gpu*) echo '35, 100, 8092, 0' ;;\n"
        "  *) echo 'Fake GPU, GPU-fake, 1.0, 8192 MiB' ;;\n"
        "esac\n",
        encoding="utf-8",
    )
    executable.chmod(0o755)


def test_batch8_is_blocked_on_an_8gb_gpu_without_fallback() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        binary_dir = Path(temp_dir)
        _fake_nvidia_smi(binary_dir)
        completed = subprocess.run(
            [
                "bash",
                "-c",
                'source "$1"; harness_require_batch_hardware 8 12288',
                "bash",
                str(HARNESS),
            ],
            env={**os.environ, "PATH": f"{binary_dir}:{os.environ['PATH']}"},
            check=False,
            capture_output=True,
            text=True,
        )

        assert completed.returncode == 3
        assert "blocked_hardware" in completed.stderr
        assert "No fallback batch size was selected" in completed.stderr


def test_heartbeat_writes_atomic_health_and_hardware_snapshot() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        binary_dir = root / "bin"
        binary_dir.mkdir()
        _fake_nvidia_smi(binary_dir)
        heartbeat_path = root / "heartbeat.tsv"
        completed = subprocess.run(
            ["bash", str(HEARTBEAT), "--once"],
            env={
                **os.environ,
                "PATH": f"{binary_dir}:{os.environ['PATH']}",
                "HEARTBEAT_PATH": str(heartbeat_path),
                "WATCH_PID": str(os.getpid()),
                "DISK_PATH": str(root),
                "MIN_FREE_GIB": "0",
            },
            check=False,
            capture_output=True,
            text=True,
        )

        assert completed.returncode == 0, completed.stderr
        snapshot = heartbeat_path.read_text(encoding="utf-8")
        assert "health\thealthy" in snapshot
        assert "watch_process_alive\ttrue" in snapshot
        assert "gpu_temperature_memory_used_free_utilization\t35, 100, 8092, 0" in snapshot
        assert not heartbeat_path.with_name(f"{heartbeat_path.name}.tmp").exists()


def test_running_status_from_an_old_boot_is_stale() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        status_path = Path(temp_dir) / "status.tsv"
        status_path.write_text(
            "key\tvalue\nstatus\trunning\npid\t1\nboot_id\tdefinitely-an-old-boot\n",
            encoding="utf-8",
        )
        completed = subprocess.run(
            [
                "bash",
                "-c",
                'source "$1"; harness_running_state "$2"',
                "bash",
                str(HARNESS),
                str(status_path),
            ],
            check=False,
            capture_output=True,
            text=True,
        )

        assert completed.returncode == 0
        assert completed.stdout == "stale"
