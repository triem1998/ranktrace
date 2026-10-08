"""Helpers shared by the profilers and the analysis tools."""
import subprocess
from pathlib import Path


def run_logged(cmd: list[str], log: Path) -> None:
    """Run a command with its output in `log`. A failure is reported, not raised."""
    with open(log, "w") as f:
        if subprocess.run(cmd, stdout=f, stderr=f).returncode:
            print(f"failed, see {log}")
