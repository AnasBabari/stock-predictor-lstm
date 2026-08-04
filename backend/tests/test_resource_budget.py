import json
import subprocess
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "check_resource_budget.py"


def test_resource_budget_catches_exit_137():
    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--json",
            sys.executable,
            "-c",
            "import sys; sys.exit(137)",
        ],
        text=True,
        capture_output=True,
        timeout=20,
    )
    assert completed.returncode == 1
    assert "oom_killed" in json.loads(completed.stdout)["failures"]
