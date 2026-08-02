import json
import subprocess
import sys


def test_resource_budget_catches_exit_137():
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/check_resource_budget.py",
            "--json",
            sys.executable,
            "-c",
            "import sys; sys.exit(137)",
        ],
        cwd=".",
        text=True,
        capture_output=True,
        timeout=20,
    )
    assert completed.returncode == 1
    assert "oom_killed" in json.loads(completed.stdout)["failures"]
