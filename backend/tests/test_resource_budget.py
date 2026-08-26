import json
import subprocess
import sys
from pathlib import Path

import scripts.check_resource_budget as resource_budget

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


def test_rss_bytes_uses_psutil_when_proc_is_unavailable(monkeypatch):
    class _Memory:
        rss = 123456

    class _Process:
        def __init__(self, pid):
            assert pid == 42

        @staticmethod
        def memory_info():
            return _Memory()

        @staticmethod
        def children(*, recursive):
            assert recursive is True
            return []

        @staticmethod
        def is_running():
            return True

    monkeypatch.setattr(resource_budget.Path, "exists", lambda _self: False)
    monkeypatch.setattr("psutil.Process", _Process)
    assert resource_budget.rss_bytes(42) == 123456
