"""CLI smoke tests for the scripts/run_holdout.py multi-window holdout runner."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "run_holdout.py"


def test_run_holdout_rejects_missing_snapshot(tmp_path: Path) -> None:
    result = subprocess.run(
        [sys.executable, str(SCRIPT), str(tmp_path / "nope.csv"), "--family", "ridge"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 1
    assert "does not exist" in result.stderr


def test_run_holdout_rejects_unknown_family(tmp_path: Path) -> None:
    snapshot = tmp_path / "snapshot.csv"
    snapshot.write_text("date,Close\n2022-01-03,100.0\n", encoding="utf-8")
    result = subprocess.run(
        [sys.executable, str(SCRIPT), str(snapshot), "--family", "not_a_family", "--horizon", "5"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 1
    assert "Unknown candidate family" in result.stderr


def test_run_holdout_requires_family(tmp_path: Path) -> None:
    snapshot = tmp_path / "snapshot.csv"
    snapshot.write_text("date,Close\n2022-01-03,100.0\n", encoding="utf-8")
    result = subprocess.run(
        [sys.executable, str(SCRIPT), str(snapshot), "--horizon", "5"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert "--family" in result.stderr
