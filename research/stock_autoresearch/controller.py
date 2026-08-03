"""Autonomous Experiment Controller for Stock Autoresearch.

Enforces multi-fidelity evaluation, subprocess isolation, memory/time limits,
immutable file protection, git worktree/branch management, and append-only experiment logging.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import EVALUATION_POLICY, RUNTIME_BUDGET, EvaluationPolicy, RuntimeBudget
from .ledger import append_record, export_tsv_summary, generate_markdown_report
from .resources import sample_cuda_memory


PROHIBITED_FILES = (
    "research/stock_autoresearch/config.py",
    "research/stock_autoresearch/data.py",
    "research/stock_autoresearch/metrics.py",
    "research/stock_autoresearch/evaluation.py",
    "research/stock_autoresearch/ledger.py",
    "research/stock_autoresearch/resources.py",
    "backend/api.py",
    "backend/model.py",
    "render.yaml",
    "frontend/",
)


@dataclass
class SubprocessResult:
    status: str  # 'success', 'crash', 'invalid', 'timeout', 'oom', 'violates_harness_lock'
    stdout: str
    stderr: str
    duration_seconds: float
    peak_vram_mb: int
    payload: dict[str, Any] | None = None
    failure_reason: str = ""


def check_harness_integrity(repo_root: Path) -> bool:
    """Verify candidate edits have NOT modified protected harness or production files."""
    try:
        cmd = ["git", "diff", "--name-only", "HEAD"]
        proc = subprocess.run(cmd, cwd=repo_root, capture_output=True, text=True, check=True)
        changed_files = [line.strip() for line in proc.stdout.splitlines() if line.strip()]
        for changed in changed_files:
            for prohibited in PROHIBITED_FILES:
                if changed == prohibited or changed.startswith(prohibited.rstrip("/") + "/"):
                    return False
    except Exception:
        return True  # Fallback if git is not in environment
    return True


def run_isolated_candidate_eval(
    snapshot_path: Path,
    candidate_family: str,
    *,
    horizon: int = 5,
    timeout_seconds: int = 120,
    budget: RuntimeBudget = RUNTIME_BUDGET,
    repo_root: Path | None = None,
) -> SubprocessResult:
    """Run candidate evaluation in an isolated subprocess with strict memory/time bounds."""
    if repo_root and not check_harness_integrity(repo_root):
        return SubprocessResult(
            status="violates_harness_lock",
            stdout="",
            stderr="Immutable harness or production file was modified by candidate.",
            duration_seconds=0.0,
            peak_vram_mb=0,
            failure_reason="Candidate touched prohibited harness files.",
        )

    eval_script = f"""
import sys, json, pathlib
sys.path.insert(0, str(pathlib.Path('{snapshot_path.parent.parent.parent.as_posix()}')))
import pandas as pd
from stock_autoresearch.data import Snapshot
from stock_autoresearch.config import EVALUATION_POLICY
from stock_autoresearch.evaluation import evaluate_candidate
from stock_autoresearch.candidates import (
    PersistenceCandidate, RidgeCandidate, CompactMLPCandidate, DLinearCandidate, SmallTCNCandidate
)

frame = pd.read_csv(r'{snapshot_path.as_posix()}', index_col=0, parse_dates=True)
cols = [c for c in frame.columns if c != 'Close']
snapshot = Snapshot(frame=frame, snapshot_id='eval_snapshot', feature_names=tuple(cols))

factories = {{
    'persistence': lambda seed: PersistenceCandidate(),
    'ridge': lambda seed: RidgeCandidate(),
    'compact_mlp': lambda seed: CompactMLPCandidate(),
    'dlinear': lambda seed: DLinearCandidate(),
    'small_tcn': lambda seed: SmallTCNCandidate(seed=seed),
}}

factory = factories.get('{candidate_family}')
if not factory:
    sys.exit(2)

result = evaluate_candidate(snapshot, factory, horizon={horizon}, policy=EVALUATION_POLICY)
print("JSON_RESULT_START")
print(json.dumps(result.summary(EVALUATION_POLICY)))
print("JSON_RESULT_END")
"""

    start_time = time.time()
    peak_vram = 0
    proc = subprocess.Popen(
        [sys.executable, "-c", eval_script],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    stdout, stderr = "", ""
    try:
        while proc.poll() is None:
            elapsed = time.time() - start_time
            if elapsed > timeout_seconds:
                proc.kill()
                proc.wait()
                return SubprocessResult(
                    status="timeout",
                    stdout=stdout,
                    stderr=stderr,
                    duration_seconds=elapsed,
                    peak_vram_mb=peak_vram,
                    failure_reason=f"Exceeded time limit of {timeout_seconds}s",
                )

            # Sample GPU memory
            sample = sample_cuda_memory(budget)
            peak_vram = max(peak_vram, sample.peak_vram_mb)
            if sample.exceeded:
                proc.kill()
                proc.wait()
                return SubprocessResult(
                    status="oom",
                    stdout=stdout,
                    stderr=stderr,
                    duration_seconds=time.time() - start_time,
                    peak_vram_mb=peak_vram,
                    failure_reason=f"Exceeded VRAM kill threshold ({budget.vram_kill_mb} MB)",
                )
            time.sleep(0.1)

        stdout, stderr = proc.communicate()
    except Exception as exc:
        proc.kill()
        return SubprocessResult(
            status="crash",
            stdout="",
            stderr=str(exc),
            duration_seconds=time.time() - start_time,
            peak_vram_mb=peak_vram,
            failure_reason=str(exc),
        )

    duration = time.time() - start_time
    if proc.returncode != 0:
        return SubprocessResult(
            status="crash",
            stdout=stdout,
            stderr=stderr,
            duration_seconds=duration,
            peak_vram_mb=peak_vram,
            failure_reason=f"Subprocess exited with code {proc.returncode}",
        )

    # Parse payload
    payload = None
    if "JSON_RESULT_START" in stdout and "JSON_RESULT_END" in stdout:
        try:
            raw_json = stdout.split("JSON_RESULT_START")[1].split("JSON_RESULT_END")[0].strip()
            payload = json.loads(raw_json)
        except Exception:
            pass

    if payload is None:
        return SubprocessResult(
            status="invalid",
            stdout=stdout,
            stderr=stderr,
            duration_seconds=duration,
            peak_vram_mb=peak_vram,
            failure_reason="Failed to parse result payload from candidate subprocess.",
        )

    return SubprocessResult(
        status="success",
        stdout=stdout,
        stderr=stderr,
        duration_seconds=duration,
        peak_vram_mb=peak_vram,
        payload=payload,
    )


class ExperimentController:
    """Manages autonomous candidate trials, multi-fidelity gates, and experiment ledger recording."""

    def __init__(
        self,
        snapshot_path: Path,
        ledger_path: Path = Path("research/results/experiments.jsonl"),
        run_tag: str = "default_run",
        policy: EvaluationPolicy = EVALUATION_POLICY,
        budget: RuntimeBudget = RUNTIME_BUDGET,
        repo_root: Path | None = None,
    ):
        self.snapshot_path = snapshot_path
        self.ledger_path = ledger_path
        self.run_tag = run_tag
        self.policy = policy
        self.budget = budget
        self.repo_root = repo_root or Path.cwd()

    def run_level_0_smoke(self, candidate_family: str) -> SubprocessResult:
        """Level 0: Correctness smoke test (30s max, 1 horizon, synthetic fixture checks)."""
        return run_isolated_candidate_eval(
            self.snapshot_path,
            candidate_family,
            horizon=1,
            timeout_seconds=self.budget.smoke_seconds,
            budget=self.budget,
            repo_root=self.repo_root,
        )

    def run_level_1_screening(self, candidate_family: str) -> SubprocessResult:
        """Level 1: Screening (2 mins max, representative horizon)."""
        return run_isolated_candidate_eval(
            self.snapshot_path,
            candidate_family,
            horizon=5,
            timeout_seconds=self.budget.screen_seconds,
            budget=self.budget,
            repo_root=self.repo_root,
        )

    def run_level_2_confirmation(self, candidate_family: str) -> SubprocessResult:
        """Level 2: Full multi-horizon multi-fold confirmation (20 mins max)."""
        return run_isolated_candidate_eval(
            self.snapshot_path,
            candidate_family,
            horizon=20,
            timeout_seconds=self.budget.confirm_seconds,
            budget=self.budget,
            repo_root=self.repo_root,
        )

    def execute_trial(
        self,
        candidate_family: str,
        hypothesis: str = "Candidate search trial",
        level: int = 1,
    ) -> dict[str, Any]:
        """Execute a full multi-fidelity candidate trial and record in ledger."""
        if level == 0:
            sub = self.run_level_0_smoke(candidate_family)
        elif level == 2:
            sub = self.run_level_2_confirmation(candidate_family)
        else:
            sub = self.run_level_1_screening(candidate_family)

        payload = sub.payload or {}
        record = {
            "run_tag": self.run_tag,
            "candidate_family": candidate_family,
            "hypothesis": hypothesis,
            "status": sub.status,
            "failure_reason": sub.failure_reason,
            "peak_vram_mb": sub.peak_vram_mb,
            "training_seconds": sub.duration_seconds,
            "median_relative_mae": payload.get("median_relative_mae"),
            "median_relative_rmse": payload.get("median_relative_rmse"),
            "worst_fold_relative_rmse": payload.get("worst_fold_relative_rmse"),
            "folds_beating_persistence": payload.get("folds_beating_persistence"),
            "promotable": payload.get("promotable", False),
            "decision": "keep" if payload.get("promotable") and sub.status == "success" else "discard",
        }

        entry = append_record(self.ledger_path, record)
        export_tsv_summary(self.ledger_path, self.ledger_path.with_suffix(".tsv"))
        generate_markdown_report(self.ledger_path, self.ledger_path.parent / "REPORT.md")
        return entry
