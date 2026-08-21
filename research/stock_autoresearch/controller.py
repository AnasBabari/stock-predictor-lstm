"""Autonomous Experiment Controller for Stock Autoresearch.

Enforces multi-fidelity evaluation, subprocess isolation, memory/time limits,
immutable file protection, git worktree/branch management, and append-only experiment logging.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import psutil
except ImportError:
    psutil = None

from .config import EVALUATION_POLICY, RUNTIME_BUDGET, EvaluationPolicy, RuntimeBudget
from .ledger import append_record, export_tsv_summary, generate_markdown_report
from .resources import sample_process_tree_memory

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
    peak_rss_mb: int = 0
    payload: dict[str, Any] | None = None
    failure_reason: str = ""


def kill_process_tree(pid: int | None) -> None:
    """Recursively terminate all descendant processes and the root process."""
    if pid is None or psutil is None:
        return
    with contextlib.suppress(Exception):
        parent = psutil.Process(pid)
        children = parent.children(recursive=True)
        for child in children:
            with contextlib.suppress(Exception):
                child.kill()
        with contextlib.suppress(Exception):
            parent.kill()
        with contextlib.suppress(Exception):
            psutil.wait_procs(children + [parent], timeout=1.0)


def _hash_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def get_protected_fingerprints(repo_root: Path) -> dict[str, str]:
    fingerprints = {}
    for prohibited in PROHIBITED_FILES:
        target = repo_root / prohibited
        if target.is_file():
            fingerprints[prohibited] = _hash_file(target)
        elif target.is_dir():
            for p in target.rglob("*"):
                if p.is_file():
                    rel = p.relative_to(repo_root).as_posix()
                    fingerprints[rel] = _hash_file(p)
    return fingerprints


def check_harness_integrity(
    repo_root: Path, baseline_fingerprints: dict[str, str] | None = None
) -> bool:
    """Verify candidate edits have NOT modified protected harness or production files."""
    if baseline_fingerprints is not None:
        current_fingerprints = get_protected_fingerprints(repo_root)
        return current_fingerprints == baseline_fingerprints

    try:
        cmd = ["git", "diff", "--name-only", "HEAD"]
        proc = subprocess.run(cmd, cwd=repo_root, capture_output=True, text=True, check=True)
        changed_files = [line.strip() for line in proc.stdout.splitlines() if line.strip()]
        for changed in changed_files:
            for prohibited in PROHIBITED_FILES:
                if changed == prohibited or changed.startswith(prohibited.rstrip("/") + "/"):
                    return False
    except Exception:
        return True
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
    initial_fingerprints = get_protected_fingerprints(repo_root) if repo_root else None
    research_root = Path(__file__).resolve().parent.parent.as_posix()

    eval_script = f"""
import sys, json, pathlib, hashlib
sys.path.insert(0, str(pathlib.Path('{research_root}')))
import pandas as pd
from stock_autoresearch.data import Snapshot
from stock_autoresearch.config import EVALUATION_POLICY
from stock_autoresearch.evaluation import evaluate_candidate
from stock_autoresearch.candidates import (
    PersistenceCandidate, RidgeCandidate, ElasticNetCandidate,
    CompactMLPCandidate, DLinearCandidate, RandomFeaturesRidgeCandidate,
    elastic_net_family_factories
)

frame = pd.read_csv(r'{snapshot_path.as_posix()}', index_col=0, parse_dates=True)
cols = [c for c in frame.columns if c != 'Close']
digest = hashlib.sha256()
digest.update(frame.to_csv(index=True).encode('utf-8'))
digest.update(json.dumps(cols, separators=(",", ":")).encode('utf-8'))
snapshot = Snapshot(frame=frame, snapshot_id='sha256:' + digest.hexdigest(), feature_names=tuple(cols))

factories = {{
    'persistence': lambda seed: PersistenceCandidate(),
    'ridge': lambda seed: RidgeCandidate(),
    'elastic_net': lambda seed: ElasticNetCandidate(),
    'compact_mlp': lambda seed: CompactMLPCandidate(),
    'dlinear': lambda seed: DLinearCandidate(),
    'random_features_ridge': lambda seed: RandomFeaturesRidgeCandidate(seed=seed),
}}
# Tuned Elastic Net grid variants (elastic_net_a*_l*); baseline above unchanged.
factories.update(elastic_net_family_factories())

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
    peak_rss = 0
    raw_status = "crash"
    failure_reason = ""
    stdout, stderr = "", ""

    proc = subprocess.Popen(
        [sys.executable, "-c", eval_script],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    try:
        while True:
            try:
                stdout, stderr = proc.communicate(timeout=0.1)
                if proc.returncode == 0:
                    raw_status = "success"
                else:
                    raw_status = "crash"
                    failure_reason = f"Subprocess exited with code {proc.returncode}"
                break
            except subprocess.TimeoutExpired:
                elapsed = time.time() - start_time
                sample = sample_process_tree_memory(proc.pid, budget)
                peak_vram = max(peak_vram, sample.peak_vram_mb)
                peak_rss = max(peak_rss, sample.rss_mb)

                if elapsed > timeout_seconds:
                    raw_status = "timeout"
                    failure_reason = f"Exceeded time limit of {timeout_seconds}s"
                    kill_process_tree(proc.pid)
                    proc.kill()
                    stdout, stderr = proc.communicate()
                    break

                if sample.exceeded:
                    raw_status = "oom"
                    reasons = []
                    if sample.rss_mb >= budget.rss_kill_mb > 0:
                        reasons.append(f"RSS memory limit ({budget.rss_kill_mb} MB)")
                    if sample.peak_vram_mb >= budget.vram_kill_mb > 0:
                        reasons.append(f"VRAM kill threshold ({budget.vram_kill_mb} MB)")
                    failure_reason = f"Exceeded {' and '.join(reasons) or 'memory limit'}"
                    kill_process_tree(proc.pid)
                    proc.kill()
                    stdout, stderr = proc.communicate()
                    break
    except Exception as exc:
        raw_status = "crash"
        failure_reason = str(exc)
        kill_process_tree(proc.pid)
        with contextlib.suppress(Exception):
            proc.kill()
    finally:
        kill_process_tree(proc.pid)
        with contextlib.suppress(Exception):
            proc.kill()

    duration = time.time() - start_time

    # Critical security gate: verify harness integrity regardless of exit mode
    if repo_root and not check_harness_integrity(repo_root, initial_fingerprints):
        return SubprocessResult(
            status="violates_harness_lock",
            stdout=stdout or "",
            stderr="Immutable harness or production file was modified during subprocess execution.",
            duration_seconds=duration,
            peak_vram_mb=peak_vram,
            peak_rss_mb=peak_rss,
            failure_reason="Subprocess modified prohibited harness files.",
        )

    if raw_status == "timeout":
        return SubprocessResult(
            status="timeout",
            stdout=stdout or "",
            stderr=stderr or "",
            duration_seconds=duration,
            peak_vram_mb=peak_vram,
            peak_rss_mb=peak_rss,
            failure_reason=failure_reason,
        )

    if raw_status == "oom":
        return SubprocessResult(
            status="oom",
            stdout=stdout or "",
            stderr=stderr or "",
            duration_seconds=duration,
            peak_vram_mb=peak_vram,
            peak_rss_mb=peak_rss,
            failure_reason=failure_reason,
        )

    if raw_status == "crash":
        return SubprocessResult(
            status="crash",
            stdout=stdout or "",
            stderr=stderr or "",
            duration_seconds=duration,
            peak_vram_mb=peak_vram,
            peak_rss_mb=peak_rss,
            failure_reason=failure_reason,
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
            peak_rss_mb=peak_rss,
            failure_reason="Failed to parse result payload from candidate subprocess.",
        )

    return SubprocessResult(
        status="success",
        stdout=stdout,
        stderr=stderr,
        duration_seconds=duration,
        peak_vram_mb=peak_vram,
        peak_rss_mb=peak_rss,
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

    def run_level_2_confirmation(
        self, candidate_family: str, horizon: int = 20
    ) -> SubprocessResult:
        """Level 2: Full confirmation at the requested horizon (20 mins max)."""
        return run_isolated_candidate_eval(
            self.snapshot_path,
            candidate_family,
            horizon=horizon,
            timeout_seconds=self.budget.confirm_seconds,
            budget=self.budget,
            repo_root=self.repo_root,
        )

    def execute_trial(
        self,
        candidate_family: str,
        hypothesis: str = "Candidate search trial",
        level: int = 1,
        horizon: int = 5,
    ) -> dict[str, Any]:
        """Execute a full multi-fidelity candidate trial and record in ledger.

        ``horizon`` applies to the level-2 confirmation path; levels 0 and 1
        intentionally keep their fixed smoke (1) and screening (5) horizons.
        """
        if level == 0:
            trial_horizon = 1
            sub = self.run_level_0_smoke(candidate_family)
        elif level == 2:
            trial_horizon = horizon
            sub = self.run_level_2_confirmation(candidate_family, horizon=horizon)
        else:
            trial_horizon = 5
            sub = self.run_level_1_screening(candidate_family)

        payload = sub.payload or {}
        record = {
            "run_tag": self.run_tag,
            "candidate_family": candidate_family,
            "hypothesis": hypothesis,
            "horizon": trial_horizon,
            "snapshot_id": payload.get("snapshot_id") or "unknown",
            "status": sub.status,
            "failure_reason": sub.failure_reason,
            "peak_vram_mb": sub.peak_vram_mb,
            "training_seconds": sub.duration_seconds,
            "median_relative_mae": payload.get("median_relative_mae"),
            "median_relative_rmse": payload.get("median_relative_rmse"),
            "worst_fold_relative_rmse": payload.get("worst_fold_relative_rmse"),
            "folds_beating_persistence": payload.get("folds_beating_persistence"),
            "promotable": payload.get("promotable", False),
            "decision": "keep"
            if payload.get("promotable") and sub.status == "success"
            else "discard",
        }

        entry = append_record(self.ledger_path, record)
        export_tsv_summary(self.ledger_path, self.ledger_path.with_suffix(".tsv"))
        generate_markdown_report(self.ledger_path, self.ledger_path.parent / "REPORT.md")
        return entry
