"""Fail-closed validation for the methodology gate record and harness integrity."""

from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _load_gate_module():
    spec = importlib.util.spec_from_file_location(
        "check_methodology_gate", ROOT / "scripts" / "check_methodology_gate.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


GATE = _load_gate_module()

RECORDED = "a" * 40
FREEZE = "b" * 40


def _record_text(recorded: str = RECORDED, freeze: str = FREEZE) -> str:
    battery = "\n".join(f"- {cmd}" for cmd in GATE.REQUIRED_BATTERY)
    return (
        f"recorded_sha: {recorded}\n"
        f"freeze_record_commit: {freeze}\n\n"
        "## Full check battery\n\n"
        f"{battery}\n"
    )


def _runner(mapping):
    """Build a run_git stand-in from {subcommand: output | callable | exception}."""

    def run(*args):
        sub = args[0]
        outcome = mapping.get(sub, GATE.GitError(f"unexpected git subcommand: {sub}"))
        if isinstance(outcome, Exception):
            raise outcome
        if callable(outcome):
            return outcome(*args)
        return outcome

    return run


def _revparse():
    # Mirror `git rev-parse <sha>^{commit}` output: full sha without suffix.
    return lambda *args: args[1].replace("^{commit}", "")


def test_valid_record_passes():
    errors = GATE.validate(
        _record_text(),
        run_git=_runner(
            {
                "rev-parse": _revparse(),
                "merge-base": RECORDED,
                "diff": "",
            }
        ),
    )
    assert errors == []


def test_missing_freeze_record_commit_is_an_error():
    text = _record_text().replace(f"freeze_record_commit: {FREEZE}\n", "")
    errors = GATE.validate(text, run_git=_runner({}))
    assert any("freeze_record_commit" in error for error in errors)


def test_short_or_garbage_freeze_sha_is_rejected():
    text = _record_text(freeze="deadbeef")
    errors = GATE.validate(text, run_git=_runner({}))
    assert any("freeze_record_commit" in error for error in errors)


def test_unknown_shas_fail_closed():
    boom = GATE.GitError("not a repository")
    errors = GATE.validate(_record_text(), run_git=_runner({"rev-parse": boom}))
    assert any("does not exist" in error for error in errors)


def test_freeze_not_descending_from_recorded_fails():
    errors = GATE.validate(
        _record_text(),
        run_git=_runner(
            {
                "rev-parse": lambda *a: a[1],
                "merge-base": "c" * 40,
                "diff": "",
            }
        ),
    )
    assert any("not an ancestor of freeze_record_commit" in error for error in errors)


def test_git_failure_during_diff_fails_closed():
    errors = GATE.validate(
        _record_text(),
        run_git=_runner(
            {
                "rev-parse": _revparse(),
                "merge-base": RECORDED,
                "diff": GATE.GitError("diff unavailable"),
            }
        ),
    )
    assert any("failed" in error for error in errors)


def test_stale_guarded_paths_are_reported():
    errors = GATE.validate(
        _record_text(),
        run_git=_runner(
            {
                "rev-parse": _revparse(),
                "merge-base": RECORDED,
                "diff": "frontend/src/ml/trainingWorker.js\nREADME.md\n",
            }
        ),
    )
    assert any("methodology evidence is stale" in error for error in errors)


def test_current_repository_record_validates():
    """Integration: the committed record must satisfy the stricter gate."""
    text = (ROOT / "docs" / "METHODOLOGY_GATE.md").read_text(encoding="utf-8")
    assert GATE.validate(text) == []


def test_harness_integrity_fails_closed_without_git(tmp_path):
    """Regression: a failing git inspection used to return True (fail open)."""
    from stock_autoresearch.controller import check_harness_integrity

    bare = tmp_path / "not-a-repo"
    bare.mkdir()
    assert check_harness_integrity(bare) is False


def test_harness_integrity_still_detects_drift_with_fingerprints(tmp_path):
    from stock_autoresearch.controller import (
        check_harness_integrity,
        get_protected_fingerprints,
    )

    target = tmp_path / "research" / "stock_autoresearch" / "config.py"
    target.parent.mkdir(parents=True)
    target.write_text("ORIGINAL = True\n", encoding="utf-8")
    baseline = get_protected_fingerprints(tmp_path)
    assert check_harness_integrity(tmp_path, baseline) is True
    target.write_text("ORIGINAL = False\n", encoding="utf-8")
    assert check_harness_integrity(tmp_path, baseline) is False
