"""Fail CI when global-model serving methodology evidence is stale relative to its recorded full check.

The serving methodology is a contract: global volatility forecasts must come from
certified offline ONNX releases with verified Ed25519 signatures, exact feature order,
causal Deployable Schema v5 inputs, and explicit fail-closed abstentions. Every
methodology-affecting change invalidates the pinned evidence unless the full check
battery (unit suite, build, TFJS-free bundle verification, and Playwright contract e2e)
is re-run and the record in docs/METHODOLOGY_GATE.md is updated with the new SHA.

Usage:
  python scripts/check_methodology_gate.py
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GATE = ROOT / "docs" / "METHODOLOGY_GATE.md"

# Paths whose drift invalidates pinned evidence. These mirror the CI jobs
# that own each artifact so evidence can never silently outlive its method.
GUARDED = (
    "frontend/src/ml/",
    "frontend/src/hooks/useForecast.js",
    "frontend/src/components/ModelCard.jsx",
    "frontend/src/components/MetricsCard.jsx",
    "frontend/src/components/StockChart.jsx",
    "frontend/src/App.jsx",
    "frontend/e2e/fixtures.js",
    "frontend/e2e/fixtures.spec.js",
    "frontend/e2e/server-contract.spec.js",
    "frontend/package.json",
    "frontend/package-lock.json",
    "frontend/playwright.config.js",
    "backend/routes/forecasts.py",
    "backend/routes/models.py",
    "backend/services/volatility_forecast.py",
    "backend/services/volatility_snapshot.py",
    "backend/release/bundle.py",
)

REQUIRED_BATTERY = (
    "npm run test:run",
    "npm run build",
    "npm run check:production-bundle",
    "server-contract.spec.js",
    "fixtures.spec.js",
)


def guarded_name(name: str) -> bool:
    """True when a changed path sits under any guarded methodology location."""
    relative = name.replace("\\", "/")
    return any(relative.startswith(prefix) for prefix in GUARDED)


class GitError(RuntimeError):
    """A git command required by the gate failed — the gate fails closed."""


def git(*args: str) -> str:
    try:
        return subprocess.run(
            ["git", *args],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (subprocess.CalledProcessError, OSError) as exc:
        raise GitError(f"git {' '.join(args)} failed: {exc}") from exc


SHA_PATTERN = r"[0-9a-f]{7,40}"


def validate(text: str, run_git=None) -> list[str]:
    """Validate the gate record; returns a list of human-readable errors.

    ``run_git`` is injectable so tests can simulate git failures without a
    real repository. Any git failure is an error (fail closed), never a
    silent pass.
    """
    if run_git is None:
        run_git = git
    errors: list[str] = []

    def resolve(label: str, value: str) -> str | None:
        try:
            return run_git("rev-parse", f"{value}^{{commit}}")
        except GitError:
            errors.append(f"{label} {value} does not exist in git history.")
            return None

    recorded_match = re.search(rf"recorded_sha:\s*({SHA_PATTERN})", text)
    freeze_match = re.search(r"freeze_record_commit:\s*([0-9a-f]{40})", text)

    if not recorded_match:
        errors.append(f"{GATE.relative_to(ROOT)} must record recorded_sha.")
        return errors
    recorded = recorded_match.group(1)
    if recorded == "????":
        errors.append(
            "recorded_sha is still the placeholder; run the battery and update the record."
        )
        return errors

    recorded_full = resolve("recorded_sha", recorded)
    if not re.search(r"freeze_record_commit:\s*[0-9a-f]{40}", text):
        errors.append(
            f"{GATE.relative_to(ROOT)} must record freeze_record_commit (full 40-hex SHA "
            "of the commit that last wrote this record)."
        )
        return errors
    freeze = freeze_match.group(1)
    freeze_full = resolve("freeze_record_commit", freeze)

    if recorded_full is None or freeze_full is None:
        return errors

    # The battery-verified tree must be contained in the record-writing
    # commit's history: evidence can only be pinned by a later commit.
    try:
        base = run_git("merge-base", recorded_full, freeze_full)
    except GitError as exc:
        errors.append(f"git merge-base failed: {exc}")
        return errors
    if base != recorded_full:
        errors.append(
            f"recorded_sha {recorded} is not an ancestor of freeze_record_commit; "
            "the freeze record must descend from the battery-verified tree."
        )

    # And the record-writing commit must itself be contained in the branch
    # being validated — otherwise a valid-looking freeze from a sibling
    # branch would certify unrelated code.
    try:
        head_base = run_git("merge-base", freeze_full, "HEAD")
    except GitError as exc:
        errors.append(f"git merge-base failed: {exc}")
        return errors
    if head_base != freeze_full:
        errors.append(
            "freeze_record_commit is not an ancestor of HEAD; the current "
            "branch does not contain the recorded freeze."
        )

    # The freeze commit must actually have written the gate document.
    try:
        freeze_touched = run_git("diff", "--name-only", f"{freeze_full}^", freeze_full)
    except GitError as exc:
        errors.append(f"git diff failed: {exc}")
        return errors
    if "docs/METHODOLOGY_GATE.md" not in {
        name.replace("\\", "/") for name in freeze_touched.splitlines()
    }:
        errors.append(
            "freeze_record_commit did not modify docs/METHODOLOGY_GATE.md; "
            "it does not look like a record-writing commit."
        )

    try:
        changed = run_git("diff", "--name-only", recorded, "HEAD")
    except GitError as exc:
        errors.append(f"git diff failed: {exc}")
        return errors
    drifted = [name for name in changed.splitlines() if guarded_name(name)]
    if drifted:
        errors.append(
            f"methodology evidence is stale: {len(drifted)} guarded file(s) changed "
            f"since recorded_sha {recorded}."
        )
        errors.append("  npx vitest run && npm run build (frontend)")
        errors.append("  npm run check:production-bundle")
        errors.append("  npx playwright test e2e/server-contract.spec.js e2e/fixtures.spec.js")
        errors.append(
            "Then bump recorded_sha to the verified HEAD and let the follow-up "
            "commit pin freeze_record_commit."
        )

    for required in REQUIRED_BATTERY:
        if required not in text:
            errors.append(f"{GATE.relative_to(ROOT)} battery listing must mention {required!r}.")
    return errors


def main() -> int:
    if not GATE.exists():
        print(f"ERROR: {GATE.relative_to(ROOT)} is missing.", file=sys.stderr)
        return 1
    text = GATE.read_text(encoding="utf-8")
    try:
        errors = validate(text)
    except GitError as exc:
        # Unavailable git history must fail the gate, never skip it.
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    if errors:
        print("\n".join(f"ERROR: {error}" for error in errors), file=sys.stderr)
        return 1
    recorded = re.search(rf"recorded_sha:\s*({SHA_PATTERN})", text).group(1)
    print(f"Methodology evidence is current (recorded at {recorded}).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
