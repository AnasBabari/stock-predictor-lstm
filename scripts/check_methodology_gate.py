"""Fail CI when browser-methodology evidence is stale relative to its recorded full check.

The browser evaluation methodology is a contract: metrics must come from
untouched evaluation windows, scalers fitted only to fitting observations,
direction baselines derived from pre-evaluation labels, and evidence reported
per forecast day. Every methodology-affecting change invalidates the pinned
evidence unless the full check battery (unit suite, build, contract e2e,
real-training e2e, temporal-isolation e2e) is re-run and the record in
docs/METHODOLOGY_GATE.md is updated with the new SHA.

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
    "frontend/e2e/browser-real-training.spec.js",
    "frontend/e2e/browser-temporal-isolation.spec.js",
    "frontend/e2e/fixtures.js",
    "frontend/src/components/",
    "frontend/src/App.jsx",
    "frontend/package.json",
    "frontend/package-lock.json",
    "frontend/playwright.config.js",
)

REQUIRED_BATTERY = (
    "npx vitest run",
    "npm run build",
    "server-contract.spec.js",
    "fixtures.spec.js",
    "browser-real-training.spec.js",
    "browser-temporal-isolation.spec.js",
    "--workers=1",
)


def guarded_name(name: str) -> bool:
    """True when a changed path sits under any guarded methodology location."""
    relative = name.replace("\\", "/")
    return any(relative.startswith(prefix) for prefix in GUARDED)


def git(*args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def main() -> int:
    errors: list[str] = []
    if not GATE.exists():
        errors.append(f"{GATE.relative_to(ROOT)} is missing.")
    else:
        text = GATE.read_text(encoding="utf-8")
        match = re.search(r"recorded_sha:\s*([0-9a-f]{7,40}|\?{4})", text)
        if not match:
            errors.append(f"{GATE.relative_to(ROOT)} must record recorded_sha.")
        else:
            recorded = match.group(1)
            if recorded == "????":
                errors.append(
                    "recorded_sha is still the placeholder; run the battery and update the record."
                )
            else:
                try:
                    recorded_full = git("rev-parse", f"{recorded}^{{commit}}")
                except subprocess.CalledProcessError:
                    errors.append(f"recorded_sha {recorded} does not exist in git history.")
                else:
                    base = git("merge-base", "HEAD", recorded_full)
                    if base != recorded_full:
                        errors.append(
                            f"recorded_sha {recorded} is not an ancestor of HEAD; "
                            "re-run the battery and update the record at HEAD."
                        )
                    changed = git("diff", "--name-only", recorded, "HEAD")
                    drifted = [name for name in changed.splitlines() if guarded_name(name)]
                    if drifted:
                        errors.append(
                            f"methodology evidence is stale: {len(drifted)} guarded file(s) changed "
                            f"since recorded_sha {recorded}."
                        )
                        errors.append("Rerun the full battery and update docs/METHODOLOGY_GATE.md:")
                        errors.append("  npx vitest run && npm run build (frontend)")
                        errors.append(
                            "  npx playwright test e2e/server-contract.spec.js e2e/fixtures.spec.js"
                        )
                        errors.append(
                            "  npx playwright test e2e/browser-real-training.spec.js --workers=1"
                        )
                        errors.append(
                            "  npx playwright test e2e/browser-temporal-isolation.spec.js --workers=1"
                        )
                        errors.append(
                            "Then bump recorded_sha to the new HEAD and commit the record."
                        )
        for required in REQUIRED_BATTERY:
            if required not in text:
                errors.append(
                    f"{GATE.relative_to(ROOT)} battery listing must mention {required!r}."
                )
    if errors:
        print("\n".join(f"ERROR: {error}" for error in errors), file=sys.stderr)
        return 1
    match = re.search(r"recorded_sha:\s*([0-9a-f]{7,40})", GATE.read_text(encoding="utf-8"))
    print(f"Methodology evidence is current (recorded at {match.group(1)}).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
