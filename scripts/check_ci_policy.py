"""Fail CI when required gates or dependency reproducibility controls drift."""

from __future__ import annotations

import re
import sys
from pathlib import Path

import tomllib

ROOT = Path(__file__).resolve().parents[1]
WORKFLOWS = list((ROOT / ".github" / "workflows").glob("*.yml"))
workflow_text = "\n".join(path.read_text(encoding="utf-8") for path in WORKFLOWS)

errors: list[str] = []
if "@latest" in workflow_text:
    errors.append("GitHub workflows may not execute mutable @latest tools.")

# Third-party actions are executable supply-chain inputs.  Keep the reviewable
# workflow references immutable rather than relying on mutable major tags.
for action_ref in re.findall(
    r"^\s*-?\s*uses:\s*[^\s#]+@([^\s#]+)", workflow_text, re.MULTILINE
):
    if not re.fullmatch(r"[0-9a-f]{40}", action_ref):
        errors.append("GitHub Actions must be pinned to a full commit SHA.")
        break

for required in (
    "ruff check",
    "ruff format --check",
    "mypy",
    "bandit",
    "pip-audit",
    "pytest",
    "npm run test:run",
    "npm run build",
    "docker compose",
    "uv lock --project backend --check",
    "uv sync --project backend --frozen",
    "npm ci",
    "scripts/check_api_docs.py",
    "scripts/check_text_hygiene.py",
):
    if required not in workflow_text:
        errors.append(f"Missing CI gate: {required}")


def requirement_names(path: Path) -> set[str]:
    names = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            names.add(
                re.split(r"[<>=!~\[]", line, maxsplit=1)[0].lower().replace("_", "-")
            )
    return names


project = tomllib.loads(
    (ROOT / "backend" / "pyproject.toml").read_text(encoding="utf-8")
)
runtime_names = {
    re.split(r"[<>=!~\[]", item, maxsplit=1)[0].lower().replace("_", "-")
    for item in project["project"]["dependencies"]
}
dev_names = {
    re.split(r"[<>=!~\[]", item, maxsplit=1)[0].lower().replace("_", "-")
    for item in project["dependency-groups"]["dev"]
}
if runtime_names != requirement_names(ROOT / "backend" / "requirements.txt"):
    errors.append(
        "requirements.txt names have drifted from pyproject runtime dependencies."
    )
if dev_names != requirement_names(ROOT / "backend" / "requirements-dev.txt"):
    errors.append(
        "requirements-dev.txt names have drifted from the dev dependency group."
    )
if (ROOT / "backend" / "uv.lock").stat().st_size < 1000:
    errors.append("backend/uv.lock is missing or incomplete.")

if errors:
    print("\n".join(f"ERROR: {error}" for error in errors), file=sys.stderr)
    raise SystemExit(1)
print("CI policy checks passed.")
