"""Fail CI when required gates or dependency reproducibility controls drift."""

from __future__ import annotations

import re
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORKFLOWS = list((ROOT / ".github" / "workflows").glob("*.yml"))
workflow_text = "\n".join(path.read_text(encoding="utf-8") for path in WORKFLOWS)

errors: list[str] = []
if "@latest" in workflow_text:
    errors.append("GitHub workflows may not execute mutable @latest tools.")

# Third-party actions are executable supply-chain inputs.  Keep the reviewable
# workflow references immutable rather than relying on mutable major tags.
for action_ref in re.findall(r"^\s*-?\s*uses:\s*[^\s#]+@([^\s#]+)", workflow_text, re.MULTILINE):
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


def job_blocks(text: str) -> dict[str, str]:
    """Split a workflow body into its top-level jobs keyed by job name."""
    blocks = re.split(r"^  ([a-z][a-z0-9-]*):\s*$", text, flags=re.MULTILINE)
    jobs: dict[str, str] = {}
    for index in range(1, len(blocks), 2):
        jobs[blocks[index]] = blocks[index + 1]
    return jobs


def requirement_names(path: Path) -> set[str]:
    names = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            names.add(re.split(r"[<>=!~\[]", line, maxsplit=1)[0].lower().replace("_", "-"))
    return names


CI = ROOT / ".github" / "workflows" / "ci.yml"
REAL_TRAINING = ROOT / ".github" / "workflows" / "frontend-real-training-e2e.yml"

if CI.exists():
    ci_jobs = job_blocks(CI.read_text(encoding="utf-8"))
    unit_build = ci_jobs.get("frontend-unit-build", "")
    contract_e2e = ci_jobs.get("frontend-contract-e2e", "")
    compose_smoke = ci_jobs.get("compose-smoke", "")

    if not unit_build:
        errors.append("ci.yml must define a frontend-unit-build job.")
    else:
        for gate in ("npm ci", "npm run test:run", "npm run build"):
            if gate not in unit_build:
                errors.append(f"frontend-unit-build job must run {gate!r}.")

    if not contract_e2e:
        errors.append("ci.yml must define a frontend-contract-e2e job.")
    elif "server-contract.spec.js" not in contract_e2e:
        errors.append("frontend-contract-e2e job must run the server-contract spec.")
    elif "browser-real-training.spec.js" in contract_e2e:
        errors.append("frontend-contract-e2e must never run the real-training spec.")

    if (
        "backend" not in compose_smoke
        or "policy" not in compose_smoke
        or "frontend-unit-build" not in compose_smoke
        or "frontend-contract-e2e" not in compose_smoke
    ):
        errors.append(
            "compose-smoke must depend on backend, policy, frontend-unit-build, and frontend-contract-e2e."
        )
else:
    errors.append("ci.yml is missing.")

if not REAL_TRAINING.exists():
    errors.append("frontend-real-training-e2e.yml is missing.")
else:
    real_training = REAL_TRAINING.read_text(encoding="utf-8")
    for required in (
        "workflow_dispatch:",
        "schedule:",
        "timeout-minutes:",
        "browser-real-training.spec.js",
        "--workers=1",
        "actions/upload-artifact@",
    ):
        if required not in real_training:
            errors.append(f"frontend-real-training-e2e.yml must include {required!r}.")

project = tomllib.loads((ROOT / "backend" / "pyproject.toml").read_text(encoding="utf-8"))
runtime_names = {
    re.split(r"[<>=!~\[]", item, maxsplit=1)[0].lower().replace("_", "-")
    for item in project["project"]["dependencies"]
}
dev_names = {
    re.split(r"[<>=!~\[]", item, maxsplit=1)[0].lower().replace("_", "-")
    for item in project["dependency-groups"]["dev"]
}
if runtime_names != requirement_names(ROOT / "backend" / "requirements.txt"):
    errors.append("requirements.txt names have drifted from pyproject runtime dependencies.")
if dev_names != requirement_names(ROOT / "backend" / "requirements-dev.txt"):
    errors.append("requirements-dev.txt names have drifted from the dev dependency group.")
if (ROOT / "backend" / "uv.lock").stat().st_size < 1000:
    errors.append("backend/uv.lock is missing or incomplete.")

render_text = (ROOT / "render.yaml").read_text(encoding="utf-8")
if (
    "--frozen" not in render_text
    or "uv sync" not in render_text
    or "pip install -r requirements.txt" in render_text
):
    errors.append("render.yaml must install runtime dependencies from the frozen backend/uv.lock.")
if "uv==" not in render_text:
    errors.append("render.yaml must pin the uv version used for the lockfile install.")

if errors:
    print("\n".join(f"ERROR: {error}" for error in errors), file=sys.stderr)
    raise SystemExit(1)
print("CI policy checks passed.")
