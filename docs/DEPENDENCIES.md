# Dependency Warning Inventory

This inventory records a local baseline captured on 2026-07-28. It separates security findings from deprecations and available updates; it is not a claim that dependencies are permanently vulnerability-free. Re-run the commands below when dependency changes or new advisories warrant review.

## Baseline checks

| Check | Result |
| --- | --- |
| `uv lock --project backend --check` | Passed; lockfile is consistent |
| `uv run --project backend pip-audit` | Passed; no known Python vulnerabilities |
| `npm ci` (from `frontend/`) | Passed; lockfile installs reproducibly |
| `npm audit` (from `frontend/`) | Passed; 0 vulnerabilities across 291 installed packages |
| Docker image scan | Not run; no image scanner is configured in CI |

The raw command output was kept outside the repository during the review. The backend runtime dependency tree intentionally excludes TensorFlow; TensorFlow is available only through the opt-in `training` group for offline research. The frontend dependencies are pure React and Chart.js; TensorFlow.js has been completely retired. `pip-audit` reported no advisory for the resolved environment.

### Declaration change (2026-08-28)

Three packages that research code imported **directly** but that were declared only transitively have been added to the backend `dev` dependency group:

| Package | Constraint | Why it is now declared |
| --- | --- | --- |
| `exchange-calendars` | `>=4.13.0,<5.0.0` | Imported directly by research calendar code; previously satisfied only via `pandas-market-calendars` |
| `threadpoolctl` | `>=3.5.0,<4.0.0` | Required to pin thread counts for reproducible neural runs |
| `pyyaml` | `>=6.0.0,<7.0.0` | Imported directly by research configuration code |

This is a **declaration** change, not a version change: `uv lock` was regenerated and the resolved package count was unchanged at 120, confirming all three were already in the tree. A clean `uv sync --frozen` into an empty environment was verified. `pip-audit` still reports no advisory. The drift detector in `scripts/check_ci_policy.py` requires these names to also appear in `backend/requirements-dev.txt`; they do.

## Classified findings

| Package and version | Category | Reachability | Advisory/severity | Action |
| --- | --- | --- | --- | --- |
| `whatwg-encoding@3.1.1` via `jsdom@25.0.1` → `html-encoding-sniffer@4.0.0` | Deprecated transitive package | Frontend tests only; not shipped in the production bundle | No CVE reported by `npm audit`; npm recommends `@exodus/bytes` | Monitor. Do not add an override: the maintained path is a future compatible `jsdom` parent upgrade. |
| `@types/react@18.3.31`, `@types/react-dom@18.3.7` | Outdated development packages | Type tooling only | No vulnerability reported | Selective review when React 19 compatibility is planned; no automatic major upgrade. |
| `react@18.3.1`, `react-dom@18.3.1` | Outdated runtime packages | Production frontend | No vulnerability reported | Keep React 18 until a separately tested React/Vite compatibility upgrade is scheduled. |
| `jsdom@25.0.1` | Outdated development package | Frontend test environment only | No vulnerability reported | Monitor. `jsdom@29.1.1` is a major upgrade with changed dependencies and peer requirements; evaluate with Vitest before changing the manifest. |
| `react-doctor@0.1.6` | Outdated development tool | Optional developer diagnostic only | No vulnerability reported | Upgrade only as a separate tooling change after checking its compatibility with the current React/Vite stack. |

## Deferred warning review

`whatwg-encoding@3.1.1` is the latest published version in its package line and is pulled by the pinned `jsdom` parent. It is a deprecation notice rather than a security vulnerability, and the affected package is reachable only through the test environment. Revisit by 2026-10-28, or earlier if `jsdom` publishes a compatible parent release that removes it or an advisory affects the package.

No dependency was upgraded in this inventory pass. Broad major upgrades, `npm audit fix --force`, manual lockfile edits, and unsupported overrides were intentionally avoided.
