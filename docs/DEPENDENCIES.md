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

The raw command output was kept outside the repository during the review. The backend dependency tree contains the declared runtime and development tools; `pip-audit` reported no advisory for the resolved environment.

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
