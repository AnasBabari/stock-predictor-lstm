# Render/Vercel deployment gate

The release gate keeps the free-tier Render API and Vercel frontend deployable without silently spending provider resources or shipping a broken browser-training path.

## Provider settings and secrets

Render service:

- Use `render.yaml` with `plan: free`, `healthCheckPath: /health`, `autoDeploy: checksPass`, PR previews enabled, and monorepo build filters.
- Set `CORS_ORIGIN` to the production Vercel origin.
- Set `PREVIEW_CORS_ORIGIN_REGEX` only for preview services, for example `https://[a-z0-9-]+\.vercel\.app`.
- Keep `SERVER_MODELS_ENABLED=false` and browser training enabled unless deliberately rolling back.

GitHub variables/secrets used by deployment-gate workflows:

- `RENDER_PREVIEW_BASE_URL` or manual `render_base_url` input for preview smoke.
- `RENDER_PRODUCTION_BASE_URL` for protected production smoke.
- `VERCEL_PREVIEW_URL` or manual `vercel_preview_url` input for browser E2E.
- `RENDER_ROLLBACK_SERVICE_ID` only documents the rollback target; automatic rollback is intentionally disabled.
- Provider API tokens must stay in GitHub environments or provider dashboards and are not required for local verification.

Branch protection should require the normal CI workflow plus the deployment gate jobs that are relevant for the target environment. Production smoke should be run from the protected `production` environment after Render and Vercel have deployed the same commit.

## Required external repository settings (not provable from code)

Repository files cannot verify GitHub settings; the gates below depend on
them being configured once by the owner. If they are missing, CI runs but
does not actually protect `main`:

1. **Branch protection on `main`:** require the status checks named
   `backend`, `frontend-unit-build`, `frontend-contract-e2e`, `research`,
   `policy`, and `compose-smoke` (plus any enabled `deployment-gate` jobs),
   with "require branches to be up to date" enabled.
2. **Render `autoDeploy: checksPass`:** Render only blocks deploys on failed
   checks if those GitHub statuses are reported to Render (GitHub App /
   OAuth integration connected). Without the integration, autodeploy happens
   regardless of CI outcome.
3. **CODEOWNERS is advisory on its own:** review requirements come from
   branch protection ("Require review from Code Owners"). The CODEOWNERS file
   covers `.github/`, `scripts/check_*.py`, and `research/program.md` so a
   PR cannot weaken the policy gates it is judged by.
4. **Dependabot** (`.github/dependabot.yml`) opens update PRs; it does not
   merge them. pip updates must re-run `uv lock --project backend` so
   `uv lock --check` stays green.

## Free-tier constraints

Render free instances can sleep, restart, and have tight memory/CPU budgets. The resource harness fails on startup errors, exit `127`, exit `137`, restarts, and peak RSS above 400 MiB inside a 512 MiB budget. The live smoke checker uses short timeouts and sanitized JSON failures so logs do not expose provider headers or credentials.

Vercel preview E2E uses a deterministic Quick-profile fixture instead of live Yahoo data. It verifies browser worker training, non-flat output, metric labels, IndexedDB cache reload behavior, direction probabilities, and captures traces/screenshots/video on failure.

## Local parity

Run from PowerShell:

```powershell
./scripts/verify_release.ps1
```

Optional live smoke:

```powershell
$env:RENDER_BASE_URL="https://your-render-preview.onrender.com"
$env:VERCEL_PREVIEW_URL="https://your-vercel-preview.vercel.app"
./scripts/verify_release.ps1
```

If Playwright is not installed, local browser E2E is skipped. CI runs it only when a Vercel preview URL is supplied and the frontend lockfile includes the Playwright dependency.
