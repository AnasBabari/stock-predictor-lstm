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
