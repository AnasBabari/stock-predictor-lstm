# Render/Vercel deployment gate

The release gate keeps the free-tier Render API and Vercel frontend deployable without silently spending provider resources or shipping an unverified global-volatility model.

## Provider settings and secrets

Render service:

- Use `render.yaml` with `plan: free`, `healthCheckPath: /health`, `autoDeploy: checksPass`, PR previews enabled, and monorepo build filters.
- Set `CORS_ORIGIN` to the production Vercel origin.
- Set `PREVIEW_CORS_ORIGIN_REGEX` only for preview services, for example `https://[a-z0-9-]+\.vercel\.app`.
- Set `VITE_VOLATILITY_SERVING_ENABLED=true` on the Vercel production project.
- Keep the repository-pinned `VOLATILITY_PUBLIC_KEY_PATH` on Render. Set `VOLATILITY_SERVING_REQUIRED=true` only after certification passes. For Render's ephemeral free filesystem, publish the deterministic ZIP produced by `scripts/package_volatility_release.py` at an immutable HTTPS URL and set the paired `VOLATILITY_RELEASE_ARCHIVE_URL` and `VOLATILITY_RELEASE_ARCHIVE_SHA256`. Disk-backed deployments may use `VOLATILITY_RELEASE_DIR` instead.
- Keep `SERVER_MODELS_ENABLED=false` and `VITE_BROWSER_TRAINING_ENABLED=false` in production. Browser TFJS is a rollback/migration path, not the production learned-model contract.

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

Vercel preview E2E uses a deterministic fixture for the UI contract. Production smoke separately verifies the signed global-volatility response, seven dated quantile values, release identity, and explicit abstention behavior when a release or horizon is unavailable.

Render previews use `--forecast-contract global_volatility_abstention` until
certification succeeds. That contract requires the structured 503 abstention
and rejects any browser/baseline learned-model advertisement. A manual
production run uses the same strict-abstention contract by default. Set the
workflow's `certified_release` input to `true` only after locked certification,
signed-bundle publication, and Render configuration are complete; that switches
the gate to `--forecast-contract global_volatility` and requires `/ready` plus
`/models` to report the verified release as `ready` before accepting a forecast.

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

## Release smoke checklist

1. `/health` reports the expected commit and environment.
2. `/ready` is 200 only when market data and the required signed release are available.
3. `/models` reports `global_volatility.status=ready`, the model id, and certified horizons; `browser_training.status=disabled`.
4. `/api/v2/forecast?ticker=MSFT&horizon=7` returns seven strictly increasing dates, p05/p50/p95 arrays, and `locked_purged_walk_forward` evidence.
5. A failed horizon or tampered bundle returns structured 503 abstention; no baseline is relabelled as learned.
6. Vercel renders the volatility-only labels and retains the response evidence in exports.

## Ephemeral Render release procedure

This procedure is forbidden until the locked certification outcome has overall `status = passed`.

1. After the future panel contains 252 target-complete origins on or after
   2026-08-27, open the v7 reserve once. Use a new, empty output directory; a
   failed run is immutable evidence and must never be deleted and repeated.

   ```powershell
   python scripts/certify_prospective_volatility_candidate.py `
     --candidate-dir C:\path\to\prospective-v7-candidate `
     --development-report C:\path\to\prospective-development-report.json `
     --development-panel-dir C:\path\to\panel-ending-2026-08-21 `
     --certification-panel-dir C:\path\to\future-immutable-panel `
     --example-cache-root C:\path\to\volatility-example-cache `
     --output-dir C:\path\to\v7-locked-certification `
     --open-locked-holdout
   ```

   Continue only when `locked-certification.json` has overall
   `status = passed` and the command created `candidate/`. Prefix drift,
   insufficient future sessions, a failed horizon, NMM/MSFT degradation, or a
   repeated non-empty output directory must stop the release.

   If a passed report exists but candidate promotion was interrupted, recover
   materialization without reopening the reserve:

   ```powershell
   python scripts/materialize_prospective_certification.py `
     --candidate-dir C:\path\to\prospective-v7-candidate `
     --development-report C:\path\to\prospective-development-report.json `
     --development-panel-dir C:\path\to\panel-ending-2026-08-21 `
     --certification-dir C:\path\to\v7-locked-certification
   ```

2. Assemble the signed ONNX release from that exact materialized candidate
   with `scripts/assemble_volatility_release.py`; keep the private Ed25519 key
   off Git and off Render.

   ```powershell
   python scripts/assemble_volatility_release.py `
     --candidate-dir C:\path\to\v7-locked-certification\candidate `
     --output-dir C:\path\to\signed-release `
     --private-key-path C:\secure\volatility-v1.private.pem `
     --public-key-path backend\release_keys\volatility-v1.public.pem
   ```

3. Verify the bundle locally with the pinned public key and package it without extra files:

   ```powershell
   python scripts/package_volatility_release.py `
     --release-dir C:\path\to\signed-release `
     --public-key-path backend\release_keys\volatility-v1.public.pem `
     --output C:\path\to\stocklstm-volatility-v7.zip
   ```

4. Upload the ZIP to an immutable HTTPS object or GitHub Release asset. Record the command's exact `archive_sha256`; do not use a mutable `latest` URL.
5. Confirm the Blueprint supplies
   `VOLATILITY_PUBLIC_KEY_PATH=backend/release_keys/volatility-v1.public.pem`.
   The public key may be distributed; the matching private key must remain off
   Git, Render, release archives, and build logs.
6. Set the immutable archive URL and SHA-256, then enable `VOLATILITY_SERVING_REQUIRED=true` in the same deployment change.
7. Require `/ready`, `/models`, a certified MSFT request, a certified NMM request, and a deliberately incorrect digest smoke test before declaring the deployment complete.

The backend downloads into a digest-named directory under `VOLATILITY_RELEASE_CACHE_DIR` (default platform temp storage). Restarts may download again; requests never train and never write model state. URL credentials, private signing keys, failed candidates, and unsigned prospective candidates must not enter the service configuration.
