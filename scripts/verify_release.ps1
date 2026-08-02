param(
  [string]$BaseUrl = $env:RENDER_BASE_URL,
  [string]$ExpectedCommit = $env:GITHUB_SHA,
  [string]$CorsOrigin = $env:VERCEL_PREVIEW_URL,
  [switch]$SkipFrontend,
  [switch]$SkipBackend
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

Write-Host "Checking backend lockfile and policy gates..."
uv lock --project backend --check
python scripts/check_ci_policy.py
python scripts/check_text_hygiene.py

if (-not $SkipBackend) {
  Write-Host "Running focused backend deployment tests..."
  uv run --project backend pytest backend/tests/test_deployment_identity.py backend/tests/test_check_deployment.py backend/tests/test_resource_budget.py backend/tests/test_api.py -q
  uv run --project backend python scripts/check_resource_budget.py --json
}

if (-not $SkipFrontend) {
  Write-Host "Running frontend tests and build..."
  Push-Location frontend
  npm ci
  npm run test:run
  npm run build
  if (Get-Command playwright -ErrorAction SilentlyContinue) {
    npm run e2e
  } else {
    Write-Host "Playwright CLI is not installed; skipping browser E2E locally."
  }
  Pop-Location
}

if ($BaseUrl) {
  Write-Host "Running deployment smoke against $BaseUrl..."
  $args = @("scripts/check_deployment.py", "--base-url", $BaseUrl, "--json", "--timeout", "60", "--restart-window", "2")
  if ($ExpectedCommit) { $args += @("--expected-commit", $ExpectedCommit) }
  if ($CorsOrigin) { $args += @("--cors-origin", $CorsOrigin) }
  python @args
} else {
  Write-Host "No RENDER_BASE_URL supplied; skipping live deployment smoke."
}

git diff --check
Write-Host "Release verification completed. Provider dashboards/previews are still required for live E2E."
