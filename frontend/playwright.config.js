import { defineConfig, devices } from '@playwright/test';

const baseURL = process.env.VERCEL_PREVIEW_URL || process.env.E2E_BASE_URL || 'http://127.0.0.1:4173';

export default defineConfig({
  testDir: './e2e',
  timeout: 120_000,
  expect: { timeout: 15_000 },
  reporter: [['list'], ['html', { outputFolder: 'playwright-report', open: 'never' }]],
  use: {
    baseURL,
    trace: 'retain-on-failure',
    screenshot: 'only-on-failure',
    video: 'retain-on-failure',
  },
  projects: [{ name: 'chromium', use: { ...devices['Desktop Chrome'] } }],
  webServer: process.env.VERCEL_PREVIEW_URL || process.env.E2E_BASE_URL ? undefined : {
    command: 'npm run preview -- --host 127.0.0.1 --port 4173',
    url: 'http://127.0.0.1:4173',
    // Reusing an arbitrary process on 4173 can test a stale production bundle
    // and create false passes/failures. Opt in only when the caller knowingly
    // owns that server.
    reuseExistingServer: process.env.PLAYWRIGHT_REUSE_SERVER === 'true',
    timeout: 120_000,
  },
});
