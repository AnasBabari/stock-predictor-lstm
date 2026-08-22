import { expect, test } from '@playwright/test';
import { deterministicSnapshot, rejectedForecastSnapshot } from './fixtures.js';

// Real end-to-end browser training: this spec builds an actual TensorFlow.js
// model inside Chromium with deterministic fixtures. The server-forecast
// registry is intercepted (unavailable 200 => sanctioned browser fallback) so
// the app's own test harness never talks to a real backend and no request can
// leak to the local proxy (which is not running in CI).
async function mockApi(page, ticker, snapshot) {
  await page.route('**/api/v1/search?query=*', (route) => route.fulfill({ json: [{ ticker, name: `${ticker} Corp.`, type: 'Equity' }] }));
  await page.route('**/api/v1/server-forecasts/**', (route) =>
    route.fulfill({ json: { available: false, reason: 'missing', fallback: 'browser_training' } })
  );
  await page.route(`**/api/v1/training-data?ticker=${ticker}`, (route) => route.fulfill({ json: snapshot }));
  await page.route(`**/api/v1/info?ticker=${ticker}`, (route) => route.fulfill({ json: { longName: `${ticker} Corp.`, sector: 'Technology' } }));
  await page.route('**/api/v1/predict**', (route) => route.fulfill({ status: 503, json: { detail: `baseline must not be used for ${ticker} fixture` } }));
}

async function prepare(page, ticker, snapshot) {
  await mockApi(page, ticker, snapshot);
  await page.goto('/');
  await page.locator('#trainingProfile').selectOption('quick');
  await page.getByRole('combobox', { name: 'Search stock ticker or company name' }).fill(ticker);
}

test('trains a real price model in the browser and reloads it from IndexedDB', async ({ page }) => {
  test.setTimeout(180_000);
  // Learnable trending fixture: the price model must be promoted rather than
  // falling back to the flat persistence baseline.
  await prepare(page, 'MSFT', deterministicSnapshot('MSFT'));
  await page.getByRole('button', { name: 'Predict', exact: true }).click();
  await expect(page.getByText('Price Forecast Metrics')).toBeVisible({ timeout: 150_000 });
  await expect(page.getByText(/flat|baseline fallback/i)).toHaveCount(0);

  await page.reload();
  await page.getByRole('combobox', { name: 'Search stock ticker or company name' }).fill('MSFT');
  await page.getByRole('button', { name: 'Predict', exact: true }).click();
  await expect(page.getByText(/Cached on this device/i)).toBeVisible({ timeout: 30_000 });
});

test('trains a real direction v2 model and falls back to the base-rate baseline', async ({ page }) => {
  test.setTimeout(240_000);
  // Unlearnable random-walk fixture: the three-way model cannot demonstrate
  // Brier skill over the pre-evaluation base rate, so the gate must visibly
  // fall back to it. Deterministic by construction (seeded walk).
  const unlearnable = rejectedForecastSnapshot('CRSH');
  await prepare(page, 'CRSH', unlearnable);
  const trendButton = page.getByRole('button', { name: 'Trend Forecast' });
  await trendButton.click();
  await expect(trendButton).toHaveAttribute('aria-pressed', 'true');
  await page.getByRole('button', { name: 'Predict', exact: true }).click();
  await expect(page.getByText('Trend Forecast Metrics')).toBeVisible({ timeout: 200_000 });

  // Fallback notice uses the structured status label; per-day direction
  // evidence is gone under the v2 cumulative contract.
  await expect(
    page.getByText(/Experimental model did not beat persistence|base rate displayed/i).first()
  ).toBeVisible({ timeout: 30_000 });
  await expect(page.getByText('Direction by forecast day')).toHaveCount(0);

  // The decision card shows a single three-way call.
  await expect(page.locator('#statsBar')).toContainText(/Up|Down|Neutral/);
});
