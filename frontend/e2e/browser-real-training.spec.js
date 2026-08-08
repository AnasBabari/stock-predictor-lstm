import { expect, test } from '@playwright/test';
import { deterministicSnapshot } from './fixtures.js';

// Real end-to-end browser training: this spec builds an actual TensorFlow.js
// model inside Chromium with the deterministic fixture. The server-forecast
// registry is intercepted (unavailable 200 => sanctioned browser fallback) so
// the app's own test harness never talks to a real backend and no request can
// leak to the local proxy (which is not running in CI).
async function mockApi(page) {
  await page.route('**/api/v1/search?query=*', (route) => route.fulfill({ json: [{ ticker: 'MSFT', name: 'Microsoft Corp.', type: 'Equity' }] }));
  await page.route('**/api/v1/server-forecasts/**', (route) =>
    route.fulfill({ json: { available: false, reason: 'missing', fallback: 'browser_training' } })
  );
  await page.route('**/api/v1/training-data?ticker=MSFT', (route) => route.fulfill({ json: deterministicSnapshot('MSFT') }));
  await page.route('**/api/v1/info?ticker=MSFT', (route) => route.fulfill({ json: { longName: 'Microsoft Corp.', sector: 'Technology' } }));
  await page.route('**/api/v1/predict**', (route) => route.fulfill({ status: 503, json: { detail: 'baseline should not be used in quick fixture' } }));
}

async function prepare(page) {
  await mockApi(page);
  await page.goto('/');
  await page.locator('#trainingProfile').selectOption('quick');
  await page.getByRole('combobox', { name: 'Search stock ticker or company name' }).fill('MSFT');
}

test('trains a real price model in the browser and reloads it from IndexedDB', async ({ page }) => {
  test.setTimeout(180_000);
  await prepare(page);
  await page.getByRole('button', { name: 'Predict', exact: true }).click();
  await expect(page.getByText('Price Forecast Metrics')).toBeVisible({ timeout: 150_000 });
  // The deterministic fixture is learnable, so the price model must be promoted
  // rather than falling back to the flat persistence baseline.
  await expect(page.getByText(/flat|baseline fallback/i)).toHaveCount(0);

  await page.reload();
  await page.getByRole('combobox', { name: 'Search stock ticker or company name' }).fill('MSFT');
  await page.getByRole('button', { name: 'Predict', exact: true }).click();
  await expect(page.getByText(/Cached on this device/i)).toBeVisible({ timeout: 30_000 });
});

test('trains a real direction model and falls back to the majority baseline', async ({ page }) => {
  test.setTimeout(180_000);
  await prepare(page);
  const trendButton = page.getByRole('button', { name: 'Trend Forecast' });
  await trendButton.click();
  await expect(trendButton).toHaveAttribute('aria-pressed', 'true');
  await page.getByRole('button', { name: 'Predict', exact: true }).click();
  await expect(page.getByText('Trend Forecast Metrics')).toBeVisible({ timeout: 150_000 });
  // The fixture's direction signal is dominated by a positive daily drift, so
  // the learned direction model cannot beat the majority-class baseline on the
  // untouched holdout: the gate must visibly fall back to the baseline.
  await expect(page.getByText(/majority class displayed/i)).toBeVisible({ timeout: 30_000 });
  // Direction evidence is also reported per forecast day.
  await expect(page.getByText('Direction by forecast day')).toBeVisible();
  await expect(page.locator('.horizon-metrics-table')).toContainText('Day');
});