import { expect, test } from '@playwright/test';
import { deterministicSnapshot } from './fixtures.js';

async function mockApi(page) {
  await page.route('**/api/v1/search?query=*', (route) => route.fulfill({ json: [{ ticker: 'MSFT', name: 'Microsoft Corp.', type: 'Equity' }] }));
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

test('Vercel preview trains a price model and reloads it from IndexedDB', async ({ page }) => {
  test.setTimeout(180_000);
  await prepare(page);
  await page.getByRole('button', { name: 'Predict', exact: true }).click();
  await expect(page.getByText('Price Forecast Metrics')).toBeVisible({ timeout: 150_000 });
  await expect(page.getByText(/flat|baseline fallback/i)).toHaveCount(0);

  await page.reload();
  await page.getByRole('combobox', { name: 'Search stock ticker or company name' }).fill('MSFT');
  await page.getByRole('button', { name: 'Predict', exact: true }).click();
  await expect(page.getByText(/Cached on this device/i)).toBeVisible({ timeout: 30_000 });
});

test('Vercel preview trains an independent direction model', async ({ page }) => {
  test.setTimeout(180_000);
  await prepare(page);
  const trendButton = page.getByRole('button', { name: 'Trend Forecast' });
  await trendButton.click();
  await expect(trendButton).toHaveAttribute('aria-pressed', 'true');
  await page.getByRole('button', { name: 'Predict', exact: true }).click();
  await expect(page.getByText('Trend Forecast Metrics')).toBeVisible({ timeout: 150_000 });
  // The synthetic fixture has an almost-perfect up-day majority, so the learned
  // direction model cannot beat the majority-class baseline: the gate must
  // visibly fall back to the baseline forecast.
  await expect(page.getByText(/majority class displayed/i)).toBeVisible({ timeout: 30_000 });
});