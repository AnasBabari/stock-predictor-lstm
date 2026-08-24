import { expect, test } from '@playwright/test';
import { rejectedForecastSnapshot } from './fixtures.js';

// End-to-end contract for a REJECTED price forecast (overhaul slice 1/2):
// the decision path must be labelled as the no-change baseline, the raw
// learned path must stay visible as a diagnostic, and the reloaded cached
// artifact must be identified as rejected evidence — never as an active
// browser artifact.
async function mockApi(page) {
  await page.route('**/api/v1/search?query=*', (route) => route.fulfill({ json: [{ ticker: 'IGC', name: 'ICG Corp.', type: 'Equity' }] }));
  await page.route('**/api/v1/server-forecasts/**', (route) =>
    route.fulfill({ json: { available: false, reason: 'missing', fallback: 'browser_training' } })
  );
  await page.route('**/api/v1/training-data?ticker=IGC', (route) => route.fulfill({ json: rejectedForecastSnapshot('IGC') }));
  await page.route('**/api/v1/info?ticker=IGC', (route) => route.fulfill({ json: { longName: 'ICG Corp.', sector: 'Consumer' } }));
  await page.route('**/api/v1/predict**', (route) => route.fulfill({ status: 503, json: { detail: 'baseline must come from the rejected local model in this spec' } }));
}

async function prepare(page) {
  await mockApi(page);
  await page.goto('/');
  await page.locator('#trainingProfile').selectOption('quick');
  await page.getByRole('combobox', { name: 'Search stock ticker or company name' }).fill('IGC');
}

async function readPriceModelMetadata(page) {
  return page.evaluate(async () => {
    const request = indexedDB.open('stocklstm-browser-models', 1);
    const database = await new Promise((resolve, reject) => {
      request.onsuccess = () => resolve(request.result);
      request.onerror = () => reject(request.error || new Error('IndexedDB unavailable.'));
    });
    try {
      const transaction = database.transaction('metadata', 'readonly');
      const entries = await new Promise((resolve, reject) => {
        const getAll = transaction.objectStore('metadata').getAll();
        getAll.onsuccess = () => resolve(getAll.result);
        getAll.onerror = () => reject(getAll.error || new Error('metadata read failed.'));
      });
      return (
        entries.find((entry) => entry.kind === 'model' && entry.forecast_type === 'price') || null
      );
    } finally {
      database.close();
    }
  });
}

test('rejected price model: model forecast primary, experimental badge displayed, benchmark hidden by default', async ({ page }) => {
  test.setTimeout(240_000);

  // ── First run: train on the unlearnable fixture ────────────────────
  await prepare(page);
  await page.getByRole('button', { name: 'Predict', exact: true }).click();
  await expect(page.getByText('Price Forecast Metrics')).toBeVisible({ timeout: 200_000 });

  // Validation badge reflects EXPERIMENTAL status without replacing the forecast with a flat baseline
  await expect(
    page.getByRole('status').filter({ hasText: /EXPERIMENTAL/i }).first()
  ).toBeVisible({ timeout: 30_000 });

  // Model forecast is primary and rendered; persistence benchmark is hidden by default
  const benchmarkCheckbox = page.getByRole('checkbox', { name: /Show benchmark/i });
  await expect(benchmarkCheckbox).toBeVisible();
  await expect(benchmarkCheckbox).not.toBeChecked();

  const firstRun = await readPriceModelMetadata(page);
  expect(firstRun).not.toBeNull();
  expect(firstRun.promotion_summary?.promoted).toBe(false);

  // ── Reload: cache hit must preserve experimental validation status ───────────
  await page.reload();
  await prepare(page);
  await page.getByRole('button', { name: 'Predict', exact: true }).click();
  await expect(page.getByText('Price Forecast Metrics')).toBeVisible({ timeout: 60_000 });

  const reloaded = await readPriceModelMetadata(page);
  expect(reloaded).not.toBeNull();
  expect(reloaded.promotion_summary?.promoted).toBe(false);

  // Badge persists across reload
  await expect(
    page.getByRole('status').filter({ hasText: /EXPERIMENTAL/i }).first()
  ).toBeVisible();
});
