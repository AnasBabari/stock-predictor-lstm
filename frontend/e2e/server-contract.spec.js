import { expect, test } from '@playwright/test';
import { deterministicSnapshot, installStubBrowserWorker, serverForecastPayload } from './fixtures.js';

// Contract spec for the hybrid path: server-pretrained forecast when the
// server responds, browser training when it falls back, and the trend->direction
// API mapping. All server APIs are intercepted and the browser-training worker
// is stubbed, so this runs anywhere in seconds without building TF.js models.

function mockApi(page, forecastHandler) {
  const calls = { server: [], trainingData: 0, search: 0 };
  page.route('**/api/v1/search?query=*', (route) => {
    calls.search += 1;
    return route.fulfill({ json: [{ ticker: 'MSFT', name: 'Microsoft Corp.', type: 'Equity' }] });
  });
  page.route('**/api/v1/info?ticker=MSFT', (route) =>
    route.fulfill({ json: { longName: 'Microsoft Corp.', sector: 'Technology' } })
  );
  page.route('**/api/v1/training-data?ticker=MSFT', (route) => {
    calls.trainingData += 1;
    return route.fulfill({ json: deterministicSnapshot('MSFT') });
  });
  page.route('**/api/v1/server-forecasts/**', (route) => {
    calls.server.push(route.request().url());
    return forecastHandler(route);
  });
  page.route('**/api/v1/predict**', (route) =>
    route.fulfill({ status: 503, json: { detail: 'baseline should not be used in fixtures' } })
  );
  return calls;
}

async function prepare(page, calls) {
  await installStubBrowserWorker(page);
  await page.goto('/');
  await page.locator('#trainingProfile').selectOption('quick');
  await page.getByRole('combobox', { name: 'Search stock ticker or company name' }).fill('MSFT');
  return calls;
}

test('server price forecast is used verbatim and browser training is skipped', async ({ page }) => {
  test.setTimeout(30_000);
  const calls = mockApi(page, (route) => route.fulfill({ json: serverForecastPayload('MSFT', 7) }));
  await prepare(page, calls);

  await page.getByRole('button', { name: 'Predict', exact: true }).click();
  await expect(page.getByText('Price Forecast Metrics')).toBeVisible({ timeout: 60_000 });
  await expect(page.getByText(/server_pretrained|server-pretrained/i).first()).toBeVisible({ timeout: 15_000 });

  expect(calls.server).toHaveLength(1);
  expect(calls.server[0]).toContain('forecast_type=price');
  // The server answered; the browser must never fetch training data or train.
  expect(calls.trainingData).toBe(0);
});

test('server fallback (missing) sends the request to browser training', async ({ page }) => {
  test.setTimeout(60_000);
  const calls = mockApi(page, (route) =>
    route.fulfill({ json: { available: false, reason: 'missing', fallback: 'browser_training' } })
  );
  await prepare(page, calls);

  await page.getByRole('button', { name: 'Predict', exact: true }).click();
  await expect(page.getByText('Price Forecast Metrics')).toBeVisible({ timeout: 150_000 });
  expect(calls.trainingData).toBe(1);
});

test('server 503 (infrastructure, hybrid) degrades to browser training', async ({ page }) => {
  test.setTimeout(60_000);
  const calls = mockApi(page, (route) =>
    route.fulfill({
      status: 503,
      json: {
        detail: {
          available: false,
          code: 'registry_unavailable',
          message: 'Server forecast infrastructure is unavailable.',
          fallback: 'browser_training',
        },
      },
    })
  );
  await prepare(page, calls);

  await page.getByRole('button', { name: 'Predict', exact: true }).click();
  await expect(page.getByText('Price Forecast Metrics')).toBeVisible({ timeout: 150_000 });
  expect(calls.trainingData).toBe(1);
});

test('server 503 without a browser fallback (server_pretrained) surfaces an error, never trains', async ({ page }) => {
  test.setTimeout(60_000);
  const calls = mockApi(page, (route) =>
    route.fulfill({
      status: 503,
      json: {
        detail: {
          available: false,
          code: 'signature_verification_failed',
          message: 'Server forecast bundle verification failed.',
          fallback: null,
        },
      },
    })
  );
  await prepare(page, calls);

  await page.getByRole('button', { name: 'Predict', exact: true }).click();
  await expect(page.getByText(/bundle verification failed|Server forecast/i).first()).toBeVisible({
    timeout: 15_000,
  });
  // Fail closed: the browser must NOT silently take over training.
  expect(calls.trainingData).toBe(0);
});

test('UI trend request maps to API direction and lands in the browser trend path', async ({ page }) => {
  test.setTimeout(60_000);
  let requestedUrl = null;
  const calls = mockApi(page, (route) => {
    requestedUrl = route.request().url();
    return route.fulfill({
      json: { available: false, reason: 'unsupported_forecast_type', fallback: 'browser_training' },
    });
  });
  await prepare(page, calls);

  const trendButton = page.getByRole('button', { name: 'Trend Forecast' });
  await trendButton.click();
  await expect(trendButton).toHaveAttribute('aria-pressed', 'true');
  await page.getByRole('button', { name: 'Predict', exact: true }).click();

  await expect(page.getByText('Trend Forecast Metrics')).toBeVisible({ timeout: 150_000 });
  // The wire contract is "direction"; the server response must not be a price forecast.
  expect(requestedUrl).toContain('forecast_type=direction');
  expect(calls.trainingData).toBe(1);
});