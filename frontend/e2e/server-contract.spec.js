import { expect, test } from '@playwright/test';
import { activeVolatilityForecastPayload } from './fixtures.js';

function mockVolatilityApi(page, forecastHandler) {
  const calls = { forecast: [], search: 0, info: 0 };
  page.route('**/api/v1/search?query=*', (route) => {
    calls.search += 1;
    return route.fulfill({ json: [{ ticker: 'MSFT', name: 'Microsoft Corp.', type: 'Equity' }] });
  });
  page.route('**/api/v1/info?ticker=MSFT', (route) => {
    calls.info += 1;
    return route.fulfill({ json: { longName: 'Microsoft Corp.', sector: 'Technology' } });
  });
  page.route('**/api/v1/volatility/forecast?*', (route) => {
    calls.forecast.push(route.request().url());
    return forecastHandler(route);
  });
  return calls;
}

test('active volatility baseline is rendered with an honest scenario range', async ({ page }) => {
  test.setTimeout(30_000);
  const calls = mockVolatilityApi(page, (route) =>
    route.fulfill({ json: activeVolatilityForecastPayload('MSFT', 5) })
  );
  await page.goto('/');
  await page.getByRole('combobox', { name: 'Search stock ticker or company name' }).fill('MSFT');
  await page.getByRole('button', { name: 'Predict', exact: true }).click();

  await expect(page.locator('#metricsCard')).toBeVisible({ timeout: 15_000 });
  await expect(page.getByText('BASELINE').first()).toBeVisible({ timeout: 10_000 });
  await expect(page.getByText(/Historical vs Causal Volatility Scenario/i)).toBeVisible({ timeout: 10_000 });

  expect(calls.forecast).toHaveLength(1);
  expect(calls.forecast[0]).toContain('ticker=MSFT');
  expect(calls.forecast[0]).toContain('horizon=5');
});

test('server 503 abstention surfaces truthful message and does not substitute a baseline', async ({ page }) => {
  test.setTimeout(30_000);
  const calls = mockVolatilityApi(page, (route) =>
    route.fulfill({
      status: 503,
      json: {
        detail: {
          code: 'abstain_no_certified_model',
          message: 'No certified global model is available yet.',
        },
      },
    })
  );
  await page.goto('/');
  await page.getByRole('combobox', { name: 'Search stock ticker or company name' }).fill('MSFT');
  await page.getByRole('button', { name: 'Predict', exact: true }).click();

  await expect(
    page.getByText(/legacy global model is unavailable|active volatility forecast/i).first()
  ).toBeVisible({ timeout: 15_000 });

  expect(calls.forecast).toHaveLength(1);
});

test('server integrity failure fails closed with error toast/message', async ({ page }) => {
  test.setTimeout(30_000);
  const calls = mockVolatilityApi(page, (route) =>
    route.fulfill({
      status: 503,
      json: {
        detail: {
          code: 'artifact_integrity_failure',
          message: 'Release digest mismatch.',
        },
      },
    })
  );
  await page.goto('/');
  await page.getByRole('combobox', { name: 'Search stock ticker or company name' }).fill('MSFT');
  await page.getByRole('button', { name: 'Predict', exact: true }).click();

  await expect(
    page.getByText(/volatility forecast service is temporarily unavailable|integrity failure/i).first()
  ).toBeVisible({ timeout: 15_000 });

  expect(calls.forecast).toHaveLength(1);
});

test('uncertified horizon returns 503 and displays clean error without baseline fallback', async ({ page }) => {
  test.setTimeout(30_000);
  const calls = mockVolatilityApi(page, (route) =>
    route.fulfill({
      status: 503,
      json: {
        detail: {
          code: 'certified_horizon_unavailable',
          message: 'The selected horizon is not certified.',
        },
      },
    })
  );
  await page.goto('/');
  await page.getByRole('combobox', { name: 'Search stock ticker or company name' }).fill('MSFT');
  await page.getByRole('button', { name: 'Predict', exact: true }).click();

  await expect(
    page.getByText(/selected volatility horizon is not available/i).first()
  ).toBeVisible({ timeout: 15_000 });

  expect(calls.forecast).toHaveLength(1);
});
