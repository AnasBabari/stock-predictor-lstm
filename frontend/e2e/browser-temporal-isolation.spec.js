import { expect, test } from '@playwright/test';
import { prepareDirectionData, preparePriceData, resolveHorizon } from '../src/ml/preprocessing.js';
import { deterministicSnapshot } from './fixtures.js';

// Temporal-isolation integration suite. The fitting scaler, all training
// sequences, the holdout evaluation, the final refit, and the latest-input
// prediction window consume feature rows only up to the final refit's scaler
// boundary: (sampleCount - 1) + WINDOW_SIZE in raw coordinates. Corrupting
// every feature row strictly beyond that closed interval must leave every
// displayed metric and the stored model scaler bit-identical, because the
// worker stages nothing from those rows. The fixture uses seeded
// initialization, so two identical trainings are exactly reproducible.

async function mockApp(page, trainingData) {
  await page.route('**/api/v1/search?query=*', (route) => route.fulfill({ json: [{ ticker: 'MSFT', name: 'Microsoft Corp.', type: 'Equity' }] }));
  await page.route('**/api/v1/server-forecasts/**', (route) =>
    route.fulfill({ json: { available: false, reason: 'missing', fallback: 'browser_training' } })
  );
  await page.route('**/api/v1/training-data?ticker=MSFT', (route) => route.fulfill({ json: trainingData }));
  await page.route('**/api/v1/info?ticker=MSFT', (route) => route.fulfill({ json: { longName: 'Microsoft Corp.', sector: 'Technology' } }));
  await page.route('**/api/v1/predict**', (route) => route.fulfill({ status: 503, json: { detail: 'baseline must not be used on the isolation fixture' } }));
}

async function prepare(page, forecastType) {
  await page.goto('/');
  await page.locator('#trainingProfile').selectOption('quick');
  await page.getByRole('combobox', { name: 'Search stock ticker or company name' }).fill('MSFT');
  if (forecastType === 'trend') {
    await page.getByRole('button', { name: 'Trend Forecast' }).click();
    await expect(page.getByRole('button', { name: 'Trend Forecast' })).toHaveAttribute('aria-pressed', 'true');
  }
}

function corruptSnapshot(source, firstCorruptFeatureRow) {
  const corrupted = structuredClone(source);
  for (let row = firstCorruptFeatureRow; row < corrupted.features.length; row += 1) {
    corrupted.features[row] = corrupted.features[row].map((_, column) => {
      const magnitude = (1 + (column % 3)) * 1e4;
      return column % 2 === 0 ? magnitude : -magnitude;
    });
  }
  return corrupted;
}

async function trainAndCollectState(browser, forecastType, trainingData) {
  const context = await browser.newContext();
  const page = await context.newPage();
  try {
    await mockApp(page, trainingData);
    await prepare(page, forecastType);
    await page.getByRole('button', { name: 'Predict', exact: true }).click();
    const heading = forecastType === 'trend' ? 'Trend Forecast Metrics' : 'Price Forecast Metrics';
    await expect(page.getByText(heading)).toBeVisible({ timeout: 150_000 });
    const scaler = await page.evaluate(async () => {
      const request = indexedDB.open('stocklstm-browser-models', 1);
      const database = await new Promise((resolve, reject) => {
        request.onsuccess = () => resolve(request.result);
        request.onerror = () => reject(request.error || new Error('IndexedDB is unavailable.'));
      });
      try {
        const transaction = database.transaction('metadata', 'readonly');
        const store = transaction.objectStore('metadata');
        const entries = await new Promise((resolve, reject) => {
          const getAll = store.getAll();
          getAll.onsuccess = () => resolve(getAll.result);
          getAll.onerror = () => reject(getAll.error || new Error('Could not read model metadata.'));
        });
        const model = entries.find((entry) => entry.kind === 'model');
        return model?.scaler ?? null;
      } finally {
        database.close();
      }
    });
    return {
      metrics: await page.locator('#metricsCard .metric-value').allTextContents(),
      scaler,
    };
  } finally {
    await context.close();
  }
}

function finalRefitBoundary(snapshot, forecastType, horizon) {
  const featureSpan = forecastType === 'trend' ? snapshot.features.length - 1 : snapshot.features.length;
  const sampleCount = featureSpan - 60 - horizon + 1;
  const prepared = forecastType === 'price'
    ? preparePriceData(snapshot, sampleCount, horizon)
    : prepareDirectionData(snapshot, sampleCount, horizon);
  return prepared.scalerRawEndExclusive;
}

test('price: corruption beyond the final-refit boundary cannot move metrics', async ({ browser }) => {
  test.setTimeout(300_000);
  const clean = deterministicSnapshot('MSFT');
  const horizon = resolveHorizon(7);
  const boundary = finalRefitBoundary(clean, 'price', horizon);
  expect(boundary).toBeLessThan(clean.features.length);
  const corrupted = corruptSnapshot(clean, boundary);

  const corruptedState = await trainAndCollectState(browser, 'price', corrupted);
  expect(corruptedState.metrics.length).toBeGreaterThan(4);
  expect(corruptedState.scaler).not.toBeNull();
  const cleanState = await trainAndCollectState(browser, 'price', clean);
  expect(cleanState.metrics).toEqual(corruptedState.metrics);
  expect(cleanState.scaler.median).toEqual(corruptedState.scaler.median);
  expect(cleanState.scaler.iqr).toEqual(corruptedState.scaler.iqr);
});

test('trend: corruption beyond the shifted boundary cannot move trend metrics', async ({ browser }) => {
  test.setTimeout(300_000);
  const clean = deterministicSnapshot('MSFT');
  const horizon = resolveHorizon(7);
  const boundary = finalRefitBoundary(clean, 'trend', horizon);
  const firstCorrupt = boundary + 1;
  const corrupted = corruptSnapshot(clean, firstCorrupt);

  const corruptedState = await trainAndCollectState(browser, 'trend', corrupted);
  expect(corruptedState.metrics.length).toBeGreaterThan(0);
  expect(corruptedState.scaler).not.toBeNull();
  const cleanState = await trainAndCollectState(browser, 'trend', clean);
  expect(cleanState.metrics).toEqual(corruptedState.metrics);
  expect(cleanState.scaler.median).toEqual(corruptedState.scaler.median);
  expect(cleanState.scaler.iqr).toEqual(corruptedState.scaler.iqr);
});