import { expect, test } from '@playwright/test';
import {
  FEATURE_NAMES_V4,
  businessDatesAfter,
  deterministicSnapshot,
  serverForecastPayload,
} from './fixtures.js';

// Pure Node assertions (no browser) that lock the deterministic fixture to the
// contract the browser-training pipeline depends on. Runs in seconds as part
// of the cheap contract job alongside server-contract.spec.js.

const ROWS = 480;
const FEATURE_COUNT = FEATURE_NAMES_V4.length;

function dailyLogReturns(prices) {
  return prices.slice(1).map((price, index) => Math.log(price / prices[index]));
}

function isStrictlyIncreasing(values) {
  return values.every((value, index) => index === 0 || value > values[index - 1]);
}

function isFiniteArray(values) {
  return values.every((value) => Number.isFinite(Number(value)));
}

test('deterministic snapshot preserves the canonical feature contract', () => {
  const snapshot = deterministicSnapshot('MSFT');
  expect(snapshot.snapshot_id).toBe('MSFT-quick-fixture-v6');
  expect(snapshot.schema_version).toBe(4);
  expect(snapshot.window_size).toBe(60);
  expect(snapshot.output_width).toBe(30);
  expect(snapshot.feature_names).toEqual(FEATURE_NAMES_V4);
  expect(snapshot.feature_names.length).toBe(FEATURE_COUNT);

  expect(snapshot.features).toHaveLength(ROWS);
  expect(snapshot.dates).toHaveLength(ROWS);
  expect(snapshot.historical_prices).toHaveLength(ROWS);
  for (const row of snapshot.features) {
    expect(row).toHaveLength(FEATURE_COUNT);
    expect(row.every((value) => Number.isFinite(Number(value)))).toBe(true);
  }
  expect(snapshot.features.flat().some((value) => Number.isNaN(Number(value)))).toBe(false);
  expect(isFiniteArray(snapshot.historical_prices)).toBe(true);
  expect(snapshot.historical_prices.every((price) => Number(price) > 0)).toBe(true);
});

test('snapshot dates are strictly increasing unique business days ahead of future dates', () => {
  const snapshot = deterministicSnapshot('MSFT');
  expect(isStrictlyIncreasing(snapshot.dates)).toBe(true);
  expect(new Set(snapshot.dates).size).toBe(ROWS);
  expect(snapshot.dates.every((date) => /^\d{4}-\d{2}-\d{2}$/.test(date))).toBe(true);
  expect(snapshot.future_dates.length).toBe(30);
  expect(isStrictlyIncreasing(snapshot.future_dates)).toBe(true);
  const finalHistorical = new Date(`${snapshot.dates[snapshot.dates.length - 1]}T00:00:00Z`);
  for (const future of snapshot.future_dates) {
    expect(new Date(`${future}T00:00:00Z`).getTime()).toBeGreaterThan(finalHistorical.getTime());
  }
});

test('snapshot Return_1D feature column equals the price history 1-day returns', () => {
  const snapshot = deterministicSnapshot('MSFT');
  const returnColumn = snapshot.features.map((row) => row[FEATURE_NAMES_V4.indexOf('Return_1D')]);
  expect(returnColumn).toHaveLength(ROWS);
  expect(returnColumn[0]).toBe(0);
  const priceReturns = dailyLogReturns(snapshot.historical_prices);
  for (let t = 1; t < ROWS; t += 1) {
    expect(returnColumn[t]).toBeCloseTo(priceReturns[t - 1], 12);
  }
  expect(returnColumn.every((value) => Number.isFinite(value))).toBe(true);
});

test('snapshot carries a non-constant feature column spanning both signs', () => {
  const snapshot = deterministicSnapshot('MSFT');
  const vixColumn = snapshot.features.map((row) => row[FEATURE_NAMES_V4.indexOf('VIX_Return_1D')]);
  expect(new Set(vixColumn).size).toBeGreaterThan(1);
  expect(vixColumn.some((value) => value > 0)).toBe(true);
  expect(vixColumn.some((value) => value < 0)).toBe(true);
});

test('snapshot price history is a deterministic upward drift', () => {
  const snapshot = deterministicSnapshot('MSFT');
  const returns = dailyLogReturns(snapshot.historical_prices);
  expect(returns).toHaveLength(ROWS - 1);
  expect(returns.every((value) => Number.isFinite(value))).toBe(true);
  const mean = returns.reduce((sum, value) => sum + value, 0) / returns.length;
  expect(mean).toBeGreaterThan(0);
});

test('snapshot rows are non-constant and fully deterministic', () => {
  const first = deterministicSnapshot('MSFT');
  const second = deterministicSnapshot('MSFT');
  expect(second).toEqual(first);
  const distinctRows = new Set(first.features.map((row) => JSON.stringify(row)));
  expect(distinctRows.size).toBeGreaterThan(1);
  expect(first.features[0]).not.toEqual(first.features[1]);
});

test('server forecast payload keeps strictly increasing dates and positive prices', () => {
  const payload = serverForecastPayload('MSFT', 7);
  expect(payload.forecast_days).toBe(7);
  expect(payload.future_dates).toHaveLength(7);
  expect(payload.predicted_prices).toHaveLength(7);
  expect(isStrictlyIncreasing(payload.future_dates)).toBe(true);
  expect(isStrictlyIncreasing(payload.historical_dates)).toBe(true);
  const finalHistorical = new Date(`${payload.historical_dates[payload.historical_dates.length - 1]}T00:00:00Z`);
  expect(new Date(`${payload.future_dates[0]}T00:00:00Z`).getTime()).toBeGreaterThan(finalHistorical.getTime());
  expect(payload.predicted_prices.every((value) => Number.isFinite(value) && value > 0)).toBe(true);
  expect(payload.historical_prices.every((value) => Number.isFinite(value) && value > 0)).toBe(true);
});

test('business date generator emits only weekdays', () => {
  const dates = businessDatesAfter('2026-01-01', 60);
  for (const date of dates) {
    const weekday = new Date(`${date}T00:00:00Z`).getUTCDay();
    expect(weekday).not.toBe(0);
    expect(weekday).not.toBe(6);
  }
  expect(isStrictlyIncreasing(dates)).toBe(true);
});