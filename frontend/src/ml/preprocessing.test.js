import {
  FEATURE_NAMES,
  FEATURE_SCHEMA_VERSION,
  MODEL_VERSION,
  OUTPUT_WIDTH,
  TARGET_MODE,
  WINDOW_SIZE,
  fitRobustScaler,
  modelKey,
  prepareDirectionData,
  preparePriceData,
  resolveHorizon,
  scaleRows,
  validateSnapshot,
} from './preprocessing';

function makeSnapshot(rowCount = 200, closeValues) {
  const featureNames = FEATURE_NAMES;
  const features = Array.from({ length: rowCount }, (_, row) =>
    featureNames.map((_, column) => column === 0 ? 0.001 * row : row + column + 1),
  );
  const closes = closeValues || Array.from({ length: rowCount }, (_, row) => 100 + row);
  return {
    ticker: 'TEST',
    schema_version: FEATURE_SCHEMA_VERSION,
    snapshot_id: 'snapshot-test',
    feature_names: featureNames,
    window_size: WINDOW_SIZE,
    output_width: OUTPUT_WIDTH,
    dates: Array.from({ length: rowCount }, (_, row) => `2025-01-${String(row + 1).padStart(2, '0')}`),
    features,
    historical_prices: closes,
    future_dates: Array.from({ length: OUTPUT_WIDTH }, (_, row) => `2026-01-${String(row + 1).padStart(2, '0')}`),
    data_snapshot: {},
  };
}

test('validates the schema-v4 browser snapshot contract', () => {
  const snapshot = makeSnapshot();
  expect(() => validateSnapshot(snapshot)).not.toThrow();
  expect(() => validateSnapshot({ ...snapshot, feature_names: snapshot.feature_names.slice(1) })).toThrow(
    /schema/i,
  );
  expect(() => validateSnapshot({ ...snapshot, schema_version: 3 })).toThrow(/schema version/i);
});

test('builds 60-day sequences with exact cumulative log-return targets', () => {
  const closes = Array.from({ length: 200 }, (_, row) => 100 * Math.exp(0.001 * row));
  const prepared = preparePriceData(makeSnapshot(200, closes), undefined, 3);
  expect(prepared.inputs[0]).toHaveLength(WINDOW_SIZE);
  expect(prepared.inputs[0][0]).toHaveLength(FEATURE_NAMES.length);
  expect(prepared.targets[0]).toHaveLength(3);
  expect(prepared.horizon).toBe(3);
  const firstTarget = prepared.targets[0];
  expect(firstTarget[0]).toBeCloseTo(Math.log(closes[WINDOW_SIZE] / closes[WINDOW_SIZE - 1]), 12);
  expect(firstTarget[2]).toBeCloseTo(Math.log(closes[WINDOW_SIZE + 2] / closes[WINDOW_SIZE - 1]), 12);
  expect(prepared.origins[0]).toBeCloseTo(closes[WINDOW_SIZE - 1], 12);
  expect(prepared.trainCount).toBe(prepared.split - 3 + 1);
});

test('reconstructs prices from predicted cumulative returns starting at the latest close', () => {
  const closes = Array.from({ length: 200 }, (_, row) => 100 + row);
  const prepared = preparePriceData(makeSnapshot(200, closes), undefined, 7);
  const horizon = prepared.horizon;
  const sample = prepared.inputs.length - 1;
  const origin = prepared.origins[sample];
  const predictedReturns = [0.01, -0.005, 0.02, 0.0, 0.015, -0.01, 0.008].slice(0, horizon);
  const predictedPrices = predictedReturns.map((value) => origin * Math.exp(value));
  expect(predictedPrices[0]).toBeCloseTo(origin * Math.exp(0.01), 12);
  expect(predictedPrices[horizon - 1]).toBeCloseTo(origin * Math.exp(0.008), 12);
});

test('applies robust train-only scaling and exposes no future influence', () => {
  const snapshot = makeSnapshot(200);
  const prepared = preparePriceData(snapshot, undefined, 7);
  const trainRows = snapshot.features.slice(0, prepared.split + WINDOW_SIZE);
  const expected = fitRobustScaler(trainRows);
  expect(prepared.scaler.median[0]).toBeCloseTo(expected.median[0], 12);
  const testRowMax = Math.max(...snapshot.features.map((row) => row[0]));
  expect(prepared.scaler.iqr[0]).toBeLessThan(testRowMax);
  const scaled = scaleRows(snapshot.features, prepared.scaler);
  expect(scaled.length).toBe(snapshot.features.length);
  expect(scaled.every((row) => row.every(Number.isFinite))).toBe(true);
});

test('snaps requested days to supported horizon-specific models', () => {
  expect(resolveHorizon(1)).toBe(1);
  expect(resolveHorizon(2)).toBe(3);
  expect(resolveHorizon(3)).toBe(3);
  expect(resolveHorizon(4)).toBe(5);
  expect(resolveHorizon(7)).toBe(7);
  expect(resolveHorizon(8)).toBe(14);
  expect(resolveHorizon(14)).toBe(14);
  expect(resolveHorizon(15)).toBe(30);
  expect(resolveHorizon(30)).toBe(30);
  expect(resolveHorizon(31)).toBe(30);
});

test('includes horizon and target mode in the cache identity', () => {
  const snapshot = makeSnapshot();
  expect(modelKey(snapshot, 'price', 'research', 'webgpu', 3)).toContain(MODEL_VERSION);
  expect(modelKey(snapshot, 'price', 'research', 'webgpu', 3)).toContain(TARGET_MODE);
  expect(modelKey(snapshot, 'price', 'research', 'webgpu', 3)).toMatch(/\/4\/TEST\/price\/research\/webgpu\/[0-9a-f]{8}\/snapshot-test\/60\/3$/);
  expect(modelKey(snapshot, 'price', 'research', 'webgpu', 7)).not.toBe(modelKey(snapshot, 'price', 'research', 'webgpu', 3));
  expect(modelKey(snapshot, 'price')).not.toBe(modelKey(snapshot, 'direction'));
});

test('aligns direction returns to the shifted 60-day feature windows', () => {
  const prepared = prepareDirectionData(makeSnapshot());
  expect(prepared.inputs[0]).toHaveLength(WINDOW_SIZE);
  expect(prepared.targets[0]).toHaveLength(OUTPUT_WIDTH);
  expect(prepared.targets.flat().every((value) => value === 0 || value === 1)).toBe(true);
});

test('horizon-specific models use more samples than the fixed 30-day design', () => {
  const prepared = preparePriceData(makeSnapshot(), undefined, 3);
  const full = preparePriceData(makeSnapshot(), undefined, 30);
  expect(prepared.inputs.length).toBeGreaterThan(full.inputs.length);
});
