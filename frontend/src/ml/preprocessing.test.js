import {
  FEATURE_NAMES,
  FEATURE_SCHEMA_VERSION,
  MODEL_VERSION,
  OUTPUT_WIDTH,
  TARGET_MODE,
  WINDOW_SIZE,
  fitRobustScaler,
  fittingScalerBounds,
  modelKey,
  prepareDirectionData,
  preparePriceData,
  resolveHorizon,
  scaleRows,
  sequencePartition,
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
  const trainRows = snapshot.features.slice(0, (prepared.trainCount - 1) + WINDOW_SIZE);
  const expected = fitRobustScaler(trainRows);
  expect(prepared.scaler.median[0]).toBeCloseTo(expected.median[0], 12);
  const testRowMax = Math.max(...snapshot.features.map((row) => row[0]));
  expect(prepared.scaler.iqr[0]).toBeLessThan(testRowMax);
  const scaled = scaleRows(snapshot.features, prepared.scaler);
  expect(scaled.length).toBe(snapshot.features.length);
  expect(scaled.every((row) => row.every(Number.isFinite))).toBe(true);
});

test('computes exact zero-origin scaler bounds from fitting sequence indices', () => {
  const sequenceCount = 101;
  const bounds = fittingScalerBounds(0, sequenceCount);
  expect(bounds.fitSequenceStart).toBe(0);
  expect(bounds.fitSequenceEndExclusive).toBe(sequenceCount);
  expect(bounds.fitSequenceCount).toBe(sequenceCount);
  expect(bounds.scalerRawStart).toBe(0);
  expect(bounds.scalerRawEndExclusive).toBe(sequenceCount - 1 + WINDOW_SIZE);
  expect(bounds.hasFittingSequences).toBe(true);

  const prepared = preparePriceData(makeSnapshot(200), undefined, 7);
  expect(prepared.fitSequenceEndExclusive).toBe(prepared.trainCount);
  expect(prepared.scalerRawEndExclusive).toBe(prepared.trainCount - 1 + WINDOW_SIZE);
  const expanded = fitRobustScaler(
    makeSnapshot(200).features.slice(0, prepared.scalerRawEndExclusive),
  );
  expect(prepared.scaler.median).toEqual(expanded.median);
  expect(prepared.scaler.iqr).toEqual(expanded.iqr);
});

test('supports non-zero-origin research fold boundaries', () => {
  const first = fittingScalerBounds(50, 110);
  expect(first.scalerRawStart).toBe(50);
  expect(first.scalerRawEndExclusive).toBe(109 + WINDOW_SIZE);
  const second = fittingScalerBounds(180, 380);
  expect(second.scalerRawEndExclusive).toBe(379 + WINDOW_SIZE);
  expect(second.hasFittingSequences).toBe(true);
});

test('handles empty fitting ranges without arithmetic', () => {
  const empty = fittingScalerBounds(5, 5);
  expect(empty.hasFittingSequences).toBe(false);
  expect(empty.fitSequenceCount).toBe(0);
  expect(empty.scalerRawEndExclusive).toBe(5);
  expect(fittingScalerBounds(10, 4).hasFittingSequences).toBe(false);
  expect(() => preparePriceData(makeSnapshot(200), 0, 7)).toThrow(/No fitting sequences/);
});

test('keeps short and long horizon scaler boundaries inside the fitting partition', () => {
  for (const horizon of [1, 30]) {
    const prepared = preparePriceData(makeSnapshot(300), undefined, horizon);
    expect(prepared.trainCount).toBe(prepared.split - horizon + 1);
    expect(prepared.scalerRawEndExclusive).toBe((prepared.trainCount - 1) + WINDOW_SIZE);
    expect(prepared.scalerRawEndExclusive).toBeGreaterThan(prepared.split);
    expect(prepared.scalerRawEndExclusive).toBeLessThan(prepared.split + WINDOW_SIZE);
  }
});

test('last included raw row changes the scaler, first excluded row cannot', () => {
  const prepared = preparePriceData(makeSnapshot(200), undefined, 7);
  const lastIncluded = prepared.scalerRawEndExclusive - 1;
  const firstExcluded = prepared.scalerRawEndExclusive;
  const expanded = fitRobustScaler(makeSnapshot(200).features.slice(0, prepared.scalerRawEndExclusive));
  expect(prepared.scaler.median).toEqual(expanded.median);
  const altered = makeSnapshot(200);
  altered.features[lastIncluded] = altered.features[lastIncluded].map(() => 0);
  const changed = preparePriceData(altered, undefined, 7);
  expect(changed.scaler.median[0]).not.toBe(prepared.scaler.median[0]);
  const untouched = makeSnapshot(200);
  untouched.features[firstExcluded] = untouched.features[firstExcluded].map(() => 0);
  const unchanged = preparePriceData(untouched, undefined, 7);
  expect(unchanged.scaler.median[0]).toBe(prepared.scaler.median[0]);
  expect(unchanged.scaler.iqr[0]).toBe(prepared.scaler.iqr[0]);
});

test('direction scaler uses shifted matrix rows within the same sequence bounds', () => {
  const snapshot = makeSnapshot(200);
  const direction = prepareDirectionData(snapshot, undefined, 7);
  expect(direction.scalerRawEndExclusive).toBe(direction.trainCount - 1 + WINDOW_SIZE);
  const rawRows = snapshot.features
    .slice(1, direction.scalerRawEndExclusive + 1)
    .map((row) => row.map((value) => Number(value)));
  const expected = fitRobustScaler(rawRows);
  expect(direction.scaler.median).toEqual(expected.median);
});

test('supports an explicit fitting end (research fold) in both coordinates', () => {
  for (const prepare of [preparePriceData, prepareDirectionData]) {
    const snapshot = makeSnapshot(400);
    const prepared = prepare(snapshot, 300, 5);
    expect(prepared.fitSequenceEndExclusive).toBe(300);
    expect(prepared.scalerRawEndExclusive).toBe(299 + WINDOW_SIZE);
    expect(prepared.scalerRawEndExclusive).toBeLessThan(prepared.inputs.length + WINDOW_SIZE);
  }
});

test('reports partition counts in the actual sequence coordinates', () => {
  const price = sequencePartition(makeSnapshot(300), 'price', 7);
  const direction = sequencePartition(makeSnapshot(300), 'direction', 7);
  expect(direction.sampleCount).toBe(price.sampleCount - 1);
  expect(direction.trainCount).toBe(price.trainCount - 1);
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

test('produces horizon-specific target widths for every sample', () => {
  const snapshot = makeSnapshot();
  const prepared = preparePriceData(snapshot, undefined, 5);
  expect(prepared.horizon).toBe(5);
  expect(prepared.targets.length).toBeGreaterThan(0);
  expect(prepared.targets.every((target) => target.length === 5)).toBe(true);
  const full = preparePriceData(snapshot);
  expect(full.horizon).toBe(OUTPUT_WIDTH);
  expect(full.targets.every((target) => target.length === OUTPUT_WIDTH)).toBe(true);
});

test('horizon-specific models use more samples than the fixed 30-day design', () => {
  const prepared = preparePriceData(makeSnapshot(), undefined, 3);
  const full = preparePriceData(makeSnapshot(), undefined, 30);
  expect(prepared.inputs.length).toBeGreaterThan(full.inputs.length);
});
