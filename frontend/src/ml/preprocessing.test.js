import {
  FEATURE_NAMES,
  inverseClose,
  modelKey,
  prepareDirectionData,
  preparePriceData,
  validateSnapshot,
} from './preprocessing';

function makeSnapshot(rowCount = 140) {
  const featureNames = FEATURE_NAMES;
  const features = Array.from({ length: rowCount }, (_, row) =>
    featureNames.map((_, column) => column === 3 ? 100 + row : row + column + 1),
  );
  return {
    ticker: 'TEST',
    schema_version: 3,
    snapshot_id: 'snapshot-test',
    feature_names: featureNames,
    window_size: 60,
    output_width: 30,
    close_index: 3,
    dates: Array.from({ length: rowCount }, (_, row) => `2025-01-${String(row + 1).padStart(2, '0')}`),
    features,
    historical_prices: features.map((row) => row[3]),
    future_dates: Array.from({ length: 30 }, (_, row) => `2026-01-${String(row + 1).padStart(2, '0')}`),
    data_snapshot: {},
  };
}

test('validates the 22-feature browser snapshot contract', () => {
  const snapshot = makeSnapshot();
  expect(() => validateSnapshot(snapshot)).not.toThrow();
  expect(() => validateSnapshot({ ...snapshot, feature_names: snapshot.feature_names.slice(1) })).toThrow(
    /schema/i,
  );
});

test('builds 60-day price sequences with a purged training boundary', () => {
  const prepared = preparePriceData(makeSnapshot());
  expect(prepared.inputs[0]).toHaveLength(60);
  expect(prepared.inputs[0][0]).toHaveLength(22);
  expect(prepared.targets[0]).toHaveLength(30);
  expect(prepared.trainCount).toBe(prepared.split - 30 + 1);
  expect(prepared.scaler.max[3]).toBe(199);
  expect(prepared.scaled[prepared.split + 60][3]).toBeGreaterThan(1);
});

test('aligns direction returns to the shifted 60-day feature windows', () => {
  const prepared = prepareDirectionData(makeSnapshot());
  expect(prepared.inputs[0]).toHaveLength(60);
  expect(prepared.targets[0]).toHaveLength(30);
  expect(prepared.targets.flat().every((value) => value === 0 || value === 1)).toBe(true);
});

test('uses the close column for inverse scaling and cache identity', () => {
  const snapshot = makeSnapshot();
  const prepared = preparePriceData(snapshot);
  expect(inverseClose(0, prepared.scaler, 3)).toBe(prepared.scaler.min[3]);
  expect(modelKey(snapshot, 'price')).toContain('tfjs-lstm-v1/3/TEST/price/snapshot-test/60/30');
  expect(modelKey(snapshot, 'price')).not.toBe(modelKey(snapshot, 'direction'));
});
