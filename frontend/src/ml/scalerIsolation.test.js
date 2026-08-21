import {
  FEATURE_NAMES,
  FEATURE_SCHEMA_VERSION,
  OUTPUT_WIDTH,
  WINDOW_SIZE,
  fitRobustScaler,
  fittingScalerBounds,
  prepareDirectionData,
  preparePriceData,
} from './preprocessing';

function makeSnapshot(rowCount = 200) {
  const features = Array.from({ length: rowCount }, (_, row) =>
    FEATURE_NAMES.map((_, column) => column === 0 ? 0.001 * row : row + column + 1),
  );
  return {
    ticker: 'TEST',
    schema_version: FEATURE_SCHEMA_VERSION,
    snapshot_id: 'snapshot-test',
    feature_names: FEATURE_NAMES,
    window_size: WINDOW_SIZE,
    output_width: OUTPUT_WIDTH,
    dates: Array.from({ length: rowCount }, (_, row) => `2025-01-${String(row + 1).padStart(2, '0')}`),
    features,
    historical_prices: Array.from({ length: rowCount }, (_, row) => 100 + row),
    future_dates: Array.from({ length: OUTPUT_WIDTH }, (_, row) => `2026-01-${String(row + 1).padStart(2, '0')}`),
    data_snapshot: {},
  };
}

function snapshotWithCorruptedRows(source, startRow, pattern) {
  const snapshot = structuredClone(source);
  for (let row = startRow; row < snapshot.features.length; row += 1) {
    snapshot.features[row] = FEATURE_NAMES.map((_, column) => pattern(column));
  }
  return snapshot;
}

function snapshotWithCorruptedRange(source, fromRow, toRowExclusive, pattern) {
  const snapshot = structuredClone(source);
  for (let row = fromRow; row < Math.min(toRowExclusive, snapshot.features.length); row += 1) {
    snapshot.features[row] = FEATURE_NAMES.map((_, column) => pattern(column));
  }
  return snapshot;
}

const ADVERSARIAL_PATTERNS = [
  { name: 'huge positive', apply: () => 1e6 },
  { name: 'huge negative', apply: () => -1e6 },
  { name: 'extreme positive', apply: () => 1e9 },
  { name: 'extreme negative', apply: () => -1e9 },
  { name: 'absurd positive', apply: () => 1e12 },
  { name: 'absurd negative', apply: () => -1e12 },
  { name: 'alternating signs', apply: (column) => (column % 2 === 0 ? -1e9 : 1e9) },
];

function captureFittingState(prepared) {
  return {
    scalerMedian: prepared.scaler.median,
    scalerIqr: prepared.scaler.iqr,
    fitSequenceStart: prepared.fitSequenceStart,
    fitSequenceEndExclusive: prepared.fitSequenceEndExclusive,
    fitSequenceCount: prepared.fitSequenceCount,
    scalerRawStart: prepared.scalerRawStart,
    scalerRawEndExclusive: prepared.scalerRawEndExclusive,
    trainCount: prepared.trainCount,
    split: prepared.split,
    fittingInputs: prepared.inputs.slice(0, prepared.fitSequenceEndExclusive),
    fittingTargets: prepared.targets.slice(0, prepared.fitSequenceEndExclusive),
  };
}

function assertScalerStateUnchanged(corrupted, baseline) {
  expect(corrupted.scaler.median).toEqual(baseline.scalerMedian);
  expect(corrupted.scaler.iqr).toEqual(baseline.scalerIqr);
  expect(corrupted.fitSequenceStart).toBe(baseline.fitSequenceStart);
  expect(corrupted.fitSequenceEndExclusive).toBe(baseline.fitSequenceEndExclusive);
  expect(corrupted.fitSequenceCount).toBe(baseline.fitSequenceCount);
  expect(corrupted.scalerRawStart).toBe(baseline.scalerRawStart);
  expect(corrupted.scalerRawEndExclusive).toBe(baseline.scalerRawEndExclusive);
  expect(corrupted.trainCount).toBe(baseline.trainCount);
  expect(corrupted.split).toBe(baseline.split);
  expect(corrupted.inputs.slice(0, corrupted.fitSequenceEndExclusive)).toEqual(baseline.fittingInputs);
  expect(corrupted.targets.slice(0, corrupted.fitSequenceEndExclusive)).toEqual(baseline.fittingTargets);
}

test('future feature rows cannot move holdout scaler state for price', () => {
  for (const horizon of [1, 3, 30]) {
    const baseline = captureFittingState(preparePriceData(makeSnapshot(300), undefined, horizon));
    for (const pattern of ADVERSARIAL_PATTERNS) {
      const corrupted = snapshotWithCorruptedRows(
        makeSnapshot(300),
        baseline.scalerRawEndExclusive,
        pattern.apply,
      );
      assertScalerStateUnchanged(preparePriceData(corrupted, undefined, horizon), baseline);
    }
  }
});

test('future feature rows cannot move holdout scaler state for direction', () => {
  for (const horizon of [1, 3, 30]) {
    const baseline = captureFittingState(prepareDirectionData(makeSnapshot(300), undefined, horizon));
    const firstExcludedFeatureRow = baseline.scalerRawEndExclusive + 1;
    for (const pattern of ADVERSARIAL_PATTERNS) {
      const corrupted = snapshotWithCorruptedRows(
        makeSnapshot(300),
        firstExcludedFeatureRow,
        pattern.apply,
      );
      assertScalerStateUnchanged(prepareDirectionData(corrupted, undefined, horizon), baseline);
    }
  }
});

test('future feature rows cannot move research fold scaler state', () => {
  for (const horizon of [5, 30]) {
    const snapshot = makeSnapshot(400);
    const priceBaseline = captureFittingState(preparePriceData(snapshot, 300, horizon));
    const directionBaseline = captureFittingState(prepareDirectionData(snapshot, 300, horizon));
    expect(priceBaseline.scalerRawEndExclusive).toBe(299 + WINDOW_SIZE);
    expect(directionBaseline.scalerRawEndExclusive).toBe(299 + WINDOW_SIZE);
    for (const pattern of ADVERSARIAL_PATTERNS) {
      const priceCorrupted = snapshotWithCorruptedRows(
        makeSnapshot(400),
        priceBaseline.scalerRawEndExclusive,
        pattern.apply,
      );
      assertScalerStateUnchanged(preparePriceData(priceCorrupted, 300, horizon), priceBaseline);
      const directionCorrupted = snapshotWithCorruptedRows(
        makeSnapshot(400),
        directionBaseline.scalerRawEndExclusive + 1,
        pattern.apply,
      );
      assertScalerStateUnchanged(prepareDirectionData(directionCorrupted, 300, horizon), directionBaseline);
    }
  }
});

test('direction matrix shift consumes the same feature rows as price', () => {
  const snapshot = makeSnapshot(300);
  const price = preparePriceData(snapshot, undefined, 7);
  const direction = prepareDirectionData(snapshot, undefined, 7);
  expect(direction.scalerRawEndExclusive + 1).toBe(price.scalerRawEndExclusive);
  expect(price.scalerRawEndExclusive - price.fitSequenceEndExclusive)
    .toBe(direction.scalerRawEndExclusive - direction.fitSequenceEndExclusive);
});

test('corrupting every fitting raw row moves the scaler, corrupted excluded rows cannot', () => {
  for (const horizon of [1, 30]) {
    const snapshot = makeSnapshot(300);
    const priceBaseline = preparePriceData(snapshot, undefined, horizon);
    const directionBaseline = prepareDirectionData(snapshot, undefined, horizon);
    for (const prepare of [
      { fn: preparePriceData, reference: priceBaseline, scalerRawStart: priceBaseline.scalerRawStart, scalerRawEndExclusive: priceBaseline.scalerRawEndExclusive },
      { fn: prepareDirectionData, reference: directionBaseline, scalerRawStart: directionBaseline.scalerRawStart + 1, scalerRawEndExclusive: directionBaseline.scalerRawEndExclusive + 1 },
    ]) {
      const inside = snapshotWithCorruptedRange(
        snapshot,
        prepare.scalerRawStart,
        prepare.scalerRawEndExclusive,
        () => 1e12,
      );
      const outside = snapshotWithCorruptedRange(
        snapshot,
        prepare.scalerRawEndExclusive,
        snapshot.features.length,
        () => -1e12,
      );
      const moved = prepare.fn(inside, undefined, horizon);
      expect(moved.scaler.median[0]).not.toBe(prepare.reference.scaler.median[0]);
      const unchanged = prepare.fn(outside, undefined, horizon);
      expect(unchanged.scaler.median).toEqual(prepare.reference.scaler.median);
      expect(unchanged.scaler.iqr).toEqual(prepare.reference.scaler.iqr);
    }
  }
});

test('non-zero-origin research fold bounds keep the fold scaler isolated', () => {
  const source = makeSnapshot(400);
  const bounds = fittingScalerBounds(50, 200);
  expect(bounds.scalerRawStart).toBe(50);
  expect(bounds.scalerRawEndExclusive).toBe(199 + WINDOW_SIZE);
  const foldScaler = fitRobustScaler(
    source.features.slice(bounds.scalerRawStart, bounds.scalerRawEndExclusive),
  );
  for (const pattern of [() => 1e12, () => -1e9, (column) => (column % 2 === 0 ? -1e9 : 1e6)]) {
    const prefixOnly = snapshotWithCorruptedRange(source, 0, bounds.scalerRawStart, pattern);
    const suffixOnly = snapshotWithCorruptedRange(source, bounds.scalerRawEndExclusive, source.features.length, pattern);
    const bothSides = snapshotWithCorruptedRange(suffixOnly, 0, bounds.scalerRawStart, pattern);
    for (const corrupted of [prefixOnly, suffixOnly, bothSides]) {
      const recalculated = fitRobustScaler(
        corrupted.features.slice(bounds.scalerRawStart, bounds.scalerRawEndExclusive),
      );
      expect(recalculated.median).toEqual(foldScaler.median);
      expect(recalculated.iqr).toEqual(foldScaler.iqr);
    }
  }
});
