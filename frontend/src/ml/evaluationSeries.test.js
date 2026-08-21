import { describe, expect, it } from 'vitest';
import { OUTPUT_WIDTH, WINDOW_SIZE, buildEvaluationSeries, sequencePartition } from './preprocessing';

function makeSnapshot(rows, horizon = 7) {
  const dates = [];
  const features = [];
  const prices = [];
  for (let i = 0; i < rows; i += 1) {
    const day = new Date(Date.UTC(2024, 0, 1 + i));
    dates.push(day.toISOString().slice(0, 10));
    features.push(new Array(28).fill(((i % 7) + 1) / 7));
    prices.push(100 + i * 0.5);
  }
  return {
    ticker: 'TEST',
    schema_version: 4,
    snapshot_id: 'snapshot-series',
    feature_names: new Array(28).fill('f').map((f, i) => `${f}${i}`),
    window_size: WINDOW_SIZE,
    output_width: OUTPUT_WIDTH,
    dates,
    features,
    historical_prices: prices,
    future_dates: Array.from({ length: OUTPUT_WIDTH }, (_, i) =>
      new Date(Date.UTC(2024, 0, rows + 1 + i)).toISOString().slice(0, 10)
    ),
  };
}

describe('buildEvaluationSeries', () => {
  it('aligns actual/model/persistence on true target dates for the final step', () => {
    const rows = 300;
    const horizon = 7;
    const snapshot = makeSnapshot(rows, horizon);
    const selection = sequencePartition(snapshot, 'price', horizon);
    // Fabricate a prepared-like object with the expected split/inputs length.
    const preparedSelection = { ...selection, inputs: { length: selection.sampleCount } };
    const count = selection.split === undefined ? 0 : selection.sampleCount - selection.split;
    const evaluated = {
      actualPrices: Array.from({ length: count }, (_, s) => [1, 2, 3, 4, 5, 6, 100 + s]),
      predictedPrices: Array.from({ length: count }, (_, s) => [1, 2, 3, 4, 5, 6, 200 + s]),
      persistencePrices: Array.from({ length: count }, (_, s) => [1, 2, 3, 4, 5, 6, 300 + s]),
    };

    const series = buildEvaluationSeries(snapshot, preparedSelection, evaluated, horizon);

    expect(series).not.toBeNull();
    expect(series.horizon).toBe(horizon);
    expect(series.step).toBe(horizon - 1);
    expect(series.metric_scope).toBe('untouched_post_purge_holdout');
    expect(series.dates.length).toBe(count);
    expect(series.actual[0]).toBe(100);
    expect(series.model[0]).toBe(200);
    expect(series.persistence[0]).toBe(300);
    // Target date of the first evaluation origin must exist in the snapshot.
    const targetRow = WINDOW_SIZE + selection.split + horizon - 1;
    expect(series.dates[0]).toBe(snapshot.dates[targetRow]);
    expect(series.truncated).toBeUndefined();
  });

  it('caps long histories to the most recent origins and reports the truncation', () => {
    const rows = 1200;
    const horizon = 5;
    const snapshot = makeSnapshot(rows, horizon);
    const selection = sequencePartition(snapshot, 'price', horizon);
    const preparedSelection = { ...selection, inputs: { length: selection.sampleCount } };
    const count = selection.sampleCount - selection.split;
    const evaluated = {
      actualPrices: Array.from({ length: count }, (_, s) => [s, s, s, s, s]),
      predictedPrices: Array.from({ length: count }, (_, s) => [s, s, s, s, s]),
      persistencePrices: Array.from({ length: count }, (_, s) => [s, s, s, s, s]),
    };
    const series = buildEvaluationSeries(snapshot, preparedSelection, evaluated, horizon, 50);
    expect(series.dates.length).toBe(50);
    expect(series.truncated).toBe(count - 50);
    // Last retained point is the latest origin.
    expect(series.model[series.model.length - 1]).toBe(count - 1);
  });

  it('returns null for missing or mismatched series data', () => {
    const snapshot = makeSnapshot(200, 3);
    const selection = sequencePartition(snapshot, 'price', 3);
    const preparedSelection = { ...selection };
    expect(buildEvaluationSeries(snapshot, preparedSelection, {}, 3)).toBeNull();
    expect(
      buildEvaluationSeries(
        snapshot,
        preparedSelection,
        { actualPrices: [[1]], predictedPrices: [[1]] },
        3
      )
    ).toBeNull();
  });
});
