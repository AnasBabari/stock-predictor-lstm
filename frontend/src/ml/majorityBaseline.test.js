import {
  FEATURE_NAMES,
  FEATURE_SCHEMA_VERSION,
  OUTPUT_WIDTH,
  WINDOW_SIZE,
  prepareDirectionData,
} from './preprocessing';
import { classificationMetrics, directionMajority } from './evaluation';

const ROW_COUNT = 300;

function makeSnapshot(prices) {
  return {
    ticker: 'TEST',
    schema_version: FEATURE_SCHEMA_VERSION,
    snapshot_id: 'snapshot-majority',
    feature_names: FEATURE_NAMES,
    window_size: WINDOW_SIZE,
    output_width: OUTPUT_WIDTH,
    dates: Array.from({ length: ROW_COUNT }, (_, index) => `day-${index}`),
    features: Array.from({ length: ROW_COUNT }, (_, row) =>
      FEATURE_NAMES.map((_, column) => column === 0 ? 0.001 * row : row + column + 1)),
    historical_prices: prices,
    future_dates: Array.from({ length: OUTPUT_WIDTH }, (_, index) => `future-${index}`),
  };
}

function baselinePrices() {
  const prices = [100];
  for (let index = 1; index < ROW_COUNT; index += 1) {
    prices.push(prices[index - 1] * (index % 3 === 0 ? 1.02 : 0.98));
  }
  return prices;
}

test('reports the positive class when and only when its rate clears half', () => {
  expect(directionMajority([[1], [1], [0]])).toEqual({ label: 1, rate: 2 / 3 });
  expect(directionMajority([[0], [0], [1]])).toEqual({ label: 0, rate: 1 / 3 });
  expect(directionMajority([[1], [0]])).toEqual({ label: 1, rate: 0.5 });
  expect(directionMajority([])).toEqual({ label: 1, rate: 0.5 });
});

test('classification metrics honor an externally supplied majority label', () => {
  const actual = [[1, 0], [1, 1]];
  const predicted = [[1.4, -0.2], [0.8, 0.2]];
  const inSet = classificationMetrics(actual, predicted, 'browser_purged_holdout');
  expect(inSet.naive_baseline).toBe(0.75);
  const preEvaluation = classificationMetrics(actual, predicted, 'browser_purged_holdout', 0);
  expect(preEvaluation.naive_baseline).toBe(0.25);
  expect(preEvaluation.accuracy).toBe(inSet.accuracy);
});

test('majority is derived from pre-evaluation labels, never the evaluation window', () => {
  const baseline = prepareDirectionData(makeSnapshot(baselinePrices()), undefined, 7);
  const split = baseline.split;

  // Corrupt prices strictly after the last price any pre-split training
  // sequence can reference (sequence rows shift by WINDOW_SIZE inside target
  // construction), so only evaluation-window labels can change.
  const flipped = baselinePrices();
  const flipFrom = WINDOW_SIZE + split + 7;
  for (let row = flipFrom; row < ROW_COUNT; row += 1) flipped[row] *= 1.5;
  const corrupted = prepareDirectionData(makeSnapshot(flipped), undefined, 7);

  const majorityBefore = directionMajority(baseline.targets.slice(0, split));
  const majorityAfter = directionMajority(corrupted.targets.slice(0, split));
  expect(majorityAfter).toEqual(majorityBefore);

  // The corrupted labels themselves do move: the mutation is real and the
  // invariance above is not vacuous.
  const labelsBefore = flatten2D(baseline.targets.slice(split));
  const labelsAfter = flatten2D(corrupted.targets.slice(split));
  expect(labelsBefore.length).toBe(labelsAfter.length);
  expect(labelsBefore.every((value, index) => value === labelsAfter[index])).toBe(false);
});

function flatten2D(rows) {
  return rows.flatMap((row) => row.map((value) => Number(value) > 0.5 ? 1 : 0));
}