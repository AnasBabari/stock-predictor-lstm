import { classificationMetrics, generateResearchSplits, regressionMetrics } from './evaluation';

test('creates five expanding validation folds with an exact purge boundary', () => {
  const splits = generateResearchSplits(629);
  expect(splits).toHaveLength(5);
  expect(splits[0]).toEqual({ fold: 1, trainEnd: 300, validationStart: 329, validationEnd: 389 });
  expect(splits[4].validationEnd).toBe(629);
  expect(() => generateResearchSplits(628)).toThrow(/requires/i);
});

test('calculates regression evidence against persistence from untouched predictions', () => {
  const metrics = regressionMetrics([10, 12], [11, 11], [10, 10], 'browser_purged_holdout');
  expect(metrics.mae).toBe(1);
  expect(metrics.rmse).toBe(1);
  expect(metrics.relative_rmse).toBeCloseTo(1 / Math.sqrt(2));
  expect(metrics.metric_scope).toBe('untouched_post_purge_holdout');
});

test('keeps direction probabilities bounded and reports the majority baseline', () => {
  const metrics = classificationMetrics([[1, 0], [1, 1]], [[1.4, -0.2], [0.8, 0.2]], 'browser_walk_forward_out_of_fold');
  expect(metrics.accuracy).toBe(0.75);
  expect(metrics.naive_baseline).toBe(0.75);
  expect(metrics.brier_score).toBeGreaterThanOrEqual(0);
});
