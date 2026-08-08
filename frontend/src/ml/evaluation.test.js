import {
  classificationMetrics,
  directionalAccuracy,
  generateResearchSplits,
  horizonClassificationMetrics,
  horizonRegressionMetrics,
  regressionMetrics,
} from './evaluation';

test('creates five expanding validation folds with an exact purge boundary', () => {
  const splits = generateResearchSplits(629);
  expect(splits).toHaveLength(5);
  expect(splits[0]).toEqual({ fold: 1, trainEnd: 300, validationStart: 329, validationEnd: 389 });
  expect(splits[4].validationEnd).toBe(629);
  expect(() => generateResearchSplits(628)).toThrow(/requires/i);
});

test('uses a horizon-specific purge for research splits', () => {
  const splits = generateResearchSplits(629, { purge: 2 });
  expect(splits[0].trainEnd).toBe(splits[0].validationStart - 2);
  expect(() => generateResearchSplits(601, { purge: 2 })).toThrow(/requires/i);
});

test('calculates regression evidence against persistence from untouched predictions', () => {
  const metrics = regressionMetrics([10, 12], [11, 11], [10, 10], 'browser_purged_holdout');
  expect(metrics.mae).toBe(1);
  expect(metrics.rmse).toBe(1);
  expect(metrics.relative_rmse).toBeCloseTo(1 / Math.sqrt(2));
  expect(metrics.metric_scope).toBe('untouched_post_purge_holdout');
});

test('reports pooled and per-horizon metrics for cumulative return targets', () => {
  const actual = [[0.01, 0.02, 0.03], [0.0, 0.01, 0.02]];
  const predicted = [[0.012, 0.022, 0.032], [-0.001, 0.009, 0.019]];
  const persistence = [[0, 0, 0], [0, 0, 0]];
  const result = horizonRegressionMetrics(actual, predicted, persistence, 3, 'browser_walk_forward_out_of_fold');
  expect(result.per_horizon).toHaveLength(3);
  expect(result.per_horizon[0].horizon).toBe(1);
  expect(result.per_horizon[0].rows).toBe(2);
  expect(result.per_horizon[1].relative_rmse).toBeCloseTo(
    Math.sqrt((0.002 ** 2 + 0.001 ** 2) / 2) / Math.sqrt((0.02 ** 2 + 0.01 ** 2) / 2),
    12,
  );
  expect(result.pooled.mae).toBeGreaterThan(0);
  expect(result.evaluation_rows).toBe(2);
});

test('persistence for returns is exactly a zero-return forecast', () => {
  const actual = [[0.01, 0.02], [0.0, 0.01]];
  const result = horizonRegressionMetrics(actual, actual, [[0, 0], [0, 0]], 2, 'browser_purged_holdout');
  expect(result.pooled.relative_mae).toBeLessThan(0.001);
  expect(result.pooled.relative_rmse).toBeLessThan(0.001);
});

test('measures directional accuracy from predicted returns', () => {
  expect(directionalAccuracy([[0.01, -0.01]], [[0.02, -0.02]])).toBe(1);
  expect(directionalAccuracy([[0.01, -0.01]], [[-0.02, -0.02]])).toBe(0.5);
  expect(directionalAccuracy([[0.01]], [[NaN]])).toBeNull();
});

test('keeps direction probabilities bounded and reports the majority baseline', () => {
  const metrics = classificationMetrics([[1, 0], [1, 1]], [[1.4, -0.2], [0.8, 0.2]], 'browser_walk_forward_out_of_fold');
  expect(metrics.accuracy).toBe(0.75);
  expect(metrics.naive_baseline).toBe(0.75);
  expect(metrics.brier_score).toBeGreaterThanOrEqual(0);
});

test('reports direction evidence separately for each forecast day', () => {
  const actual = [[1, 0, 1], [0, 1, 1]];
  const predicted = [[0.9, 0.1, 0.8], [0.2, 0.9, 0.4]];
  const evidence = horizonClassificationMetrics(actual, predicted, 'browser_walk_forward_out_of_fold');
  expect(evidence.per_horizon).toHaveLength(3);
  expect(evidence.per_horizon[0].horizon).toBe(1);
  expect(evidence.per_horizon[0].rows).toBe(2);
  expect(evidence.per_horizon[0].accuracy).toBe(1);
  expect(evidence.per_horizon[1].accuracy).toBe(1);
  expect(evidence.per_horizon[2].accuracy).toBe(0.5);
  expect(evidence.per_horizon[2].naive_baseline).toBe(1);
  expect(evidence.pooled.accuracy).toBeCloseTo(5 / 6);
  expect(evidence.pooled.evaluation_rows).toBe(6);
});

test('per-horizon direction evidence honors the pre-evaluation majority label', () => {
  const actual = [[1, 0, 1], [0, 1, 1]];
  const predicted = [[0.9, 0.1, 0.8], [0.2, 0.9, 0.4]];
  const evidence = horizonClassificationMetrics(actual, predicted, 'browser_purged_holdout', 0);
  expect(evidence.per_horizon[0].naive_baseline).toBe(0.5);
  expect(evidence.per_horizon[1].naive_baseline).toBe(0.5);
  expect(evidence.per_horizon[2].naive_baseline).toBe(0);
});
