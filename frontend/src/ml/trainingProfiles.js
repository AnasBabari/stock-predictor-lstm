export const TRAINING_PROFILES = Object.freeze({
  quick: Object.freeze({
    id: 'quick', label: 'Quick', description: 'Fastest',
    lstmUnits: [16, 8], denseUnits: 8, dropout: 0.2,
    epochs: 12, patience: 3, folds: 1,
    expectedSeconds: [30, 90], metricSource: 'browser_purged_holdout',
  }),
  balanced: Object.freeze({
    id: 'balanced', label: 'Balanced', description: 'Recommended',
    lstmUnits: [32, 16], denseUnits: 16, dropout: 0.2,
    epochs: 25, patience: 5, folds: 1,
    expectedSeconds: [120, 600], metricSource: 'browser_purged_holdout',
  }),
  research: Object.freeze({
    id: 'research', label: 'Research', description: 'Five-fold benchmark',
    lstmUnits: [32, 16], denseUnits: 16, dropout: 0.2,
    epochs: 25, patience: 5, folds: 5,
    validationHorizon: 60, minTrainSamples: 300,
    expectedSeconds: [600, 2700], metricSource: 'browser_walk_forward_out_of_fold',
  }),
});

export function resolveTrainingProfile(profile) {
  const resolved = TRAINING_PROFILES[profile];
  if (!resolved) throw new Error('Unknown browser training profile.');
  return resolved;
}

export function defaultTrainingProfile(device = globalThis.navigator) {
  const memory = Number(device?.deviceMemory || 0);
  const cores = Number(device?.hardwareConcurrency || 0);
  const mobile = /Android|iPhone|iPad|iPod|Mobile/i.test(device?.userAgent || '');
  return mobile || (memory > 0 && memory <= 4) || (cores > 0 && cores <= 4) ? 'quick' : 'balanced';
}

export function expectedDurationLabel(profile) {
  const [minimum, maximum] = resolveTrainingProfile(profile).expectedSeconds;
  const format = (seconds) => seconds < 120 ? `${seconds} sec` : `${Math.round(seconds / 60)} min`;
  return `${format(minimum)}–${format(maximum)}`;
}
