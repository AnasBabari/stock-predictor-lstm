import { defaultTrainingProfile, expectedDurationLabel, resolveTrainingProfile } from './trainingProfiles';

test('uses quick on constrained or mobile devices and balanced on capable desktops', () => {
  expect(defaultTrainingProfile({ deviceMemory: 4, hardwareConcurrency: 4, userAgent: 'Desktop' })).toBe('quick');
  expect(defaultTrainingProfile({ deviceMemory: 16, hardwareConcurrency: 12, userAgent: 'Desktop' })).toBe('balanced');
  expect(defaultTrainingProfile({ deviceMemory: 16, hardwareConcurrency: 12, userAgent: 'Mobile' })).toBe('quick');
});

test('locks profile architecture and duration contracts', () => {
  expect(resolveTrainingProfile('quick').lstmUnits).toEqual([16, 8]);
  expect(resolveTrainingProfile('balanced').lstmUnits).toEqual([32, 16]);
  expect(resolveTrainingProfile('balanced').epochs).toBe(25);
  expect(resolveTrainingProfile('research').folds).toBe(5);
  expect(expectedDurationLabel('research')).toContain('min');
  expect(() => resolveTrainingProfile('unknown')).toThrow(/unknown/i);
});
