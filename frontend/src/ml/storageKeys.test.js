import { modelKey } from './preprocessing';
import { CHECKPOINT_PREFIX, isVersionedKey } from './storageKeys';

const snapshot = {
  ticker: 'AAPL',
  schema_version: 4,
  snapshot_id: 'snapshot-xyz',
  feature_names: ['a', 'b'],
};

test('accepts keys built for the current model and architecture versions', () => {
  const key = modelKey(snapshot, 'price', 'balanced', 'cpu', 7);
  expect(isVersionedKey(key)).toBe(true);
  expect(isVersionedKey(`${CHECKPOINT_PREFIX}/${key}`)).toBe(true);
});

test('reclaims orphaned keys from older version namespaces', () => {
  const key = modelKey(snapshot, 'price', 'balanced', 'cpu', 7);
  const olderModel = key.replace('tfjs-return-lstm-v4', 'tfjs-return-lstm-v3');
  const olderArchitecture = key.replace('local-return-lstm-v3', 'local-return-lstm-v2');
  const preSchema = key.split('/').map((part, index) => (index === 3 ? '3' : part)).join('/');
  expect(isVersionedKey(olderModel)).toBe(false);
  expect(isVersionedKey(olderArchitecture)).toBe(false);
  expect(isVersionedKey(preSchema)).toBe(false);
  expect(isVersionedKey(`${CHECKPOINT_PREFIX}/${olderModel}`)).toBe(false);
});

test('rejects unrelated or empty history keys without throwing', () => {
  expect(isVersionedKey('stocklstm-random')).toBe(false);
  expect(isVersionedKey('')).toBe(false);
  expect(isVersionedKey(undefined)).toBe(false);
  expect(isVersionedKey(null)).toBe(false);
  expect(isVersionedKey(42)).toBe(false);
});