import { ARCHITECTURE_VERSION, FEATURE_SCHEMA_VERSION, MODEL_VERSION, TARGET_MODE } from './preprocessing';

export const CHECKPOINT_PREFIX = 'checkpoint';
const CURRENT_KEY_PREFIX = `${MODEL_VERSION}/${ARCHITECTURE_VERSION}/${TARGET_MODE}/${FEATURE_SCHEMA_VERSION}/`;
const CHECKPOINT_KEY_PREFIX = `${CHECKPOINT_PREFIX}/${CURRENT_KEY_PREFIX}`;

export function isVersionedKey(key) {
  const value = String(key ?? '');
  return value.startsWith(CURRENT_KEY_PREFIX) || value.startsWith(CHECKPOINT_KEY_PREFIX);
}