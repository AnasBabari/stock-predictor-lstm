import { ARCHITECTURE_VERSION, FEATURE_SCHEMA_VERSION, MODEL_VERSION, TARGET_MODE } from './preprocessing';

export const CHECKPOINT_PREFIX = 'checkpoint';
const CURRENT_KEY_PREFIX = `${MODEL_VERSION}/${ARCHITECTURE_VERSION}/${TARGET_MODE}/${FEATURE_SCHEMA_VERSION}/`;
const CHECKPOINT_KEY_PREFIX = `${CHECKPOINT_PREFIX}/${CURRENT_KEY_PREFIX}`;

// Reserved top-level namespaces (overhaul slice 2). Local browser-trained
// artifacts keep the versioned prefix above; future global production
// artifacts and diagnostics live in their own namespaces so a rejected or
// research object can never satisfy an active-production lookup.
export const ARTIFACT_NAMESPACES = Object.freeze({
  localTrained: CURRENT_KEY_PREFIX,
  checkpoint: CHECKPOINT_KEY_PREFIX,
  productionGlobal: 'production_global_artifact/',
  promotedLocalCalibrator: 'promoted_local_calibrator/',
  rejectedEvidence: 'rejected_evidence/',
});

export function namespaceOf(key) {
  const value = String(key ?? '');
  for (const namespace of Object.values(ARTIFACT_NAMESPACES)) {
    if (value.startsWith(namespace)) return namespace;
  }
  return null;
}

// A rejected artifact may be kept for diagnostics but must never satisfy an
// active lookup. Absence of a verdict means a legacy cache entry; callers
// re-evaluate promotion deterministically from stored metrics, so legacy
// entries remain safe.
export function isRejectedArtifact(metadata) {
  const summary = metadata?.promotion_summary;
  return summary?.applicable !== false && summary?.promoted === false;
}

export function isVersionedKey(key) {
  const value = String(key ?? '');
  return (
    value.startsWith(CURRENT_KEY_PREFIX) ||
    value.startsWith(CHECKPOINT_KEY_PREFIX) ||
    Object.values(ARTIFACT_NAMESPACES).some(
      (namespace) => namespace !== CURRENT_KEY_PREFIX && namespace !== CHECKPOINT_KEY_PREFIX &&
        value.startsWith(namespace)
    )
  );
}