import { describe, expect, it } from 'vitest';
import {
  ARTIFACT_NAMESPACES,
  isRejectedArtifact,
  isVersionedKey,
  namespaceOf,
} from './storageKeys';

describe('artifact namespaces', () => {
  it('reserves disjoint namespaces for future artifact classes', () => {
    const values = Object.values(ARTIFACT_NAMESPACES);
    expect(new Set(values).size).toBe(values.length);
    expect(ARTIFACT_NAMESPACES.productionGlobal).toBe('production_global_artifact/');
    expect(ARTIFACT_NAMESPACES.rejectedEvidence).toBe('rejected_evidence/');
    // Local trained keys keep their versioned prefix.
    expect(ARTIFACT_NAMESPACES.localTrained.startsWith('tfjs-return-lstm-v4/')).toBe(true);
  });

  it.each([
    ['production_global_artifact/global-v1/model.json', 'production_global_artifact/'],
    ['promoted_local_calibrator/MSFT/v1', 'promoted_local_calibrator/'],
    ['rejected_evidence/tfjs-return-lstm-v4/x', 'rejected_evidence/'],
  ])('namespaceOf(%s) -> %s', (key, expected) => {
    expect(namespaceOf(key)).toBe(expected);
  });

  it('returns null namespace for foreign keys', () => {
    expect(namespaceOf('something_else')).toBeNull();
  });
});

describe('isVersionedKey with reserved namespaces', () => {
  it('still accepts local versioned and checkpoint keys', () => {
    const localKey = `${ARTIFACT_NAMESPACES.localTrained}TEST/price/balanced/any/hash/snap/60/7`;
    expect(isVersionedKey(localKey)).toBe(true);
    expect(isVersionedKey(`${ARTIFACT_NAMESPACES.checkpoint}rest`)).toBe(true);
  });

  it('recognises reserved production/calibrator/rejected keys', () => {
    for (const key of [
      `${ARTIFACT_NAMESPACES.productionGlobal}bundle/model.json`,
      `${ARTIFACT_NAMESPACES.promotedLocalCalibrator}cal`,
      `${ARTIFACT_NAMESPACES.rejectedEvidence}probe`,
    ]) {
      expect(isVersionedKey(key)).toBe(true);
    }
  });
});

describe('isRejectedArtifact', () => {
  it('flags artifacts whose persisted verdict is a rejection', () => {
    expect(isRejectedArtifact({ promotion_summary: { promoted: false, applicable: true } })).toBe(true);
  });

  it('treats promoted, non-applicable and legacy entries as not rejected', () => {
    expect(isRejectedArtifact({ promotion_summary: { promoted: true } })).toBe(false);
    expect(isRejectedArtifact({ promotion_summary: { promoted: false, applicable: false } })).toBe(false);
    expect(isRejectedArtifact({})).toBe(false);
    expect(isRejectedArtifact(undefined)).toBe(false);
  });
});
