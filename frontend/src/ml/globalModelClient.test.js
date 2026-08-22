import { describe, expect, it, vi } from 'vitest';
import {
  fetchVerifiedArtifact,
  isGlobalModelEnabled,
  validateCatalog,
} from './globalModelClient';

const SHA_A = 'a'.repeat(64);
const SHA_B = 'b'.repeat(64);

function validCatalog(overrides = {}) {
  return {
    schema_version: 1,
    signature: 's'.repeat(128),
    recorded_sha: 'abcdef1234567',
    artifacts: [
      { name: 'global-v1', url: '/models/global/global-v1/model.json', sha256: SHA_A, horizons: [1, 5, 7] },
      { name: 'global-v1-30d', url: '/models/global/30d.bin', sha256: SHA_B, horizons: [14, 30] },
    ],
    ...overrides,
  };
}

describe('validateCatalog', () => {
  it('accepts a well-formed catalog', () => {
    const catalog = validCatalog();
    expect(validateCatalog(catalog)).toEqual(catalog);
  });

  it.each([
    ['null payload', null],
    ['wrong schema', { ...validCatalog(), schema_version: 2 }],
    ['empty artifacts', { ...validCatalog(), artifacts: [] }],
    ['bad checksum format', validCatalog({ artifacts: [{ name: 'x', url: '/u', sha256: 'nope', horizons: [1] }] })],
    ['artifact missing horizons', validCatalog({ artifacts: [{ name: 'x', url: '/u', sha256: SHA_A }] })],
    ['missing signature', (() => { const c = validCatalog(); delete c.signature; return c; })()],
    ['missing recorded_sha', (() => { const c = validCatalog(); delete c.recorded_sha; return c; })()],
  ])('rejects %s', (_label, raw) => {
    expect(() => validateCatalog(raw)).toThrow();
  });
});

describe('fetchVerifiedArtifact', () => {
  it('returns buffer when checksum matches', async () => {
    const payload = new TextEncoder().encode('model-bytes');
    const digest = await crypto.subtle.digest('SHA-256', payload);
    const hex = [...new Uint8Array(digest)].map((b) => b.toString(16).padStart(2, '0')).join('');
    const fakeFetch = vi.fn().mockResolvedValue({ ok: true, arrayBuffer: () => Promise.resolve(payload) });
    const buf = await fetchVerifiedArtifact('/u', hex, fakeFetch);
    expect(new Uint8Array(buf)).toEqual(new Uint8Array(payload));
  });

  it('throws on checksum mismatch', async () => {
    const payload = new TextEncoder().encode('tampered');
    const fakeFetch = vi.fn().mockResolvedValue({ ok: true, arrayBuffer: () => Promise.resolve(payload) });
    await expect(fetchVerifiedArtifact('/u', SHA_A, fakeFetch)).rejects.toThrow(/checksum mismatch/);
  });

  it('throws on HTTP failure', async () => {
    const fakeFetch = vi.fn().mockResolvedValue({ ok: false, status: 404 });
    await expect(fetchVerifiedArtifact('/u', SHA_A, fakeFetch)).rejects.toThrow(/404/);
  });
});

describe('isGlobalModelEnabled', () => {
  it('is false unless the flag is explicitly true', () => {
    // Default env has no VITE_GLOBAL_MODEL_ENABLED — must be off.
    expect(isGlobalModelEnabled()).toBe(false);
  });
});
