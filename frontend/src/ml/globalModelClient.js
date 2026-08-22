/**
 * Global model artifact loader (slice 12).
 *
 * Fetches a pinned catalog, verifies sha256 checksums of every artifact
 * entry via WebCrypto, and exposes tfjs inference behind an explicit
 * feature flag. Fail-closed: any missing field, checksum mismatch, or
 * disabled flag leaves globalMode inactive and callers fall back to the
 * browser-training path.
 */

export const GLOBAL_CATALOG_URL = '/models/global/catalog.json';

export function validateCatalog(raw) {
  if (!raw || typeof raw !== 'object') throw new Error('Global catalog is not an object.');
  if (raw.schema_version !== 1) throw new Error('Unsupported global catalog schema.');
  if (!Array.isArray(raw.artifacts) || raw.artifacts.length === 0) {
    throw new Error('Global catalog has no artifacts.');
  }
  for (const entry of raw.artifacts) {
    if (!entry.name || typeof entry.url !== 'string' || !/^[a-f0-9]{64}$/.test(entry.sha256)) {
      throw new Error(`Invalid catalog entry: ${JSON.stringify(entry).slice(0, 120)}`);
    }
    if (!Array.isArray(entry.horizons) || entry.horizons.length === 0) {
      throw new Error(`Artifact ${entry.name} lists no horizons.`);
    }
  }
  if (typeof raw.signature !== 'string' || raw.signature.length < 64) {
    throw new Error('Global catalog has no valid signature field.');
  }
  if (typeof raw.recorded_sha !== 'string' || raw.recorded_sha.length < 7) {
    throw new Error('Global catalog has no recorded_sha provenance.');
  }
  return raw;
}

async function sha256Hex(buffer) {
  const digest = await crypto.subtle.digest('SHA-256', buffer);
  return [...new Uint8Array(digest)].map((b) => b.toString(16).padStart(2, '0')).join('');
}

export async function fetchVerifiedArtifact(url, expectedSha256, fetchImpl = fetch) {
  const response = await fetchImpl(url);
  if (!response.ok) throw new Error(`Artifact download failed (${response.status}).`);
  const buffer = await response.arrayBuffer();
  const actual = await sha256Hex(buffer);
  if (actual !== expectedSha256) {
    throw new Error(
      `Artifact checksum mismatch: expected ${expectedSha256}, got ${actual}.`
    );
  }
  return buffer;
}

export function isGlobalModelEnabled() {
  return import.meta.env.VITE_GLOBAL_MODEL_ENABLED === 'true';
}

/**
 * Load the first artifact whose horizons include the requested days.
 * Returns { model, catalog, artifact } or null when disabled/unavailable.
 */
export async function loadGlobalModel(days, tf, fetchImpl = fetch) {
  if (!isGlobalModelEnabled()) return null;
  let catalog;
  try {
    const response = await fetchImpl(GLOBAL_CATALOG_URL);
    if (!response.ok) return null;
    catalog = validateCatalog(await response.json());
  } catch {
    return null; // fail closed — caller falls back to browser training
  }
  for (const artifact of catalog.artifacts) {
    if (!artifact.horizons.includes(Number(days))) continue;
    try {
      const buffer = await fetchVerifiedArtifact(artifact.url, artifact.sha256, fetchImpl);
      const model = await tf.loadLayersModel(tf.io.fromMemory(buffer));
      return { model, catalog, artifact };
    } catch {
      return null;
    }
  }
  return null;
}
