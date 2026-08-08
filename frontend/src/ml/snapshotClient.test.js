import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { createSnapshotClient } from './snapshotClient';

function jsonResponse(body, ok = true, status = 200) {
  return {
    ok,
    status,
    json: () => Promise.resolve(body),
  };
}

describe('snapshot client', () => {
  beforeEach(() => {
    vi.useFakeTimers();
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  it('fetches the training snapshot and caches it for the TTL window', async () => {
    const fetchImpl = vi.fn(() => Promise.resolve(jsonResponse({ ticker: 'MSFT', snapshot_id: 's1' })));
    const client = createSnapshotClient({ ttlMs: 60_000, fetchImpl });

    const first = await client.fetchTrainingSnapshot('MSFT');
    expect(fetchImpl).toHaveBeenCalledTimes(1);
    expect(first.snapshot_id).toBe('s1');

    const second = await client.fetchTrainingSnapshot('MSFT');
    expect(fetchImpl).toHaveBeenCalledTimes(1);
    expect(second).toBe(first);
  });

  it('coalesces concurrent identical requests into a single fetch', async () => {
    const fetchImpl = vi.fn(() => Promise.resolve(jsonResponse({ ticker: 'MSFT', snapshot_id: 's1' })));
    const client = createSnapshotClient({ ttlMs: 60_000, fetchImpl });

    const [a, b, c] = await Promise.all([
      client.fetchTrainingSnapshot('MSFT'),
      client.fetchTrainingSnapshot('MSFT'),
      client.fetchTrainingSnapshot('MSFT'),
    ]);
    expect(fetchImpl).toHaveBeenCalledTimes(1);
    expect(a).toBe(b);
    expect(b).toBe(c);
  });

  it('refetches after the TTL window expires', async () => {
    const fetchImpl = vi.fn(() => Promise.resolve(jsonResponse({ ticker: 'MSFT', snapshot_id: 's1' })));
    const client = createSnapshotClient({ ttlMs: 60_000, fetchImpl });

    await client.fetchTrainingSnapshot('MSFT');
    vi.advanceTimersByTime(60_001);
    await client.fetchTrainingSnapshot('MSFT');
    expect(fetchImpl).toHaveBeenCalledTimes(2);
  });

  it('refetches immediately after a failure and never caches it', async () => {
    const fetchImpl = vi.fn()
      .mockRejectedValueOnce(new Error('network down'))
      .mockResolvedValueOnce(jsonResponse({ ticker: 'MSFT', snapshot_id: 's1' }));
    const client = createSnapshotClient({ ttlMs: 60_000, fetchImpl });

    await expect(client.fetchTrainingSnapshot('MSFT')).rejects.toThrow('network down');
    const snapshot = await client.fetchTrainingSnapshot('MSFT');
    expect(snapshot.snapshot_id).toBe('s1');
    expect(fetchImpl).toHaveBeenCalledTimes(2);
  });

  it('surfaces non-ok responses as detailed errors', async () => {
    const fetchImpl = vi.fn(() => Promise.resolve(jsonResponse({ detail: 'Ticker is invalid.' }, false, 400)));
    const client = createSnapshotClient({ fetchImpl });

    await expect(client.fetchTrainingSnapshot('MSFT')).rejects.toThrow('Ticker is invalid.');
  });

  it('normalizes ticker casing in the cache key and the URL', async () => {
    const fetchImpl = vi.fn(() => Promise.resolve(jsonResponse({ ticker: 'MSFT' })));
    const client = createSnapshotClient({ baseUrl: 'https://x.test', fetchImpl });

    await Promise.all([
      client.fetchTrainingSnapshot('msft'),
      client.fetchTrainingSnapshot('MSFT '),
    ]);
    expect(fetchImpl).toHaveBeenCalledTimes(1);
    expect(fetchImpl).toHaveBeenCalledWith('https://x.test/api/v1/training-data?ticker=MSFT', expect.anything());
  });

  it('propagates abort and allows a later retry', async () => {
    const controller = new AbortController();
    const fetchImpl = vi.fn((url, options) => new Promise((resolve, reject) => {
      options.signal.addEventListener('abort', () => {
        reject(new DOMException('The operation was aborted.', 'AbortError'));
      }, { once: true });
    }));
    const client = createSnapshotClient({ fetchImpl });

    const pending = client.fetchTrainingSnapshot('MSFT', controller.signal);
    controller.abort();
    await expect(pending).rejects.toMatchObject({ name: 'AbortError' });

    fetchImpl.mockResolvedValue(jsonResponse({ ticker: 'MSFT', snapshot_id: 's1' }));
    const snapshot = await client.fetchTrainingSnapshot('MSFT');
    expect(snapshot.snapshot_id).toBe('s1');
  });

  it('clear() drops cached values so the next call refetches', async () => {
    const fetchImpl = vi.fn(() => Promise.resolve(jsonResponse({ ticker: 'MSFT' })));
    const client = createSnapshotClient({ ttlMs: 60_000, fetchImpl });

    await client.fetchTrainingSnapshot('MSFT');
    client.clear();
    await client.fetchTrainingSnapshot('MSFT');
    expect(fetchImpl).toHaveBeenCalledTimes(2);
  });
});