import { modelKey, validateSnapshot } from './preprocessing';

let worker;
let sequence = 0;
const pending = new Map();

function getWorker() {
  if (!worker) {
    worker = new Worker(new URL('./trainingWorker.js', import.meta.url), { type: 'module' });
    worker.onmessage = (event) => {
      const { id, type, result, message, ...progress } = event.data || {};
      const request = pending.get(id);
      if (!request) return;
      if (type === 'progress') {
        request.onProgress?.(progress);
      } else if (type === 'complete') {
        pending.delete(id);
        request.cleanup?.();
        request.resolve(result);
      } else if (type === 'error') {
        pending.delete(id);
        request.cleanup?.();
        request.reject(new Error(message || 'Browser training failed.'));
      }
    };
    worker.onerror = () => {
      for (const request of pending.values()) {
        request.cleanup?.();
        request.reject(new Error('Browser training worker failed.'));
      }
      pending.clear();
      worker = undefined;
    };
  }
  return worker;
}

export function browserTrainingSupported() {
  // IndexedDB is optional: the worker can still train a session-only model when
  // browser persistence is unavailable or the quota is exhausted.
  return typeof Worker !== 'undefined';
}

export function forecastCacheIdentity(snapshot, forecastType) {
  validateSnapshot(snapshot);
  return modelKey(snapshot, forecastType);
}

export function trainBrowserForecast({ snapshot, forecastType, days, signal, onProgress }) {
  if (!browserTrainingSupported()) {
    return Promise.reject(new Error('Browser training is unavailable on this device.'));
  }
  validateSnapshot(snapshot);
  const id = `browser-${Date.now()}-${sequence++}`;
  const activeWorker = getWorker();
  return new Promise((resolve, reject) => {
    const abort = () => {
      activeWorker.postMessage({ id, type: 'cancel' });
      pending.delete(id);
      reject(new DOMException('Browser training was cancelled.', 'AbortError'));
    };
    const cleanup = () => signal?.removeEventListener('abort', abort);
    pending.set(id, { resolve, reject, onProgress, cleanup });
    if (signal) {
      if (signal.aborted) {
        abort();
        return;
      }
      signal.addEventListener('abort', abort, { once: true });
    }
    activeWorker.postMessage({ id, type: 'forecast', snapshot, forecastType, days });
  });
}

export function clearBrowserModelCache() {
  if (typeof Worker === 'undefined') return Promise.reject(new Error('Browser workers are unavailable.'));
  const id = `clear-${Date.now()}-${sequence++}`;
  const activeWorker = getWorker();
  return new Promise((resolve, reject) => {
    pending.set(id, { resolve, reject });
    activeWorker.postMessage({ id, type: 'clear-cache' });
  });
}

export function disposeBrowserTrainingWorker() {
  if (worker) worker.terminate();
  worker = undefined;
  for (const request of pending.values()) {
    request.cleanup?.();
    request.reject(new Error('Browser training stopped.'));
  }
  pending.clear();
}