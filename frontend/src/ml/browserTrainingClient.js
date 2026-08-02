import { modelKey, validateSnapshot } from './preprocessing';
import { resolveTrainingProfile } from './trainingProfiles';

let worker;
let sequence = 0;
const pending = new Map();
const activeForecasts = new Map();

function resetWorker(message = 'Browser training worker failed.') {
  worker?.terminate();
  worker = undefined;
  for (const request of pending.values()) {
    for (const subscriber of request.subscribers || []) {
      subscriber.cleanup?.();
      subscriber.reject(new Error(message));
    }
    request.subscribers?.clear();
  }
  pending.clear();
  activeForecasts.clear();
}

function getWorker() {
  if (!worker) {
    worker = new Worker(new URL('./trainingWorker.js', import.meta.url), { type: 'module' });
    worker.onmessage = (event) => {
      const { id, type, result, message, ...progress } = event.data || {};
      const request = pending.get(id);
      if (!request) return;
      if (type === 'progress') {
        for (const subscriber of request.subscribers) subscriber.onProgress?.(progress);
        return;
      }
      pending.delete(id);
      activeForecasts.delete(request.identity);
      for (const subscriber of request.subscribers) {
        subscriber.cleanup?.();
        if (type === 'complete') subscriber.resolve(result);
        else subscriber.reject(new Error(message || 'Browser training failed.'));
      }
    };
    worker.onerror = () => resetWorker();
  }
  return worker;
}

export function browserTrainingSupported() {
  return typeof Worker !== 'undefined';
}

export function forecastCacheIdentity(snapshot, forecastType, profile = 'balanced', backend = 'any') {
  validateSnapshot(snapshot);
  resolveTrainingProfile(profile);
  return modelKey(snapshot, forecastType, profile, backend);
}

function subscribe(request, signal, onProgress) {
  return new Promise((resolve, reject) => {
    const subscriber = { resolve, reject, onProgress };
    const abort = () => {
      request.subscribers.delete(subscriber);
      subscriber.cleanup?.();
      reject(new DOMException('Browser training was cancelled.', 'AbortError'));
      if (request.subscribers.size === 0) {
        getWorker().postMessage({ id: request.id, type: 'cancel' });
        pending.delete(request.id);
        activeForecasts.delete(request.identity);
      }
    };
    subscriber.cleanup = () => signal?.removeEventListener('abort', abort);
    request.subscribers.add(subscriber);
    if (signal) {
      if (signal.aborted) abort();
      else signal.addEventListener('abort', abort, { once: true });
    }
  });
}

export function trainBrowserForecast({ snapshot, forecastType, days, profile = 'balanced', signal, onProgress }) {
  if (!browserTrainingSupported()) {
    return Promise.reject(new Error('Browser training is unavailable on this device.'));
  }
  validateSnapshot(snapshot);
  resolveTrainingProfile(profile);
  const identity = `${snapshot.snapshot_id}/${snapshot.ticker}/${forecastType}/${profile}/${Number(days)}`;
  let request = activeForecasts.get(identity);
  if (!request) {
    const id = `browser-${Date.now()}-${sequence++}`;
    request = { id, identity, subscribers: new Set() };
    pending.set(id, request);
    activeForecasts.set(identity, request);
    getWorker().postMessage({ id, type: 'forecast', snapshot, forecastType, days, profile });
  }
  return subscribe(request, signal, onProgress);
}

export function clearBrowserModelCache() {
  if (!browserTrainingSupported()) return Promise.reject(new Error('Browser workers are unavailable.'));
  const id = `clear-${Date.now()}-${sequence++}`;
  const request = { id, identity: id, subscribers: new Set() };
  pending.set(id, request);
  const promise = subscribe(request);
  getWorker().postMessage({ id, type: 'clear-cache' });
  return promise;
}

export function disposeBrowserTrainingWorker() {
  resetWorker('Browser training stopped.');
}
