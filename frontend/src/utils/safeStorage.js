/**
 * Storage access that never throws, even when localStorage is blocked
 * (private mode, disabled cookies, quota errors). All helpers return
 * null/undefined-safe values and swallow QuotaExceededError variants.
 */
export function safeGet(key) {
  try {
    return window.localStorage.getItem(key);
  } catch {
    return null;
  }
}

export function safeSet(key, value) {
  try {
    window.localStorage.setItem(key, value);
    return true;
  } catch {
    return false;
  }
}

export function safeRemove(key) {
  try {
    window.localStorage.removeItem(key);
    return true;
  } catch {
    return false;
  }
}
