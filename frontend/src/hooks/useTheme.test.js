import { describe, expect, it, vi } from 'vitest';
import { renderHook } from '@testing-library/react';
import { useTheme } from './useTheme';

describe('useTheme with blocked storage', () => {
  it('falls back to the default theme when localStorage throws', () => {
    const getItem = vi.spyOn(Storage.prototype, 'getItem').mockImplementation(() => {
      throw new DOMException('Access denied', 'SecurityError');
    });
    const setItem = vi.spyOn(Storage.prototype, 'setItem').mockImplementation(() => {
      throw new DOMException('Access denied', 'SecurityError');
    });
    const { result } = renderHook(() => useTheme());
    expect(result.current.theme).toBe('dark');
    expect(typeof result.current.toggleTheme).toBe('function');
    getItem.mockRestore();
    setItem.mockRestore();
  });
});
