import { useCallback, useEffect, useState } from 'react';
import { safeGet, safeSet } from '../utils/safeStorage';

const THEME_KEY = 'stock_lstm_theme';

export function useTheme() {
  const [theme, setTheme] = useState(() => {
    return safeGet(THEME_KEY) || 'dark';
  });

  useEffect(() => {
    document.documentElement.setAttribute('data-theme', theme);
    safeSet(THEME_KEY, theme);
  }, [theme]);

  const toggleTheme = useCallback(() => {
    setTheme((prev) => (prev === 'dark' ? 'light' : 'dark'));
  }, []);

  return { theme, toggleTheme };
}
