import { defineConfig, loadEnv } from 'vite';
import react from '@vitejs/plugin-react';
import { resolve } from 'node:path';

export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), '');
  const legacyBrowserBuild = mode === 'test' || (
    env.VITE_BROWSER_TRAINING_ENABLED === 'true'
    && env.VITE_VOLATILITY_SERVING_ENABLED === 'false'
  );

  return {
    plugins: [
      {
        name: 'stocklstm-disable-browser-training-in-production',
        enforce: 'pre',
        resolveId(source) {
          if (!legacyBrowserBuild && source === '../ml/browserTrainingClient') {
            return resolve(process.cwd(), 'src/ml/browserTrainingDisabled.js');
          }
          return null;
        },
      },
      react(),
    ],
    define: {
      __STOCKLSTM_LEGACY_BROWSER_BUILD__: JSON.stringify(legacyBrowserBuild),
    },
    test: {
      environment: 'jsdom',
      globals: true,
      setupFiles: './src/test/setup.js',
      testTimeout: 30_000,
      exclude: ['e2e/**', 'node_modules/**', 'dist/**'],
    },
    server: {
      // Local-development convenience only. Vercel's static production build
      // uses VITE_API_URL and does not inherit this proxy.
      port: 5500,
      host: true,
      proxy: {
        '/api': {
          target: 'http://127.0.0.1:8000',
          changeOrigin: true,
        },
      },
    },
    preview: {
      port: 4173,
      host: true,
      proxy: {
        '/api': {
          target: 'http://127.0.0.1:8000',
          changeOrigin: true,
        },
      },
    },
  };
});
