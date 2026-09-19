import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig(() => {
  return {
    plugins: [
      react(),
    ],
    test: {
      environment: 'jsdom',
      globals: true,
      setupFiles: './src/test/setup.js',
      testTimeout: 30_000,
      exclude: ['e2e/**', 'node_modules/**', 'dist/**'],
      coverage: {
        provider: 'v8',
        reporter: ['text', 'json-summary', 'html'],
        // Production code only. The test directory, build config and
        // generated stockpiles are excluded by hand because vitest's
        // default ignores data files and stories but not the whole tree.
        include: ['src/**/*.{js,jsx}'],
        exclude: [
          'src/test/**',
          'src/data/**',
          'src/**/*.test.{js,jsx}',
          'src/**/index.js',
        ],
        thresholds: {
          // Documented in docs/quality.md. Coverage must not silently
          // regress below these on PRs.
          lines: 50,
          functions: 50,
          branches: 35,
          statements: 50,
        },
      },
    },
    server: {
      // Local-development convenience only. Vercel's static production build
      // uses VITE_API_URL and does not inherit this proxy.
      port: 5500,
      host: true,
      proxy: {
        '/api': {
          target: process.env.VITE_API_URL || 'http://127.0.0.1:8000',
          changeOrigin: true,
        },
        '/health': {
          target: process.env.VITE_API_URL || 'http://127.0.0.1:8000',
          changeOrigin: true,
        },
      },
    },
    preview: {
      port: 4173,
      host: true,
      proxy: {
        '/api': {
          target: process.env.VITE_API_URL || 'http://127.0.0.1:8000',
          changeOrigin: true,
        },
        '/health': {
          target: process.env.VITE_API_URL || 'http://127.0.0.1:8000',
          changeOrigin: true,
        },
      },
    },
  };
});
