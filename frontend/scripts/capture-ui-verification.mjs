import { chromium, devices } from '@playwright/test';

import { installFixtures } from '../e2e/uiFixtures.js';

async function submit(page) {
  await page.goto('/');
  await page.getByLabel('Stock ticker').fill('MSFT');
  await page.getByRole('button', { name: /view outlook/i }).click();
  await page.getByText(/Day 7 estimate:/i).waitFor({ state: 'visible', timeout: 15_000 });
  await page.locator('#chartContainer canvas').waitFor({ state: 'visible' });
  // Allow the chart's documented 250ms transition to complete before capture.
  await page.waitForTimeout(500);
}

const browser = await chromium.launch({ headless: true });
try {
  const desktop = await browser.newPage({ viewport: { width: 1440, height: 1000 }, baseURL: 'http://127.0.0.1:4173' });
  await installFixtures(desktop);
  await submit(desktop);
  await desktop.screenshot({ path: '../docs/verification/ui-success-desktop.png', fullPage: true });
  await desktop.getByRole('button', { name: /expand chart/i }).click();
  await desktop.getByRole('dialog', { name: /expanded price chart/i }).waitFor();
  await desktop.waitForTimeout(500);
  await desktop.screenshot({ path: '../docs/verification/ui-expanded-desktop.png' });
  await desktop.keyboard.press('Escape');
  await desktop.close();

  const mobile = await browser.newPage({ ...devices['iPhone 13'], baseURL: 'http://127.0.0.1:4173' });
  await installFixtures(mobile);
  await submit(mobile);
  await mobile.screenshot({ path: '../docs/verification/ui-success-mobile.png', fullPage: true });
  await mobile.close();

  const partial = await browser.newPage({ viewport: { width: 1440, height: 1000 }, baseURL: 'http://127.0.0.1:4173' });
  await installFixtures(partial, 10);
  await submit(partial);
  await partial.getByText(/10-session outlook unavailable/i).waitFor({ state: 'visible', timeout: 15_000 });
  await partial.screenshot({ path: '../docs/verification/ui-partial-volatility-desktop.png', fullPage: true });
  await partial.close();

  const partialMobile = await browser.newPage({ ...devices['iPhone 13'], baseURL: 'http://127.0.0.1:4173' });
  await installFixtures(partialMobile, 10);
  await submit(partialMobile);
  await partialMobile.getByText(/10-session outlook unavailable/i).waitFor();
  await partialMobile.screenshot({ path: '../docs/verification/ui-partial-volatility-mobile.png', fullPage: true });
  await partialMobile.close();
} finally {
  await browser.close();
}
