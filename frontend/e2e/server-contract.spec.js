import { expect, test } from '@playwright/test';
import { installFixtures } from './uiFixtures.js';

async function submit(page) {
  await page.goto('/');
  await page.getByLabel('Stock ticker').fill('MSFT');
  await page.getByRole('button', { name: /view outlook/i }).click();
  await expect(page.locator('#chartContainer canvas')).toBeVisible();
}

test('first visit waits for a stock selection without a placeholder chart', async ({ page }) => {
  const calls = [];
  page.on('request', (request) => calls.push(request.url()));
  await installFixtures(page);
  await page.goto('/');
  await expect(page.getByLabel('Stock ticker')).toHaveValue('');
  await expect(page.locator('#chartContainer')).toHaveCount(0);
  expect(calls.some((url) => /\/api\/v1\/(history|forecast)/.test(url))).toBe(false);
});

test('valid estimate, modal keyboard containment and supporting tab navigation', async ({ page }) => {
  await installFixtures(page);
  await submit(page);
  await expect(page.locator('.chart-estimate-bar')).toContainText('$456.00');
  const expand = page.getByRole('button', { name: 'Expand chart', exact: true });
  await expand.click();
  const dialog = page.getByRole('dialog', { name: 'Expanded price chart' });
  const close = dialog.getByRole('button', { name: 'Close expanded chart' });
  await expect(close).toBeFocused();
  await page.keyboard.press('Tab');
  await expect(close).toBeFocused();
  await page.keyboard.press('Shift+Tab');
  await expect(close).toBeFocused();
  expect(await page.getByLabel('Stock ticker').evaluate((element) => Boolean(element.closest('[inert]')))).toBe(true);
  await expect.poll(async () => {
    const canvas = await dialog.locator('canvas').boundingBox();
    const bounds = await dialog.boundingBox();
    return canvas.y + canvas.height <= bounds.y + bounds.height;
  }).toBe(true);
  await page.keyboard.press('Escape');
  await expect(dialog).toHaveCount(0);
  await expect(expand).toBeFocused();
  expect(await page.locator('body').evaluate((element) => element.style.overflow)).not.toBe('hidden');
  await page.getByRole('tab', { name: 'Overview', exact: true }).focus();
  await page.keyboard.press('ArrowRight');
  await expect(page.getByRole('tab', { name: /News/ })).toBeFocused();
  await expect(page.getByText('[Example fixture] Company update for layout verification')).toBeVisible();
  await page.keyboard.press('End');
  await expect(page.getByRole('tab', { name: 'Performance', exact: true })).toBeFocused();
  await page.keyboard.press('Home');
  await expect(page.getByRole('tab', { name: 'Overview', exact: true })).toBeFocused();
  await page.keyboard.press('ArrowLeft');
  await expect(page.getByRole('tab', { name: 'Performance', exact: true })).toBeFocused();
});

test('one unavailable horizon retains the other results and exposes its status', async ({ page }) => {
  await installFixtures(page, 10);
  await submit(page);
  await expect(page.getByText(/10-session outlook unavailable/)).toBeVisible();
  await expect(page.getByText('5 sessions', { exact: true })).toBeVisible();
  await expect(page.getByText('20 sessions', { exact: true })).toBeVisible();
  await page.getByText('About this outlook', { exact: true }).click();
  await expect(page.getByText(/Evaluation: historical validation panel/)).toHaveCount(2);
});

for (const [name, overrides] of [
  ['null bounds that previously produced a false $55 estimate', { lower_prices: Array(7).fill(null), upper_prices: Array(7).fill(110) }],
  ['truncated path', { lower_prices: [440], upper_prices: [460] }],
  ['missing origin', { data_as_of: null }],
  ['inconsistent origin', { current_price: 315.34 }],
]) {
  test(`rejects ${name} while keeping history visible`, async ({ page }) => {
    await installFixtures(page, null, overrides);
    await submit(page);
    await expect(page.getByText(/Day 7 estimate:/)).toHaveCount(0);
    await expect(page.locator('.chart-mismatch-alert')).toBeVisible();
    await expect(page.locator('#chartContainer canvas')).toBeVisible();
    await expect(page.locator('.t212-price')).toHaveText('$450.00');
  });
}

test('touch scrolling can pass the chart without trapping the page', async ({ browser, baseURL }) => {
  const context = await browser.newContext({ baseURL, viewport: { width: 390, height: 844 }, isMobile: true, hasTouch: true });
  const page = await context.newPage();
  await installFixtures(page);
  await submit(page);
  await page.locator('#chartContainer canvas').scrollIntoViewIfNeeded();
  const canvas = await page.locator('#chartContainer canvas').boundingBox();
  const startY = Math.min(canvas.y + canvas.height - 20, 750);
  const startScroll = await page.evaluate(() => window.scrollY);
  const client = await context.newCDPSession(page);
  await client.send('Input.dispatchTouchEvent', { type: 'touchStart', touchPoints: [{ x: 180, y: startY }] });
  for (let step = 1; step <= 6; step++) {
    await client.send('Input.dispatchTouchEvent', { type: 'touchMove', touchPoints: [{ x: 180, y: startY - step * 40 }] });
  }
  await client.send('Input.dispatchTouchEvent', { type: 'touchEnd', touchPoints: [] });
  await expect.poll(() => page.evaluate(() => window.scrollY)).toBeGreaterThan(startScroll + 60);
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth)).toBe(true);
  await context.close();
});
