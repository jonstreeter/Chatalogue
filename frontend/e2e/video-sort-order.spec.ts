import { test, expect } from '@playwright/test';

test.describe('Video Sort Order', () => {
  test('newest videos appear first in the channel video list', async ({ page }) => {
    // Go to channels
    await page.goto('/');
    await page.waitForLoadState('networkidle');

    // Find and open the primary channel
    const card = page.locator('.glass-panel').filter({ hasText: /chatalogue|porchtime/i });
    await expect(card.first()).toBeVisible({ timeout: 10_000 });
    await card.first().getByRole('link', { name: 'Open' }).click();
    await page.waitForLoadState('networkidle');

    // Should be on the Videos tab by default
    await expect(page).toHaveURL(/\/channel\/\d+$/);

    // Wait for video list to load — look for any video card/row
    // The video list should have items
    const videoItems = page.locator('[class*="glass-panel"], [class*="rounded"]').filter({ hasText: /Lobstermaxxing/i });

    // The newest video "Lobstermaxxing: Frame Mogging" (2026-02-20) should be visible
    // without scrolling (it should be near the top)
    await expect(page.getByText(/Lobstermaxxing/i).first()).toBeVisible({ timeout: 15_000 });

    // "Soft White Lies" (2026-02-12) should also be visible
    await expect(page.getByText(/Soft White Lies/i).first()).toBeVisible({ timeout: 5_000 });

    // "Secret Combinations" (2026-02-06) should also be visible
    await expect(page.getByText(/Secret Combinations/i).first()).toBeVisible({ timeout: 5_000 });

    // Verify ordering: Lobstermaxxing should appear BEFORE Soft White Lies in the DOM
    const allText = await page.locator('body').innerText();
    const lobsterIdx = allText.indexOf('Lobstermaxxing');
    const softWhiteIdx = allText.indexOf('Soft White Lies');
    const secretIdx = allText.indexOf('Secret Combinations');

    expect(lobsterIdx).toBeGreaterThan(-1);
    expect(softWhiteIdx).toBeGreaterThan(-1);
    expect(secretIdx).toBeGreaterThan(-1);

    // Newest should come first
    expect(lobsterIdx).toBeLessThan(softWhiteIdx);
    expect(softWhiteIdx).toBeLessThan(secretIdx);
  });
});
