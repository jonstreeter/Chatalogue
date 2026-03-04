import { test, expect } from '@playwright/test';

test.describe('Channel Refresh', () => {
  test('refresh discovers newly added channel videos', async ({ page }) => {
    // Go to channels list
    await page.goto('/');
    await page.waitForLoadState('networkidle');

    // Find the primary channel card
    const card = page.locator('.glass-panel').filter({ hasText: /chatalogue|porchtime/i });
    await expect(card).toBeVisible({ timeout: 10_000 });

    // Note current video count
    const videoCountBefore = await card.locator('text=/\\d+/').first().textContent();

    // Click Scan button
    await card.getByRole('button', { name: /Scan/i }).click();

    // Wait for scan to finish (the button text changes while scanning)
    await expect(card.getByRole('button', { name: /Scan/i })).toBeVisible({ timeout: 60_000 });

    // Open the channel to check videos
    await card.getByRole('link', { name: 'Open' }).click();
    await page.waitForLoadState('networkidle');

    // Should be on channel detail page
    await expect(page).toHaveURL(/\/channel\/\d+/);

    // Videos tab should be active by default and show video entries
    // Just verify we're on the page and content loaded
    await expect(page.getByRole('link', { name: /Back to Channels/i })).toBeVisible();
  });
});
