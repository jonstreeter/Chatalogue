import { test, expect } from '@playwright/test';
import { ChannelsPage } from './pages/ChannelsPage';

test.describe('Channels List', () => {
  let channelsPage: ChannelsPage;

  test.beforeEach(async ({ page }) => {
    channelsPage = new ChannelsPage(page);
    await channelsPage.goto();
  });

  test('loads the channels page', async () => {
    await channelsPage.expectLoaded();
  });

  test('shows add channel form', async () => {
    await expect(channelsPage.addChannelInput).toBeVisible();
    await expect(channelsPage.addChannelButton).toBeVisible();
  });

  test('shows import archive button', async () => {
    await expect(channelsPage.importButton).toBeVisible();
  });

  test('can navigate to a channel', async ({ page }) => {
    const count = await channelsPage.getChannelCount();
    if (count > 0) {
      // Click the first "Open" link
      await page.locator('.glass-panel').first().getByRole('link', { name: 'Open' }).click();
      await expect(page).toHaveURL(/\/channel\/\d+/);
    }
  });
});
