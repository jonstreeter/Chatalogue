import { test, expect } from '@playwright/test';
import { ChannelsPage } from './pages/ChannelsPage';
import { ChannelDetailPage } from './pages/ChannelDetailPage';
import { DeleteChannelModal } from './pages/DeleteChannelModal';

test.describe('Channel Detail', () => {
  // These tests require at least one channel to exist.
  // Run against a live dev server with data.

  test('shows export and delete buttons', async ({ page }) => {
    // Navigate to channels, then open the first one
    const channelsPage = new ChannelsPage(page);
    await channelsPage.goto();

    const count = await channelsPage.getChannelCount();
    test.skip(count === 0, 'No channels available to test');

    await page.locator('.glass-panel').first().getByRole('link', { name: 'Open' }).click();
    await page.waitForLoadState('networkidle');

    const detail = new ChannelDetailPage(page);
    await detail.expectLoaded();
  });

  test('tab navigation works', async ({ page }) => {
    const channelsPage = new ChannelsPage(page);
    await channelsPage.goto();

    const count = await channelsPage.getChannelCount();
    test.skip(count === 0, 'No channels available to test');

    await page.locator('.glass-panel').first().getByRole('link', { name: 'Open' }).click();
    await page.waitForLoadState('networkidle');

    const detail = new ChannelDetailPage(page);

    // Click through tabs
    for (const tab of ['Transcripts', 'Speakers', 'Search', 'Videos']) {
      await detail.switchTab(tab);
      await page.waitForLoadState('networkidle');
    }
  });

  test('delete modal opens and requires confirmation', async ({ page }) => {
    const channelsPage = new ChannelsPage(page);
    await channelsPage.goto();

    const count = await channelsPage.getChannelCount();
    test.skip(count === 0, 'No channels available to test');

    await page.locator('.glass-panel').first().getByRole('link', { name: 'Open' }).click();
    await page.waitForLoadState('networkidle');

    const detail = new ChannelDetailPage(page);
    await detail.clickDelete();

    const modal = new DeleteChannelModal(page);
    await modal.expectVisible();
    await modal.expectDeleteDisabled();

    // Type wrong name — button should stay disabled
    await modal.typeConfirmation('wrong name');
    await modal.expectDeleteDisabled();

    // Cancel should close the modal
    await modal.cancel();
    await expect(modal.heading).not.toBeVisible();
  });

  test('export triggers a download', async ({ page }) => {
    const channelsPage = new ChannelsPage(page);
    await channelsPage.goto();

    const count = await channelsPage.getChannelCount();
    test.skip(count === 0, 'No channels available to test');

    await page.locator('.glass-panel').first().getByRole('link', { name: 'Open' }).click();
    await page.waitForLoadState('networkidle');

    const detail = new ChannelDetailPage(page);

    // Listen for download event
    const [download] = await Promise.all([
      page.waitForEvent('download'),
      detail.clickExport(),
    ]);

    expect(download.suggestedFilename()).toMatch(/_archive\.json$/);
  });
});
