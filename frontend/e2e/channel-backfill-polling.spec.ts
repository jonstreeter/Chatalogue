import { test, expect } from '@playwright/test';

test('backfill keeps polling unchanged progress and clears on completion', async ({ page }) => {
  let channelReads = 0;
  let complete = false;
  await page.route('**/channels/9', async route => {
    channelReads += 1;
    await route.fulfill({ json: {
      id: 9, name: 'Polling regression', source_type: 'youtube', status: 'active',
      sync_status_detail: complete ? 'Channel is up to date.' : 'Finishing metadata backfill (yt-dlp fallback 1/47)...',
      sync_progress: complete ? 100 : 88,
    } });
  });
  await page.route('**/videos/list?*', route => route.fulfill({ json: [{
    id: 1, channel_id: 9, youtube_id: 'example', title: 'Example video',
    published_at: '2026-10-20T00:00:00', processed: false, status: 'pending',
  }] }));
  await page.goto('/channel/9');
  await expect(page.getByText('Metadata backfill in progress')).toBeVisible();
  const initialReads = channelReads;
  // Three poll periods prove that an unchanged response does not stop the loop.
  await expect.poll(() => channelReads, { timeout: 20_000 }).toBeGreaterThanOrEqual(initialReads + 3);
  complete = true;
  await expect(page.getByText('Metadata backfill in progress')).toBeHidden({ timeout: 10_000 });
  const completedReads = channelReads;
  // Allow a full poll period to establish that completion cancels polling.
  await page.waitForTimeout(5500);
  expect(channelReads).toBe(completedReads);
});
