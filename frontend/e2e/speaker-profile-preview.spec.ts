import { expect, test } from '@playwright/test';

const speakerId = process.env.PLAYWRIGHT_SPEAKER_ID ?? '26';

test('speaker profile preview can be closed and reopened/switched', async ({ page }) => {
  await page.goto(`/speakers/${speakerId}`);

  await expect(page.getByRole('heading', { name: 'Voice Profiles' })).toBeVisible();

  const previewButtons = page.locator('button:has-text("Preview Clip")');
  await expect(previewButtons.first()).toBeVisible();

  await previewButtons.first().click();

  const previewBanner = page.locator('text=Previewing profile #');
  await expect(previewBanner).toBeVisible();

  const closeButton = page.getByRole('button', { name: /^Close$/ });
  await expect(closeButton).toBeVisible();
  await closeButton.click();
  await expect(previewBanner).toHaveCount(0);

  await previewButtons.first().click();
  await expect(previewBanner).toBeVisible();

  const remainingPreviewButtons = page.locator('button:has-text("Preview Clip")');
  if (await remainingPreviewButtons.count()) {
    await remainingPreviewButtons.first().click();
    await expect(previewBanner).toBeVisible();
  }
});

