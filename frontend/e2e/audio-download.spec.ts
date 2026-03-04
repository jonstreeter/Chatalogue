import { test, expect } from '@playwright/test';

test.describe('Audio Download', () => {
  test('can trigger processing for a video and verify download starts', async ({ page, request }) => {
    // Check the video is in pending state via API
    const videoRes = await request.get('http://localhost:8000/videos/198');
    const video = await videoRes.json();
    expect(video.title).toContain('Lobstermaxxing');

    // If already processed, skip
    if (video.status === 'completed') {
      test.skip(true, 'Video already processed');
    }

    // Trigger processing via API
    const processRes = await request.post('http://localhost:8000/videos/198/process');
    expect(processRes.ok()).toBeTruthy();

    // Poll for download to start (status should change from 'pending' to 'downloading')
    let attempts = 0;
    let currentStatus = video.status;
    while (attempts < 30 && currentStatus === 'pending') {
      await page.waitForTimeout(2000);
      const statusRes = await request.get('http://localhost:8000/videos/198');
      const statusData = await statusRes.json();
      currentStatus = statusData.status;
      attempts++;
    }

    // The video should have moved past 'pending' — downloading, transcribing, or further
    expect(['downloading', 'transcribing', 'diarizing', 'completed']).toContain(currentStatus);
  });
});
