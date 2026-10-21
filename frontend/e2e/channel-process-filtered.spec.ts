import { test, expect } from '@playwright/test';

for (const scenario of ['unprocessed', 'failed', 'search', 'combined', 'completed', 'all', 'pagination']) {
  test(`Process All respects ${scenario} filters`, async ({ page }) => {
    const videos = [
      { id: 1, title: 'Match pending', status: 'pending' },
      { id: 2, title: 'Other pending', status: 'pending' },
      { id: 3, title: 'Match failed', status: 'failed' },
      { id: 4, title: 'Match completed', status: 'completed', processed: true },
      { id: 5, title: 'Match muted', status: 'pending', muted: true },
      { id: 6, title: 'Match restricted', status: 'pending', access_restricted: true },
    ].map(video => ({ channel_id: 9, processed: false, published_at: '2026-10-20', ...video }));
    if (scenario === 'pagination') {
      videos.push(...Array.from({ length: 251 }, (_, i) => ({
        id: i + 10, title: `Paged ${i}`, status: 'pending', channel_id: 9,
        processed: false, published_at: '2026-10-20',
      })));
    }
    const requests: string[] = [];
    await page.route('**/channels/9', route => route.fulfill({ json: {
      id: 9, name: 'Filter regression', source_type: 'manual', status: 'active',
    } }));
    await page.route('**/system/setup-status', route => route.fulfill({ json: { setup_completed: true } }));
    await page.route('**/videos/list?*', route => route.fulfill({ json: videos }));
    await page.route('**/*/process', route => {
      requests.push(new URL(route.request().url()).pathname);
      return route.fulfill({ json: { status: 'processing_queued' } });
    });
    await page.route('**/channels/9/process-all', route => {
      requests.push('channel-wide');
      return route.fulfill({ json: { queued: 3 } });
    });
    page.on('dialog', dialog => dialog.accept());
    await page.goto('/channel/9');
    if (['unprocessed', 'failed', 'completed', 'combined'].includes(scenario)) {
      const label = scenario === 'combined' ? 'Unprocessed' : scenario[0].toUpperCase() + scenario.slice(1);
      await page.getByRole('button', { name: label, exact: true }).click();
    }
    if (['search', 'combined', 'pagination'].includes(scenario)) {
      await page.getByPlaceholder('Search title or description...').fill(scenario === 'pagination' ? 'Paged' : 'Match');
    }
    const expected = scenario === 'pagination' ? videos.slice(6).map(v => v.id)
      : scenario === 'unprocessed' ? [1, 2]
      : scenario === 'failed' ? [3]
      : scenario === 'search' ? [1, 3]
      : scenario === 'combined' ? [1]
      : scenario === 'completed' ? [] : [1, 2, 3];
    if (!expected.length) {
      await expect(page.getByRole('button', { name: /Process All/ })).toHaveCount(0);
    } else {
      await page.getByRole('button', { name: `Process All (${expected.length})`, exact: true }).click();
      await expect(page.getByRole('button', { name: /Queueing/ })).toHaveCount(0);
    }
    expect(requests.sort()).toEqual(expected.map(id => `/videos/${id}/process`).sort());
  });
}
