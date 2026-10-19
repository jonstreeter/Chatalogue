import { test, expect, type Page } from '@playwright/test';

test.use({ launchOptions: process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH
  ? { executablePath: process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH } : {} });

const origin = 'http://127.0.0.1:5175';

// Local PCM fixture: Chromium decodes real media; no production media is requested.
function audioFixture() {
  const samples = 16000 * 15;
  const wav = Buffer.alloc(44 + samples * 2);
  wav.write('RIFF'); wav.writeUInt32LE(wav.length - 8, 4); wav.write('WAVEfmt ', 8);
  wav.writeUInt32LE(16, 16); wav.writeUInt16LE(1, 20); wav.writeUInt16LE(1, 22);
  wav.writeUInt32LE(16000, 24); wav.writeUInt32LE(32000, 28);
  wav.writeUInt16LE(2, 32); wav.writeUInt16LE(16, 34);
  wav.write('data', 36); wav.writeUInt32LE(samples * 2, 40);
  return wav;
}

async function fixture(page: Page) {
  page.on('pageerror', error => console.error('Browser error:', error.message));
  const data = { saved: { hf_token: '', transcription_engine: 'auto', setup_wizard_completed: true } as Record<string, unknown>,
    state: 'absent', failure: '', actions: [] as Record<string, unknown>[], saves: 0, pollReady: false };
  await page.route('**/*', async route => {
    const req = route.request();
    const url = new URL(req.url());
    if (url.origin === origin) return route.continue();
    const path = url.pathname;
    if (path === '/settings/db-health') return route.fulfill({ json: { database: { connections: {}, pool: {}, query_metrics: {}, provider: 'fixture', database_url: 'fixture' }, queue_depth: {} } });
    if (path === '/system/setup-status') return route.fulfill({ json: { setup_completed: true } });
    if (path === '/settings') {
      if (req.method() === 'POST') { data.saved = req.postDataJSON(); data.saves++; }
      return route.fulfill({ json: data.saved });
    }
    if (path === '/settings/test-transcription-engine') {
      const body = req.postDataJSON(); data.actions.push(body);
      if (body.engine !== 'moss') return route.fulfill({ json: { status: 'ok', state: 'ready' } });
      const action = body.moss_action || 'load';
      if (action === 'load') data.state = data.failure ? 'error' : 'loading';
      if (action === 'unload') data.state = 'installed';
      if (action === 'status' && data.state === 'loading' && data.pollReady) data.state = 'ready';
      return route.fulfill({ json: { status: data.state === 'error' ? 'error' : data.state === 'ready' ? 'ok' : 'pending',
        state: data.state, model_loaded: data.state === 'ready', process_running: ['loading', 'ready'].includes(data.state),
        detail: data.state === 'loading' ? 'Loading MOSS inference' : data.state === 'ready' ? 'MOSS inference ready' : 'MOSS unloaded',
        downloaded_bytes: 1048576, total_bytes: 2097152, error: data.failure || null,
        requested_engine: 'moss', resolved_engine: 'moss' } });
    }
    if (path === '/jobs') return route.fulfill({ json: [{ id: 901, video_id: 1, job_type: 'process', status: 'transcribing', progress: 40,
      created_at: '2026-10-19T12:00:00', video_title: 'MOSS fixture',
      payload_json: JSON.stringify({ transcription_engine_requested: 'moss', transcription_engine_used: 'moss' }) }] });
    if (path === '/videos/1') return route.fulfill({ json: { id: 1, title: 'MOSS segment fixture', status: 'completed',
      media_source_type: 'upload', media_kind: 'audio', duration: 15, channel_id: null, upload_date: '2026-10-19T12:00:00' } });
    if (path === '/videos/1/transcript-quality') return route.fulfill({ json: { quality_score: 100, quality_profile: 'fixture', metrics: {}, reasons: [] } });
    if (path === '/videos/1/segments') return route.fulfill({ json: [
      { id: 11, video_id: 1, start_time: 3, end_time: 10, speaker_id: 101, speaker: 'Alice', text: 'First native speaker turn.', words: null },
      { id: 12, video_id: 1, start_time: 5, end_time: 8, speaker_id: 102, speaker: 'Bob', text: 'Nested overlapping speaker turn.', words: null },
    ] });
    if (path === '/videos/1/media') {
      const wav = audioFixture();
      const range = req.headers().range?.match(/^bytes=(\d+)-(\d*)$/);
      if (!range) return route.fulfill({ contentType: 'audio/wav', headers: { 'Accept-Ranges': 'bytes' }, body: wav });
      const start = Number(range[1]); const end = range[2] ? Math.min(Number(range[2]), wav.length - 1) : wav.length - 1;
      return route.fulfill({ status: 206, contentType: 'audio/wav',
        headers: { 'Accept-Ranges': 'bytes', 'Content-Range': `bytes ${start}-${end}/${wav.length}` }, body: wav.subarray(start, end + 1) });
    }
    if (path.includes('worker')) return route.fulfill({ json: { status: 'online' } });
    if (path === '/ollama/models') return route.fulfill({ json: { models: [] } });
    return route.fulfill({ json: path === '/channels' || path.includes('campaign') || path.includes('summary') || path.startsWith('/videos/1/') ? [] : {} });
  });
  return data;
}

test('MOSS capability notice, read-only setup/status, failure retry, unload and explicit save', async ({ page }) => {
  const data = await fixture(page);
  await page.goto(`${origin}/settings`);
  const engine = page.getByRole('combobox', { name: 'Transcription Engine', exact: true });
  await engine.selectOption('moss');
  const panel = page.getByRole('region', { name: 'MOSS model', exact: true });
  await expect(panel.getByText(/segment timing only/)).toContainText('90 minutes');
  await expect(panel.getByText(/segment timing only/)).toContainText('whole input');
  await expect(panel.getByText('Not installed', { exact: true })).toBeVisible();
  expect(data.saves).toBe(0);
  data.failure = 'MOSS download interrupted; reconnect and retry setup';
  await panel.getByRole('button', { name: 'Set up and test MOSS' }).click();
  await expect(panel.getByRole('alert')).toHaveText(data.failure);
  data.failure = '';
  await panel.getByRole('button', { name: 'Set up and test MOSS' }).click();
  await expect(panel.getByRole('status')).toHaveText('Loading MOSS inference');
  await expect(panel.getByRole('button', { name: 'Set up and test MOSS' })).toBeDisabled();
  await expect(panel.getByRole('progressbar')).toHaveAttribute('value', '1048576');
  data.pollReady = true;
  await expect(panel.getByText('Ready', { exact: true })).toBeVisible({ timeout: 10000 });
  expect(data.saves).toBe(0);
  expect(data.saved.transcription_engine).toBe('auto');
  expect(data.actions.some(action => action.engine === 'moss' && action.moss_action === 'status')).toBe(true);
  await page.getByRole('button', { name: 'Save Changes' }).click();
  await expect(page.getByText('Settings saved', { exact: false })).toBeVisible();
  expect(data.saved.transcription_engine).toBe('moss');
  await page.reload();
  await expect(engine).toHaveValue('moss');
  await expect(panel.getByText('Ready', { exact: true })).toBeVisible();
  await page.screenshot({ path: '../output/playwright/moss-settings.png', fullPage: true });
  await panel.getByRole('button', { name: 'Unload MOSS model' }).click();
  await expect(panel.getByText('Unloaded', { exact: true })).toBeVisible();
  expect(data.actions.some(action => action.moss_action === 'unload')).toBe(true);
  for (const value of ['auto', 'whisper', 'parakeet', 'phonon']) {
    await engine.selectOption(value); await expect(engine).toHaveValue(value);
  }
});

test('wizard MOSS setup polls without saving selection', async ({ page }) => {
  const data = await fixture(page);
  await page.goto(`${origin}/settings`);
  await page.getByRole('button', { name: /^System Logging/ }).click();
  await page.getByRole('button', { name: 'Re-run Setup Wizard' }).click();
  await page.getByRole('button', { name: 'Get Started' }).click();
  await page.getByRole('radio', { name: /^MOSS/ }).check();
  await expect(page.getByText(/MOSS provides segment timing only/)).toContainText('90 minutes');
  await expect(page.getByText('All engines support word-level timestamps.', { exact: true })).toHaveCount(0);
  await page.getByRole('button', { name: 'Set up and test MOSS' }).click();
  await expect(page.getByRole('status')).toHaveText('Loading MOSS inference');
  expect(data.saves).toBe(0);
  data.pollReady = true;
  await expect(page.getByText('Engine ready!', { exact: false })).toBeVisible({ timeout: 10000 });
  expect(data.saves).toBe(0);
  await page.screenshot({ path: '../output/playwright/moss-wizard.png', fullPage: true });
  await page.getByRole('button', { name: 'Next', exact: true }).click();
  await page.getByRole('button', { name: 'Skip', exact: true }).click();
  await page.getByRole('button', { name: 'Skip', exact: true }).click();
  await page.getByRole('button', { name: 'Open Chatalogue', exact: true }).click();
  expect(data.saved.transcription_engine).toBe('moss');
});

test('queue provenance and overlapping native segment seek/playback without word clocks', async ({ page }) => {
  await fixture(page);
  await page.goto(`${origin}/jobs`);
  await expect(page.getByText('MOSS', { exact: true }).first()).toBeVisible();
  await page.screenshot({ path: '../output/playwright/moss-queue.png', fullPage: true });
  await page.goto(`${origin}/video/1`);
  const first = page.locator('#seg-11'); const nested = page.locator('#seg-12');
  await expect(first.getByText('Alice', { exact: true })).toBeVisible();
  await expect(nested.getByText('Bob', { exact: true })).toBeVisible();
  await expect(first.locator('p')).toHaveText('First native speaker turn.');
  await expect(nested.locator('p')).toHaveText('Nested overlapping speaker turn.');
  await expect(first.locator('p span')).toHaveCount(0);
  await expect(nested.locator('p span')).toHaveCount(0);
  const media = page.locator('audio').first();
  await expect.poll(() => media.evaluate((el: HTMLAudioElement) => el.readyState)).toBeGreaterThanOrEqual(2);
  await media.evaluate((el: HTMLAudioElement) => { el.muted = true; });
  await nested.locator('p').click();
  await expect.poll(() => media.evaluate((el: HTMLAudioElement) => el.currentTime)).toBeGreaterThanOrEqual(4.9);
  await expect.poll(() => media.evaluate((el: HTMLAudioElement) => el.paused)).toBe(false);
  await expect(first).toHaveClass(/bg-blue-50/);
  await expect(nested).toHaveClass(/bg-blue-50/);
  await page.screenshot({ path: '../output/playwright/moss-segment-playback.png', fullPage: true });
  await first.locator('p').click();
  await expect.poll(() => media.evaluate((el: HTMLAudioElement) => el.currentTime)).toBeLessThan(5);
});
