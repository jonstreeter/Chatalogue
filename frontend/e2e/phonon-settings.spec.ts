import { test, expect } from '@playwright/test';

test.use({ launchOptions: process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH
  ? { executablePath: process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH } : {} });

// All API requests are intercepted. These tests never contact the installed app.
test('optional Phonon settings test unsaved credentials and survive save/reload', async ({ page }) => {
  let saved: Record<string, unknown> = { hf_token: '', transcription_engine: 'auto', setup_wizard_completed: true };
  let probe: Record<string, unknown> = {};
  let failure = '';
  await page.route('http://127.0.0.1:8011/**', async route => {
    const req = route.request();
    const path = new URL(req.url()).pathname;
    if (path === '/system/setup-status') return route.fulfill({ json: { setup_completed: true } });
    if (path === '/settings/test-transcription-engine') {
      probe = req.postDataJSON();
      return route.fulfill({ json: failure
        ? { status: 'error', requested_engine: 'phonon', error: failure }
        : { status: 'ok', requested_engine: 'phonon', resolved_engine: 'phonon', device: 'test CUDA runtime' } });
    }
    if (path === '/settings') {
      if (req.method() === 'POST') saved = req.postDataJSON();
      return route.fulfill({ json: saved });
    }
    return route.fulfill({ json: path === '/ollama/models' ? { models: [] } : {} });
  });
  await page.goto('http://127.0.0.1:5175/settings');
  const engine = page.getByRole('combobox', { name: 'Transcription Engine', exact: true });
  await engine.selectOption('phonon');
  await expect(page.getByRole('heading', { name: 'Whisper routing options' })).toBeVisible();
  await page.getByLabel('Local Phonon URL').fill('http://localhost:8010');
  const key = page.getByLabel('Phonon API key');
  await expect(key).toHaveAttribute('type', 'password');
  await key.fill('browser-test-key');
  await page.getByText('Advanced connection settings', { exact: true }).click();
  await page.getByLabel('Inference deadline (seconds)').fill('90');
  await page.getByRole('button', { name: 'Test Selected Engine' }).click();
  await expect(page.getByText('test CUDA runtime', { exact: false })).toBeVisible();
  expect(probe.phonon_api_key).toBe('browser-test-key');
  expect(probe.phonon_url).toBe('http://localhost:8010');
  expect(saved.transcription_engine).toBe('auto');
  await page.getByRole('button', { name: 'Save Changes' }).click();
  await expect(page.getByText('Settings saved', { exact: false })).toBeVisible();
  expect(saved.phonon_request_timeout_seconds).toBe(90);
  await page.reload();
  await expect(engine).toHaveValue('phonon');
  await expect(page.getByLabel('Local Phonon URL')).toHaveValue('http://localhost:8010');
  await expect(key).toHaveValue('browser-test-key');
  failure = 'Phonon authentication failed (HTTP 401)';
  await page.getByRole('button', { name: 'Test Selected Engine' }).click();
  await expect(page.getByText('Phonon authentication failed (HTTP 401)')).toBeVisible();
  failure = 'Phonon runtime is unreachable';
  await page.getByRole('button', { name: 'Test Selected Engine' }).click();
  await expect(page.getByText('Phonon runtime is unreachable', { exact: true })).toBeVisible();
  await page.screenshot({ path: 'test-results/phonon-settings.png', fullPage: true });
  for (const value of ['auto', 'parakeet', 'whisper']) {
    await engine.selectOption(value);
    await expect(engine).toHaveValue(value);
  }
});

test('queue displays actual Phonon and routed Whisper provenance', async ({ page }) => {
  await page.route('http://127.0.0.1:8011/**', async route => {
    const path = new URL(route.request().url()).pathname;
    if (path === '/system/setup-status') return route.fulfill({ json: { setup_completed: true } });
    if (path.includes('worker')) return route.fulfill({ json: { status: 'online' } });
    if (path === '/jobs') return route.fulfill({ json: [
      { id: 901, video_id: 1, job_type: 'process', status: 'transcribing', progress: 40, created_at: '2026-10-15T12:00:00', video_title: 'Phonon test audio', payload_json: JSON.stringify({ transcription_engine_requested: 'phonon', transcription_engine_used: 'phonon' }) },
      { id: 902, video_id: 2, job_type: 'process', status: 'transcribing', progress: 20, created_at: '2026-10-15T12:00:01', video_title: 'Spanish test audio', payload_json: JSON.stringify({ transcription_engine_requested: 'phonon', transcription_engine_used: 'whisper' }) },
    ] });
    return route.fulfill({ json: path === '/channels' || path.includes('campaign') || path.includes('summary') ? [] : {} });
  });
  await page.goto('http://127.0.0.1:5175/jobs');
  await expect(page.getByText('Phonon-2', { exact: true }).first()).toBeVisible();
  await expect(page.getByText('Whisper', { exact: true }).first()).toBeVisible();
  await page.screenshot({ path: 'test-results/phonon-queue.png', fullPage: true });
});
