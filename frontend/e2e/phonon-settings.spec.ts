import { test, expect } from '@playwright/test';

test.use({ launchOptions: process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH
  ? { executablePath: process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH } : {} });

// All API requests are intercepted. These tests never contact the installed app.
test('managed Phonon setup, progress, unload and save/reload', async ({ page }) => {
  let saved: Record<string, unknown> = { hf_token: '', transcription_engine: 'auto', setup_wizard_completed: true };
  let action: Record<string, unknown> = {};
  let state = 'absent';
  let failure = '';
  await page.route('**/*', async route => {
    const req = route.request();
    const url = new URL(req.url());
    if (url.origin === 'http://127.0.0.1:5175') return route.continue();
    const path = url.pathname;
    if (path === '/system/setup-status') return route.fulfill({ json: { setup_completed: true } });
    if (path === '/settings/test-transcription-engine') {
      const body = req.postDataJSON();
      if (body.phonon_action === 'load') { action = body; state = failure ? 'error' : 'installing'; }
      else if (body.phonon_action === 'unload') state = 'installed';
      else if (state === 'installing') state = 'ready';
      return route.fulfill({ json: { status: state === 'ready' ? 'ok' : state === 'error' ? 'error' : 'idle',
        state, requested_engine: 'phonon', resolved_engine: 'phonon', model_loaded: state === 'ready',
        process_running: state === 'ready', detail: state === 'installing' ? 'Downloading Phonon' : state === 'ready' ? 'Phonon is loaded and ready' : 'Phonon is unloaded',
        runtime: state === 'ready' ? { gpu: 'Test NVIDIA GPU', dtype: 'bfloat16' } : {}, error: failure || null } });
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
  const panel = page.getByRole('region', { name: 'Phonon model' });
  await expect(page.getByLabel('Local Phonon URL')).toHaveCount(0);
  await expect(page.getByLabel('Phonon API key')).toHaveCount(0);
  await expect(page.getByRole('heading', { name: 'Whisper routing options' })).toBeVisible();
  const ready = page.waitForResponse(async response => response.url().endsWith('/settings/test-transcription-engine')
    && response.request().postDataJSON().phonon_action === 'status' && (await response.json()).state === 'ready');
  await panel.getByRole('button', { name: /^(Set up|Load) and test Phonon$/ }).click();
  await expect(panel.getByRole('status')).toHaveText('Downloading Phonon');
  await ready;
  await expect(panel.getByText('Ready', { exact: true })).toBeVisible();
  await expect(panel.getByText('Test NVIDIA GPU', { exact: false })).toBeVisible();
  expect(action.phonon_action).toBe('load');
  expect(action).not.toHaveProperty('phonon_url');
  expect(action).not.toHaveProperty('phonon_api_key');
  expect(saved.transcription_engine).toBe('auto');
  await panel.getByText('Advanced model options', { exact: true }).click();
  await page.getByLabel('Inference deadline (seconds)').fill('90');
  await page.getByRole('button', { name: 'Save Changes' }).click();
  await expect(page.getByText('Settings saved', { exact: false })).toBeVisible();
  expect(saved.phonon_request_timeout_seconds).toBe(90);
  expect(saved).not.toHaveProperty('phonon_api_key');
  await page.reload();
  await expect(engine).toHaveValue('phonon');
  await expect(panel.getByText('Ready', { exact: true })).toBeVisible();
  await page.screenshot({ path: 'test-results/phonon-settings.png', fullPage: true });
  await panel.getByRole('button', { name: 'Unload Phonon model' }).click();
  await expect(panel.getByText('Unloaded', { exact: true })).toBeVisible();
  failure = 'Phonon model download interrupted or offline; reconnect and retry setup';
  await panel.getByRole('button', { name: /^(Set up|Load) and test Phonon$/ }).click();
  await expect(panel.getByRole('alert')).toHaveText(failure);
  for (const value of ['auto', 'parakeet', 'whisper']) {
    await engine.selectOption(value);
    await expect(engine).toHaveValue(value);
  }
});

test('queue displays actual Phonon and routed Whisper provenance', async ({ page }) => {
  await page.route('**/*', async route => {
    const url = new URL(route.request().url());
    if (url.origin === 'http://127.0.0.1:5175') return route.continue();
    const path = url.pathname;
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
