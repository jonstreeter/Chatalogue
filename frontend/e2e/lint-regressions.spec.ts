import { test, expect } from '@playwright/test';
import { AxiosError, AxiosHeaders } from 'axios';
import { getErrorDetail, getErrorMessage, getErrorStatus } from '../src/lib/errors';

test.use({ launchOptions: process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH
    ? { executablePath: process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH } : {} });

test('API failures retain server detail and safely handle other thrown values', () => {
    const error = new AxiosError('Request failed', undefined, undefined, undefined, {
        status: 422, statusText: 'Invalid', headers: {}, config: { headers: new AxiosHeaders() },
        data: { detail: 'The clip range is invalid' },
    });
    expect(getErrorDetail(error)).toBe('The clip range is invalid');
    expect(getErrorStatus(error)).toBe(422);
    expect(getErrorMessage(error)).toBe('Request failed');
    for (const value of [null, undefined, 'plain failure', Symbol('failure'), { response: { data: { detail: 'unverified' } } }]) {
        expect(getErrorDetail(value)).toBeUndefined();
        expect(getErrorStatus(value)).toBeUndefined();
        expect(getErrorMessage(value)).toBeUndefined();
    }
    expect(getErrorMessage(new Error('Native failure'))).toBe('Native failure');
});

test('draft and reconstruction visibility changes preserve hook ordering', async ({ page }) => {
    const failures: string[] = [];
    page.on('pageerror', error => failures.push(error.message));
    await page.route('http://127.0.0.1:8011/**', route => route.abort());
    await page.goto('http://127.0.0.1:5175/e2e/fixtures/lint-hooks.html');
    for (let index = 0; index < 2; index++) {
        await page.getByRole('button', { name: 'Open draft', exact: true }).click();
        await expect(page.getByText('Clip Editor', { exact: true })).toBeVisible();
        await expect(page.locator('input[type="text"]').first()).toHaveValue('Regression clip');
        await page.locator('input[type="text"]').first().fill('Edited title');
        await page.getByRole('button', { name: 'Close draft', exact: true }).click();
        await expect(page.getByText('Clip Editor', { exact: true })).toHaveCount(0);
        await page.getByRole('button', { name: 'Toggle reconstruction', exact: true }).click();
        await expect(page.getByText('Voice Queue', { exact: true })).toBeVisible();
        await page.getByRole('button', { name: 'Toggle reconstruction', exact: true }).click();
        await expect(page.getByText('Voice Queue', { exact: true })).toHaveCount(0);
    }
    expect(failures).toEqual([]);
});
