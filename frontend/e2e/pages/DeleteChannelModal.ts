import { Page, Locator, expect } from '@playwright/test';

export class DeleteChannelModal {
  readonly page: Page;
  readonly heading: Locator;
  readonly confirmInput: Locator;
  readonly deleteButton: Locator;
  readonly cancelButton: Locator;
  readonly activeJobsWarning: Locator;

  constructor(page: Page) {
    this.page = page;
    this.heading = page.getByRole('heading', { name: /Delete Channel/i });
    this.confirmInput = page.getByPlaceholder('Channel name...');
    this.deleteButton = page.getByRole('button', { name: /Delete Permanently/i });
    this.cancelButton = page.getByRole('button', { name: /Cancel/i });
    this.activeJobsWarning = page.getByText(/Cannot delete/);
  }

  async expectVisible() {
    await expect(this.heading).toBeVisible();
  }

  async expectDeleteDisabled() {
    await expect(this.deleteButton).toBeDisabled();
  }

  async expectDeleteEnabled() {
    await expect(this.deleteButton).toBeEnabled();
  }

  async typeConfirmation(text: string) {
    await this.confirmInput.fill(text);
  }

  async confirmDelete() {
    await this.deleteButton.click();
  }

  async cancel() {
    await this.cancelButton.click();
  }

  async expectStatRow(label: string, value: string) {
    const row = this.page.locator('div').filter({ hasText: label }).filter({ hasText: value });
    await expect(row.first()).toBeVisible();
  }
}
