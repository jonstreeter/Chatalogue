import { Page, Locator, expect } from '@playwright/test';
import { BasePage } from './BasePage';

export class ChannelDetailPage extends BasePage {
  readonly backLink: Locator;
  readonly exportButton: Locator;
  readonly deleteButton: Locator;
  readonly tabNav: Locator;

  constructor(page: Page) {
    super(page);
    this.backLink = page.getByRole('link', { name: /Back to Channels/i });
    this.exportButton = page.getByRole('button', { name: /Export Archive/i });
    this.deleteButton = page.getByRole('button', { name: /Delete Channel/i });
    this.tabNav = page.locator('.glass-panel nav');
  }

  async goto(channelId: number) {
    await super.goto(`/channel/${channelId}`);
    await this.waitForLoad();
  }

  async expectLoaded() {
    await expect(this.backLink).toBeVisible();
    await expect(this.exportButton).toBeVisible();
    await expect(this.deleteButton).toBeVisible();
  }

  async switchTab(tabName: string) {
    await this.tabNav.getByRole('link', { name: tabName }).click();
  }

  async clickDelete() {
    await this.deleteButton.click();
  }

  async clickExport() {
    await this.exportButton.click();
  }
}
