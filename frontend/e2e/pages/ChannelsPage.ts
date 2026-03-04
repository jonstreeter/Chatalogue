import { Page, Locator, expect } from '@playwright/test';
import { BasePage } from './BasePage';

export class ChannelsPage extends BasePage {
  readonly heading: Locator;
  readonly addChannelInput: Locator;
  readonly addChannelButton: Locator;
  readonly importButton: Locator;
  readonly channelCards: Locator;

  constructor(page: Page) {
    super(page);
    this.heading = page.getByRole('heading', { name: 'Channels' });
    this.addChannelInput = page.getByPlaceholder('https://youtube.com/@...');
    this.addChannelButton = page.getByRole('button', { name: /Add Channel/i });
    this.importButton = page.getByRole('button', { name: /Import Archive/i });
    this.channelCards = page.locator('.glass-panel');
  }

  async goto() {
    await super.goto('/');
    await this.waitForLoad();
  }

  async expectLoaded() {
    await expect(this.heading).toBeVisible();
  }

  async getChannelCount() {
    return this.channelCards.count();
  }

  async openChannel(name: string) {
    const card = this.channelCards.filter({ hasText: name });
    await card.getByRole('link', { name: 'Open' }).click();
  }

  async scanChannel(name: string) {
    const card = this.channelCards.filter({ hasText: name });
    await card.getByRole('button', { name: /Scan/i }).click();
  }
}
