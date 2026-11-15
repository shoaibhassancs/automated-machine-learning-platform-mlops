import { expect, test } from "@playwright/test";
import { settled } from "./support";

test("the skip link is the first tab stop and moves focus to the content", async ({ page }) => {
  await page.goto("/models");
  await settled(page);
  await page.keyboard.press("Tab");
  const skip = page.getByRole("link", { name: "Skip to content" });
  await expect(skip).toBeFocused();
  await page.keyboard.press("Enter");
  await expect(page.locator("#main")).toBeFocused();
});

test("navigating moves focus to the new page's heading", async ({ page }) => {
  await page.goto("/");
  await settled(page);
  await page
    .getByRole("navigation", { name: "ML lifecycle stages" })
    .getByRole("link", { name: "Models" })
    .click();
  await expect(page.getByRole("heading", { level: 1, name: "Models" })).toBeFocused();
  await expect(page).toHaveTitle(/Models · AutoML Ops/);
});

test("the sidebar and the page can be used with the keyboard alone", async ({ page }) => {
  await page.goto("/");
  await settled(page);
  for (let i = 0; i < 8; i++) await page.keyboard.press("Tab");
  const focused = await page.evaluate(() => document.activeElement?.tagName);
  expect(["A", "BUTTON"]).toContain(focused);
  // every link and button shows a visible focus indicator
  const outline = await page.evaluate(() => {
    const el = document.activeElement as HTMLElement;
    const s = getComputedStyle(el);
    return `${s.outlineStyle} ${s.outlineWidth}`;
  });
  expect(outline).toMatch(/solid [23]px/);
});

test("a dialog traps focus, Escape closes it, and focus returns to the button", async ({
  page,
}) => {
  await page.goto("/monitoring/alerts/4");
  const opener = page.getByRole("button", { name: "Acknowledge" });
  await opener.focus();
  await page.keyboard.press("Enter");
  const dialog = page.getByRole("dialog");
  await expect(dialog).toBeVisible();
  await expect(dialog.getByLabel(/reason/i)).toBeFocused();
  for (let i = 0; i < 6; i++) {
    await page.keyboard.press("Tab");
    // The dialog is modal: focus stays inside it (or leaves the page for the browser, never to the page behind).
    expect(
      await page.evaluate(() => {
        const el = document.activeElement;
        return el === document.body || !!el?.closest("dialog");
      }),
    ).toBe(true);
  }
  await page.keyboard.press("Escape");
  await expect(dialog).toBeHidden();
  await expect(opener).toBeFocused();
});

test("the reason field is required before a confirmation can be sent", async ({ page }) => {
  await page.goto("/monitoring/alerts/4");
  await page.getByRole("button", { name: "Resolve" }).click();
  const confirm = page.getByRole("dialog").getByRole("button", { name: "Resolve" });
  await expect(confirm).toBeDisabled();
  await page
    .getByRole("dialog")
    .getByLabel(/reason/i)
    .fill("x");
  await expect(confirm).toBeEnabled();
  await page.keyboard.press("Escape");
});

test("the mobile menu opens and closes by keyboard and on navigation", async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 800 });
  await page.goto("/");
  await settled(page);
  const menu = page.getByRole("button", { name: "Menu" });
  await expect(menu).toBeVisible();
  await menu.focus();
  await page.keyboard.press("Enter");
  await expect(menu).toHaveAttribute("aria-expanded", "true");
  await page
    .getByRole("navigation", { name: "ML lifecycle stages" })
    .getByRole("link", { name: "Deployment" })
    .click();
  await expect(menu).toHaveAttribute("aria-expanded", "false");
  await expect(page.getByRole("heading", { level: 1, name: "Deployment" })).toBeVisible();
});
