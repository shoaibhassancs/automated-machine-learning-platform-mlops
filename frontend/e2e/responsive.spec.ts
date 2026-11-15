import { expect, test } from "@playwright/test";
import { mkdirSync } from "node:fs";
import { SEEDED_ROUTES, settled } from "./support";

// Layout check at the three review widths: no horizontal page scroll, content readable, and a
// screenshot of every screen saved for the manual visual review (test-results/screens).
const WIDTHS = [1280, 900, 390];
mkdirSync("test-results/screens", { recursive: true });

for (const width of WIDTHS) {
  test.describe(`${width}px wide`, () => {
    test.use({ viewport: { width, height: 900 } });
    for (const route of [...SEEDED_ROUTES, "/monitoring/reports/4", "/monitoring/alerts/4"]) {
      test(`${route}`, async ({ page }) => {
        await page.goto(route);
        await settled(page);
        await page.waitForTimeout(500);
        const overflow = await page.evaluate(
          () => document.documentElement.scrollWidth - window.innerWidth,
        );
        expect(overflow, "the page must not scroll sideways").toBeLessThanOrEqual(1);
        await expect(page.getByRole("heading", { level: 1 })).toBeVisible();
        const name = (route.replace(/[^a-z0-9]+/gi, "_") || "home").replace(/^_/, "");
        await page.screenshot({
          path: `test-results/screens/${name}_${width}.png`,
          fullPage: true,
        });
      });
    }
  });
}
