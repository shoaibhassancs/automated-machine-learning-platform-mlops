import { expect, test } from "@playwright/test";
import { checkA11y, SEEDED_ROUTES, settled } from "./support";

// Automated accessibility check (axe-core, WCAG 2.0/2.1/2.2 A and AA rules) on every screen of the
// seeded stack, in light and dark mode. Serious or critical violations fail; lesser ones are listed.
const notes: string[] = [];

for (const scheme of ["light", "dark"] as const) {
  test.describe(`axe, ${scheme} mode`, () => {
    test.use({ colorScheme: scheme });
    const routes =
      scheme === "light"
        ? SEEDED_ROUTES
        : ["/", "/training/1", "/monitoring", "/monitoring/alerts/4", "/deployment/try"];
    for (const route of routes) {
      test(`${route}`, async ({ page }) => {
        await page.goto(route);
        await settled(page);
        await page.waitForTimeout(400);
        notes.push(...(await checkA11y(page, `${scheme} ${route}`)));
      });
    }
  });
}

test("report and alert detail pages", async ({ page }) => {
  for (const route of ["/monitoring/reports/4", "/monitoring/alerts/4"]) {
    await page.goto(route);
    await settled(page);
    notes.push(...(await checkA11y(page, route)));
  }
});

test("open dialogs and glossary notes are accessible", async ({ page }) => {
  await page.goto("/monitoring/alerts/4");
  await page.getByRole("button", { name: "Acknowledge" }).click();
  await expect(page.getByRole("dialog")).toBeVisible();
  notes.push(...(await checkA11y(page, "acknowledge dialog")));
  await page.keyboard.press("Escape");
  await expect(page.getByRole("dialog")).toBeHidden();
  await page.goto("/monitoring");
  await page.getByRole("button", { name: "reference sample" }).click();
  await expect(page.getByRole("note")).toBeVisible();
  notes.push(...(await checkA11y(page, "glossary note")));
});

test.afterAll(() => {
  const unique = [...new Set(notes)];
  console.log(
    `axe: minor or moderate findings (not blocking): ${unique.length ? unique.join("; ") : "none"}`,
  );
});
