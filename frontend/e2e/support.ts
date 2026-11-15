import AxeBuilder from "@axe-core/playwright";
import { expect, type Page } from "@playwright/test";

/** A small, reproducible classification dataset: y depends on x (with noise), z is noise. */
export function toyCsv(rows = 300, seed = 7): string {
  let state = seed;
  const rand = () => {
    state = (state * 1664525 + 1013904223) % 4294967296;
    return state / 4294967296;
  };
  const colors = ["red", "green", "blue"];
  const lines = ["x,z,color,y"];
  for (let i = 0; i < rows; i++) {
    const x = (rand() - 0.5) * 6;
    const z = (rand() - 0.5) * 2;
    const color = colors[Math.floor(rand() * 3)]!;
    const y = x + (rand() - 0.5) * 2 > 0 ? "pos" : "neg";
    lines.push(`${x.toFixed(3)},${z.toFixed(3)},${color},${y}`);
  }
  return lines.join("\n") + "\n";
}

export const AXE_TAGS = ["wcag2a", "wcag2aa", "wcag21a", "wcag21aa", "wcag22aa"];

/** Run axe on the current page. Serious or critical violations fail the test; the rest are returned
 *  so they can be listed in the notes. */
export async function checkA11y(page: Page, label: string) {
  const result = await new AxeBuilder({ page }).withTags(AXE_TAGS).analyze();
  const blocking = result.violations.filter(
    (v) => v.impact === "serious" || v.impact === "critical",
  );
  expect(
    blocking.map(
      (v) =>
        `${label}: ${v.id} (${v.impact}) ${v.nodes
          .slice(0, 2)
          .map((n) => n.target.join(" "))
          .join(" | ")}`,
    ),
  ).toEqual([]);
  return result.violations
    .filter((v) => !blocking.includes(v))
    .map((v) => `${label}: ${v.id} (${v.impact})`);
}

export async function settled(page: Page) {
  await page.waitForLoadState("networkidle");
  await expect(page.locator("main")).toBeVisible();
}

export const SEEDED_ROUTES = [
  "/",
  "/datasets",
  "/datasets/abalone-pool/v/1",
  "/datasets/abalone-pool/v/1/profile",
  "/datasets/abalone-pool/v/1/validation",
  "/datasets/abalone-pool/v/1/task",
  "/training",
  "/training/new",
  "/training/1",
  "/experiments",
  "/models",
  "/models/abalone",
  "/models/abalone/v/1",
  "/deployment",
  "/deployment/try",
  "/monitoring",
  "/monitoring/reports",
  "/monitoring/alerts",
  "/monitoring/predictions",
];
