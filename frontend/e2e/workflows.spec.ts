import { expect, test } from "@playwright/test";
import { checkA11y, settled } from "./support";

// The seeded stack holds the demo story: a trained Abalone model that is deployed, 3,400 simulated
// predictions with a drift episode, and open alerts. These tests change its state, so they run last.
test.describe.configure({ mode: "serial" });

test("the lifecycle overview reflects the real state", async ({ page }) => {
  await page.goto("/");
  const strip = page.getByRole("navigation", { name: "ML lifecycle", exact: true });
  await expect(strip).toContainText("1 dataset");
  await expect(strip).toContainText("abalone v1 is live");
  await expect(strip).toContainText("open alert");
  await expect(page.getByRole("link", { name: "Review alerts" })).toBeVisible();
});

test("a finished job shows its winner, charts and the final test with its limitation", async ({
  page,
}) => {
  await page.goto("/training/1");
  await expect(page.getByText("Selected: Ridge regression")).toBeVisible();
  await expect(page.getByRole("img", { name: /Predicted against actual/ }).first()).toBeVisible();
  await expect(page.getByRole("img", { name: /Residuals/ }).first()).toBeVisible();
  await expect(
    page.getByText(/repeated use across jobs can influence model choices/),
  ).toBeVisible();
  // the chart's numbers can be read as a table
  await page.getByText("Show the data as a table").first().click();
  await expect(page.getByRole("table", { name: "Cross-validation scores" })).toContainText(
    "Ridge regression",
  );
});

test("monitoring: simulated traffic is badged, with activity, performance, drift and alerts", async ({
  page,
}) => {
  await page.goto("/monitoring");
  await expect(page.getByText("You are looking at simulated demo traffic")).toBeVisible();
  await expect(page.getByRole("img", { name: /Predictions per/ })).toBeVisible();
  await expect(page.getByRole("table", { name: "Live and offline scores" })).toContainText("RMSE");
  await expect(page.getByText("2 of 8")).toBeVisible();
  await expect(page.getByRole("table", { name: "Open alerts" })).toContainText("Prediction shift");
  // the time-range filter works and keeps the page usable
  await page.getByRole("button", { name: "Last hour" }).click();
  await expect(page.getByRole("button", { name: "Last hour" })).toHaveAttribute(
    "aria-pressed",
    "true",
  );
  await settled(page);
  // real and simulated are separate
  await page.getByRole("button", { name: "Real requests" }).click();
  await expect(page.getByText("No predictions in this range")).toBeVisible();
});

test("run monitoring now, then read the report", async ({ page }) => {
  await page.goto("/monitoring");
  await page.getByRole("button", { name: "Run monitoring now" }).click();
  await expect(page.getByText("Monitoring run finished.")).toBeVisible();
  await page.getByRole("link", { name: "Open the full report" }).click();
  await expect(page.getByRole("heading", { name: /Monitoring report #/ })).toBeVisible();
  await expect(page.getByRole("table", { name: "Drift per input column" })).toContainText(
    "Whole_weight",
  );
  await expect(page.getByRole("table", { name: "Alert rules" })).toContainText("Performance drop");
  await expect(page.getByText(/Verdict: triggered/)).toBeVisible();
  await expect(page.getByText("Simulated").first()).toBeVisible();
  await checkA11y(page, "report");
});

test("alert lifecycle: acknowledge, then resolve, each with a reason", async ({ page }) => {
  await page.goto("/monitoring/alerts");
  const open = page
    .getByRole("table", { name: /Alerts/ })
    .getByRole("link", { name: "Prediction shift" })
    .first();
  await open.click();
  await expect(page.getByRole("heading", { level: 1, name: "Prediction shift" })).toBeVisible();
  await page.getByRole("button", { name: "Acknowledge" }).click();
  await page
    .getByRole("dialog")
    .getByLabel(/reason/i)
    .fill("seen, checking the shifted file");
  await page.getByRole("dialog").getByRole("button", { name: "Acknowledge" }).click();
  await expect(page.getByText("Alert acknowledged.")).toBeVisible();
  await expect(page.getByText("acknowledged", { exact: true }).first()).toBeVisible();
  await expect(page.getByText(/seen, checking the shifted file/)).toBeVisible();
  await page.getByRole("button", { name: "Resolve" }).click();
  await page
    .getByRole("dialog")
    .getByLabel(/reason/i)
    .fill("demo finished");
  await page.getByRole("dialog").getByRole("button", { name: "Resolve" }).click();
  await expect(page.getByText("This alert is resolved")).toBeVisible();
  await expect(page.getByRole("button", { name: "Resolve" })).toHaveCount(0);
  await page.goto("/monitoring/alerts");
  await page.getByRole("button", { name: "Resolved" }).click();
  await expect(page.getByRole("table", { name: /Alerts/ })).toContainText("resolved");
});

test("predictions and feedback: history is paged and a label can be recorded", async ({ page }) => {
  await page.goto("/monitoring/predictions");
  await page.getByRole("button", { name: "Simulated demo traffic" }).click();
  const table = page.getByRole("table", { name: /Predictions, newest first/ });
  await expect(table.getByRole("row")).toHaveCount(26); // header + 25
  await page.getByRole("button", { name: "Load more" }).click();
  await expect(table.getByRole("row")).toHaveCount(51);
  await expect(page.getByText(/input values are not shown/i)).toBeVisible();
  await table.getByRole("button", { name: "Record true value" }).first().click();
  await page
    .getByRole("dialog")
    .getByLabel(/what actually happened/i)
    .fill("9");
  await page.getByRole("dialog").getByRole("button", { name: "Record" }).click();
  await expect(page.getByText("True value recorded.")).toBeVisible();
});

test("registry: promote the staged version to production and read its history", async ({
  page,
}) => {
  await page.goto("/models/abalone/v/1");
  await expect(page.getByText("staging", { exact: true }).first()).toBeVisible();
  await page.getByRole("button", { name: "Promote to production" }).click();
  await page
    .getByRole("dialog")
    .getByLabel(/reason/i)
    .fill("approved after review");
  await page.getByRole("dialog").getByRole("button", { name: "Promote to production" }).click();
  await expect(page.getByText("Promoted to production.")).toBeVisible();
  await page.goto("/models/abalone");
  await expect(page.getByText(/approved after review/)).toBeVisible();
  await expect(page.getByRole("table", { name: "Versions" })).toContainText("production");
});

test("an unreachable backend is explained, with a way to retry", async ({ page }) => {
  await page.route("**/127.0.0.1:18102/**", (route) => route.abort());
  await page.goto("/datasets");
  await expect(page.getByRole("alert").first()).toContainText("The backend is not reachable");
  await expect(page.getByText("Backend not reachable")).toBeVisible();
  await expect(page.getByRole("button", { name: "Try again" })).toBeVisible();
});
