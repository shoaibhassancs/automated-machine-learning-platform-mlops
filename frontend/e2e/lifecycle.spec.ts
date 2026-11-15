import { expect, test } from "@playwright/test";
import { checkA11y, toyCsv } from "./support";

// The whole lifecycle through the browser against an empty, throwaway backend:
// upload -> inspect -> configure -> train -> compare -> register -> promote -> deploy -> predict ->
// feedback -> switch -> roll back -> monitor.
test.describe.configure({ mode: "serial" });
test.setTimeout(240_000);

const csv = (name: string, content: string) => ({
  name,
  mimeType: "text/csv",
  buffer: Buffer.from(content),
});

test("an empty stack guides a new user, and every empty state is accessible", async ({ page }) => {
  const notes: string[] = [];
  await page.goto("/");
  await expect(page.getByText("Start by uploading a CSV file.")).toBeVisible();
  await expect(page.getByText("Throwaway demo stack")).toBeVisible();
  notes.push(...(await checkA11y(page, "overview (empty)")));
  const empties: [string, string][] = [
    ["/datasets", "No datasets yet"],
    ["/training", "No training jobs yet"],
    ["/experiments", "No finished jobs yet"],
    ["/models", "No registered models yet"],
    ["/deployment", "Nothing is deployed"],
    ["/monitoring", "Nothing is deployed"],
  ];
  for (const [route, text] of empties) {
    await page.goto(route);
    await expect(page.getByText(text).first()).toBeVisible();
    notes.push(...(await checkA11y(page, `${route} (empty)`)));
  }
  test.info().annotations.push({ type: "a11y-notes", description: notes.join("; ") || "none" });
});

test("a bad CSV shows the API's own message and nothing is stored", async ({ page }) => {
  await page.goto("/datasets");
  await page.getByLabel(/choose a file/i).setInputFiles(csv("empty.csv", ""));
  await page.getByRole("button", { name: /upload and inspect/i }).click();
  const alert = page.getByRole("alert");
  await expect(alert).toContainText("The upload was rejected");
  await expect(alert).toContainText(/Error code/);
  await expect(page.getByText("No datasets yet")).toBeVisible();
});

test("upload a CSV, inspect it, and configure the task", async ({ page }) => {
  await page.goto("/datasets");
  await page.getByLabel(/choose a file/i).setInputFiles(csv("toy.csv", toyCsv()));
  await expect(page.getByLabel("Dataset name")).toHaveValue("toy");
  await page.getByRole("button", { name: /upload and inspect/i }).click();
  await expect(page.getByRole("heading", { level: 1 })).toHaveText("toy · version 1");
  await expect(page.getByText("300").first()).toBeVisible();
  await expect(page.getByRole("heading", { name: "Sample rows" })).toBeVisible();
  await expect(page.getByRole("heading", { name: "Distributions" })).toBeVisible();

  await page.getByRole("link", { name: "Profile" }).click();
  await expect(page.getByRole("table", { name: "Column profile" })).toContainText("color");
  await page.getByRole("link", { name: "Validation" }).click();
  await expect(page.getByText(/No blocking problems|This dataset has errors/)).toBeVisible();

  await page.getByRole("link", { name: "Task setup" }).click();
  await page.getByLabel("Target column").selectOption("y");
  await page.getByRole("button", { name: "Check this setup" }).click();
  await expect(page.getByText("Ready to train")).toBeVisible();
  await expect(page.getByText(/Task: classification/)).toBeVisible();
  await page.getByRole("link", { name: /Train on this dataset/ }).click();
  await expect(page).toHaveURL(/\/training\/new\?.*dataset=toy/);
});

async function trainFinal(page: import("@playwright/test").Page, seed: string) {
  await page.goto("/training/new?dataset=toy&version=1&target=y&task=classification");
  await expect(page.getByText(/Ready: classification/)).toBeVisible();
  await page.getByLabel("Folds").fill("3");
  await page.getByLabel("Random seed").fill(seed);
  await page.getByLabel(/cross-validate, select, and run the final test/i).check();
  await page.getByLabel(/I understand and want to run the final test/i).check();
  await page.getByRole("button", { name: "Start training" }).click();
  await expect(page).toHaveURL(/\/training\/\d+$/);
  await expect(page.getByText("succeeded", { exact: true }).first()).toBeVisible({
    timeout: 180_000,
  });
}

test("train a final job: charts, selection rule, final test and its limitation", async ({
  page,
}) => {
  await trainFinal(page, "1");
  await expect(page.getByText(/Selected: /)).toBeVisible();
  await expect(page.getByRole("heading", { name: "Which algorithm won" })).toBeVisible();
  await expect(page.getByRole("img", { name: /Cross-validation .* by algorithm/ })).toBeVisible();
  await expect(page.getByRole("img", { name: /Confusion matrix/ }).first()).toBeVisible();
  await expect(page.getByRole("img", { name: /ROC curve/ }).first()).toBeVisible();
  await expect(page.getByRole("heading", { name: "Final test result" })).toBeVisible();
  await expect(page.getByText(/not independent or leakage-proof/)).toBeVisible();
  await page.getByText("Show the data as a table").first().click();
  await checkA11y(page, "job detail");
  await page.getByRole("button", { name: "Register as a model" }).click();
  await page.getByLabel("Model name").fill("toy");
  await page.getByRole("dialog").getByRole("button", { name: "Register" }).click();
  await expect(page).toHaveURL(/\/models\/toy$/);
  await expect(page.getByRole("table", { name: "Versions" })).toContainText("development");
});

test("a second job, registered as version 2, can be compared with the first", async ({ page }) => {
  await trainFinal(page, "2");
  await page.getByRole("button", { name: "Register as a model" }).click();
  await page.getByLabel("Model name").fill("toy");
  await page.getByRole("dialog").getByRole("button", { name: "Register" }).click();
  await expect(page).toHaveURL(/\/models\/toy$/);
  await expect(page.getByRole("table", { name: "Versions" })).toContainText("v2");

  await page.goto("/experiments");
  await expect(page.getByRole("table", { name: "Finished jobs" })).toBeVisible();
  await page.getByLabel(/Compare job 1/).check();
  await page.getByLabel(/Compare job 2/).check();
  await page.getByRole("button", { name: /Compare 2 selected/ }).click();
  await expect(page.getByRole("heading", { name: "Compare runs" })).toBeVisible();
  await expect(page.getByRole("table", { name: "Jobs being compared" })).toContainText("#2");
  await expect(page.getByRole("img", { name: /Cross-validation score/ })).toBeVisible();
});

test("promote, deploy, predict, record feedback", async ({ page }) => {
  await page.goto("/models/toy/v/1");
  await page.getByRole("button", { name: "Promote to staging" }).click();
  await page
    .getByRole("dialog")
    .getByLabel(/reason/i)
    .fill("looks good in review");
  await page.getByRole("dialog").getByRole("button", { name: "Promote to staging" }).click();
  await expect(page.getByText("staging", { exact: true }).first()).toBeVisible();
  await page.getByRole("button", { name: "Create reference sample" }).click();
  await expect(page.getByText("Sample rows")).toBeVisible();

  await page.getByRole("link", { name: "Deploy…" }).click();
  await expect(page.getByLabel("Version")).toHaveValue("1");
  await page.getByRole("button", { name: "Deploy this version" }).click();
  await page
    .getByRole("dialog")
    .getByLabel(/reason/i)
    .fill("first deployment");
  await page.getByRole("dialog").getByRole("button", { name: "Confirm" }).click();
  await expect(page.getByRole("heading", { name: /Live now: toy · v1/ })).toBeVisible();
  await expect(page.getByText(/serving in sync/)).toBeVisible({ timeout: 30_000 });

  await page.getByRole("link", { name: "Try a prediction" }).click();
  await page.getByLabel(/^x/).fill("2.5");
  await page.getByLabel(/^z/).fill("0.1");
  await page.getByLabel(/^color/).fill("red");
  await page.getByRole("button", { name: "Predict" }).click();
  await expect(page.getByRole("heading", { name: "Answer" })).toBeVisible();
  await expect(page.getByText("Model time")).toBeVisible();
  await expect(page.getByText(/^pos|^neg/).first()).toBeVisible();
  await page.getByLabel(/True y/).fill("pos");
  await page.getByRole("button", { name: "Record true value" }).click();
  await expect(page.getByText("True value recorded.")).toBeVisible();

  await page.goto("/monitoring/predictions");
  const table = page.getByRole("table", { name: /Predictions, newest first/ });
  await expect(table).toBeVisible();
  await expect(table.getByRole("row")).toHaveCount(2);
  await expect(table).toContainText("pos");
});

test("switch to version 2, then roll back to version 1", async ({ page }) => {
  await page.goto("/models/toy/v/2");
  await page.getByRole("button", { name: "Promote to staging" }).click();
  await page
    .getByRole("dialog")
    .getByLabel(/reason/i)
    .fill("candidate");
  await page.getByRole("dialog").getByRole("button", { name: "Promote to staging" }).click();
  await expect(page.getByText("staging", { exact: true }).first()).toBeVisible();

  await page.goto("/deployment");
  await page.getByRole("button", { name: "Switch to another version" }).click();
  const dialog = page.getByRole("dialog");
  await dialog.getByLabel("Model").selectOption("toy");
  await dialog.getByLabel("Version").selectOption("2");
  await dialog.getByLabel(/reason/i).fill("try the candidate");
  await dialog.getByRole("button", { name: "Confirm" }).click();
  await expect(page.getByRole("heading", { name: /Live now: toy · v2/ })).toBeVisible();

  await page.getByRole("button", { name: "Roll back" }).click();
  await page
    .getByRole("dialog")
    .getByLabel(/reason/i)
    .fill("candidate was worse");
  await page.getByRole("dialog").getByRole("button", { name: "Roll back" }).click();
  await expect(page.getByRole("heading", { name: /Live now: toy · v1/ })).toBeVisible();
  await expect(page.getByRole("table", { name: "Deployment history" })).toContainText("rollback");
});

test("monitoring on real traffic: too little data is explained, not hidden", async ({ page }) => {
  // (the rollback created a new deployment, so it starts with no traffic: send one real request)
  await page.goto("/deployment/try");
  await page.getByLabel(/^x/).fill("-1.5");
  await page.getByLabel(/^z/).fill("0.3");
  await page.getByLabel(/^color/).fill("blue");
  await page.getByRole("button", { name: "Predict" }).click();
  await expect(page.getByRole("heading", { name: "Answer" })).toBeVisible();
  await page.getByLabel(/True y/).fill("neg");
  await page.getByRole("button", { name: "Record true value" }).click();
  await expect(page.getByText("True value recorded.")).toBeVisible();

  await page.goto("/monitoring");
  await expect(page.getByRole("heading", { name: "Prediction activity" })).toBeVisible();
  await expect(page.getByText("Real requests")).toBeVisible();
  await page.getByRole("button", { name: "Run monitoring now" }).click();
  await expect(page.getByText("Monitoring run finished.")).toBeVisible();
  await expect(page.getByText(/Not enough data for a drift verdict/)).toBeVisible();
  await expect(page.getByText(/Not enough true values yet|No true values yet/)).toBeVisible();
  await checkA11y(page, "monitoring (real, small sample)");
});
