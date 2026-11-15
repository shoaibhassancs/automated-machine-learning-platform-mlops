import { defineConfig } from "@playwright/test";
import { FLOW, SEEDED } from "./e2e/stacks";

// Two throwaway backends (made by scripts/ui_stack.py: new database automl_test_e2e_*, temp folders,
// SQLite MLflow) and one Vite dev server for each. The real application and MLflow databases and
// data/ are never used. See e2e/global-setup.ts.
export default defineConfig({
  testDir: "./e2e",
  globalSetup: "./e2e/global-setup.ts",
  globalTeardown: "./e2e/global-teardown.ts",
  timeout: 60_000,
  expect: { timeout: 10_000 },
  workers: 1,
  fullyParallel: false,
  retries: 0,
  reporter: [["list"]],
  outputDir: "test-results",
  use: { trace: "retain-on-failure", screenshot: "only-on-failure", colorScheme: "light" },
  projects: [
    {
      name: "lifecycle",
      testMatch: /lifecycle\.spec\.ts/,
      use: { baseURL: FLOW.ui, browserName: "chromium" },
    },
    {
      name: "seeded",
      testMatch: /(a11y|keyboard|responsive|workflows)\.spec\.ts/,
      use: { baseURL: SEEDED.ui, browserName: "chromium" },
    },
  ],
});
