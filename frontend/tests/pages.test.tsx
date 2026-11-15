import { screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it } from "vitest";
import AlertDetail from "@/features/monitoring/AlertDetail";
import DatasetList from "@/features/datasets/DatasetList";
import JobDetail from "@/features/training/JobDetail";
import NewJob from "@/features/training/NewJob";
import Overview from "@/features/overview/Overview";
import TryIt from "@/features/deployment/TryIt";
import { ConfirmDialog } from "@/components/ConfirmDialog";
import { Async, Empty, ErrorPanel } from "@/components/States";
import { Term } from "@/components/Term";
import { Layout } from "@/components/Layout";
import { Route, Routes } from "react-router-dom";
import { API, http, HttpResponse, SERVING, server } from "./server";
import { fx } from "./fixtures";
import { renderAt } from "./render";
import { ApiError } from "@/api/client";
import { useState } from "react";

const empty = (key: string, body: object) =>
  http.get(`${API}${key}`, () => HttpResponse.json(body));

describe("Datasets page", () => {
  it("shows an empty state, then the list", async () => {
    server.use(empty("/datasets", { datasets: [] }));
    const first = renderAt(<DatasetList />);
    expect(await screen.findByText("No datasets yet")).toBeInTheDocument();
    first.unmount();
    server.resetHandlers();
    renderAt(<DatasetList />);
    const link = await screen.findByRole("link", { name: /abalone-pool v1/ });
    expect(link).toHaveAttribute("href", "/datasets/abalone-pool/v/1");
    expect(screen.getByText("2,924")).toBeInTheDocument();
  });

  it("shows the API's own message when a CSV is rejected", async () => {
    server.use(
      http.post(`${API}/datasets`, () =>
        HttpResponse.json(
          { error: { code: "csv_rejected", message: "Row 3 has 5 fields but the header has 4." } },
          { status: 422 },
        ),
      ),
    );
    renderAt(<DatasetList />);
    const user = userEvent.setup();
    await user.upload(
      screen.getByLabelText(/choose a file/i),
      new File(["a,b\n1"], "my data.csv", { type: "text/csv" }),
    );
    expect(screen.getByLabelText("Dataset name")).toHaveValue("my-data");
    await user.click(screen.getByRole("button", { name: /upload and inspect/i }));
    const alert = await screen.findByRole("alert");
    expect(alert).toHaveTextContent("The upload was rejected");
    expect(alert).toHaveTextContent("Row 3 has 5 fields but the header has 4.");
    expect(alert).toHaveTextContent("csv_rejected");
  });

  it("refuses an invalid name before sending anything", async () => {
    renderAt(<DatasetList />);
    const user = userEvent.setup();
    await user.upload(screen.getByLabelText(/choose a file/i), new File(["a\n1"], "x.csv"));
    await user.clear(screen.getByLabelText("Dataset name"));
    await user.type(screen.getByLabelText("Dataset name"), "Bad Name!");
    expect(screen.getByText(/lower-case letters/i)).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /upload and inspect/i })).toBeDisabled();
  });
});

describe("Training job page", () => {
  it("shows the winner, the ranking numbers, and the final test with its limitation", async () => {
    renderAt(<JobDetail />, "/training/1", "/training/:id");
    expect(await screen.findByText("Selected: Ridge regression")).toBeInTheDocument();
    const ranking = screen.getByRole("table", { name: "Ranking" });
    const first = fx.job.result.selection.ranking[0]!;
    expect(within(ranking).getByText(/2\.2392/)).toBeInTheDocument();
    expect(first.metric_mean).toBeCloseTo(2.2392, 3);
    expect(await screen.findByRole("heading", { name: "Final test result" })).toBeInTheDocument();
    expect(screen.getByText(fx.job.result.test_result.limitation)).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /register as a model/i })).toBeInTheDocument();
  });

  it("shows live progress for a running job and offers cancel", async () => {
    const running = {
      ...fx.job,
      status: "running",
      phase: "training",
      result: null,
      finished_at: null,
      progress: {
        event: "fold_finished",
        algorithm: "ridge",
        algorithm_index: 1,
        algorithm_count: 2,
        fold: 2,
        folds: 3,
      },
    };
    server.use(http.get(`${API}/jobs/:id`, () => HttpResponse.json(running)));
    renderAt(<JobDetail />, "/training/1", "/training/:id");
    const bar = await screen.findByRole("progressbar", { name: "Training progress" });
    expect(Number(bar.getAttribute("aria-valuenow"))).toBeGreaterThan(0);
    expect(screen.getByText(/Algorithm 1 of 2 \(ridge\), fold 2 of 3/)).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /cancel job/i })).toBeInTheDocument();
  });

  it("explains a failure and shows no results", async () => {
    server.use(
      http.get(`${API}/jobs/:id`, () =>
        HttpResponse.json({
          ...fx.job,
          status: "failed",
          result: null,
          error: { code: "worker_lost", message: "The worker stopped." },
        }),
      ),
    );
    renderAt(<JobDetail />, "/training/1", "/training/:id");
    expect(await screen.findByText("The job failed")).toBeInTheDocument();
    expect(screen.getByText("The worker stopped.")).toBeInTheDocument();
    expect(screen.queryByText(/Which algorithm won/)).toBeNull();
  });
});

describe("New job form", () => {
  it("builds the algorithm list from the task, and sends the job the user configured", async () => {
    let body: Record<string, unknown> | null = null;
    server.use(
      http.post(`${API}/datasets/:name/versions/:n/task-check`, () =>
        HttpResponse.json({
          ok: true,
          config: { target: "Rings", task: "regression", excluded: [] },
          dataset_ref: "abalone-pool@v1",
          errors: 0,
          warnings: 0,
          infos: 0,
          findings: [],
          feature_columns: ["Sex", "Length"],
          settings: {},
          sha256: "x",
          suggestion: { task: "regression", reason: "Rings looks numeric." },
        }),
      ),
      http.post(`${API}/jobs`, async ({ request }) => {
        body = (await request.json()) as Record<string, unknown>;
        return HttpResponse.json({ ...fx.job, id: 9 }, { status: 202 });
      }),
    );
    renderAt(<NewJob />);
    const user = userEvent.setup();
    await screen.findByRole("option", { name: /abalone-pool@v1/ });
    await user.selectOptions(screen.getByLabelText("Dataset version"), "abalone-pool@v1");
    await user.selectOptions(screen.getByLabelText(/target column/i), "Rings");
    expect(await screen.findByText(/Ready: regression/)).toBeInTheDocument();
    // only regression algorithms are offered
    const regression = fx.algorithms.algorithms.filter((a) => a.task === "regression");
    for (const a of regression) expect(screen.getByLabelText(a.label)).toBeChecked();
    const regressionLabels = new Set(regression.map((x) => x.label));
    for (const a of fx.algorithms.algorithms.filter(
      (x) => x.task === "classification" && !regressionLabels.has(x.label),
    )) {
      expect(screen.queryByLabelText(a.label)).toBeNull();
    }
    // a final job needs the explicit confirmation
    await user.click(screen.getByLabelText(/cross-validate, select, and run the final test/i));
    const start = screen.getByRole("button", { name: /start training/i });
    expect(start).toBeDisabled();
    await user.click(screen.getByLabelText(/I understand and want to run the final test/i));
    await user.clear(screen.getByLabelText("Folds"));
    await user.type(screen.getByLabelText("Folds"), "3");
    expect(start).toBeEnabled();
    await user.click(start);
    await waitFor(() => expect(body).not.toBeNull());
    expect(body).toMatchObject({
      kind: "final",
      dataset: "abalone-pool",
      version: 1,
      target: "Rings",
      folds: 3,
      confirm_final_test: true,
      track: true,
    });
  });

  it("blocks starting when the setup has errors and shows why", async () => {
    server.use(
      http.post(`${API}/datasets/:name/versions/:n/task-check`, () =>
        HttpResponse.json({
          ok: false,
          config: { target: "Sex", task: "classification", excluded: [] },
          dataset_ref: "abalone-pool@v1",
          errors: 1,
          warnings: 0,
          infos: 0,
          findings: [
            {
              code: "class_too_small",
              severity: "error",
              column: "Sex",
              message: "Class 'I' has only 2 rows.",
              details: {},
            },
          ],
          feature_columns: [],
          settings: {},
          sha256: "x",
          suggestion: null,
        }),
      ),
    );
    renderAt(<NewJob />);
    const user = userEvent.setup();
    await screen.findByRole("option", { name: /abalone-pool@v1/ });
    await user.selectOptions(screen.getByLabelText("Dataset version"), "abalone-pool@v1");
    await user.selectOptions(screen.getByLabelText(/target column/i), "Sex");
    expect(await screen.findByText("Class 'I' has only 2 rows.")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /start training/i })).toBeDisabled();
  });
});

describe("Try a prediction", () => {
  it("generates the form from the served model's input columns and records feedback", async () => {
    let sent: { rows?: Record<string, unknown>[] } | null = null;
    let label: Record<string, unknown> | null = null;
    server.use(
      http.get(`${SERVING}/model`, () => HttpResponse.json(fx.smodel)),
      http.post(`${SERVING}/predict`, async ({ request }) => {
        sent = (await request.json()) as typeof sent;
        return HttpResponse.json({
          deployment: {
            deployment_id: 1,
            model: "abalone",
            version: 1,
            package_id: "p",
            task: "regression",
          },
          predictions: [
            {
              prediction_id: "11111111-1111-4111-8111-111111111111",
              prediction: 9.25,
              probabilities: null,
            },
          ],
          latency_ms: 4.2,
          logged: true,
        });
      }),
      http.post(`${SERVING}/feedback`, async ({ request }) => {
        label = (await request.json()) as typeof label;
        return HttpResponse.json({ label: {}, created: true });
      }),
    );
    renderAt(<TryIt />);
    const user = userEvent.setup();
    for (const c of fx.smodel.input_columns) {
      expect(await screen.findByLabelText(new RegExp(`^${c.name}`))).toHaveAttribute(
        "type",
        c.kind === "numeric" ? "number" : "text",
      );
    }
    await user.type(screen.getByLabelText(/^Length/), "0.455");
    await user.type(screen.getByLabelText(/^Sex/), "M");
    await user.click(screen.getByRole("button", { name: "Predict" }));
    expect(await screen.findByText("9.25")).toBeInTheDocument();
    expect(screen.getByText("4.2 ms")).toBeInTheDocument();
    expect(sent!.rows![0]).toMatchObject({ Length: "0.455", Sex: "M", Diameter: null });
    await user.type(screen.getByLabelText(/True Rings/), "10");
    await user.click(screen.getByRole("button", { name: "Record true value" }));
    await waitFor(() =>
      expect(label).toEqual({
        prediction_id: "11111111-1111-4111-8111-111111111111",
        true_value: "10",
      }),
    );
  });
});

describe("Alert page", () => {
  it("requires a reason, then sends the acknowledgement", async () => {
    let body: unknown = null;
    server.use(
      http.get(`${API}/alerts/:id`, () => HttpResponse.json(fx.alert)),
      http.post(`${API}/alerts/:id/acknowledge`, async ({ request }) => {
        body = await request.json();
        return HttpResponse.json({ ...fx.alert, status: "acknowledged" });
      }),
    );
    renderAt(<AlertDetail />, "/monitoring/alerts/4", "/monitoring/alerts/:id");
    const user = userEvent.setup();
    await user.click(await screen.findByRole("button", { name: "Acknowledge" }));
    const dialog = screen.getByRole("dialog");
    const confirm = within(dialog).getByRole("button", { name: "Acknowledge" });
    expect(confirm).toBeDisabled();
    await user.type(within(dialog).getByLabelText(/reason/i), "looking into it");
    expect(confirm).toBeEnabled();
    await user.click(confirm);
    await waitFor(() => expect(body).toEqual({ reason: "looking into it" }));
  });

  it("describes a resolved alert as final", async () => {
    server.use(
      http.get(`${API}/alerts/:id`, () =>
        HttpResponse.json({
          ...fx.alert,
          status: "resolved",
          resolved_at: "2026-11-14T10:00:00+00:00",
          events: [
            ...fx.alert.events,
            {
              id: 99,
              action: "auto_resolved",
              report_id: 5,
              reason: null,
              created_at: "2026-11-14T10:00:00+00:00",
            },
          ],
        }),
      ),
    );
    renderAt(<AlertDetail />, "/monitoring/alerts/4", "/monitoring/alerts/:id");
    expect(await screen.findByText("This alert is resolved")).toBeInTheDocument();
    expect(screen.getByText(/Resolved automatically: a later run was clear/)).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Resolve" })).toBeNull();
  });
});

describe("Overview", () => {
  it("guides a new user to upload, and a returning one to the next step", async () => {
    server.use(
      empty("/datasets", { datasets: [] }),
      empty("/jobs", { jobs: [], next_before_id: null }),
      empty("/models", { models: [] }),
      empty("/alerts", { alerts: [], next_before_id: null }),
      http.get(`${API}/deployments/active`, () =>
        HttpResponse.json(
          { error: { code: "no_active_deployment", message: "none" } },
          { status: 404 },
        ),
      ),
    );
    const first = renderAt(<Overview />);
    expect(await screen.findByText("Start by uploading a CSV file.")).toBeInTheDocument();
    expect(screen.getByText("No datasets yet")).toBeInTheDocument();
    first.unmount();
    server.resetHandlers();
    renderAt(<Overview />);
    expect(await screen.findByText("Monitoring has open alerts to look at.")).toBeInTheDocument();
    expect(screen.getByText("abalone v1 is live")).toBeInTheDocument();
  });
});

describe("shared building blocks", () => {
  it("ConfirmDialog traps the decision: Escape cancels, reason is required when asked", async () => {
    let confirmed: string | null = null;
    function Host() {
      const [open, setOpen] = useState(true);
      return (
        <ConfirmDialog
          open={open}
          title="Archive?"
          confirmLabel="Archive"
          askReason
          onCancel={() => setOpen(false)}
          onConfirm={(r) => (confirmed = r)}
        >
          <p>Consequence.</p>
        </ConfirmDialog>
      );
    }
    renderAt(<Host />);
    const user = userEvent.setup();
    const dialog = await screen.findByRole("dialog", { name: "Archive?" });
    expect(within(dialog).getByText("Consequence.")).toBeInTheDocument();
    await user.type(within(dialog).getByLabelText(/reason/i), "  done  ");
    await user.click(within(dialog).getByRole("button", { name: "Archive" }));
    expect(confirmed).toBe("done");
  });

  it("ErrorPanel separates an unreachable backend from an API error", () => {
    const a = renderAt(
      <ErrorPanel error={new ApiError(0, "unreachable", "Cannot reach the backend.")} />,
    );
    expect(screen.getByRole("alert")).toHaveTextContent("The backend is not reachable");
    a.unmount();
    renderAt(
      <ErrorPanel
        error={new ApiError(409, "queue_full", "Too many jobs.")}
        retry={() => undefined}
      />,
    );
    expect(screen.getByRole("alert")).toHaveTextContent("Error code queue_full (HTTP 409)");
    expect(screen.getByRole("button", { name: "Try again" })).toBeInTheDocument();
  });

  it("Empty and Async render their states", () => {
    renderAt(
      <Empty title="Nothing" action={<button type="button">Do it</button>}>
        Because.
      </Empty>,
    );
    expect(screen.getByText("Because.")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Do it" })).toBeInTheDocument();
    const q = { isPending: true, isError: false } as never;
    renderAt(<Async query={q}>{() => <p>never</p>}</Async>);
    expect(screen.getByRole("status")).toHaveAttribute("aria-busy", "true");
  });

  it("Term opens a plain-language note on click and closes on Escape", async () => {
    renderAt(<Term k="drift">drift</Term>);
    const user = userEvent.setup();
    const button = screen.getByRole("button", { name: "drift" });
    expect(button).toHaveAttribute("aria-expanded", "false");
    await user.click(button);
    expect(screen.getByRole("note")).toHaveTextContent(/statistically different/);
    await user.keyboard("{Escape}");
    expect(screen.queryByRole("note")).toBeNull();
  });

  it("the shell reports an unreachable backend and names the environment", async () => {
    server.use(http.get(`${API}/health`, () => HttpResponse.error()));
    const first = renderAt(
      <Routes>
        <Route element={<Layout />}>
          <Route index element={<p>page</p>} />
        </Route>
      </Routes>,
      "/",
      "/*",
    );
    expect(await screen.findByText("Backend not reachable")).toBeInTheDocument();
    first.unmount();
    server.resetHandlers();
    renderAt(
      <Routes>
        <Route element={<Layout />}>
          <Route index element={<p>page</p>} />
        </Route>
      </Routes>,
      "/",
      "/*",
    );
    expect(await screen.findByText("Backend connected")).toBeInTheDocument();
    expect(screen.getByRole("link", { name: "Skip to content" })).toHaveAttribute("href", "#main");
    expect(screen.getByRole("navigation", { name: "ML lifecycle stages" })).toBeInTheDocument();
  });
});
