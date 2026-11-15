import { Link } from "react-router-dom";
import { useActiveDeployment, useAlerts, useDatasets, useJobs, useModels } from "@/api/queries";
import { Loading } from "@/components/States";
import { Callout, PageHeader, Panel, Section } from "@/components/ui";
import { plural } from "@/lib/format";

const STAGES = [
  {
    n: 1,
    title: "Datasets",
    to: "/datasets",
    what: "Upload a CSV. The platform keeps an immutable, checksummed version, profiles every column and flags problems before anything is trained.",
  },
  {
    n: 2,
    title: "Training",
    to: "/training",
    what: "Several algorithms are cross-validated on the training rows. One is selected by a rule you can read, then tested once on rows held back.",
  },
  {
    n: 3,
    title: "Models",
    to: "/models",
    what: "A trained package is registered as a version and moves through development, staging and production, with every change recorded.",
  },
  {
    n: 4,
    title: "Deployment",
    to: "/deployment",
    what: "One version is served over a REST API. You can try predictions, switch versions, and roll back without retraining.",
  },
  {
    n: 5,
    title: "Monitoring",
    to: "/monitoring",
    what: "Live predictions are compared with the training data (drift) and, once true outcomes arrive, with the offline score. Alerts open and close on their own.",
  },
];

export default function Overview() {
  const datasets = useDatasets();
  const jobs = useJobs({}, 20);
  const models = useModels();
  const active = useActiveDeployment();
  const openAlerts = useAlerts({ status: "open" }, 100);

  const ready = [datasets, jobs, models, active, openAlerts].every((q) => !q.isPending);
  const failed = [datasets, jobs, models, active, openAlerts].find((q) => q.isError);

  const nDatasets = datasets.data?.datasets.length ?? 0;
  const jobList = jobs.data?.jobs ?? [];
  const running = jobList.filter((j) => j.status === "running" || j.status === "queued").length;
  const succeeded = jobList.filter((j) => j.status === "succeeded").length;
  const modelList = models.data?.models ?? [];
  const alertCount = openAlerts.data?.alerts.length ?? 0;
  const deployed = active.data ?? null;

  const state = [
    nDatasets ? plural(nDatasets, "dataset") : "No datasets yet",
    running
      ? `${plural(running, "job")} in progress`
      : jobList.length
        ? `${plural(succeeded, "finished job")}`
        : "No jobs yet",
    modelList.length ? plural(modelList.length, "registered model") : "No models yet",
    deployed ? `${deployed.model} v${deployed.version} is live` : "Nothing deployed",
    deployed
      ? alertCount
        ? plural(alertCount, "open alert")
        : "No open alerts"
      : "Needs a deployment",
  ];

  let next: { text: string; to: string; cta: string } | null = null;
  if (ready && !failed) {
    if (!nDatasets)
      next = { text: "Start by uploading a CSV file.", to: "/datasets", cta: "Upload a dataset" };
    else if (!jobList.length)
      next = {
        text: "You have data. Train some models on it.",
        to: "/training/new",
        cta: "Start a training job",
      };
    else if (!modelList.length)
      next = {
        text: "Register a finished final job as a model version.",
        to: "/training",
        cta: "Open training jobs",
      };
    else if (!deployed)
      next = {
        text: "Deploy a registered version so it can answer requests.",
        to: "/deployment",
        cta: "Deploy a model",
      };
    else if (alertCount)
      next = {
        text: "Monitoring has open alerts to look at.",
        to: "/monitoring/alerts",
        cta: "Review alerts",
      };
    else
      next = {
        text: "The model is live. Try a prediction or check monitoring.",
        to: "/deployment/try",
        cta: "Try a prediction",
      };
  }

  return (
    <div className="stack-lg">
      <PageHeader
        title="AutoML Ops"
        lead="Manage the whole machine-learning lifecycle for tabular CSV data in one place: from a raw file to a monitored, deployed model. Follow the five stages from left to right."
      />
      {failed && !ready ? null : null}
      <Section
        title="The lifecycle"
        lead="Each stage links to its page. The line under each title is where your project stands right now."
      >
        {!ready ? (
          <Loading label="Loading project state" />
        ) : (
          <nav className="lifecycle" aria-label="ML lifecycle">
            {STAGES.map((s, i) => (
              <Link key={s.to} to={s.to} className="lc-step">
                <span className="lc-num">Step {s.n}</span>
                <span className="lc-title">{s.title}</span>
                <span className="lc-state">{state[i]}</span>
              </Link>
            ))}
          </nav>
        )}
        {failed && (
          <div style={{ marginTop: 16 }}>
            <Callout tone="warn" title="Some project state could not be loaded">
              <p>
                The backend did not answer every request. Check that the API is running (make api).
              </p>
            </Callout>
          </div>
        )}
      </Section>

      {next && (
        <Panel>
          <div className="row-between">
            <div>
              <h2>What to do next</h2>
              <p className="muted">{next.text}</p>
            </div>
            <Link to={next.to} className="btn primary">
              {next.cta}
            </Link>
          </div>
        </Panel>
      )}

      <Section title="What each stage does">
        <ol className="stack" style={{ margin: 0, padding: 0, listStyle: "none" }}>
          {STAGES.map((s) => (
            <li key={s.to} className="row" style={{ alignItems: "flex-start", flexWrap: "nowrap" }}>
              <span className="lc-num" style={{ width: 56, flex: "none", paddingTop: 3 }}>
                Step {s.n}
              </span>
              <div>
                <strong>
                  <Link to={s.to} style={{ color: "inherit" }}>
                    {s.title}
                  </Link>
                </strong>
                <p className="muted">{s.what}</p>
              </div>
            </li>
          ))}
        </ol>
      </Section>

      <Callout title="About the numbers you will see">
        <p>
          Training scores come from cross-validation on the training rows. The one final test result
          is shown only for the selected model and carries a note that reusing the held-out rows
          across jobs can influence choices. Drift is a reason to look, not proof that a model is
          wrong. Demo traffic is always labelled <em>simulated</em>.
        </p>
      </Callout>
    </div>
  );
}
