import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useState } from "react";
import { Link } from "react-router-dom";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { control, unwrap } from "@/api/client";
import { useActiveDeployment, useAlerts } from "@/api/queries";
import { ChartFrame } from "@/components/ChartFrame";
import { DataTable } from "@/components/DataTable";
import { Async, Empty, ErrorPanel } from "@/components/States";
import { Term } from "@/components/Term";
import { useToast } from "@/components/Toast";
import { Callout, Metric, PageHeader, Panel, Section, StatusBadge } from "@/components/ui";
import { clock, metricLabel, num, plural, signedPct, when } from "@/lib/format";
import {
  RangeSwitch,
  ruleName,
  rangeSince,
  ScopeBadge,
  ScopeSwitch,
  summaryOf,
  useReport,
  type Range,
  type Scope,
} from "./common";

export function activitySeries(
  buckets: {
    start: string;
    real: { predictions: number; labelled: number };
    simulated: { predictions: number; labelled: number };
  }[],
  scope: Scope,
  bucket: string,
) {
  // Trim long empty stretches at both ends (keeping two buckets of context) so a short burst of
  // traffic is not squeezed into one thin bar. Totals are unchanged: only empty buckets are dropped.
  const used = buckets.map((b, i) => (b[scope].predictions > 0 ? i : -1)).filter((i) => i >= 0);
  const from = used.length ? Math.max(0, used[0]! - 2) : 0;
  const to = used.length ? Math.min(buckets.length, used[used.length - 1]! + 3) : buckets.length;
  return buckets.slice(from, to).map((b) => {
    const c = b[scope];
    const d = new Date(b.start);
    return {
      start: b.start,
      label:
        bucket === "day"
          ? d.toLocaleDateString("en-GB", { day: "numeric", month: "short" })
          : clock(b.start),
      labelled: c.labelled,
      unlabelled: c.predictions - c.labelled,
      total: c.predictions,
    };
  });
}

export default function MonitoringOverview() {
  const active = useActiveDeployment();
  return (
    <div className="stack-lg">
      <PageHeader
        title="Monitoring"
        lead={
          <>
            Is the deployed model still seeing the kind of data it was trained on, and is it still
            performing? Traffic is compared with a fixed{" "}
            <Term k="reference sample">reference sample</Term>; where true outcomes are known, live
            scores are compared with the offline test score.
          </>
        }
      />
      <Async query={active} label="Loading deployment">
        {(dep) =>
          dep ? (
            <Dashboard
              id={dep.id}
              label={`${dep.model} v${dep.version}`}
              model={dep.model}
              version={dep.version}
              task={dep.task}
            />
          ) : (
            <Panel>
              <Empty
                title="Nothing is deployed"
                action={
                  <Link to="/deployment" className="btn primary">
                    Go to Deployment
                  </Link>
                }
              >
                Monitoring watches the live model. Deploy a version first.
              </Empty>
            </Panel>
          )
        }
      </Async>
    </div>
  );
}

function Dashboard({
  id,
  label,
  model,
  version,
}: {
  id: number;
  label: string;
  model: string;
  version: number;
  task: string;
}) {
  const client = useQueryClient();
  const toast = useToast();
  const [scope, setScope] = useState<Scope | null>(null);
  const [range, setRange] = useState<Range>("all");

  const totals = useQuery({
    queryKey: ["activity-all", id],
    queryFn: () =>
      unwrap(
        control.GET("/deployments/{deployment_id}/activity", {
          params: { path: { deployment_id: id }, query: { scope: "all" } },
        }),
      ),
    refetchInterval: 15_000,
  });
  const defaultScope: Scope =
    totals.data &&
    totals.data.totals.real.predictions === 0 &&
    totals.data.totals.simulated.predictions > 0
      ? "simulated"
      : "real";
  const s: Scope = scope ?? defaultScope;

  const activity = useQuery({
    queryKey: ["activity", id, s, range],
    enabled: !!totals.data,
    queryFn: () =>
      unwrap(
        control.GET("/deployments/{deployment_id}/activity", {
          params: {
            path: { deployment_id: id },
            query: { scope: s, ...(rangeSince(range) ? { since: rangeSince(range)! } : {}) },
          },
        }),
      ),
    refetchInterval: 15_000,
  });
  const performance = useQuery({
    queryKey: ["performance", id, s],
    enabled: !!totals.data,
    queryFn: () =>
      unwrap(
        control.GET("/deployments/{deployment_id}/performance", {
          params: { path: { deployment_id: id }, query: { scope: s } },
        }),
      ),
    refetchInterval: 15_000,
  });
  const status = useQuery({
    queryKey: ["monitoring", id],
    queryFn: () =>
      unwrap(
        control.GET("/deployments/{deployment_id}/monitoring", {
          params: { path: { deployment_id: id } },
        }),
      ),
    refetchInterval: 15_000,
  });
  const latest = s === "real" ? status.data?.latest_real : status.data?.latest_simulated;
  const report = useReport(latest?.id);
  const alerts = useAlerts({ deployment_id: id, scope: s }, 50);
  const live = alerts.data?.alerts.filter((a) => a.status !== "resolved") ?? [];

  const run = useMutation({
    mutationFn: () =>
      unwrap(
        control.POST("/deployments/{deployment_id}/monitor", {
          params: { path: { deployment_id: id } },
          body: { scope: s },
        }),
      ),
    onSuccess: async () => {
      toast("Monitoring run finished.");
      await client.invalidateQueries();
    },
  });

  const summary = summaryOf(report.data);
  const noReference = status.data && !status.data.reference_id;
  const both = totals.data
    ? totals.data.totals.real.predictions + totals.data.totals.simulated.predictions
    : 0;

  return (
    <div className="stack-lg">
      <div className="row-between">
        <div className="row">
          <ScopeSwitch value={s} onChange={setScope} />
          <span className="muted">
            Watching <strong>{label}</strong>
          </span>
        </div>
        <button
          type="button"
          className="btn primary"
          onClick={() => run.mutate()}
          disabled={run.isPending || !!noReference || !totals.data}
        >
          {run.isPending ? "Running…" : "Run monitoring now"}
        </button>
      </div>
      {run.isError && <ErrorPanel error={run.error} title="The monitoring run could not start" />}
      {noReference && (
        <Callout tone="warn" title="This version has no drift reference yet">
          <p>
            Create one first on the <Link to={`/models/${model}/v/${version}`}>version page</Link>,
            then run monitoring.
          </p>
        </Callout>
      )}
      {s === "simulated" && (
        <Callout title="You are looking at simulated demo traffic">
          <p>
            It was replayed from a separate monitoring file and is kept apart from real requests
            everywhere.
          </p>
        </Callout>
      )}
      {s === "real" && totals.data && totals.data.totals.real.predictions === 0 && both > 0 && (
        <Callout title="No real requests yet">
          <p>
            Only simulated traffic exists (
            {plural(totals.data.totals.simulated.predictions, "prediction")}). Switch to simulated
            traffic above, or <Link to="/deployment/try">send a prediction yourself</Link>.
          </p>
        </Callout>
      )}

      <Section
        title="Prediction activity"
        actions={<RangeSwitch value={range} onChange={setRange} />}
        lead="How many predictions the live model answered, and how many already have a true value."
      >
        <Async query={activity} label="Loading activity" rows={4}>
          {(a) => {
            const data = activitySeries(a.buckets, s, a.bucket);
            const total = a.totals[s];
            return (
              <Panel>
                <div className="stack">
                  <div className="metrics">
                    <Metric label="Predictions" value={num(total.predictions)} />
                    <Metric label="With a true value" value={num(total.labelled)} />
                    <Metric
                      label="Typical model time"
                      value={a.latency_ms.mean === null ? "n/a" : `${num(a.latency_ms.mean, 1)} ms`}
                      sub={
                        a.latency_ms.p95 !== null
                          ? `95% under ${num(a.latency_ms.p95, 1)} ms`
                          : undefined
                      }
                    />
                  </div>
                  {total.predictions === 0 ? (
                    <Empty title="No predictions in this range">
                      Change the range, or send some traffic.
                    </Empty>
                  ) : (
                    <ChartFrame
                      title={`Predictions per ${a.bucket}`}
                      summary={`${num(total.predictions)} predictions across ${data.length} ${a.bucket} buckets; ${num(total.labelled)} have a true value.`}
                      table={
                        <DataTable
                          caption="Predictions per bucket"
                          hideCaption
                          rows={data.filter((d) => d.total > 0)}
                          rowKey={(d) => d.start}
                          columns={[
                            { key: "t", header: "Time (UTC)", cell: (d) => d.label },
                            {
                              key: "n",
                              header: "Predictions",
                              numeric: true,
                              cell: (d) => num(d.total),
                            },
                            {
                              key: "l",
                              header: "With true value",
                              numeric: true,
                              cell: (d) => num(d.labelled),
                            },
                          ]}
                        />
                      }
                    >
                      <ResponsiveContainer width="100%" height={220}>
                        <BarChart
                          data={data}
                          margin={{ top: 4, right: 8, left: -8, bottom: 0 }}
                          accessibilityLayer
                        >
                          <CartesianGrid stroke="var(--chart-grid)" vertical={false} />
                          <XAxis dataKey="label" interval="preserveStartEnd" minTickGap={24} />
                          <YAxis allowDecimals={false} />
                          <Tooltip />
                          <Legend />
                          <Bar
                            dataKey="labelled"
                            name="With a true value"
                            stackId="a"
                            fill="var(--chart-3)"
                          />
                          <Bar
                            dataKey="unlabelled"
                            name="Awaiting a true value"
                            stackId="a"
                            fill="var(--chart-1)"
                          />
                        </BarChart>
                      </ResponsiveContainer>
                    </ChartFrame>
                  )}
                </div>
              </Panel>
            );
          }}
        </Async>
      </Section>

      <Section
        title="Live performance"
        lead={
          <>
            Measured only on predictions whose true outcome has been sent back (
            <Term k="feedback">feedback</Term>). It is a different set of rows from the offline
            test, so differences are a prompt to look, not proof.
          </>
        }
      >
        <Async query={performance} label="Loading performance" rows={3}>
          {(p) => (
            <Panel>
              <div className="stack">
                <div className="metrics">
                  <Metric label="Predictions" value={num(p.counts.predictions)} />
                  <Metric
                    label="With a true value"
                    value={num(p.counts.labelled)}
                    sub={`${num(p.counts.unlabelled)} still unlabelled`}
                  />
                </div>
                {p.status === "ok" && p.live ? (
                  <DataTable
                    caption="Live and offline scores"
                    hideCaption
                    rows={Object.keys(p.live.metrics).map((k) => ({ k }))}
                    rowKey={(r) => r.k}
                    columns={[
                      { key: "m", header: "Score", cell: (r) => metricLabel(r.k) },
                      {
                        key: "l",
                        header: "Live",
                        numeric: true,
                        cell: (r) => num(p.live!.metrics[r.k], 3),
                      },
                      {
                        key: "o",
                        header: "Offline test",
                        numeric: true,
                        cell: (r) => num(p.offline_reference.metrics?.[r.k], 3),
                      },
                    ]}
                  />
                ) : (
                  <Callout
                    tone="warn"
                    title={
                      p.status === "insufficient_labels"
                        ? "Not enough true values yet"
                        : p.status === "no_labels"
                          ? "No true values yet"
                          : "Nothing to score yet"
                    }
                  >
                    <p>
                      {p.notes[0] ??
                        `Live scores appear once at least ${p.min_labelled} predictions have a true value.`}{" "}
                      Record them on the{" "}
                      <Link to="/monitoring/predictions">Predictions and feedback</Link> page.
                    </p>
                  </Callout>
                )}
              </div>
            </Panel>
          )}
        </Async>
      </Section>

      <Section
        title="Drift"
        lead={
          <>
            <Term k="drift">Drift</Term> means the data or the answers now look different from the
            training sample. The latest report covers the most recent <Term k="window">window</Term>{" "}
            of predictions.
          </>
        }
        actions={<Link to="/monitoring/reports">All reports</Link>}
      >
        {!latest ? (
          <Panel>
            <Empty title="No monitoring report yet">
              Press “Run monitoring now” to compare recent traffic with the reference.
            </Empty>
          </Panel>
        ) : latest.status !== "ok" || !summary ? (
          <Callout
            tone="warn"
            title={
              latest.status === "insufficient_data"
                ? "Not enough data for a drift verdict"
                : "The last run did not finish"
            }
          >
            <p>{summary?.reason ?? latest.error ?? "See the report for details."}</p>
            <p>
              <Link to={`/monitoring/reports/${latest.id}`}>Open report #{latest.id}</Link>
            </p>
          </Callout>
        ) : (
          <Panel>
            <div className="stack">
              <div className="row-between">
                <span className="row">
                  <ScopeBadge scope={latest.scope} />
                  <span className="muted">
                    Report #{latest.id} · {when(latest.finished_at)} ·{" "}
                    {plural(latest.window_rows, "prediction")}
                  </span>
                </span>
                <Link to={`/monitoring/reports/${latest.id}`} className="btn small">
                  Open the full report
                </Link>
              </div>
              <div className="metrics">
                <Metric
                  label="Drifted features"
                  value={`${summary.dataset_drift?.drifted_features ?? 0} of ${summary.dataset_drift?.evaluated_features ?? 0}`}
                  sub={`alert at ${num((summary.dataset_drift?.threshold ?? 0.5) * 100)}% or more`}
                />
                <Metric
                  label="Predictions shifted?"
                  value={summary.prediction?.drifted ? "Yes" : "No"}
                />
                <Metric
                  label="Performance"
                  value={
                    summary.performance?.verdict
                      ? summary.performance.verdict
                      : summary.performance?.evaluated
                        ? "evaluated"
                        : "not evaluated"
                  }
                  sub={
                    summary.performance?.relative_worsening !== undefined
                      ? `${signedPct(summary.performance.relative_worsening)} vs offline`
                      : undefined
                  }
                />
              </div>
            </div>
          </Panel>
        )}
      </Section>

      <Section title="Open alerts" actions={<Link to="/monitoring/alerts">All alerts</Link>}>
        <Panel flush={live.length > 0}>
          {live.length === 0 ? (
            <Empty title="No open alerts">
              Alerts open when a rule fires and close by themselves when the condition clears.
            </Empty>
          ) : (
            <DataTable
              caption="Open alerts"
              hideCaption
              rows={live}
              rowKey={(a) => a.id}
              columns={[
                {
                  key: "r",
                  header: "Alert",
                  cell: (a) => (
                    <Link className="row-link" to={`/monitoring/alerts/${a.id}`}>
                      {ruleName(a.rule)}
                    </Link>
                  ),
                },
                { key: "s", header: "Severity", cell: (a) => <StatusBadge status={a.severity} /> },
                { key: "t", header: "Status", cell: (a) => <StatusBadge status={a.status} /> },
                { key: "o", header: "Seen", numeric: true, cell: (a) => `${a.occurrences}×` },
                { key: "u", header: "Last update", cell: (a) => when(a.updated_at) },
              ]}
            />
          )}
        </Panel>
      </Section>
    </div>
  );
}
