import { useParams } from "react-router-dom";
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
import type { FeatureDrift, ReportSummary } from "@/api/shapes";
import { ChartFrame } from "@/components/ChartFrame";
import { DataTable } from "@/components/DataTable";
import { Async } from "@/components/States";
import { Term } from "@/components/Term";
import {
  Badge,
  BarCell,
  Callout,
  KV,
  Metric,
  PageHeader,
  Panel,
  Section,
  StatusBadge,
} from "@/components/ui";
import { metricLabel, num, pct, signedPct, when } from "@/lib/format";
import { RULES, ScopeBadge, summaryOf, useReport } from "./common";

export function predictionBars(p: NonNullable<ReportSummary["prediction"]>) {
  const names = [...new Set([...Object.keys(p.reference), ...Object.keys(p.current)])];
  return names.map((k) => ({
    name: k,
    reference: p.reference[k] ?? 0,
    current: p.current[k] ?? 0,
  }));
}

function FeatureTable({ features }: { features: FeatureDrift[] }) {
  return (
    <DataTable
      caption="Drift per input column"
      hideCaption
      rows={features}
      rowKey={(f) => f.name}
      columns={[
        { key: "n", header: "Column", cell: (f) => <strong>{f.name}</strong> },
        {
          key: "t",
          header: "Test",
          cell: (f) =>
            f.method === "ks" ? (
              <Term k="KS test">KS test</Term>
            ) : (
              <Term k="Jensen-Shannon distance">Jensen-Shannon</Term>
            ),
        },
        {
          key: "s",
          header: "Difference (vs threshold)",
          cell: (f) =>
            f.status === "ok" && f.statistic !== null ? (
              <BarCell
                value={f.method === "js" ? f.statistic : f.statistic}
                max={f.method === "js" ? Math.max(0.3, f.statistic) : Math.max(0.5, f.statistic)}
                mark={f.method === "js" ? f.threshold : undefined}
                hot={f.drifted}
                text={num(f.statistic, 3)}
              />
            ) : (
              <span className="muted">no data</span>
            ),
        },
        {
          key: "p",
          header: <Term k="p-value">p-value</Term>,
          numeric: true,
          cell: (f) => (f.p_value === null ? "n/a" : num(f.p_value, 4)),
        },
        {
          key: "d",
          header: "Verdict",
          cell: (f) =>
            f.status !== "ok" ? (
              <Badge>not evaluated</Badge>
            ) : f.drifted ? (
              <Badge tone="warn">drifted</Badge>
            ) : (
              <Badge tone="ok">no drift</Badge>
            ),
        },
      ]}
    />
  );
}

function PredictionPanel({ p }: { p: NonNullable<ReportSummary["prediction"]> }) {
  const classes = p.method === "js";
  return (
    <Panel>
      <div className="stack">
        <div className="row">
          {p.drifted ? <Badge tone="warn">shifted</Badge> : <Badge tone="ok">no shift</Badge>}
          <span className="muted">
            {p.method === "js" ? "Jensen-Shannon distance" : "KS statistic"} {num(p.statistic, 3)}
            {p.p_value !== null ? `, p ${num(p.p_value, 4)}` : ""} (threshold {num(p.threshold)})
          </span>
        </div>
        {classes ? (
          <ChartFrame
            title="Share of each predicted class"
            help="Training sample against recent traffic. Bars that differ a lot mean the model's answers have changed."
            summary={`Predicted class shares, reference against current.`}
            table={
              <DataTable
                caption="Predicted class shares"
                hideCaption
                rows={predictionBars(p)}
                rowKey={(r) => r.name}
                columns={[
                  { key: "n", header: "Class", cell: (r) => r.name },
                  { key: "r", header: "Reference", numeric: true, cell: (r) => pct(r.reference) },
                  { key: "c", header: "Current", numeric: true, cell: (r) => pct(r.current) },
                ]}
              />
            }
          >
            <ResponsiveContainer width="100%" height={240}>
              <BarChart
                data={predictionBars(p)}
                margin={{ top: 4, right: 8, left: -8, bottom: 0 }}
                accessibilityLayer
              >
                <CartesianGrid stroke="var(--chart-grid)" vertical={false} />
                <XAxis dataKey="name" />
                <YAxis tickFormatter={(v: number) => `${Math.round(v * 100)}%`} />
                <Tooltip formatter={(v) => pct(Number(v))} />
                <Legend />
                <Bar dataKey="reference" name="Training sample" fill="var(--chart-1)" />
                <Bar dataKey="current" name="Recent traffic" fill="var(--chart-2)" />
              </BarChart>
            </ResponsiveContainer>
          </ChartFrame>
        ) : (
          <DataTable
            caption="Predicted values: training sample against recent traffic"
            rows={["mean", "std", "min", "max"].map((k) => ({ k }))}
            rowKey={(r) => r.k}
            columns={[
              { key: "k", header: "Statistic", cell: (r) => r.k },
              {
                key: "r",
                header: "Training sample",
                numeric: true,
                cell: (r) => num(p.reference[r.k], 3),
              },
              {
                key: "c",
                header: "Recent traffic",
                numeric: true,
                cell: (r) => num(p.current[r.k], 3),
              },
            ]}
          />
        )}
      </div>
    </Panel>
  );
}

export default function ReportDetail() {
  const { id = "" } = useParams();
  const query = useReport(Number(id));
  return (
    <div className="stack-lg">
      <Async query={query} label="Loading report">
        {(r) => {
          const s = summaryOf(r);
          return (
            <>
              <PageHeader
                crumbs={[
                  { to: "/monitoring", label: "Monitoring" },
                  { to: "/monitoring/reports", label: "Reports" },
                  { label: `#${r.id}` },
                ]}
                title={`Monitoring report #${r.id}`}
                lead={`Compares the ${num(r.window_rows)} most recent predictions with the training reference sample.`}
                actions={
                  <>
                    <ScopeBadge scope={r.scope} />
                    <StatusBadge status={r.status} />
                  </>
                }
              />
              {r.status !== "ok" || !s ? (
                <Callout
                  tone="warn"
                  title={
                    r.status === "insufficient_data"
                      ? "Not enough data for a verdict"
                      : "This run failed"
                  }
                >
                  <p>{s?.reason ?? r.error ?? "No further detail was recorded."}</p>
                </Callout>
              ) : (
                <>
                  <Callout title="How to read this report">
                    <p>
                      Drift means recent data or answers look different from the training sample. It
                      is a reason to look, not proof that the model is wrong.{" "}
                      {s.traffic.note.charAt(0).toUpperCase() + s.traffic.note.slice(1)}.
                    </p>
                  </Callout>
                  <Section
                    title="Rules"
                    lead="Each rule gives one verdict. An alert is opened or kept open when a rule fires, and closes itself when it is evaluated and clear."
                  >
                    <Panel flush>
                      <DataTable
                        caption="Alert rules"
                        hideCaption
                        rows={s.rules}
                        rowKey={(x) => x.rule}
                        columns={[
                          {
                            key: "r",
                            header: "Rule",
                            cell: (x) => (
                              <>
                                <strong>{RULES[x.rule]?.name ?? x.rule}</strong>
                                <div className="muted">{RULES[x.rule]?.explain}</div>
                              </>
                            ),
                          },
                          {
                            key: "v",
                            header: "Verdict",
                            cell: (x) =>
                              !x.evaluated ? (
                                <Badge>not evaluated</Badge>
                              ) : x.triggered ? (
                                <Badge tone="bad">fired</Badge>
                              ) : (
                                <Badge tone="ok">clear</Badge>
                              ),
                          },
                        ]}
                      />
                    </Panel>
                  </Section>
                  {s.features && (
                    <Section
                      title="Input columns"
                      lead={`${s.dataset_drift?.drifted_features ?? 0} of ${s.dataset_drift?.evaluated_features ?? 0} columns drifted. Numeric columns use the KS test, categorical columns the Jensen-Shannon distance.`}
                    >
                      <Panel flush>
                        <FeatureTable features={s.features} />
                      </Panel>
                    </Section>
                  )}
                  {s.prediction && (
                    <Section title="Predictions">
                      <PredictionPanel p={s.prediction} />
                    </Section>
                  )}
                  {s.performance && (
                    <Section title="Performance on labelled rows">
                      <Panel>
                        {s.performance.evaluated && s.performance.live !== undefined ? (
                          <div className="stack">
                            <div className="metrics">
                              <Metric
                                label={`Live ${metricLabel(s.performance.metric ?? "")}`}
                                value={num(s.performance.live, 3)}
                                sub={`${num(s.performance.window_labelled)} labelled rows in this window`}
                              />
                              <Metric label="Offline test" value={num(s.performance.offline, 3)} />
                              <Metric
                                label="Change"
                                value={signedPct(s.performance.relative_worsening)}
                                sub="positive = worse"
                              />
                            </div>
                            <p>
                              Verdict: <strong>{s.performance.verdict}</strong>.{" "}
                              {s.performance.verdict === "triggered"
                                ? `Worse than the ${pct(s.performance.threshold, 0)} threshold, and a bootstrap (${s.performance.interval?.resamples} resamples) says it is unlikely to be chance.`
                                : s.performance.verdict === "clear"
                                  ? "Not worse than the threshold."
                                  : ""}
                            </p>
                          </div>
                        ) : (
                          <p className="muted">
                            {s.performance.reason ??
                              `Not evaluated: fewer than ${r.thresholds && (r.thresholds as { min_labelled?: number }).min_labelled} labelled predictions in this window, or the result is within sampling noise.`}{" "}
                            {s.performance.verdict === "inconclusive"
                              ? "The result was worse than the threshold but within sampling noise, so no verdict was made and any open alert is left as it is."
                              : ""}
                          </p>
                        )}
                      </Panel>
                    </Section>
                  )}
                </>
              )}
              <Section title="Run details">
                <Panel>
                  <KV
                    items={[
                      ["Run", when(r.finished_at)],
                      [
                        "Predictions looked at",
                        `${num(r.window_rows)}${r.window_start ? `, from ${when(r.window_start)} to ${when(r.window_end)}` : ""}`,
                      ],
                      [
                        "Thresholds used",
                        <span key="t" className="mono">
                          {JSON.stringify(r.thresholds)}
                        </span>,
                      ],
                    ]}
                  />
                </Panel>
              </Section>
            </>
          );
        }}
      </Async>
    </div>
  );
}
