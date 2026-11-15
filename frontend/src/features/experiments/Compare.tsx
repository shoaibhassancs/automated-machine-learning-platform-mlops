import { useQueries } from "@tanstack/react-query";
import { Link, useSearchParams } from "react-router-dom";
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
import type { JobResult } from "@/api/shapes";
import { ChartFrame } from "@/components/ChartFrame";
import { DataTable } from "@/components/DataTable";
import { ErrorPanel, Loading } from "@/components/States";
import { Callout, PageHeader, Panel, Section } from "@/components/ui";
import { metricLabel, num } from "@/lib/format";
import { jobKindLabel } from "../training/labels";

const COLORS = [
  "var(--chart-1)",
  "var(--chart-2)",
  "var(--chart-3)",
  "var(--chart-4)",
  "var(--border-strong)",
];

export interface CompareRow {
  algorithm: string;
  label: string;
  scores: Record<number, { mean: number; std: number } | undefined>;
}

/** Algorithm-by-job matrix of the selection metric (cross-validation means). */
export function compareMatrix(jobs: { id: number; result: JobResult | null }[]): CompareRow[] {
  const rows = new Map<string, CompareRow>();
  for (const j of jobs) {
    for (const r of j.result?.selection?.ranking ?? []) {
      const row = rows.get(r.algorithm) ?? { algorithm: r.algorithm, label: r.label, scores: {} };
      row.scores[j.id] = { mean: r.metric_mean, std: r.metric_std };
      rows.set(r.algorithm, row);
    }
  }
  return [...rows.values()];
}

export default function Compare() {
  const [search] = useSearchParams();
  const ids = (search.get("jobs") ?? "")
    .split(",")
    .map(Number)
    .filter((n) => Number.isInteger(n) && n > 0)
    .slice(0, 5);
  const queries = useQueries({
    queries: ids.map((id) => ({
      queryKey: ["job", id],
      queryFn: () => unwrap(control.GET("/jobs/{job_id}", { params: { path: { job_id: id } } })),
    })),
  });
  const loading = queries.some((q) => q.isPending);
  const failed = queries.find((q) => q.isError);
  const jobs = queries.flatMap((q) => (q.data ? [q.data] : []));
  const parsed = jobs.map((j) => ({ job: j, result: (j.result ?? null) as JobResult | null }));
  const metrics = new Set(parsed.map((p) => p.result?.selection?.metric).filter(Boolean));
  const tasks = new Set(parsed.map((p) => p.job.config.task));
  const comparable = metrics.size === 1 && tasks.size === 1;
  const metric = [...metrics][0] ?? "";
  const matrix = compareMatrix(parsed.map((p) => ({ id: p.job.id, result: p.result })));
  const chartData = matrix.map((r) => ({
    name: r.label,
    ...Object.fromEntries(ids.map((id) => [`job${id}`, r.scores[id]?.mean ?? null])),
  }));

  return (
    <div className="stack-lg">
      <PageHeader
        crumbs={[{ to: "/experiments", label: "Experiments" }, { label: "Compare" }]}
        title="Compare runs"
        lead="The same algorithms across different jobs, on cross-validation scores only."
      />
      {ids.length < 2 ? (
        <Callout tone="warn">
          <p>
            Choose at least two jobs on the <Link to="/experiments">experiment history</Link> page.
          </p>
        </Callout>
      ) : loading ? (
        <Loading label="Loading jobs" />
      ) : failed ? (
        <ErrorPanel error={failed.error} />
      ) : (
        <>
          {!comparable && (
            <Callout tone="warn" title="These jobs are not directly comparable">
              <p>
                They use different tasks or selection metrics, so scores are shown side by side but
                should not be ranked against each other.
              </p>
            </Callout>
          )}
          <Section title="The jobs">
            <Panel flush>
              <DataTable
                caption="Jobs being compared"
                hideCaption
                rows={parsed}
                rowKey={(p) => p.job.id}
                columns={[
                  {
                    key: "id",
                    header: "Job",
                    cell: (p) => (
                      <Link className="row-link" to={`/training/${p.job.id}`}>
                        #{p.job.id}
                      </Link>
                    ),
                  },
                  { key: "d", header: "Dataset", cell: (p) => p.job.dataset },
                  { key: "k", header: "Type", cell: (p) => jobKindLabel(p.job.kind) },
                  {
                    key: "w",
                    header: "Winner",
                    cell: (p) => p.result?.selection?.selected?.label ?? "none",
                  },
                  {
                    key: "m",
                    header: "Metric",
                    cell: (p) => metricLabel(p.result?.selection?.metric ?? "n/a"),
                  },
                  {
                    key: "s",
                    header: "Settings",
                    cell: (p) =>
                      `${p.job.config.settings.folds} folds, seed ${p.job.config.settings.seed}`,
                  },
                ]}
              />
            </Panel>
          </Section>
          <Section
            title={`${metric ? metricLabel(metric) : "Score"} by algorithm and job`}
            lead="Cross-validation means. A blank cell means that algorithm was not part of that job."
          >
            <Panel>
              <ChartFrame
                title="Cross-validation score"
                summary={`${matrix.length} algorithms across ${parsed.length} jobs.`}
                table={
                  <DataTable
                    caption="Scores by algorithm and job"
                    hideCaption
                    rows={matrix}
                    rowKey={(r) => r.algorithm}
                    columns={[
                      { key: "a", header: "Algorithm", cell: (r) => r.label },
                      ...ids.map((id) => ({
                        key: String(id),
                        header: `Job #${id}`,
                        numeric: true,
                        cell: (r: CompareRow) =>
                          r.scores[id]
                            ? `${num(r.scores[id]!.mean, 4)} ± ${num(r.scores[id]!.std, 4)}`
                            : "",
                      })),
                    ]}
                  />
                }
              >
                <ResponsiveContainer width="100%" height={300}>
                  <BarChart
                    data={chartData}
                    margin={{ top: 8, right: 16, left: 0, bottom: 8 }}
                    accessibilityLayer
                  >
                    <CartesianGrid stroke="var(--chart-grid)" vertical={false} />
                    <XAxis dataKey="name" />
                    <YAxis />
                    <Tooltip formatter={(v) => num(Number(v), 4)} />
                    <Legend />
                    {ids.map((id, i) => (
                      <Bar
                        key={id}
                        dataKey={`job${id}`}
                        name={`Job #${id}`}
                        fill={COLORS[i % COLORS.length]}
                        radius={[3, 3, 0, 0]}
                      />
                    ))}
                  </BarChart>
                </ResponsiveContainer>
              </ChartFrame>
            </Panel>
          </Section>
        </>
      )}
    </div>
  );
}
