import { useQuery } from "@tanstack/react-query";
import { useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { control, unwrap } from "@/api/client";
import { useJobs } from "@/api/queries";
import type { JobResult } from "@/api/shapes";
import { DataTable } from "@/components/DataTable";
import { Async, Empty } from "@/components/States";
import { Callout, PageHeader, Panel, StatusBadge } from "@/components/ui";
import { metricLabel, num, when } from "@/lib/format";
import { jobKindLabel } from "../training/labels";

const MAX_COMPARE = 5;

export interface JobRow {
  id: number;
  dataset: string;
  kind: string;
  created_at: string;
  selected: string | null;
  metric: string | null;
  score: number | null;
}

export default function Experiments() {
  const jobs = useJobs({ status: "succeeded" }, 100);
  const [picked, setPicked] = useState<number[]>([]);
  const navigate = useNavigate();
  const details = useQuery({
    queryKey: ["experiment-details", jobs.data?.jobs.map((j) => j.id).join(",")],
    enabled: !!jobs.data,
    queryFn: async () => {
      const out: Record<number, JobRow> = {};
      for (const j of jobs.data!.jobs) {
        const d = await unwrap(
          control.GET("/jobs/{job_id}", { params: { path: { job_id: j.id } } }),
        );
        const r = (d.result ?? null) as JobResult | null;
        const top = r?.selection?.ranking[0];
        out[j.id] = {
          id: j.id,
          dataset: j.dataset,
          kind: j.kind,
          created_at: j.created_at,
          selected: r?.selection?.selected?.label ?? null,
          metric: r?.selection?.metric ?? null,
          score: top?.metric_mean ?? null,
        };
      }
      return out;
    },
  });
  const toggle = (id: number) =>
    setPicked((cur) =>
      cur.includes(id)
        ? cur.filter((x) => x !== id)
        : cur.length < MAX_COMPARE
          ? [...cur, id]
          : cur,
    );

  return (
    <div className="stack-lg">
      <PageHeader
        title="Experiment history"
        lead="Every finished job with its winning algorithm and score. Tick two or more jobs to compare them side by side."
        actions={
          <button
            type="button"
            className="btn primary"
            disabled={picked.length < 2}
            onClick={() => navigate(`/experiments/compare?jobs=${picked.join(",")}`)}
          >
            Compare {picked.length >= 2 ? picked.length : ""} selected
          </button>
        }
      />
      <Callout title="Comparisons use cross-validation scores only">
        <p>
          Scores are averages over folds on the training rows. Final test results are not used to
          compare or rank runs.
        </p>
      </Callout>
      <Async query={jobs} label="Loading experiments">
        {(data) =>
          data.jobs.length === 0 ? (
            <Panel>
              <Empty
                title="No finished jobs yet"
                action={
                  <Link className="btn primary" to="/training/new">
                    Start a training job
                  </Link>
                }
              >
                Finished jobs appear here so you can compare them.
              </Empty>
            </Panel>
          ) : (
            <Panel flush>
              <DataTable
                caption="Finished jobs"
                hideCaption
                rows={data.jobs}
                rowKey={(j) => j.id}
                columns={[
                  {
                    key: "pick",
                    header: <span className="sr-only">Select to compare</span>,
                    cell: (j) => (
                      <input
                        type="checkbox"
                        aria-label={`Compare job ${j.id}`}
                        checked={picked.includes(j.id)}
                        disabled={!picked.includes(j.id) && picked.length >= MAX_COMPARE}
                        onChange={() => toggle(j.id)}
                      />
                    ),
                  },
                  {
                    key: "id",
                    header: "Job",
                    cell: (j) => (
                      <Link className="row-link" to={`/training/${j.id}`}>
                        #{j.id}
                      </Link>
                    ),
                  },
                  { key: "d", header: "Dataset", cell: (j) => j.dataset },
                  { key: "k", header: "Type", cell: (j) => jobKindLabel(j.kind) },
                  {
                    key: "w",
                    header: "Winner",
                    cell: (j) =>
                      details.data?.[j.id]?.selected ?? (details.isPending ? "…" : "n/a"),
                  },
                  {
                    key: "m",
                    header: "Best CV score",
                    numeric: true,
                    cell: (j) => {
                      const d = details.data?.[j.id];
                      return d?.score !== null && d?.score !== undefined
                        ? `${num(d.score, 4)} (${metricLabel(d.metric ?? "")})`
                        : "n/a";
                    },
                  },
                  { key: "s", header: "Status", cell: (j) => <StatusBadge status={j.status} /> },
                  { key: "c", header: "Created", cell: (j) => when(j.created_at) },
                ]}
              />
            </Panel>
          )
        }
      </Async>
    </div>
  );
}
